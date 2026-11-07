"""Integration smoke test: imports app.py with stubbed streamlit/nba_api/plotly, runs a real
evaluation on synthetic data, and renders the Final Assessment. Run: python -m unittest -v test_integration"""
from __future__ import annotations

import sys
import types
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


class FakeSt(types.ModuleType):
    """Recording stand-in for streamlit (the real package isn't installed in the test sandbox)."""

    def __init__(self) -> None:
        super().__init__("streamlit")
        self.calls: list = []
        self.session_state: dict = {}

    def cache_data(self, *a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)

    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [MagicMock() for _ in range(n)]

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: self.calls.append((name, a, k))


def load_app():
    sys.modules["streamlit"] = FakeSt()
    for name in ("plotly", "plotly.graph_objects", "nba_api", "nba_api.stats",
                 "nba_api.stats.static", "nba_api.stats.endpoints"):
        sys.modules[name] = MagicMock()
    sys.modules.pop("app", None)
    import app
    return app


def synthetic_logs(app, mean=28.0):
    rng = np.random.default_rng(1)
    rows = []
    for season, n, start in (("prior", 40, "2025-12-01"), ("current", 12, "2026-10-25")):
        for d in pd.date_range(start, periods=n, freq="2D"):
            row = {"DATE": d.strftime("%Y-%m-%d"), "OPPONENT": "vs. BOS", "MINUTES": 34.0 + rng.normal(0, 1), "SEASON": season}
            for col in app.STAT_COL.values():
                row[col] = max(0.0, mean + rng.normal(0, 4)) if col == "POINTS" else 5.0 + rng.normal(0, 1)
            rows.append(row)
    return pd.DataFrame(rows)


ALLOWED = lambda rank_value: pd.DataFrame(  # BOS softest when rank_value high
    {c: [rank_value] + [1.0] * 29 for c in ("POINTS", "REBOUNDS", "ASSISTS", "3-POINTERS MADE", "PTS + REB + AST",
                                          "STEALS", "BLOCKS", "PTS + AST", "PTS + REB", "REB + AST")}
    | {"G": [20] * 30}, index=["BOS"] + [f"T{i:02d}" for i in range(29)])

CLEAN_INJ = {"minutes_extra": 0.0, "usage_mult": 1.0, "table": [], "notes": [], "usg": 28.0, "team": "Dallas",
             "listed": [], "feed_ok": True, "checked": True, "self_status": ""}


class TestIntegration(unittest.TestCase):
    def run_eval(self, app, line=22.5, soft=True, spread=6.5, inj=None, opponent="Boston Celtics", over_odds=-120, under_odds=100):
        allowed = ALLOWED(1.15 if soft else 0.85)
        patches = [
            patch.object(app, "fetch_player_logs", return_value=(synthetic_logs(app), None)),
            patch.object(app, "load_team_scoring", return_value=pd.DataFrame()),
            patch.object(app, "opp_multiplier", return_value=(1.0, None)),
            patch.object(app, "load_team_game_dates", return_value=[]),
            patch.object(app, "load_team_abbr_meta", return_value=[("2", "Boston Celtics", "BOS")]),
            patch.object(app, "load_opp_allowed", return_value=allowed),
            patch.object(app, "load_opp_allowed_recent", return_value=allowed),
            patch.object(app, "injury_adjustment", return_value=inj or dict(CLEAN_INJ)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return app.evaluate_nba_prop("Test Player", "Points", line, over_odds, under_odds, opponent,
                                     date(2026, 11, 7), None, None, None, spread)

    def test_six_point_present_and_sharp_when_everything_lines_up(self):
        app = load_app()
        res = self.run_eval(app)
        self.assertNotIn("error", res)
        six = res["six_point"]
        self.assertEqual(six["best"], "Over")
        best = six["Over"]
        self.assertEqual(len(best["checks"]), 6)
        self.assertEqual([c["passed"] for c in best["checks"]][:2], [True, True])   # baby line, edge
        self.assertTrue(best["checks"][3]["passed"])                                 # BOS rank 1 of 30, recent too
        self.assertTrue(best["checks"][4]["passed"])
        self.assertTrue(best["checks"][5]["passed"])
        # implied chance: -120/+100 devig favors Over (>50%) -> full house
        self.assertEqual((best["score"], best["tier"]), (6, "SHARP"))
        self.assertEqual(set(six), {"Over", "Under", "best"})

    def test_downgrades(self):
        app = load_app()
        res = self.run_eval(app, spread=13.5)
        self.assertEqual(res["six_point"][res["six_point"]["best"]]["tier"], "LEAN")
        res = self.run_eval(app, spread=None)
        game = [c for c in res["six_point"]["Over"]["checks"] if c["key"] == "game_script"][0]
        self.assertFalse(game["verified"])
        res = self.run_eval(app, opponent=None)
        d = [c for c in res["six_point"]["Over"]["checks"] if c["key"] == "defensive_rank"][0]
        self.assertFalse(d["passed"])
        self.assertFalse(d["verified"])

    def test_injury_flag_fails_check_five(self):
        app = load_app()
        inj = dict(CLEAN_INJ, listed=[{"name": "Co-Star", "status": "OUT", "p_out": 1.0, "mpg": 31.0}])
        res = self.run_eval(app, inj=inj)
        c = [c for c in res["six_point"]["Over"]["checks"] if c["key"] == "injury"][0]
        self.assertFalse(c["passed"])
        self.assertIn("Co-Star", c["detail"])

    def test_tough_defense_fails_over_rank_check(self):
        app = load_app()
        res = self.run_eval(app, soft=False)
        d = [c for c in res["six_point"]["Over"]["checks"] if c["key"] == "defensive_rank"][0]
        self.assertFalse(d["passed"])

    def test_existing_fields_untouched(self):
        app = load_app()
        res = self.run_eval(app)
        for key in ("verdict", "best_side", "ev_pct", "context_checks", "sides", "opp_detail", "injury", "mean", "median"):
            self.assertIn(key, res)
        self.assertIn("recent_rank", res["opp_detail"])

    def test_render_final_assessment_writes_all_six_rows(self):
        app = load_app()
        res = self.run_eval(app)
        res["uid"] = "x"
        app.st.calls.clear()
        app.render_final_assessment(res)
        md = [c[1][0] for c in app.st.calls if c[0] == "markdown"]
        rows = [m for m in md if m[:1] in "✅❌⚪"]
        self.assertEqual(len(rows), 6)
        self.assertTrue(any("Final Assessment" in m for m in md))

    def test_fetch_game_spread_uses_event_endpoint_and_handles_errors(self):
        app = load_app()
        ok = MagicMock(status_code=200)
        ok.json.return_value = {"bookmakers": [{"markets": [{"key": "spreads", "outcomes": [
            {"name": "A", "point": -4.5}, {"name": "B", "point": 4.5}]}]}]}
        with patch.object(app.requests, "get", return_value=ok) as g:
            self.assertEqual(app.fetch_game_spread("evt", "KEY"), (4.5, None))
            self.assertIn("/events/evt/odds", g.call_args[0][0])
            self.assertEqual(g.call_args.kwargs["params"]["markets"], "spreads")
        with patch.object(app.requests, "get", return_value=MagicMock(status_code=422)):
            spread, err = app.fetch_game_spread("evt", "KEY")
            self.assertIsNone(spread)
            self.assertIn("422", err)
        with patch.object(app.requests, "get", side_effect=RuntimeError("boom")):
            self.assertIsNone(app.fetch_game_spread("evt", "KEY")[0])


if __name__ == "__main__":
    unittest.main()
