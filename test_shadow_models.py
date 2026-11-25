"""Tests for the shot-based shadow models (Points and 3-Pointers Made)."""
import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

import shadow_models as sm
from test_integration import CLEAN_INJ, ALLOWED, load_app


class TestMath(unittest.TestCase):
    def test_shrunk_pct(self):
        self.assertAlmostEqual(sm.shrunk_pct(0, 0, 0, 0, "3")[0], 0.36)
        self.assertAlmostEqual(sm.shrunk_pct(0, 0, 0, 0, "FT")[0], 0.78)
        hot, _ = sm.shrunk_pct(15, 25, 120, 330, "3")                       # 60% on 25 tries, 36% last year
        self.assertTrue(0.36 < hot < 0.42)                                   # a hot stretch barely moves it
        steady, n = sm.shrunk_pct(150, 400, 160, 400, "3")
        self.assertAlmostEqual(steady, (150 + 250 * ((160 + 36) / 500)) / 650)
        self.assertEqual(n, 650)
        two, _ = sm.shrunk_pct(60, 100, 0, 0, "2")                           # twos settle faster (K 150)
        self.assertAlmostEqual(two, (60 + 150 * 0.54) / 250)

    def test_threes_draws(self):
        att = np.full(20000, 6.0)
        m = sm.makes_from_attempts(att, 0.40, 10000)
        self.assertAlmostEqual(m.mean(), 2.4, places=1)
        self.assertGreater(sm.makes_from_attempts(att, 0.40, 50).std(), m.std())     # unsure % -> wider
        s = sm.summarize(m, 2.5, attempts=6.0)
        self.assertAlmostEqual(s["p_over"] + s["p_under"], 100.0)
        self.assertEqual(s["attempts"], 6.0)

    def test_points_mean_and_three_heavy_is_wider(self):
        mins = np.full(40000, 30.0)
        pct = {"2": (0.54, 1e6), "3": (0.36, 1e6), "FT": (0.80, 1e6)}
        inside = sm.simulate_points(mins, 30.0, 12.0, 1.0, 4.0, 0.15, pct)["points"]
        deep = sm.simulate_points(mins, 30.0, 12.0, 11.0, 4.0, 0.15, pct)["points"]
        self.assertAlmostEqual(inside.mean(), 2 * 11 * 0.54 + 3 * 1 * 0.36 + 4 * 0.80, delta=0.2)
        self.assertAlmostEqual(deep.mean(), 2 * 1 * 0.54 + 3 * 11 * 0.36 + 4 * 0.80, delta=0.2)
        self.assertGreater(deep.std() / deep.mean(), inside.std() / inside.mean())    # threes make scoring lumpier
        self.assertAlmostEqual(sm.usage_cv(0.30, 15.0), np.sqrt(0.09 - 1 / 15))
        self.assertEqual(sm.usage_cv(0.1, 15.0), sm.CV_U_MIN)


class TestCompare(unittest.TestCase):
    def frame(self, stat, n=200):
        rng = np.random.default_rng(0)
        actual = rng.poisson(12, n).astype(float)
        return pd.DataFrame({"STATUS": "Graded", "STAT": stat, "LINE": 12.5, "ACTUAL": actual, "P_OVER": 45.0,
                             "SHADOW_P_OVER": np.where(actual > 12.5, 70.0, 30.0), "PM_OVER": 44.0, "MU": 12.0, "SHADOW_MU": 12.0})

    def test_points_and_threes_kept_apart(self):
        d = pd.concat([self.frame("Points"), self.frame("Pts + Reb", 50), self.frame("3-Pointers Made", 30)], ignore_index=True)
        pts = sm.compare(d, "POINTS").set_index("MODEL")
        self.assertEqual(pts.at["Current model", "PROPS"], 200)                     # combos left out
        self.assertLess(pts.at["Shot-based shadow", "LOG_LOSS"], pts.at["Current model", "LOG_LOSS"])
        self.assertEqual(sm.compare(d, "3-POINT")["PROPS"].max(), 30)
        calls = []
        st = type("St", (), {})()
        for nm in ("markdown", "caption", "dataframe", "success", "info"):
            setattr(st, nm, (lambda n_: (lambda *a, **k: calls.append(n_)))(nm))
        sm.render(st, d)
        self.assertIn("success", calls)                                              # points: 200 props, shadow better
        self.assertIn("caption", calls)                                              # threes: only 30 so far
        self.assertIsNone(sm.compare(d.drop(columns=["SHADOW_P_OVER"]), "POINTS"))


class TestInApp(unittest.TestCase):
    def logs(self, app, three_share=0.4):
        rng = np.random.default_rng(1)
        rows = []
        for season, n, start in (("prior", 40, "2025-12-01"), ("current", 12, "2026-10-25")):
            for d in pd.date_range(start, periods=n, freq="2D"):
                fga = float(rng.poisson(14))
                tpa = float(rng.binomial(int(fga), three_share))
                fta = float(rng.poisson(4))
                tpm, twm, ftm = float(rng.binomial(int(tpa), 0.37)), float(rng.binomial(int(fga - tpa), 0.53)), float(rng.binomial(int(fta), 0.8))
                row = {"DATE": d.strftime("%Y-%m-%d"), "OPPONENT": "vs. BOS", "MINUTES": 32.0, "SEASON": season}
                for col in app.STAT_COL.values():
                    row[col] = 5.0
                row.update({"FGA": fga, "3PA": tpa, "FTA": fta, "FGM": tpm + twm, "FTM": ftm, "3-POINTERS MADE": tpm,
                            "POINTS": 2 * twm + 3 * tpm + ftm})
                rows.append(row)
        return pd.DataFrame(rows)

    def evaluate(self, app, logs, stat, line):
        allowed = ALLOWED(1.0)
        with patch.object(app, "fetch_player_logs", return_value=(logs, None)), \
             patch.object(app, "load_team_scoring", return_value=pd.DataFrame()), \
             patch.object(app, "opp_multiplier", return_value=(1.0, None)), \
             patch.object(app, "load_team_game_dates", return_value=[]), \
             patch.object(app, "load_team_abbr_meta", return_value=[("2", "Boston Celtics", "BOS")]), \
             patch.object(app, "load_opp_allowed", return_value=allowed), \
             patch.object(app, "load_opp_allowed_recent", return_value=allowed), \
             patch.object(app, "injury_adjustment", return_value=dict(CLEAN_INJ)), \
             patch.object(app, "opp_position_detail", return_value=None):
            return app.evaluate_nba_prop("Test Player", stat, line, -110, -110, "Boston Celtics", date(2026, 11, 7))

    def test_points_shadow(self):
        app = load_app()
        logs = self.logs(app)
        res = self.evaluate(app, logs, "Points", 17.5)
        self.assertNotIn("error", res)
        sh = res["shadow"]
        self.assertAlmostEqual(sh["mu"], logs["POINTS"].mean(), delta=2.0)
        self.assertAlmostEqual(sh["three_share"], 40.0, delta=6.0)
        old = self.evaluate(app, logs.drop(columns=["FGM", "FTM"]), "Points", 17.5)       # old cached logs: no shadow, same pick
        self.assertIsNone(old["shadow"])
        self.assertEqual((old["best_side"], old["p_model"]), (res["best_side"], res["p_model"]))
        import snapshots
        row = snapshots.make_row(res, {}, "Points", 17.5, -110, -110, "test", "now", "2026-11-07")
        self.assertAlmostEqual(row["SHADOW_P_OVER"], sh["p_over"])

    def test_threes_shadow_and_other_stats_none(self):
        app = load_app()
        logs = self.logs(app)
        res = self.evaluate(app, logs, "3-Pointers Made", 2.5)
        self.assertAlmostEqual(res["shadow"]["attempts"], logs["3PA"].mean(), delta=1.0)
        self.assertIsNone(self.evaluate(app, logs, "Rebounds", 4.5)["shadow"])


if __name__ == "__main__":
    unittest.main()
