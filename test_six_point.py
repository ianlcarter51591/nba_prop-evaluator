"""Tests for six_point.py (pure logic). Run: python -m unittest -v test_six_point"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import six_point as sp

CLEAN_INJ = {"checked": True, "feed_ok": True, "self_status": "", "listed": []}


def score(side="Over", stat="POINTS", line=24.5, proj=28.0, p_market=0.55,
          season=5, recent=8, n=30, inj=None, spread=6.5):
    return sp.score_prop(side, stat, line, proj, p_market, season, recent, n,
                         CLEAN_INJ if inj is None else inj, spread)


class TestTiers(unittest.TestCase):
    def test_tier_mapping(self):
        self.assertEqual([sp.tier_for(n) for n in range(7)],
                         ["PASS", "PASS", "PASS", "PASS", "PASS", "LEAN", "SHARP"])

    def test_perfect_over_is_sharp(self):
        r = score()
        self.assertEqual((r.score, r.tier), (6, "SHARP"))
        self.assertTrue(all(c.passed for c in r.checks))

    def test_one_miss_is_lean_two_misses_pass(self):
        self.assertEqual(score(spread=14.0).tier, "LEAN")
        self.assertEqual(score(spread=14.0, p_market=0.48).tier, "PASS")

    def test_as_dict_round_trip(self):
        d = score().as_dict()
        self.assertEqual((d["score"], d["tier"], len(d["checks"])), (6, "SHARP", 6))
        self.assertEqual(set(d["checks"][0]), {"key", "label", "passed", "detail", "verified"})


class TestChecks(unittest.TestCase):
    def test_baby_line(self):
        self.assertFalse(sp.check_line("POINTS", 12.5).passed)
        self.assertTrue(sp.check_line("POINTS", 14.5).passed)
        self.assertFalse(sp.check_line("REBOUNDS", 3.5).passed)
        self.assertTrue(sp.check_line("3-POINTERS MADE", 1.5).passed)

    def test_edge_direction_and_size(self):
        self.assertTrue(sp.check_edge("Over", "POINTS", 24.5, 28.0).passed)
        self.assertFalse(sp.check_edge("Over", "POINTS", 24.5, 25.5).passed)   # small edge
        self.assertFalse(sp.check_edge("Over", "POINTS", 24.5, 22.0).passed)   # wrong direction
        self.assertTrue(sp.check_edge("Under", "POINTS", 24.5, 21.0).passed)
        self.assertFalse(sp.check_edge("Under", "POINTS", 24.5, 28.0).passed)

    def test_edge_uses_larger_of_abs_and_pct(self):
        # 40.5 line: 6% = 2.43 > 1.5 absolute floor
        self.assertFalse(sp.check_edge("Over", "POINTS", 40.5, 42.5).passed)
        self.assertTrue(sp.check_edge("Over", "POINTS", 40.5, 43.1).passed)
        # tiny stat: absolute floor 0.4 dominates 6% of 1.5
        self.assertFalse(sp.check_edge("Over", "STEALS", 1.5, 1.8).passed)
        self.assertTrue(sp.check_edge("Over", "STEALS", 1.5, 1.95).passed)

    def test_implied_chance_is_strictly_above_half(self):
        self.assertTrue(sp.check_implied("Over", 0.5001).passed)
        self.assertFalse(sp.check_implied("Over", 0.50).passed)
        self.assertFalse(sp.check_implied("Under", 0.43).passed)

    def test_defense_over_wants_soft_under_wants_tough(self):
        self.assertTrue(sp.check_defense("Over", 3, 15, 30).passed)
        self.assertFalse(sp.check_defense("Over", 3, 16, 30).passed)    # recent not soft enough
        self.assertFalse(sp.check_defense("Over", 20, 5, 30).passed)    # season not soft
        self.assertTrue(sp.check_defense("Under", 30, 16, 30).passed)
        self.assertFalse(sp.check_defense("Under", 30, 15, 30).passed)

    def test_defense_missing_data_is_unverified_fail(self):
        for args in ((None, 5, 30), (5, None, 30), (5, 5, None), (None, None, None)):
            c = sp.check_defense("Over", *args)
            self.assertFalse(c.passed)
            self.assertFalse(c.verified)

    def test_injury_flags(self):
        inj = {"checked": True, "feed_ok": True, "self_status": "",
               "listed": [{"name": "Star", "status": "OUT", "p_out": 1.0, "mpg": 30.0}]}
        c = sp.check_injury(inj, 10.0)
        self.assertFalse(c.passed)
        self.assertIn("Star (OUT)", c.detail)

    def test_injury_ignores_bench_and_manual_active(self):
        inj = {"checked": True, "feed_ok": True, "self_status": "",
               "listed": [{"name": "Deep Bench", "status": "OUT", "p_out": 1.0, "mpg": 3.0},
                          {"name": "Fine", "status": "ACTIVE (manual)", "p_out": 0.0, "mpg": 30.0}]}
        self.assertTrue(sp.check_injury(inj, 10.0).passed)

    def test_injury_player_himself_flagged(self):
        inj = {**CLEAN_INJ, "self_status": "Questionable"}
        c = sp.check_injury(inj, 10.0)
        self.assertFalse(c.passed)
        self.assertIn("the player", c.detail)

    def test_injury_unverified_cases(self):
        for inj in ({"checked": False, "feed_ok": True, "listed": []},
                    {"checked": True, "feed_ok": False, "listed": []}, {}):
            c = sp.check_injury(inj, 10.0)
            self.assertFalse(c.passed)
            self.assertFalse(c.verified)

    def test_game_script(self):
        self.assertTrue(sp.check_game_script(9.5).passed)
        self.assertTrue(sp.check_game_script(-9.5).passed)
        self.assertFalse(sp.check_game_script(10.0).passed)
        c = sp.check_game_script(None)
        self.assertFalse(c.passed)
        self.assertFalse(c.verified)

    def test_unverifiable_checks_block_sharp(self):
        r = score(spread=None)
        self.assertEqual((r.score, r.tier), (5, "LEAN"))
        r = score(spread=None, season=None)
        self.assertEqual(r.tier, "PASS")


class TestDataHelpers(unittest.TestCase):
    def games(self):
        rows = []
        for pid in ("a", "b"):
            for d in range(1, 16):          # 15 dates
                for opp in ("BOS", "LAL"):
                    rows.append({"PID": pid, "DATE": f"2026-11-{d:02d}", "OPP": opp, "POINTS": 20.0, "ASSISTS": 5.0})
        df = pd.DataFrame(rows)
        # BOS: soft in its last 3 games only (players score 30), otherwise normal
        late = (df["OPP"] == "BOS") & (df["DATE"] >= "2026-11-13")
        df.loc[late, "POINTS"] = 30.0
        return df

    def test_recent_allowed_only_uses_last_n_games(self):
        out = sp.compute_opp_allowed_recent(self.games(), ["POINTS", "ASSISTS"], n_recent=3)
        self.assertEqual(int(out.loc["BOS", "G"]), 3)
        self.assertGreater(out.loc["BOS", "POINTS"], 1.1)            # soft recently
        self.assertLess(out.loc["LAL", "POINTS"], out.loc["BOS", "POINTS"])
        self.assertAlmostEqual(out.loc["BOS", "ASSISTS"], 1.0)

    def test_recent_window_larger_than_history_equals_all_games(self):
        out = sp.compute_opp_allowed_recent(self.games(), ["POINTS"], n_recent=99)
        self.assertEqual(int(out.loc["BOS", "G"]), 15)

    def test_recent_allowed_empty_input(self):
        empty = pd.DataFrame({"PID": ["a"], "DATE": ["2026-11-01"], "OPP": ["BOS"], "POINTS": [10.0]})
        self.assertTrue(sp.compute_opp_allowed_recent(empty, ["POINTS"], min_games=3).empty)

    def test_extract_spread(self):
        payload = {"bookmakers": [{"markets": [
            {"key": "spreads", "outcomes": [{"name": "A", "point": -7.5}, {"name": "B", "point": 7.5}]}]}]}
        self.assertEqual(sp.extract_spread(payload), 7.5)
        self.assertIsNone(sp.extract_spread({"bookmakers": []}))
        self.assertIsNone(sp.extract_spread({"bookmakers": [{"markets": [{"key": "h2h", "outcomes": []}]}]}))


if __name__ == "__main__":
    unittest.main()
