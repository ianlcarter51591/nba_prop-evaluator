"""Recommended model settings at startup / restore, and why a Best Picks row shows 'no rank'."""
import unittest
from unittest.mock import patch

import pandas as pd

import best_picks as bp
from test_integration import load_app


class TestSettings(unittest.TestCase):
    def setUp(self):
        self.app = load_app()
        self.ss = self.app.st.session_state

    def test_restore_and_list_what_is_off(self):
        self.app.restore_settings()
        self.assertEqual(self.app.settings_off(self.ss), [])
        self.assertEqual((self.ss["n0_rate"], self.ss["n0_min"], self.ss["adj_role_auto"], self.ss["adj_opp"]), (6, 2, True, True))
        self.ss.update(n0_rate=0, n0_min=0, adj_role_auto=False)
        off = self.app.settings_off(self.ss)
        self.assertEqual(len(off), 3)
        self.assertIn("Minutes weight: 0 (recommended 2)", off)
        self.assertIn("Detect role changes: off (recommended on)", off)
        self.app.restore_settings()
        self.assertEqual(self.app.settings_off(self.ss), [])

    def test_applied_calibration_is_the_new_normal(self):
        self.app.restore_settings()
        self.ss["n0_applied"], self.ss["n0_by_stat"] = (8, 3), {"POINTS": 8.0}
        self.ss.update(n0_rate=8, n0_min=3)
        self.assertEqual(self.app.settings_off(self.ss), [])                 # no nagging after Apply
        self.ss["n0_min"] = 0
        self.app.restore_settings()
        self.assertEqual((self.ss["n0_min"], self.ss["n0_by_stat"]), (3, {"POINTS": 8.0}))


class TestRankStatus(unittest.TestCase):
    def setUp(self):
        self.app = load_app()
        self.app._RANK_FAIL.clear()

    def test_reasons(self):
        a = self.app
        with patch.object(a, "position_of", return_value=None):
            self.assertEqual(a.position_rank_status("BOS", "X", "POINTS"), (None, "position unknown"))
        with patch.object(a, "position_of", return_value="PG"), patch.object(a, "opp_position_detail", return_value={"rank": 3}):
            self.assertEqual(a.position_rank_status("BOS", "X", "POINTS"), ({"rank": 3}, ""))
        boom = patch.object(a, "opp_position_detail", side_effect=RuntimeError("ESPN timeout"))
        with patch.object(a, "position_of", return_value="PG"), boom as m:
            self.assertEqual(a.position_rank_status("BOS", "X", "POINTS"), (None, a.RANK_FAILED))
            self.assertEqual(m.call_count, 2)                                 # retried once
            a.position_rank_status("BOS", "Y", "POINTS")
            self.assertEqual(m.call_count, 2)                                 # then skipped for 5 minutes, not retried per prop

    def test_best_picks_shows_the_reason(self):
        res = {"player_name": "A", "sides": {"Over": {"odds": -110, "p_model": 55, "p_market": 50, "edge_pp": 5, "ev_pct": 4},
                                             "Under": {"odds": -110, "p_model": 45, "p_market": 50, "edge_pp": -5, "ev_pct": -14}}}
        row = bp.make_row(res, "Points", 20.5, "G", "PG", "BOS", None, None, rank_note="league data didn't load: refresh to retry")
        shown = bp.display(bp.rank_picks([row], min_edge=-99))
        self.assertEqual(shown["OPP_VS_POS"][0], "no rank: league data didn't load: refresh to retry")
        ok = bp.make_row(res, "Points", 20.5, "G", "PG", "BOS", 4, 30, rank_note="ignored")
        self.assertEqual(bp.display(bp.rank_picks([ok], min_edge=-99))["OPP_VS_POS"][0], "BOS #4 of 30")
        old = bp.make_row(res, "Points", 20.5, "G", "PG", "BOS", None, None)                       # NFL / older callers
        self.assertEqual(bp.display(bp.rank_picks([old], min_edge=-99))["OPP_VS_POS"][0], "no rank")


if __name__ == "__main__":
    unittest.main()
