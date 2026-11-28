"""Tests for best_picks.py (identical copy in the NBA and NFL apps)."""
import unittest
import unittest.mock
from unittest.mock import MagicMock

import pandas as pd

import best_picks as bp


def row(player="A", side="Over", edge=8.0, rank=1, n=30, **kw):
    r = {"PLAYER": player, "POS": "G", "STAT": "Points", "SIDE": side, "LINE": 22.5, "ODDS": -110, "GAME": "X @ Y", "P_MODEL": 50 + edge, "P_MARKET": 50.0,
         "EDGE_PP": edge, "EV_%": edge * 1.5, "SIX": 5, "OPP": "UTA", "OPP_RANK": rank, "N_TEAMS": n, "PIN": ""}
    r.update(kw)
    return r


def res(ev_over=5.0, ev_under=-9.0):
    side = lambda ev, p: {"odds": -110.0, "p_model": p, "p_market": 50.0, "edge_pp": p - 50.0, "ev_pct": ev}
    return {"player_name": "A Guard", "sides": {"Over": side(ev_over, 55.0), "Under": side(ev_under, 45.0)},
            "six_point": {"Over": {"score": 5}, "Under": {"score": 2}, "best": "Over"}}


class TestMath(unittest.TestCase):
    def test_make_row_side(self):
        r = bp.make_row(res(), "Points", 22.5, "X @ Y", "G", "UTA", 3, 30)
        self.assertEqual((r["SIDE"], r["EDGE_PP"], r["SIX"], r["OPP_RANK"]), ("Over", 5.0, 5, 3))
        self.assertEqual(bp.make_row(res(-9.0, 4.0), "Points", 22.5, "", None, None, None, None)["SIDE"], "Over")    # 5/6 side wins over EV
        weak = res(-9.0, 4.0)
        weak["six_point"]["Over"]["score"] = 4
        self.assertEqual(bp.make_row(weak, "Points", 22.5, "", None, None, None, None)["SIDE"], "Under")             # no 5/6: higher EV side
        self.assertIsNone(bp.make_row({"player_name": "x"}, "Points", 1, "", None, None, None, None))

    def test_matchup_score_direction(self):
        s = bp.matchup_score(pd.Series(["Over", "Under", "Over", "Under", "Over"]), pd.Series([1, 1, 30, 30, None]), pd.Series([30, 30, 30, 30, 30]))
        self.assertEqual(list(s.round(1)), [100.0, 0.0, 0.0, 100.0, 50.0])

    def test_score_formula_and_both_flag(self):
        r = bp.rank_picks([row(edge=4.0, rank=1)], 0.6).iloc[0]                                       # EV 6% -> MODEL 75
        self.assertAlmostEqual(r["MODEL"], 100 * 6.0 / bp.EV_CAP)
        self.assertAlmostEqual(r["SCORE"], 0.6 * 75 + 0.4 * 100)
        self.assertTrue(r["BOTH"])
        self.assertAlmostEqual(bp.rank_picks([row(edge=25.0)], 1.0).iloc[0]["SCORE"], 100.0)          # EV is capped
        self.assertFalse(bp.rank_picks([row(edge=4.0, rank=1, **{"EV_%": -1.0})]).iloc[0]["BOTH"])     # not +EV
        self.assertFalse(bp.rank_picks([row(edge=1.5, rank=1)]).iloc[0]["BOTH"])                      # edge too small
        self.assertFalse(bp.rank_picks([row(edge=8.0, rank=15)]).iloc[0]["BOTH"])                     # middling matchup
        self.assertFalse(bp.rank_picks([row(edge=8.0, rank=None, n=None)]).iloc[0]["BOTH"])           # no rank is never BOTH

    def test_order_min_edge_and_weight(self):
        rows = [row("Model", edge=9.0, rank=30), row("Match", edge=2.5, rank=1), row("Low", edge=0.5, rank=1)]
        r = bp.rank_picks(rows, min_edge=1.0)
        self.assertEqual(list(r["PLAYER"]), ["Model", "Match"])                    # default: EV ranks (13.5% beats 3.75%), Low dropped
        self.assertEqual(list(bp.rank_picks(rows, 0.0, 1.0)["PLAYER"]), ["Match", "Model"])     # all matchup weight: the matchup decides
        self.assertTrue(bp.rank_picks([], 0.6).empty)
        self.assertTrue(bp.rank_picks(rows, 0.6, min_edge=50).empty)

    def test_model_score_spreads_out_instead_of_maxing(self):
        r = bp.rank_picks([row("Big", edge=6.0, ODDS=100), row("Small", edge=1.0, ODDS=100)], 1.0, w_blend=0.3)
        self.assertGreater(r.iloc[0]["MODEL"], r.iloc[1]["MODEL"])                  # was 100 / 100 under the old edge cap
        self.assertEqual(r.iloc[0]["PLAYER"], "Big")

    def test_cap_per_game(self):
        rows = [row(f"P{i}", edge=9.0 - i, GAME="A @ B" if i < 3 else "C @ D") for i in range(5)]
        r = bp.cap_per_game(bp.rank_picks(rows), 2)
        self.assertEqual(list(r["PLAYER"]), ["P0", "P1", "P3", "P4"])
        self.assertEqual(len(bp.cap_per_game(bp.rank_picks(rows), 0)), 5)

    def test_display_details_toggle(self):
        r = bp.rank_picks([row()])
        self.assertEqual(list(bp.display(r).columns), bp.SHOW_COLS)
        self.assertEqual(list(bp.display(r, details=True).columns), bp.SHOW_COLS + bp.DETAIL_COLS)
        self.assertNotIn("MODEL", bp.SHOW_COLS)

    def test_display_and_helpers(self):
        d = bp.display(bp.rank_picks([row(), row("B", rank=None, n=None, SIX=None)]))
        self.assertEqual(list(d.columns), bp.SHOW_COLS)
        self.assertEqual(d.iloc[0]["PICK"], "A Over 22.5 Points")
        self.assertEqual((d.iloc[0]["OPP_VS_POS"], d.iloc[0]["SIX"], d.iloc[0]["ODDS"]), ("UTA #1 of 30", "5/6", "-110"))
        self.assertEqual((d.iloc[1]["OPP_VS_POS"], d.iloc[1]["SIX"]), ("no rank", ""))
        self.assertEqual(bp.credit_estimate(8, 4), 32)
        self.assertTrue(bp.due(None, 15) and bp.due(0.0, 15, now=900.0))
        self.assertFalse(bp.due(0.0, 15, now=899.0) or bp.due(None, 0))
        g = pd.DataFrame({"label": ["a", "b", "c"], "day": ["2026-11-17", "2026-11-16", "2026-11-16"]})
        self.assertEqual(bp.default_games(g), ["b", "c"])


GAMES = pd.DataFrame({"id": ["g1", "g2"], "label": ["A @ B", "C @ D"], "day": ["2026-11-16", "2026-11-16"]})


def board(over=-110):
    return pd.DataFrame({"Player": ["P1", "P2", "P3"], "Line": [22.5, 5.5, None], "Over": [over, -110, -110], "Under": [-110, -110, -110]})


class TestScan(unittest.TestCase):
    def test_reuses_unchanged_props_and_survives_a_bad_board(self):
        calls = []

        def fetch(g, stat):
            if g["id"] == "g2":
                raise RuntimeError("quota")
            return board()

        def ev(g, stat, name, line, over, under):
            calls.append(name)
            return None if name == "P2" else row(name)

        cache = {}
        out = bp.scan(GAMES, ["Points"], fetch, ev, cache)
        self.assertEqual((len(out["rows"]), out["evaluated"], out["reused"], len(out["errors"])), (1, 2, 0, 1))
        out2 = bp.scan(GAMES, ["Points"], fetch, ev, cache)
        self.assertEqual((out2["evaluated"], out2["reused"], len(calls)), (0, 2, 2))             # nothing re-run, P3 (no line) skipped
        out3 = bp.scan(GAMES, ["Points"], lambda g, s: board(-125) if g["id"] == "g1" else None, ev, cache)
        self.assertEqual((out3["evaluated"], out3["reused"]), (1, 1))                            # only the prop whose price moved


def fake_st(pressed=(), multi=None, timer="Off (manual only)", view=None):
    st = MagicMock()
    st.session_state = {}
    st.button.side_effect = lambda label, key=None, **k: key in pressed
    st.columns.side_effect = lambda spec: [st] * (spec if isinstance(spec, int) else len(spec))
    st.multiselect.side_effect = lambda label, options, default=None, key=None, **k: (multi or {}).get(key, default if default is not None else [])
    st.slider.return_value = 60
    st.number_input.side_effect = lambda label, **k: k.get("value")
    st.selectbox.side_effect = lambda label, options, key=None, **k: timer
    st.fragment.side_effect = lambda **k: (lambda f: (lambda: None))          # the timer tick is not run in tests
    st.checkbox.side_effect = lambda label, value=False, **k: value
    st.radio.side_effect = lambda label, options, key=None, **k: (view or options[0])
    return st


class TestRender(unittest.TestCase):
    def kw(self, starred=None):
        return dict(stats=["Points", "Rebounds"], default_stats=["Points"], list_games=lambda: GAMES, fetch_board=lambda g, s: board(),
                    evaluate=lambda g, s, name, line, over, under: row(name, GAME=g["label"]), keep_keys=("ctx",),
                    star=(lambda rows: (starred.extend(rows["PLAYER"]) or len(rows)) if starred is not None else 0))

    def test_nothing_is_scanned_until_asked(self):
        st = fake_st()
        bp.render(st, **self.kw())
        self.assertNotIn("bp_rows", st.session_state)
        self.assertIn("2 Odds API credits", st.caption.call_args_list[0][0][0])       # 2 games x 1 stat

    def test_refresh_builds_the_list_and_restores_session_keys(self):
        st = fake_st(pressed={"bp_refresh"})
        st.session_state["ctx"] = "mine"
        kw = self.kw()
        kw["fetch_board"] = lambda g, s: (st.session_state.update(ctx="scan", other="x"), board())[1]
        bp.render(st, **kw)
        self.assertEqual(len(st.session_state["bp_rows"]), 4)
        self.assertEqual(st.session_state["ctx"], "mine")
        shown = st.dataframe.call_args[0][0]
        self.assertEqual(shown.index.name, "PICK")                                     # pinned on the left
        self.assertEqual(list(shown.columns), bp.SHOW_COLS[1:])
        self.assertEqual(len(shown), 4)

    def test_each_scan_saves_the_list_once(self):
        saved = []
        kw = self.kw()
        kw["tag"] = lambda rows: saved.append(list(rows["PLAYER"]))
        st = fake_st(pressed={"bp_refresh"})
        bp.render(st, **kw)
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(saved[0]), 4)
        st.button.side_effect = lambda label, key=None, **k: False                     # a later rerun without a scan saves nothing
        bp.render(st, **kw)
        self.assertEqual(len(saved), 1)

    def test_timer_scans_when_due_and_not_before(self):
        st = fake_st(timer="Every 15 min")
        bp.render(st, **self.kw())
        self.assertEqual(len(st.session_state["bp_rows"]), 4)                          # never scanned -> due
        n = []
        kw = self.kw()
        kw["fetch_board"] = lambda g, s: (n.append(1), board())[1]
        st.button.side_effect = lambda label, key=None, **k: False
        bp.render(st, **kw)
        self.assertEqual(n, [])                                                        # scanned seconds ago -> not due
        st.session_state["bp_time"] -= 16 * 60
        bp.render(st, **kw)
        self.assertEqual(len(n), 2)

    def test_star_buttons(self):
        starred = []
        st = fake_st(pressed={"bp_refresh", "bp_star"}, multi={"bp_star_pick": ["P1 Over 22.5 Points"]})
        bp.render(st, **self.kw(starred))
        self.assertEqual(set(starred), {"P1"})                                         # P1 in both games share the label
        starred2 = []
        st = fake_st(pressed={"bp_refresh", "bp_star_both"})
        bp.render(st, **self.kw(starred2))
        self.assertEqual(len(starred2), 4)

    def test_game_list_failure_is_reported(self):
        st = fake_st()
        kw = self.kw()
        kw["list_games"] = lambda: (_ for _ in ()).throw(RuntimeError("bad key"))
        bp.render(st, **kw)
        st.error.assert_called()


class TestClickToOpen(unittest.TestCase):
    def setUp(self):
        self.opened = []
        self.kw = dict(stats=["Points"], default_stats=["Points"], list_games=lambda: GAMES, fetch_board=lambda g, s: board(),
                       evaluate=lambda g, s, name, line, over, under: row(name, GAME=g["label"]),
                       open_pick=lambda r: (self.opened.append(r), f"Loaded {r['PLAYER']}")[1])

    def st_with_click(self, row_index):
        st = fake_st(pressed={"bp_refresh"})
        st.dataframe.side_effect = lambda df, **k: {"selection": {"rows": [] if row_index is None else [row_index]}} if k.get("on_select") else None
        return st

    def test_clicked_row_is_opened_once_with_its_game(self):
        st = self.st_with_click(1)
        bp.render(st, **self.kw)
        self.assertEqual(len(self.opened), 1)
        self.assertIn(self.opened[0]["EVENT_ID"], {"g1", "g2"})
        st.success.assert_called_with(f"Loaded {self.opened[0]['PLAYER']}")
        st.button.side_effect = lambda label, key=None, **k: False      # a later rerun with the same row still selected: not opened again
        bp.render(st, **self.kw)
        self.assertEqual(len(self.opened), 1)

    def test_no_click_and_failures(self):
        st = self.st_with_click(None)
        bp.render(st, **self.kw)
        self.assertEqual(self.opened, [])
        st = self.st_with_click(0)
        self.kw["open_pick"] = lambda r: (_ for _ in ()).throw(RuntimeError("gone"))
        bp.render(st, **self.kw)
        st.warning.assert_called()

    def test_old_streamlit_falls_back_to_a_picker(self):
        st = fake_st(pressed={"bp_refresh"})

        def df(data, **k):
            if k.get("on_select"):
                raise TypeError("unexpected keyword")
        st.dataframe.side_effect = df
        st.selectbox.side_effect = lambda label, options, key=None, **k: options[1] if key == "bp_open_pick" else "Off (manual only)"
        bp.render(st, **self.kw)
        self.assertEqual(len(self.opened), 1)

    def test_selected_row_reads_both_event_shapes(self):
        self.assertEqual(bp.selected_row({"selection": {"rows": [3]}}), 3)
        ev = MagicMock()
        ev.selection.rows = [2]
        self.assertEqual(bp.selected_row(ev), 2)
        self.assertIsNone(bp.selected_row({"selection": {"rows": []}}))
        self.assertIsNone(bp.selected_row(None))


class TestSharpView(unittest.TestCase):
    ROWS = [row("Six", SIX=6, rank=2, edge=1.0), row("FiveBadMatch", SIX=5, rank=20), row("FourGood", SIX=4, rank=1),
            row("FiveUnder", side="Under", SIX=5, rank=29, edge=0.2, **{"EV_%": -2.0}), row("FiveNoRank", SIX=5, rank=None, n=None),
            row("FiveOver", SIX=5, rank=3, edge=6.0)]

    def test_only_five_or_six_with_a_top_third_matchup(self):
        s = bp.sharp_picks(self.ROWS)
        self.assertEqual(list(s["PLAYER"]), ["Six", "FiveOver", "FiveUnder"])        # 6/6 first; tough defense counts for an Under
        self.assertTrue(s["SHARP"].all())
        self.assertIn("FiveUnder", list(s["PLAYER"]))                                 # kept even below the minimum edge / with EV < 0

    def test_view_shows_every_sharp_pick_ignoring_limits(self):
        many = [row(f"P{i}", SIX=5, rank=1, GAME="A @ B") for i in range(7)]
        st = fake_st(pressed={"bp_refresh"}, view=bp.VIEW_SHARP)
        kw = dict(stats=["Points"], default_stats=["Points"], list_games=lambda: GAMES, fetch_board=lambda g, s: board(),
                  evaluate=lambda g, s, name, line, over, under: None)
        st.session_state.update(bp_rows=many, bp_time=0.0)
        st.button.side_effect = lambda label, key=None, **k: False
        bp.render(st, **kw)
        shown = st.dataframe.call_args[0][0]
        self.assertEqual(len(shown), 7)                                               # per-game cap of 2 does not apply
        st2 = fake_st(view=bp.VIEW_ALL)
        st2.session_state.update(bp_rows=many, bp_time=0.0)
        bp.render(st2, **kw)
        self.assertEqual(len(st2.dataframe.call_args[0][0]), 2)                       # All ranked picks keeps the cap

    def test_tag_includes_sharp_picks_outside_the_top(self):
        saved = []
        rows = [row(f"Top{i}", SIX=3, rank=30, edge=9.0) for i in range(3)] + [row("Sharp", SIX=6, rank=1, edge=0.1, GAME="C @ D")]
        st = fake_st(pressed={"bp_refresh"})
        st.number_input.side_effect = lambda label, **k: 2 if label.startswith("Show top") else (0 if label.startswith("Max") else k.get("value"))
        kw = dict(stats=["Points"], default_stats=["Points"], list_games=lambda: GAMES, fetch_board=lambda g, s: board(),
                  evaluate=lambda g, s, name, line, over, under: None, tag=lambda r: saved.append(r))
        st.session_state["bp_cache"] = {}
        with unittest.mock.patch.object(bp, "scan", lambda *a, **k: {"rows": rows, "errors": [], "evaluated": 4, "reused": 0}):
            bp.render(st, **kw)
        tagged = saved[0].set_index("PLAYER")
        self.assertTrue(tagged.loc["Sharp", "SHARP"])
        self.assertEqual(tagged.loc["Sharp", "RANK"], 4)                              # its place in the full ranking


class TestHideNegativeEv(unittest.TestCase):
    def test_negative_ev_hidden_by_default_but_still_saved(self):
        rows = [row("Plus", SIX=5, rank=1, GAME="A @ B"), row("Minus", SIX=6, rank=1, GAME="C @ D", **{"EV_%": -3.0})]
        saved = []
        kw = dict(stats=["Points"], default_stats=["Points"], list_games=lambda: GAMES, fetch_board=lambda g, s: board(),
                  evaluate=lambda g, s, name, line, over, under: None, tag=lambda r: saved.append(r))
        st = fake_st(pressed={"bp_refresh"}, view=bp.VIEW_SHARP)
        with unittest.mock.patch.object(bp, "scan", lambda *a, **k: {"rows": rows, "errors": [], "evaluated": 2, "reused": 0}):
            bp.render(st, **kw)
        shown = st.dataframe.call_args[0][0]
        self.assertEqual([p.split(" ")[0] for p in shown.index], ["Plus"])
        self.assertIn("1 with negative EV hidden", " ".join(c[0][0] for c in st.caption.call_args_list))
        self.assertEqual(set(saved[0]["PLAYER"]), {"Plus", "Minus"})                  # the hidden one is still saved for grading
        st2 = fake_st(view=bp.VIEW_SHARP)
        st2.checkbox.side_effect = lambda label, value=False, **k: False if k.get("key") == "bp_hide_neg" else value
        st2.session_state.update(bp_rows=rows, bp_time=0.0)
        bp.render(st2, **kw)
        self.assertEqual(len(st2.dataframe.call_args[0][0]), 2)                        # unticked: both shown

    def test_all_hidden_message(self):
        rows = [row("Minus", SIX=6, rank=1, **{"EV_%": -3.0})]
        st = fake_st(view=bp.VIEW_SHARP)
        st.session_state.update(bp_rows=rows, bp_time=0.0)
        bp.render(st, stats=["Points"], default_stats=["Points"], list_games=lambda: GAMES, fetch_board=lambda g, s: board(),
                  evaluate=lambda g, s, name, line, over, under: None)
        self.assertIn("are hidden (negative EV or a Pinnacle gap over 3.5 pts)", st.info.call_args[0][0])


class TestPinnacleViews(unittest.TestCase):
    ROWS = [row("Agree2", PIN="Agrees", PIN_GAP=2.0, PIN_LINE=22.5), row("Agree5", PIN="Agrees", PIN_GAP=5.0, PIN_LINE=22.5),
            row("Dis", PIN="Disagrees", PIN_GAP=-3.0, PIN_LINE=22.5), row("HighPin", PIN="Agrees", PIN_GAP=3.0, PIN_LINE=23.5),
            row("LowPin", side="Under", PIN="Flat", PIN_GAP=0.4, PIN_LINE=20.5), row("NoPin", PIN="No Pinnacle")]

    def test_agrees_view_sorted_by_gap(self):
        r = bp.pinnacle_agrees(self.ROWS)
        self.assertEqual(list(r["PLAYER"]), ["Agree5", "HighPin", "Agree2"])

    def test_agrees_view_uses_pinnacle_ev(self):
        rows = [row("SmallGapPlus", PIN="Agrees", PIN_GAP=2.8, PIN_EV=1.2, **{"EV_%": -4.0}),
                row("BigGapMinus", PIN="Agrees", PIN_GAP=3.2, PIN_EV=-0.5, **{"EV_%": -3.0}),
                row("Better", PIN="Agrees", PIN_GAP=3.0, PIN_EV=2.4, **{"EV_%": -5.0})]
        self.assertEqual(list(bp.pinnacle_agrees(rows)["PLAYER"]), ["Better", "SmallGapPlus", "BigGapMinus"])   # by PIN_EV
        st = fake_st(view=bp.VIEW_PIN)
        st.session_state.update(bp_rows=rows, bp_time=0.0)
        bp.render(st, stats=["Points"], default_stats=["Points"], list_games=lambda: GAMES, fetch_board=lambda g, s: board(),
                  evaluate=lambda g, s, name, line, over, under: None)
        shown = st.dataframe.call_args[0][0]
        self.assertEqual([p.split(" ")[0] for p in shown.index], ["Better", "SmallGapPlus"])   # blended EV negative, Pinnacle EV positive: kept
        self.assertEqual(list(shown["PIN_EV"]), ["+2.4%", "+1.2%"])
        old = [row("OldApp", PIN="Agrees", PIN_GAP=2.0, **{"EV_%": 1.5})]                     # no PIN_EV (older app.py): blended EV decides
        self.assertTrue((bp.view_ev(bp.pinnacle_agrees(old), bp.VIEW_PIN) == 1.5).all())

    def test_line_differs_view(self):
        r = bp.line_differs(self.ROWS).set_index("PLAYER")
        self.assertEqual(list(bp.line_differs(self.ROWS)["PLAYER"]), ["LowPin", "HighPin"])          # 2 points away first
        self.assertEqual((r.loc["HighPin", "LINE_GAP"], r.loc["HighPin", "GAP_FAVORS"], r.loc["HighPin", "GAP_MATCH"]), (1.0, "Over", True))
        self.assertEqual((r.loc["LowPin", "GAP_FAVORS"], r.loc["LowPin", "GAP_MATCH"]), ("Under", True))
        d = bp.display(bp.line_differs(self.ROWS), extra=("PIN_VS_DK", "FAVORS")).set_index("PICK")
        self.assertEqual(d.iloc[1]["PIN_VS_DK"], "23.5 vs 22.5")
        self.assertEqual(d.iloc[1]["FAVORS"], "Over ✓")
        self.assertEqual(bp.display(bp.pinnacle_agrees(self.ROWS), extra=("PIN_EDGE",)).iloc[0]["PIN_EDGE"], "+5.0 pts")

    def test_views_render_and_tag_everything(self):
        saved = []
        st = fake_st(pressed={"bp_refresh"}, view=bp.VIEW_GAP)
        kw = dict(stats=["Points"], default_stats=["Points"], list_games=lambda: GAMES, fetch_board=lambda g, s: board(),
                  evaluate=lambda g, s, name, line, over, under: None, tag=lambda r: saved.append(r))
        with unittest.mock.patch.object(bp, "scan", lambda *a, **k: {"rows": self.ROWS, "errors": [], "evaluated": 6, "reused": 0}):
            bp.render(st, **kw)
        shown = st.dataframe.call_args[0][0]
        self.assertEqual(len(shown), 2)
        self.assertIn("FAVORS", shown.columns)
        tagged = saved[0].set_index("PLAYER")
        self.assertTrue(tagged.loc["Agree5", "PIN_AGREE"])
        self.assertEqual(tagged.loc["LowPin", "GAP_FAVORS"], "Under")
        self.assertEqual(bp.VIEWS[0], bp.VIEW_ALL)                                     # All ranked picks is the default again

    def test_big_gaps_hidden_in_agrees_view_but_saved(self):
        saved = []
        st = fake_st(pressed={"bp_refresh"}, view=bp.VIEW_PIN)
        kw = dict(stats=["Points"], default_stats=["Points"], list_games=lambda: GAMES, fetch_board=lambda g, s: board(),
                  evaluate=lambda g, s, name, line, over, under: None, tag=lambda r: saved.append(r))
        with unittest.mock.patch.object(bp, "scan", lambda *a, **k: {"rows": self.ROWS, "errors": [], "evaluated": 6, "reused": 0}):
            bp.render(st, **kw)
        shown = st.dataframe.call_args[0][0]
        self.assertEqual([p.split(" ")[0] for p in shown.index], ["HighPin", "Agree2"])          # Agree5 (5 pts) hidden
        self.assertIn("1 with a Pinnacle gap over 3.5 pts hidden", " ".join(c[0][0] for c in st.caption.call_args_list))
        self.assertIn("Agree5", set(saved[0]["PLAYER"]))                                         # still saved for grading
        st2 = fake_st(view=bp.VIEW_PIN)
        st2.checkbox.side_effect = lambda label, value=False, **k: False if k.get("key") == "bp_hide_big" else value
        st2.session_state.update(bp_rows=self.ROWS, bp_time=0.0)
        bp.render(st2, **kw)
        self.assertEqual(len(st2.dataframe.call_args[0][0]), 3)                                  # unticked: all three agrees

    def test_big_gap_marked_in_display(self):
        d = bp.display(bp.pinnacle_agrees(self.ROWS))
        pins = list(d["PIN"])
        self.assertEqual(sum("⚠️ big gap: check news" in p for p in pins), 1)
        self.assertEqual(pins[0], "Agrees ⚠️ big gap: check news")


if __name__ == "__main__":
    unittest.main()
