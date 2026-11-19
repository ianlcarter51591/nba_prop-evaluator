"""
Best Picks: one ranked list of the slate's props, by the model AND by the opponent's rank against the player's position.

Math
----
For each prop the app evaluates, the side kept is the one with the higher model EV. With the market blend on (default, see
prop_review.py) the model chance is first blended with DraftKings' no-vig chance in log-odds:
    BLEND_% = blend(P_MODEL, P_MARKET, w_blend),  EDGE_PP = BLEND_% - P_MARKET,  EV_% = 100 * (BLEND_% / 100 * decimal(ODDS) - 1)
(without the blend BLEND_% = P_MODEL and EV_% is the model's own EV). Then, on a 0-100 scale:
    MODEL    = 100 * clip(EV_% / EV_CAP, 0, 1)              EV_CAP = 8: 0% EV -> 0, 8%+ EV -> 100 (so a 6% EV pick outranks a 1% one)
    MATCHUP  = 100 * (N - R) / (N - 1)   for an Over       R = opponent's rank vs the player's position for this stat (1 = softest,
             = 100 * (R - 1) / (N - 1)   for an Under          allows the most), N = teams ranked.  No rank -> 50 (neutral).
    SCORE    = w * MODEL + (1 - w) * MATCHUP               w = the "model weight" slider, default 0.85: EV ranks, the matchup breaks ties
    BOTH ✓   = EV_% > 0 and EDGE_PP > 0 (EDGE_PP >= 2 without the blend) and MATCHUP >= 66.7 (a top-third matchup for that side)
    SHARP    = SIX >= 5 (5/6 or 6/6 on the six-point checklist) and a top-third matchup (MATCHUP >= 66.7) for that side
The default view lists EVERY SHARP prop (no minimum edge, top-N or per-game limit), sorted by SIX, then SCORE. "Hide negative-EV
picks" (on by default) hides the ones with EV_% <= 0 from the table only: every SHARP prop is still saved for grading with its EV,
so Daily results can show whether the hidden ones would have lost money.
Side: when the six-point checklist rates a side 5/6 or 6/6, that side is used (the same side the board's tag names);
otherwise the side with the higher model EV.
"All ranked picks" view: rows are sorted by SCORE, then EV_%. Props whose EDGE_PP is below the minimum edge are dropped, and at most `per_game` picks
are kept from any one game (props from the same game tend to win or lose together). The matchup weight stays low until your
graded snapshots show BOTH ✓ picks actually win more (Snapshots -> Daily results -> "Best Picks ✓ BOTH").

The UI part (render) is the same in both apps; each app passes callbacks that fetch a board and evaluate one prop.
Auto-refresh only runs while the app is open in a browser, and every refresh of a (game, stat) board costs Odds API credits.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

EV_CAP = 8.0
BOTH_EDGE = 2.0
BOTH_MATCHUP = 200.0 / 3.0
DEFAULT_WEIGHT = 0.85
DEFAULT_PER_GAME = 2
SHARP_SIX = 5
VIEWS = ["5/6 & 6/6 with a matchup edge", "All ranked picks"]
TIMER_CHOICES: Dict[str, int] = {"Off (manual only)": 0, "Every 15 min": 15, "Every 30 min": 30, "Every 60 min": 60}
ROW_COLS = ["PLAYER", "POS", "STAT", "SIDE", "LINE", "ODDS", "GAME", "P_MODEL", "P_MARKET", "EDGE_PP", "EV_%", "SIX", "OPP", "OPP_RANK", "N_TEAMS", "PIN", "EVENT_ID"]
SHOW_COLS = ["PICK", "GAME", "ODDS", "BLEND_%", "EV_%", "OPP_VS_POS", "SIX", "PIN", "BOTH"]
DETAIL_COLS = ["POS", "MODEL_%", "MARKET_%", "EDGE_PP", "MODEL", "MATCHUP", "SCORE"]


def make_row(res: Mapping[str, Any], stat: str, line: float, game: str, pos: Optional[str], opp: Optional[str],
             rank: Optional[float], n_teams: Optional[float], pin: str = "") -> Optional[Dict[str, Any]]:
    """One candidate row from an evaluation result: the six-point side when it scores 5/6 or 6/6, otherwise the side with the
    higher model EV. None when the result has no sides."""
    sides = res.get("sides") or {}
    if not {"Over", "Under"} <= set(sides):
        return None
    six_all = res.get("six_point") or {}
    best = six_all.get("best")
    if best in sides and float((six_all.get(best) or {}).get("score") or 0) >= SHARP_SIX:
        side = best
    else:
        side = max(("Over", "Under"), key=lambda s: float(sides[s].get("ev_pct", -1e9)))
    s = sides[side]
    six = (res.get("six_point") or {}).get(side) or {}
    return {"PLAYER": str(res.get("player_name", "")), "POS": pos or "", "STAT": str(stat), "SIDE": side, "LINE": float(line), "ODDS": float(s.get("odds", np.nan)),
            "GAME": str(game or ""), "P_MODEL": float(s.get("p_model", np.nan)), "P_MARKET": float(s.get("p_market", np.nan)),
            "EDGE_PP": float(s.get("edge_pp", np.nan)), "EV_%": float(s.get("ev_pct", np.nan)), "SIX": six.get("score"), "OPP": opp or "",
            "OPP_RANK": rank, "N_TEAMS": n_teams, "PIN": pin or ""}


def matchup_score(side: pd.Series, rank: pd.Series, n_teams: pd.Series) -> pd.Series:
    """0-100: how much the opponent's rank vs the position helps this side (rank 1 = softest helps Overs). Missing rank -> 50."""
    r, n = pd.to_numeric(rank, errors="coerce"), pd.to_numeric(n_teams, errors="coerce")
    soft = (n - r) / (n - 1.0)                                  # 1 at rank 1 (softest), 0 at rank N (toughest)
    val = np.where(side.astype(str).str.lower() == "under", 1.0 - soft, soft)
    ok = r.notna() & n.notna() & (n > 1)
    return pd.Series(np.where(ok, 100.0 * np.clip(val, 0.0, 1.0), 50.0), index=side.index, dtype=float)


def _decimal(odds: pd.Series) -> pd.Series:
    o = pd.to_numeric(odds, errors="coerce")
    return pd.Series(np.where(o > 0, 1.0 + o / 100.0, 1.0 + 100.0 / -o), index=odds.index).where(o.abs() >= 100)


def rank_picks(rows: Sequence[Mapping[str, Any]], w_model: float = DEFAULT_WEIGHT, min_edge: float = 0.0,
               w_blend: Optional[float] = None) -> pd.DataFrame:
    """Score and sort candidate rows (see module docstring). Returns ROW_COLS plus BLEND_%, MODEL, MATCHUP, SCORE, BOTH.
    With `w_blend` the edge, EV and BOTH use the market-blended chance; otherwise BLEND_% equals the model chance."""
    import prop_review as cal
    extra = ["BLEND_%", "MODEL", "MATCHUP", "SCORE", "BOTH", "SHARP", "RANK"]
    d = pd.DataFrame(list(rows))
    if d.empty:
        return pd.DataFrame(columns=ROW_COLS + extra)
    for c in ROW_COLS:
        if c not in d.columns:
            d[c] = np.nan
    pmod, pmkt = pd.to_numeric(d["P_MODEL"], errors="coerce"), pd.to_numeric(d["P_MARKET"], errors="coerce")
    if w_blend is None:
        d["BLEND_%"] = pmod
    else:
        d["BLEND_%"] = cal.blend(pmod.to_numpy(), pmkt.to_numpy(), float(w_blend))
        d["EDGE_PP"] = d["BLEND_%"] - pmkt
        d["EV_%"] = (d["BLEND_%"] / 100.0 * _decimal(d["ODDS"]) - 1.0) * 100.0
    edge = pd.to_numeric(d["EDGE_PP"], errors="coerce")
    d = d[edge.notna() & (edge >= float(min_edge))].copy()
    if d.empty:
        return pd.DataFrame(columns=ROW_COLS + extra)
    w = float(np.clip(w_model, 0.0, 1.0))
    e, ev = pd.to_numeric(d["EDGE_PP"], errors="coerce"), pd.to_numeric(d["EV_%"], errors="coerce")
    d["MODEL"] = 100.0 * np.clip(ev.fillna(0.0) / EV_CAP, 0.0, 1.0)
    d["MATCHUP"] = matchup_score(d["SIDE"], d["OPP_RANK"], d["N_TEAMS"])
    d["SCORE"] = w * d["MODEL"] + (1.0 - w) * d["MATCHUP"]
    model_ok = (ev > 0) & ((e >= BOTH_EDGE) if w_blend is None else (e > 0))
    has_rank = pd.to_numeric(d["OPP_RANK"], errors="coerce").notna()
    d["BOTH"] = model_ok & (d["MATCHUP"] >= BOTH_MATCHUP) & has_rank
    d["SHARP"] = (pd.to_numeric(d["SIX"], errors="coerce") >= SHARP_SIX) & (d["MATCHUP"] >= BOTH_MATCHUP) & has_rank
    d = d.sort_values(["SCORE", "EV_%"], ascending=[False, False]).reset_index(drop=True)
    d["RANK"] = np.arange(1, len(d) + 1)
    return d[ROW_COLS + extra]


def sharp_picks(rows: Sequence[Mapping[str, Any]], w_model: float = DEFAULT_WEIGHT, w_blend: Optional[float] = None) -> pd.DataFrame:
    """Every 5/6 or 6/6 prop with a top-third matchup for its side (no minimum edge), sorted by SIX, then SCORE."""
    r = rank_picks(rows, w_model, -1e9, w_blend)
    if r.empty:
        return r
    s = r[r["SHARP"]].copy()
    s["_SIX"] = pd.to_numeric(s["SIX"], errors="coerce")
    return s.sort_values(["_SIX", "SCORE"], ascending=[False, False]).drop(columns="_SIX").reset_index(drop=True)


def cap_per_game(ranked: pd.DataFrame, per_game: int) -> pd.DataFrame:
    """Keep at most `per_game` picks from each game, in ranked order (0 = no limit)."""
    if ranked.empty or int(per_game) <= 0:
        return ranked
    return ranked[ranked.groupby("GAME", sort=False).cumcount() < int(per_game)].reset_index(drop=True)


def display(ranked: pd.DataFrame, top: int = 25, details: bool = False) -> pd.DataFrame:
    """The table shown on screen, top `top` rows: SHOW_COLS (what you decide with), plus DETAIL_COLS (the scoring parts) when `details`."""
    cols = SHOW_COLS + (DETAIL_COLS if details else [])
    d = ranked.head(int(top))
    if d.empty:
        return pd.DataFrame(columns=cols)
    rk, n = pd.to_numeric(d["OPP_RANK"], errors="coerce"), pd.to_numeric(d["N_TEAMS"], errors="coerce")
    vs = np.where(rk.notna(), d["OPP"].astype(str) + " #" + rk.fillna(0).astype(int).astype(str) + " of " + n.fillna(0).astype(int).astype(str), "no rank")
    return pd.DataFrame({
        "PICK": d["PLAYER"].astype(str) + " " + d["SIDE"].astype(str) + " " + d["LINE"].map(lambda x: f"{x:g}") + " " + d["STAT"].astype(str),
        "POS": d["POS"], "GAME": d["GAME"], "ODDS": d["ODDS"].map(lambda x: "" if pd.isna(x) else f"{int(x):+d}"),
        "MODEL_%": d["P_MODEL"].round(1), "MARKET_%": d["P_MARKET"].round(1),
        "BLEND_%": pd.to_numeric(d["BLEND_%"], errors="coerce").round(1) if "BLEND_%" in d else d["P_MODEL"].round(1), "EDGE_PP": d["EDGE_PP"].round(1), "EV_%": d["EV_%"].round(1),
        "OPP_VS_POS": vs, "MODEL": d["MODEL"].round(0), "MATCHUP": d["MATCHUP"].round(0), "SCORE": d["SCORE"].round(1),
        "BOTH": d["BOTH"].map({True: "✓", False: ""}), "SIX": d["SIX"].map(lambda x: "" if x is None or pd.isna(x) else f"{int(x)}/6"), "PIN": d["PIN"]},
        columns=cols).reset_index(drop=True)


def due(last_ts: Optional[float], minutes: float, now: Optional[float] = None) -> bool:
    """True when a timed refresh should run: the timer is on and the last scan is older than `minutes` (or never ran)."""
    if not minutes or minutes <= 0:
        return False
    now = time.time() if now is None else now
    return last_ts is None or (now - float(last_ts)) >= float(minutes) * 60.0


def credit_estimate(n_games: int, n_stats: int) -> int:
    """Odds API credits one full refresh costs: one per (game, stat) board that is not already cached."""
    return int(max(n_games, 0)) * int(max(n_stats, 0))


def scan(games: pd.DataFrame, stats: Sequence[str], fetch_board: Callable[[Mapping[str, Any], str], pd.DataFrame],
         evaluate: Callable[[Mapping[str, Any], str, str, float, float, float], Optional[Dict[str, Any]]],
         cache: Dict[Any, Any], progress: Any = None) -> Dict[str, Any]:
    """Evaluate every (game, stat, player). `cache` keeps rows of props whose line and prices have not changed, so a refresh only re-runs what moved.

    Returns {"rows": [...], "errors": [...], "evaluated": n, "reused": n}.
    """
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    evaluated = reused = 0
    jobs = [(g, s) for _, g in games.iterrows() for s in stats]
    for k, (g, stat) in enumerate(jobs):
        if progress is not None:
            progress.progress(k / max(len(jobs), 1), text=f"{g.get('label', '')}: {stat} ({k + 1}/{len(jobs)})")
        try:
            board = fetch_board(g, stat)
        except Exception as e:                      # one failed board must not stop the slate
            errors.append(f"{g.get('label', '')} / {stat}: {str(e)[:90]}")
            continue
        if board is None or board.empty:
            continue
        for _, r in board.iterrows():
            if pd.isna(r.get("Line")) or pd.isna(r.get("Over")) or pd.isna(r.get("Under")):
                continue
            key = (str(g.get("id")), str(stat), str(r["Player"]), float(r["Line"]), float(r["Over"]), float(r["Under"]))
            if key in cache:
                reused += 1
                row = cache[key]
            else:
                try:
                    row = evaluate(g, stat, str(r["Player"]), float(r["Line"]), float(r["Over"]), float(r["Under"]))
                except Exception:
                    row = None
                cache[key] = row
                evaluated += 1
            if row:
                rows.append({**row, "EVENT_ID": str(g.get("id", ""))})
    if progress is not None:
        progress.progress(1.0, text="Scan complete")
    return {"rows": rows, "errors": errors, "evaluated": evaluated, "reused": reused}


def default_games(games: pd.DataFrame) -> List[str]:
    """Labels of the games on the earliest day in the list (tonight's NBA slate / the next NFL game day)."""
    if games is None or games.empty:
        return []
    days = games["day"].astype(str)
    known = days[days != ""]
    return games["label"].tolist() if known.empty else games.loc[days == known.min(), "label"].tolist()


def selected_row(event: Any) -> Optional[int]:
    """Index of the row picked in a st.dataframe selection event (None if nothing is selected)."""
    sel = getattr(event, "selection", None)
    if sel is None and isinstance(event, Mapping):
        sel = event.get("selection")
    rows = getattr(sel, "rows", None)
    if rows is None and isinstance(sel, Mapping):
        rows = sel.get("rows")
    return int(rows[0]) if rows else None


def _clickable_table(st: Any, shown: pd.DataFrame, ranked: pd.DataFrame, open_pick: Callable[[Mapping[str, Any]], str]) -> None:
    """The picks table with single-row selection: picking a row loads it into the Evaluate tab (once per pick)."""
    import zlib
    ss = st.session_state
    key = "bp_table_" + str(zlib.crc32("|".join(shown["PICK"]).encode()))      # new list -> fresh selection, so an old click never opens a different prop
    st.caption("👆 Click a row (the box at its left edge) to load that prop into the **⚡ Evaluate** tab, then open that tab.")
    try:
        event = st.dataframe(shown.set_index("PICK"), use_container_width=True, on_select="rerun", selection_mode="single-row", key=key)
    except TypeError:                                                          # Streamlit too old for row selection: fall back to a picker
        st.dataframe(shown.set_index("PICK"), use_container_width=True)
        pick = st.selectbox("Open a pick in the Evaluate tab", [""] + shown["PICK"].tolist(), key="bp_open_pick")
        event = {"selection": {"rows": [shown["PICK"].tolist().index(pick)] if pick else []}}
    i = selected_row(event)
    if i is None or i >= len(ranked):
        return
    token = (key, shown.iloc[i]["PICK"])
    if ss.get("bp_opened") == token:
        if ss.get("bp_open_msg"):
            st.success(ss["bp_open_msg"])
        return
    ss["bp_opened"] = token
    try:
        ss["bp_open_msg"] = open_pick(ranked.iloc[i].to_dict())
    except Exception as e:
        ss["bp_open_msg"] = ""
        st.warning(f"Couldn't load that prop: {str(e)[:120]}")
        return
    if ss["bp_open_msg"]:
        st.success(ss["bp_open_msg"])


def render(st: Any, *, stats: Sequence[str], default_stats: Sequence[str], list_games: Callable[[], pd.DataFrame],
           fetch_board: Callable[[Mapping[str, Any], str], pd.DataFrame],
           evaluate: Callable[[Mapping[str, Any], str, str, float, float, float], Optional[Dict[str, Any]]],
           keep_keys: Iterable[str] = (), after_scan: Optional[Callable[[], None]] = None,
           star: Optional[Callable[[pd.DataFrame], int]] = None, credits_note: Optional[Callable[[], str]] = None,
           blend_info: Optional[Callable[[], Mapping[str, Any]]] = None,
           open_pick: Optional[Callable[[Mapping[str, Any]], str]] = None,
           tag: Optional[Callable[[pd.DataFrame], None]] = None) -> None:
    """The Best Picks tab. `keep_keys` are session-state keys the callbacks overwrite; they are restored after a scan.
    `star(rows)` favorites the given ranked rows and returns how many it starred.
    `open_pick(row)` loads a clicked row into the Evaluate form and returns a message for the user.
    `tag(rows)` saves the list right after each scan (rank, side, ✓) on the props' snapshots, so Daily results can grade it."""
    ss = st.session_state
    st.markdown("One ranked list across the slate, ranked mainly by **EV** at DraftKings' price (your model blended with the market), with the "
                "opponent's rank against the player's position as the tiebreaker. ✓ in BOTH = still +EV *and* a top-third matchup.")
    if st.button("Load / reload the game list", key="bp_load") or "bp_games" not in ss:
        try:
            ss["bp_games"] = list_games()
        except Exception as e:
            ss["bp_games"] = pd.DataFrame(columns=["id", "label", "day"])
            st.error(f"Couldn't load games: {str(e)[:150]}")
    games = ss.get("bp_games")
    if games is None or games.empty:
        st.info("No upcoming games loaded. Check the Odds API key, then press Load.")
        return
    c1, c2 = st.columns(2)
    pick_stats = c1.multiselect("Stats to scan", list(stats), default=[s for s in default_stats if s in stats], key="bp_stats")
    pick_games = c2.multiselect("Games to scan", games["label"].tolist(), default=default_games(games), key="bp_games_pick")
    c3, c4, c5, c6 = st.columns(4)
    w = float(c3.slider("EV weight % (rest = matchup)", 0, 100, int(DEFAULT_WEIGHT * 100), 5, key="bp_w")) / 100.0
    use_blend = blend_info is not None and bool(st.checkbox("Respect the market (rank by the market-blended chance)", value=True, key="bp_blend",
                                                             help="Blends your model with DraftKings' no-vig price using the weight fitted from your graded props (Snapshots tab)."))
    info = blend_info() if use_blend else None
    min_edge = float(c4.number_input("Minimum edge (points)", value=0.5, step=0.5, key="bp_min_edge"))
    top = int(c5.number_input("Show top", min_value=5, max_value=200, value=25, step=5, key="bp_top"))
    timer_label = c6.selectbox("Auto-refresh", list(TIMER_CHOICES), key="bp_timer")
    view_name = st.radio("Show", VIEWS, horizontal=True, key="bp_view",
                         help="5/6 & 6/6 with a matchup edge: every prop that passes 5 or 6 of the six checks AND faces a top-third defense matchup "
                              "for that side (soft defense for an Over, tough for an Under). The edge, top-N and per-game limits don't apply to it.")
    sharp_view = view_name == VIEWS[0]
    hide_neg = sharp_view and bool(st.checkbox("Hide negative-EV picks", value=True, key="bp_hide_neg",
                                               help="Hides 5/6 and 6/6 props whose price at DraftKings is too short to profit even if the pick is right. "
                                                    "They are still saved for grading (Daily results → Best Picks 5/6+ & matchup, negative EV)."))
    c7, c8 = st.columns(2)
    per_game = int(c7.number_input("Max picks per game (0 = no limit)", min_value=0, max_value=20, value=DEFAULT_PER_GAME, step=1, key="bp_per_game",
                                   help="Props from the same game tend to win or lose together, so a list full of one game is riskier than it looks."))
    details = bool(c8.checkbox("Show scoring details (model %, market %, MODEL / MATCHUP / SCORE)", value=False, key="bp_details"))
    minutes = TIMER_CHOICES.get(timer_label, 0)
    sel = games[games["label"].isin(pick_games or [])]
    cost = credit_estimate(len(sel), len(pick_stats or []))
    st.caption(f"A full refresh pulls up to {cost} boards = up to **{cost} Odds API credits** (boards pulled in the last 10 minutes are reused for free). "
               "Auto-refresh runs only while this app is open in your browser." + (f" {credits_note()}" if credits_note else ""))
    manual = st.button("🔄 Refresh now", key="bp_refresh")
    timed = due(ss.get("bp_time"), minutes) and bool(len(sel)) and bool(pick_stats)
    just_scanned = False
    if (manual or timed) and len(sel) and pick_stats:
        just_scanned = True
        saved = {k: ss[k] for k in keep_keys if k in ss}
        missing = [k for k in keep_keys if k not in ss]
        bar = st.progress(0.0, text="Starting scan...")
        try:
            out = scan(sel, list(pick_stats), fetch_board, evaluate, ss.setdefault("bp_cache", {}), bar)
        finally:
            for k in missing:
                ss.pop(k, None)
            ss.update(saved)
            ss["bp_time"] = time.time()                 # also after a failure, so the timer never loops on an error
        ss["bp_rows"], ss["bp_errors"] = out["rows"], out["errors"]
        ss["bp_note"] = f"{out['evaluated']} props evaluated, {out['reused']} unchanged props reused."
        if after_scan:
            after_scan()
    elif manual:
        st.warning("Pick at least one stat and one game.")
    rows = ss.get("bp_rows")
    if rows is None:
        st.info("Press **Refresh now** to build the list. The first scan loads every player's game log, so it can take a few minutes.")
    else:
        wb = None if info is None else float(info["w"])
        ranked = cap_per_game(rank_picks(rows, w, min_edge, wb), per_game)
        sharp = sharp_picks(rows, w, wb)
        if just_scanned and tag is not None:
            listed = pd.concat([ranked.head(top), sharp[~sharp.set_index(["PLAYER", "STAT", "LINE"]).index.isin(
                ranked.head(top).set_index(["PLAYER", "STAT", "LINE"]).index)]], ignore_index=True)
            try:
                tag(listed)
            except Exception as e:
                st.caption(f"Couldn't save this list for grading ({str(e)[:80]}).")
        n_hidden = 0
        if sharp_view:
            ev_ok = pd.to_numeric(sharp["EV_%"], errors="coerce") > 0
            n_hidden = int((~ev_ok).sum()) if hide_neg else 0
            ranked = sharp[ev_ok].reset_index(drop=True) if hide_neg else sharp
            top = max(len(ranked), 1)
        if info is not None:
            import prop_review as cal
            st.caption(f"Edges and EV use the market-blended chance: {cal.weight_note(info)}. BOTH ✓ = still +EV after the blend and a top-third matchup.")
        age = int((time.time() - float(ss.get("bp_time", time.time()))) // 60)
        st.caption(f"Last scan {age} min ago. {ss.get('bp_note', '')} "
                   + (f"{len(ranked) + n_hidden} of {len(rows)} props are 5/6 or 6/6 with a top-third matchup"
                      + (f"; {n_hidden} with negative EV hidden." if n_hidden else ".") if sharp_view
                      else f"{len(ranked)} of {len(rows)} props clear the minimum edge."))
        for e in (ss.get("bp_errors") or [])[:5]:
            st.warning(e)
        if ranked.empty:
            st.info((f"All {n_hidden} props that are 5/6 or 6/6 with a top-third matchup have negative EV, so they're hidden. Untick "
                     "Hide negative-EV picks to see them." if n_hidden else
                     "No prop is 5/6 or 6/6 with a top-third matchup right now. Scan more stats or games, or switch to All ranked picks.")
                    if sharp_view else "No prop clears the minimum edge right now. Lower it to see more.")
        else:
            shown = display(ranked, top, details)
            if open_pick is None:
                st.dataframe(shown.set_index("PICK"), use_container_width=True)
            else:
                _clickable_table(st, shown, ranked.head(top), open_pick)
            st.caption("BLEND_% = your model's chance blended with the market; EV_% = what a bet at DraftKings' price is worth at that chance. "
                       "OPP_VS_POS: opponent and its rank vs this position for this stat (#1 = allows the most). PIN = Pinnacle cross-check for that side."
                       + (" Showing every 5/6 and 6/6 prop with a top-third matchup" + (" and positive EV." if hide_neg else
                          "; EV_% below 0 means DraftKings' price is too short even so.")
                          if sharp_view else (f" At most {per_game} pick(s) per game." if per_game else "")))
            if star is not None:
                picks = st.multiselect("⭐ Favorite picks for the Parlay tab", shown["PICK"].tolist(), key="bp_star_pick")
                s1, s2 = st.columns(2)
                chosen: Optional[pd.DataFrame] = None
                if s1.button("⭐ Favorite selected", key="bp_star") and picks:
                    chosen = ranked.head(top)[shown["PICK"].isin(picks).values]
                if s2.button("⭐ Favorite every ✓ BOTH pick shown", key="bp_star_both"):
                    chosen = ranked.head(top)[ranked.head(top)["BOTH"]]
                if chosen is not None:
                    n = star(chosen) if len(chosen) else 0
                    (st.success if n else st.warning)(f"{n} pick(s) starred: they are now in the Parlay tab." if n else "Nothing to star.")
    st.caption("Limits: the list is only as good as your model, even blended. The position rank is context the model does not use directly and "
               "hasn't been shown to predict results on its own, which is why it only breaks ties by default. Each scan's list is saved, so "
               "Snapshots -> Daily results can show how the top 10 and the ✓ BOTH picks actually did. Check the price on DraftKings before betting.")
    if minutes and hasattr(st, "fragment"):
        @st.fragment(run_every=60)
        def _tick() -> None:
            if due(st.session_state.get("bp_time"), minutes):
                st.rerun()
        try:
            _tick()
        except Exception:
            pass
    elif minutes:
        st.caption("This Streamlit version has no timer support, so auto-refresh is off: use Refresh now.")
