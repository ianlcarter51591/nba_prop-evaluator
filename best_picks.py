"""
Best Picks: one ranked list of the slate's props, by the model AND by the opponent's rank against the player's position.

Math
----
For each prop the app evaluates, the side kept is the one with the higher model EV. Then, on a 0-100 scale:
    MODEL    = 100 * clip(EDGE_PP / EDGE_CAP, 0, 1)        EDGE_PP = model % - DraftKings no-vig % (points), EDGE_CAP = 10
    MATCHUP  = 100 * (N - R) / (N - 1)   for an Over       R = opponent's rank vs the player's position for this stat (1 = softest,
             = 100 * (R - 1) / (N - 1)   for an Under          allows the most), N = teams ranked.  No rank -> 50 (neutral).
    SCORE    = w * MODEL + (1 - w) * MATCHUP               w = the "model weight" slider (default 0.60)
    BOTH     = EDGE_PP >= BOTH_EDGE and MATCHUP >= BOTH_MATCHUP   (model edge AND a top-third matchup for that side)
Rows are sorted BOTH first, then SCORE. Props whose best side has EDGE_PP below the minimum edge are dropped.

Market blend (on by default, see prop_review.py): the model chance is first blended with the market in log-odds with weight w,
    BLEND_% = blend(P_MODEL, P_MARKET, w),  EDGE_PP = BLEND_% - P_MARKET,  EV_% = BLEND_% * decimal(ODDS) - 100,
MODEL = 100 * clip(EDGE_PP / (EDGE_CAP * max(w, 0.2)), 0, 1)   (blending shrinks edges by about w, so the cap shrinks with it), and
BOTH needs a blended EV above 0 (still +EV once the market is respected) plus the top-third matchup.

The UI part (render) is the same in both apps; each app passes callbacks that fetch a board and evaluate one prop.
Auto-refresh only runs while the app is open in a browser, and every refresh of a (game, stat) board costs Odds API credits.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

EDGE_CAP = 10.0
BOTH_EDGE = 2.0
BOTH_MATCHUP = 200.0 / 3.0
DEFAULT_WEIGHT = 0.60
TIMER_CHOICES: Dict[str, int] = {"Off (manual only)": 0, "Every 15 min": 15, "Every 30 min": 30, "Every 60 min": 60}
ROW_COLS = ["PLAYER", "POS", "STAT", "SIDE", "LINE", "ODDS", "GAME", "P_MODEL", "P_MARKET", "EDGE_PP", "EV_%", "SIX", "OPP", "OPP_RANK", "N_TEAMS", "PIN"]
SHOW_COLS = ["PICK", "POS", "GAME", "ODDS", "MODEL_%", "MARKET_%", "BLEND_%", "EDGE_PP", "EV_%", "OPP_VS_POS", "MODEL", "MATCHUP", "SCORE", "BOTH", "SIX", "PIN"]


def make_row(res: Mapping[str, Any], stat: str, line: float, game: str, pos: Optional[str], opp: Optional[str],
             rank: Optional[float], n_teams: Optional[float], pin: str = "") -> Optional[Dict[str, Any]]:
    """One candidate row from an evaluation result: the side with the higher model EV. None when the result has no sides."""
    sides = res.get("sides") or {}
    if not {"Over", "Under"} <= set(sides):
        return None
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
    extra = ["BLEND_%", "MODEL", "MATCHUP", "SCORE", "BOTH"]
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
    cap = EDGE_CAP if w_blend is None else EDGE_CAP * max(float(w_blend), 0.2)
    e = pd.to_numeric(d["EDGE_PP"], errors="coerce")
    d["MODEL"] = 100.0 * np.clip(e / cap, 0.0, 1.0)
    d["MATCHUP"] = matchup_score(d["SIDE"], d["OPP_RANK"], d["N_TEAMS"])
    d["SCORE"] = w * d["MODEL"] + (1.0 - w) * d["MATCHUP"]
    model_ok = (e >= BOTH_EDGE) if w_blend is None else ((pd.to_numeric(d["EV_%"], errors="coerce") > 0) & (e > 0))
    d["BOTH"] = model_ok & (d["MATCHUP"] >= BOTH_MATCHUP) & pd.to_numeric(d["OPP_RANK"], errors="coerce").notna()
    return d.sort_values(["BOTH", "SCORE", "EV_%"], ascending=[False, False, False]).reset_index(drop=True)[ROW_COLS + extra]


def display(ranked: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    """The table shown on screen (SHOW_COLS), top `top` rows."""
    d = ranked.head(int(top))
    if d.empty:
        return pd.DataFrame(columns=SHOW_COLS)
    rk, n = pd.to_numeric(d["OPP_RANK"], errors="coerce"), pd.to_numeric(d["N_TEAMS"], errors="coerce")
    vs = np.where(rk.notna(), d["OPP"].astype(str) + " #" + rk.fillna(0).astype(int).astype(str) + " of " + n.fillna(0).astype(int).astype(str), "no rank")
    return pd.DataFrame({
        "PICK": d["PLAYER"].astype(str) + " " + d["SIDE"].astype(str) + " " + d["LINE"].map(lambda x: f"{x:g}") + " " + d["STAT"].astype(str),
        "POS": d["POS"], "GAME": d["GAME"], "ODDS": d["ODDS"].map(lambda x: "" if pd.isna(x) else f"{int(x):+d}"),
        "MODEL_%": d["P_MODEL"].round(1), "MARKET_%": d["P_MARKET"].round(1),
        "BLEND_%": pd.to_numeric(d["BLEND_%"], errors="coerce").round(1) if "BLEND_%" in d else d["P_MODEL"].round(1), "EDGE_PP": d["EDGE_PP"].round(1), "EV_%": d["EV_%"].round(1),
        "OPP_VS_POS": vs, "MODEL": d["MODEL"].round(0), "MATCHUP": d["MATCHUP"].round(0), "SCORE": d["SCORE"].round(1),
        "BOTH": d["BOTH"].map({True: "✓", False: ""}), "SIX": d["SIX"].map(lambda x: "" if x is None or pd.isna(x) else f"{int(x)}/6"), "PIN": d["PIN"]},
        columns=SHOW_COLS).reset_index(drop=True)


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
                rows.append(row)
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


def render(st: Any, *, stats: Sequence[str], default_stats: Sequence[str], list_games: Callable[[], pd.DataFrame],
           fetch_board: Callable[[Mapping[str, Any], str], pd.DataFrame],
           evaluate: Callable[[Mapping[str, Any], str, str, float, float, float], Optional[Dict[str, Any]]],
           keep_keys: Iterable[str] = (), after_scan: Optional[Callable[[], None]] = None,
           star: Optional[Callable[[pd.DataFrame], int]] = None, credits_note: Optional[Callable[[], str]] = None,
           blend_info: Optional[Callable[[], Mapping[str, Any]]] = None) -> None:
    """The Best Picks tab. `keep_keys` are session-state keys the callbacks overwrite; they are restored after a scan.
    `star(rows)` favorites the given ranked rows and returns how many it starred."""
    ss = st.session_state
    st.markdown("One ranked list across the slate: **MODEL** (your model's edge over DraftKings' no-vig price) and **MATCHUP** "
                "(the opponent's rank against the player's position for that stat). ✓ in BOTH = a model edge *and* a top-third matchup.")
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
    w = float(c3.slider("Model weight % (rest = matchup)", 0, 100, int(DEFAULT_WEIGHT * 100), 5, key="bp_w")) / 100.0
    use_blend = blend_info is not None and bool(st.checkbox("Respect the market (rank by the market-blended chance)", value=True, key="bp_blend",
                                                             help="Blends your model with DraftKings' no-vig price using the weight fitted from your graded props (Snapshots tab)."))
    info = blend_info() if use_blend else None
    min_edge = float(c4.number_input("Minimum edge (points)", value=0.5, step=0.5, key="bp_min_edge"))
    top = int(c5.number_input("Show top", min_value=5, max_value=200, value=25, step=5, key="bp_top"))
    timer_label = c6.selectbox("Auto-refresh", list(TIMER_CHOICES), key="bp_timer")
    minutes = TIMER_CHOICES.get(timer_label, 0)
    sel = games[games["label"].isin(pick_games or [])]
    cost = credit_estimate(len(sel), len(pick_stats or []))
    st.caption(f"A full refresh pulls up to {cost} boards = up to **{cost} Odds API credits** (boards pulled in the last 10 minutes are reused for free). "
               "Auto-refresh runs only while this app is open in your browser." + (f" {credits_note()}" if credits_note else ""))
    manual = st.button("🔄 Refresh now", key="bp_refresh")
    timed = due(ss.get("bp_time"), minutes) and bool(len(sel)) and bool(pick_stats)
    if (manual or timed) and len(sel) and pick_stats:
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
        ranked = rank_picks(rows, w, min_edge, None if info is None else float(info["w"]))
        if info is not None:
            import prop_review as cal
            st.caption(f"Edges and EV use the market-blended chance: {cal.weight_note(info)}. BOTH ✓ = still +EV after the blend and a top-third matchup.")
        age = int((time.time() - float(ss.get("bp_time", time.time()))) // 60)
        st.caption(f"Last scan {age} min ago. {ss.get('bp_note', '')} {len(ranked)} of {len(rows)} props clear the minimum edge.")
        for e in (ss.get("bp_errors") or [])[:5]:
            st.warning(e)
        if ranked.empty:
            st.info("No prop clears the minimum edge right now. Lower it to see more.")
        else:
            shown = display(ranked, top)
            st.dataframe(shown, use_container_width=True, hide_index=True)
            st.caption("OPP_VS_POS: opponent and its rank vs this position for this stat (#1 = allows the most). MATCHUP is high when that rank helps the side shown "
                       "(soft defense for an Over, tough defense for an Under). PIN = Pinnacle cross-check for that side.")
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
    st.caption("Limits: MODEL is your own model's edge, so the list is only as good as the model. The position rank is context the model does not "
               "use directly, and it has not been shown to predict results on its own (see the backtest tab). Check the price on DraftKings before betting.")
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
