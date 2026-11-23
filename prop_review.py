"""
Prop review tools in one file: market blend + calibration, closing-line value, and daily win %.
The same file is used by the NBA and NFL apps.

1. MARKET BLEND + CALIBRATION
-----------------------------
Market blend + calibration, fitted from your own graded snapshots.

Math
----
Every graded snapshot with a known result gives one Over/Under outcome (pushes dropped):
    y   = 1 if ACTUAL > LINE else 0
    pm  = P_OVER / (P_OVER + P_UNDER)            your model's Over chance
    pk  = PM_OVER / (PM_OVER + PM_UNDER)         DraftKings' no-vig Over chance
Blend in log-odds (logit) space, both clipped to [1%, 99%]:
    logit(pb) = w * logit(pm) + (1 - w) * logit(pk)          w = 1: model only, w = 0: market only
w_fit = the w on a 0.00-1.00 grid (step 0.05) with the lowest mean log loss  -(y ln p + (1-y) ln(1-p)).
It is shrunk toward a prior because a few dozen props can't pin it down:
    w = (n * w_fit + N0 * W_PRIOR) / (n + N0)       N0 = 150 pseudo-props, W_PRIOR = 0.30
W_PRIOR = 0.30 is an assumption (a player-prop model usually deserves less weight than the market), not a measured number;
it fades out as your graded props pile up.
Blended EV at DraftKings' price:  EV = pb * decimal_odds - 1  (pushes ignored).
Calibration: for the side the model favored, group picks by predicted chance and compare with how often that side won.
The log-loss comparison is in-sample for the blend (it was fitted on the same props), so it flatters the blend slightly.

2. CLOSING-LINE VALUE (CLV)
---------------------------
Closing-line value (CLV): did the price the app showed beat the last DraftKings price seen before the game started?

Math
----
Each board pull (Step 3 or a Best Picks refresh) is a price observation. For every saved prop of that player, stat and game date
whose snapshot was taken BEFORE the observation, and only while the game has not started, the observation becomes the new
"close": CLOSE_LINE, CLOSE_OVER, CLOSE_UNDER, CLOSE_TS and CLOSE_MIN (minutes before the start). The last one before the game wins.

For the side the app showed (Yes counts as Over):
    same line:   p_open  = DraftKings no-vig chance of that side when the snapshot was taken (PM_OVER / PM_UNDER)
                 p_close = no-vig chance of that side in the close
                 CLV_PP  = 100 * (p_close - p_open)                   market moved toward the pick when > 0
                 CLV_EV  = 100 * (p_close * decimal(price taken) - 1)  value of the price you could have taken, judged by the close
                 BEAT    = CLV_EV > 0
    line moved:  LINE_MOVE = close line - line (Over) or line - close line (Under)   > 0 = moved toward the pick
                 BEAT    = LINE_MOVE > 0   (prices across different lines are not compared)
A prop only counts once its game date has passed or it has been graded (before that the "close" may still change).
The close is only as late as your last board pull: pull once close to tip-off / kickoff, or the number is an early-close proxy.

3. DAILY RESULTS (win % by game day)
------------------------------------
Daily results: win % of the app's picks for one game day, graded automatically from the saved snapshots.

Math
----
For the chosen day and filter, using the side the app showed (SHOWN_RESULT = Hit / Miss / Push):
    WIN_% = hits / (hits + misses)        pushes are refunds and left out of the rate
"Best Picks top 10" / "Best Picks ✓ BOTH" use the list saved by the last Best Picks scan before each game, graded on the side it ranked.
"Placed bets" / "Placed parlay legs" use only props you recorded with Place Bet / "I placed this ticket", graded on the side YOU bet
(PLACED_SIDE vs ACTUAL and LINE) at the price you recorded (parlay legs: win % only).
    BREAK_EVEN_% = 52.38                  the win rate a -110 bet needs (110 / 210)
Profit, staking 1 unit on each pick at the DraftKings price the app showed for that side (OVER_ODDS or UNDER_ODDS):
    hit  -> + (decimal odds - 1)          e.g. -110 wins 0.909 u, +120 wins 1.20 u
    miss -> - 1 u                         push -> 0 (refund)
    ROI_%          = 100 * units / picks staked (hits + misses + pushes)
    PRICE_BE_%     = 100 * mean(1 / decimal odds) over the decided picks: the win rate those exact prices needed
Picks with no saved price are counted in the win % but left out of units and ROI.
Default day = the latest game day before today that has graded props (yesterday's slate for the NBA, the last game day for the NFL).
Grading runs once per session when the tab opens and props from earlier days are still pending.
"""
from __future__ import annotations

from datetime import datetime
import re
import unicodedata
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


# ====================================================================================================
# 1. MARKET BLEND + CALIBRATION
# ====================================================================================================

W_PRIOR = 0.30
N0 = 150.0
GRID = np.round(np.linspace(0.0, 1.0, 21), 2)
P_CLIP = (0.01, 0.99)
BINS = [0.50, 0.55, 0.60, 0.65, 0.70, 1.0001]
BIN_LABELS = ["50-55%", "55-60%", "60-65%", "65-70%", "70%+"]


def _logit(p: Any) -> np.ndarray:
    p = np.clip(np.asarray(p, float), *P_CLIP)
    return np.log(p / (1.0 - p))


def blend(p_model: Any, p_market: Any, w: float) -> np.ndarray:
    """Blended probability (same scale as the inputs: 0-1 or 0-100). Where the market is missing the model is returned unchanged."""
    pm, pk = np.asarray(p_model, float), np.asarray(p_market, float)
    scale = 100.0 if np.nanmax(np.r_[pm.ravel(), pk.ravel(), 0.0]) > 1.0 else 1.0
    z = w * _logit(pm / scale) + (1.0 - w) * _logit(pk / scale)
    out = scale / (1.0 + np.exp(-z))
    return np.where(np.isfinite(pk), out, pm)


def log_loss(p: Any, y: Any) -> float:
    p, y = np.clip(np.asarray(p, float), *P_CLIP), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))) if len(y) else float("nan")


def graded(snaps: Optional[pd.DataFrame]) -> pd.DataFrame:
    """One row per graded prop with Y (Over won), PM (model Over chance) and PK (market Over chance), all 0-1."""
    cols = ["Y", "PM", "PK"]
    need = {"STATUS", "ACTUAL", "LINE", "P_OVER", "P_UNDER", "PM_OVER", "PM_UNDER"}
    if snaps is None or snaps.empty or not need <= set(snaps.columns):
        return pd.DataFrame(columns=cols)
    num = lambda c: pd.to_numeric(snaps[c], errors="coerce")
    act, line = num("ACTUAL"), num("LINE")
    pm = num("P_OVER") / (num("P_OVER") + num("P_UNDER"))
    pk = num("PM_OVER") / (num("PM_OVER") + num("PM_UNDER"))
    ok = (snaps["STATUS"] == "Graded") & act.notna() & line.notna() & (act != line) & pm.between(0, 1) & pk.between(0, 1)
    return pd.DataFrame({"Y": (act > line).astype(float), "PM": pm, "PK": pk})[ok].reset_index(drop=True)


def fit_blend(snaps: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Blend weight and scores. Keys: w (use this), w_fit, n, ll_model, ll_market, ll_blend, brier_model, brier_market."""
    g = graded(snaps)
    n = len(g)
    out: Dict[str, Any] = {"w": W_PRIOR, "w_fit": None, "n": n, "ll_model": None, "ll_market": None, "ll_blend": None,
                           "brier_model": None, "brier_market": None}
    if n == 0:
        return out
    y, lm, lk = g["Y"].to_numpy(), _logit(g["PM"]), _logit(g["PK"])
    z = GRID[:, None] * lm[None, :] + (1.0 - GRID[:, None]) * lk[None, :]          # (grid, props)
    p = np.clip(1.0 / (1.0 + np.exp(-z)), *P_CLIP)
    ll = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p), axis=1)
    w_fit = float(GRID[int(np.argmin(ll))])
    w = (n * w_fit + N0 * W_PRIOR) / (n + N0)
    out.update(w=float(w), w_fit=w_fit, ll_model=log_loss(g["PM"], y), ll_market=log_loss(g["PK"], y),
               ll_blend=log_loss(blend(g["PM"], g["PK"], w), y),
               brier_model=float(np.mean((g["PM"] - y) ** 2)), brier_market=float(np.mean((g["PK"] - y) ** 2)))
    return out


def calibration_table(snaps: Optional[pd.DataFrame], w: Optional[float] = None) -> pd.DataFrame:
    """Calibration of the side the model favored: picks, average predicted %, how often it won, the market's average % (and the blend's)."""
    g = graded(snaps)
    cols = ["MODEL_SAYS", "PICKS", "MODEL_%", "WON_%", "MARKET_%"] + (["BLEND_%"] if w is not None else [])
    if g.empty:
        return pd.DataFrame(columns=cols)
    over = g["PM"] >= 0.5
    fav = np.where(over, g["PM"], 1.0 - g["PM"])
    won = np.where(over, g["Y"], 1.0 - g["Y"])
    mkt = np.where(over, g["PK"], 1.0 - g["PK"])
    d = pd.DataFrame({"BIN": pd.cut(fav, BINS, labels=BIN_LABELS, right=False), "FAV": fav, "WON": won, "MKT": mkt})
    if w is not None:
        pb = blend(g["PM"], g["PK"], w)
        d["BL"] = np.where(over, pb, 1.0 - pb)
    agg = {"PICKS": ("WON", "size"), "MODEL_%": ("FAV", "mean"), "WON_%": ("WON", "mean"), "MARKET_%": ("MKT", "mean")}
    if w is not None:
        agg["BLEND_%"] = ("BL", "mean")
    t = d.groupby("BIN", observed=True).agg(**agg).reset_index().rename(columns={"BIN": "MODEL_SAYS"})
    for c in ("MODEL_%", "WON_%", "MARKET_%", "BLEND_%"):
        if c in t:
            t[c] = (100.0 * t[c]).round(1)
    t["MODEL_SAYS"] = t["MODEL_SAYS"].astype(str)
    return t[cols]


def _decimal_odds(american: Any) -> float:
    try:
        o = float(american)
    except (TypeError, ValueError):
        return float("nan")
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / -o) if abs(o) >= 100 else float("nan")


def blend_sides(res: Mapping[str, Any], w: float) -> Dict[str, Dict[str, float]]:
    """Blended chance and EV (both in %) for Over and Under of an evaluation result ({} if it has no sides)."""
    s = res.get("sides") or {}
    if not {"Over", "Under"} <= set(s):
        return {}
    pb_over = float(blend(s["Over"]["p_model"], s["Over"]["p_market"], w))
    out = {}
    for side, pb in (("Over", pb_over), ("Under", 100.0 - pb_over)):
        d = _decimal_odds(s[side].get("odds"))
        out[side] = {"p_blend": pb, "ev_blend": 100.0 * (pb / 100.0 * d - 1.0) if np.isfinite(d) else float("nan"),
                     "p_model": float(s[side]["p_model"]), "p_market": float(s[side]["p_market"])}
    return out


def weight_note(info: Mapping[str, Any]) -> str:
    n, wf = info.get("n", 0), info.get("w_fit")
    base = f"model weight {100 * info['w']:.0f}%"
    return base + (f" (fitted {100 * wf:.0f}% from {n} graded props, pulled toward the {100 * W_PRIOR:.0f}% starting assumption)" if n else
                   f" (starting assumption: no graded props yet)")


def render_blend_banner(st: Any, res: Mapping[str, Any], info: Mapping[str, Any]) -> None:
    """One line under an evaluation: the model's chance blended with DraftKings' no-vig chance, and the EV that leaves."""
    b = blend_sides(res, float(info["w"]))
    if not b:
        return
    best = max(b, key=lambda k: b[k]["ev_blend"] if np.isfinite(b[k]["ev_blend"]) else -1e9)
    parts = [f"{k} {v['p_blend']:.1f}% (model {v['p_model']:.1f}%, market {v['p_market']:.1f}%) → EV {v['ev_blend']:+.1f}%" for k, v in b.items()]
    text = f"**Market-blended chance** — {' · '.join(parts)}. {weight_note(info)}."
    (st.success if b[best]["ev_blend"] > 0 else st.info)(text + ("" if b[best]["ev_blend"] > 0 else " Neither side is +EV once the market is respected."))


def render_calibration(st: Any, snaps: Optional[pd.DataFrame]) -> None:
    """Snapshots-tab panel: is the model calibrated, does it beat the market, and what blend weight that implies."""
    info = fit_blend(snaps)
    st.markdown("#### 🎯 Calibration and market blend")
    if info["n"] == 0:
        st.caption("No graded props with a market price yet. Once games are graded this shows whether your model's 60% picks win about 60% "
                   f"of the time, and fits how much to trust it versus DraftKings' price. Until then the blend uses {100 * W_PRIOR:.0f}% model weight.")
        return
    c = st.columns(4)
    c[0].metric("Graded props", info["n"])
    c[1].metric("Model weight used", f"{100 * info['w']:.0f}%", help="Fitted weight shrunk toward the starting assumption; see the formula in calibration.py.")
    c[2].metric("Model log loss", f"{info['ll_model']:.3f}", help="Lower is better. 0.693 = a coin flip.")
    c[3].metric("Market log loss", f"{info['ll_market']:.3f}")
    better = info["ll_model"] < info["ll_market"]
    (st.success if better else st.warning)(
        ("Your model's probabilities have been more accurate than DraftKings' no-vig price on these props."
         if better else "DraftKings' no-vig price has been more accurate than your model on these props, so the blend leans toward the market.")
        + (" With fewer than 300 graded props this can still be luck." if info["n"] < 300 else ""))
    st.markdown("**Calibration** (side the model favored): if the model is calibrated, WON_% is close to MODEL_%.")
    st.dataframe(calibration_table(snaps, info["w"]), use_container_width=True, hide_index=True)
    st.caption(f"Blend log loss {info['ll_blend']:.3f} (fitted on these same props, so slightly flattering). Brier: model {info['brier_model']:.3f}, "
               f"market {info['brier_market']:.3f}. Bins with under ~30 picks are noise.")


# ====================================================================================================
# 2. CLOSING-LINE VALUE (CLV)
# ====================================================================================================

CLOSE_TEXT = ["CLOSE_TS"]
CLOSE_NUM = ["CLOSE_LINE", "CLOSE_OVER", "CLOSE_UNDER", "CLOSE_MIN"]
RESULT_COLS = ["CLV_STATUS", "CLV_PP", "CLV_EV", "LINE_MOVE", "BEAT"]


def _norm_name(name: Any) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def _implied(odds: pd.Series) -> pd.Series:
    o = pd.to_numeric(odds, errors="coerce")
    p = np.where(o > 0, 100.0 / (o + 100.0), -o / (-o + 100.0))
    return pd.Series(p, index=odds.index).where(o.abs() >= 100)


def _decimal_series(odds: pd.Series) -> pd.Series:
    o = pd.to_numeric(odds, errors="coerce")
    return pd.Series(np.where(o > 0, 1.0 + o / 100.0, 1.0 + 100.0 / -o), index=odds.index).where(o.abs() >= 100)


def _ensure_close_cols(d: pd.DataFrame) -> pd.DataFrame:
    for c in CLOSE_TEXT:
        if c not in d.columns:
            d[c] = ""
        d[c] = d[c].fillna("").astype(str)
    for c in CLOSE_NUM:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def record_close(snaps: Optional[pd.DataFrame], board: pd.DataFrame, stat: str, game_date: str, commence_iso: str,
           observed_et: str, date_col: str = "GAME_DATE") -> Tuple[pd.DataFrame, int]:
    """Apply one board observation (Player, Line, Over, Under) as the new close of matching pending snapshots. Returns (snapshots, rows changed).

    `observed_et` is 'YYYY-MM-DD HH:MM' Eastern (same format as TS_ET). Nothing changes once the game has started."""
    if snaps is None or snaps.empty or board is None or board.empty:
        return (pd.DataFrame() if snaps is None else snaps), 0
    start = pd.to_datetime(commence_iso, utc=True, errors="coerce")
    seen = pd.Timestamp(observed_et).tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    if pd.isna(start) or pd.isna(seen) or seen >= start:
        return snaps, 0
    d = _ensure_close_cols(snaps.copy())
    b = board[["Player", "Line", "Over", "Under"]].copy()
    b = b[pd.to_numeric(b["Line"], errors="coerce").notna()]
    b["_K"] = b["Player"].map(_norm_name)
    b = b.drop_duplicates("_K", keep="last").set_index("_K")
    key = d["PLAYER"].map(_norm_name)
    ts = pd.to_datetime(d["TS_ET"], errors="coerce")
    prev = pd.to_datetime(d["CLOSE_TS"], errors="coerce")
    obs_naive = seen.tz_localize(None)
    m = ((d["STATUS"] == "Pending") & (d["STAT"].astype(str).str.upper() == str(stat).upper()) & (d[date_col].astype(str) == str(game_date))
         & key.isin(b.index) & (ts < obs_naive) & (prev.isna() | (prev <= obs_naive)))
    if not m.any():
        return d, 0
    k = key[m]
    d.loc[m, "CLOSE_LINE"] = pd.to_numeric(b.loc[k, "Line"], errors="coerce").to_numpy()
    d.loc[m, "CLOSE_OVER"] = pd.to_numeric(b.loc[k, "Over"], errors="coerce").to_numpy()
    d.loc[m, "CLOSE_UNDER"] = pd.to_numeric(b.loc[k, "Under"], errors="coerce").to_numpy()
    d.loc[m, "CLOSE_TS"] = obs_naive.strftime("%Y-%m-%d %H:%M")
    d.loc[m, "CLOSE_MIN"] = round((start - seen).total_seconds() / 60.0)
    return d, int(m.sum())


def compute_clv(snaps: Optional[pd.DataFrame], date_col: str = "GAME_DATE", today: Optional[str] = None) -> pd.DataFrame:
    """Snapshots with RESULT_COLS added (vectorized). CLV_STATUS: 'Same line', 'Line moved', 'No close' or 'Not final'."""
    if snaps is None or snaps.empty:
        return pd.DataFrame(columns=RESULT_COLS)
    d = _ensure_close_cols(snaps.copy())
    today = today or datetime.now().strftime("%Y-%m-%d")
    over = d["SHOWN_SIDE"].astype(str).str.lower().isin(["over", "yes"])
    line, cl = pd.to_numeric(d["LINE"], errors="coerce"), d["CLOSE_LINE"]
    final = (d["STATUS"] == "Graded") | ((d[date_col].astype(str) != "") & (d[date_col].astype(str) < today))
    has = cl.notna() & d["SHOWN_SIDE"].astype(str).str.lower().isin(["over", "under", "yes", "no"])
    same = has & ((cl - line).abs() < 1e-9)
    io, iu = _implied(d["CLOSE_OVER"]), _implied(d["CLOSE_UNDER"])
    p_close_over = io / (io + iu)
    pmo, pmu = pd.to_numeric(d["PM_OVER"], errors="coerce"), pd.to_numeric(d["PM_UNDER"], errors="coerce")
    p_open_over = pmo / (pmo + pmu)
    p_close = np.where(over, p_close_over, 1.0 - p_close_over)
    p_open = np.where(over, p_open_over, 1.0 - p_open_over)
    price = np.where(over, pd.to_numeric(d["OVER_ODDS"], errors="coerce"), pd.to_numeric(d["UNDER_ODDS"], errors="coerce"))
    dec = _decimal_series(pd.Series(price, index=d.index))
    d["CLV_PP"] = np.where(same, 100.0 * (p_close - p_open), np.nan)
    d["CLV_EV"] = np.where(same, 100.0 * (p_close * dec - 1.0), np.nan)
    d["LINE_MOVE"] = np.where(has & ~same, np.where(over, cl - line, line - cl), np.nan)
    beat = np.where(same, d["CLV_EV"] > 0, d["LINE_MOVE"] > 0)
    d["BEAT"] = np.where(has & (same | d["LINE_MOVE"].notna()), np.where(beat, "Y", "N"), "")
    d["CLV_STATUS"] = np.select([~has, ~final, same], ["No close", "Not final", "Same line"], "Line moved")
    return d


def clv_summary(snaps: Optional[pd.DataFrame], date_col: str = "GAME_DATE", today: Optional[str] = None) -> pd.DataFrame:
    """CLV by group: all props, the app's tier, favorites, and the Pinnacle flag."""
    cols = ["GROUP", "PROPS", "BEAT_CLOSE_%", "AVG_CLV_EV_%", "AVG_CLV_PP", "AVG_MIN_BEFORE"]
    d = compute_clv(snaps, date_col, today)
    if d.empty:
        return pd.DataFrame(columns=cols)
    d = d[d["CLV_STATUS"].isin(["Same line", "Line moved"])]
    if d.empty:
        return pd.DataFrame(columns=cols)
    groups = [("All props", pd.Series(True, index=d.index))]
    if "TIER_SHOWN" in d:
        groups += [(f"Tier: {t}", d["TIER_SHOWN"] == t) for t in sorted(set(d["TIER_SHOWN"].astype(str)) - {"", "nan"})]
    if "FAV" in d:
        groups.append(("Favorites", d["FAV"].astype(str) == "Y"))
    if "PIN_FLAG" in d:
        groups += [(f"Pinnacle: {f}", d["PIN_FLAG"] == f) for f in ("Agrees", "Disagrees")]
    rows = []
    for name, m in groups:
        s = d[m]
        if len(s):
            rows.append({"GROUP": name, "PROPS": len(s), "BEAT_CLOSE_%": round(100.0 * (s["BEAT"] == "Y").mean(), 1),
                         "AVG_CLV_EV_%": round(float(s["CLV_EV"].mean()), 2) if s["CLV_EV"].notna().any() else None,
                         "AVG_CLV_PP": round(float(s["CLV_PP"].mean()), 2) if s["CLV_PP"].notna().any() else None,
                         "AVG_MIN_BEFORE": round(float(s["CLOSE_MIN"].mean())) if s["CLOSE_MIN"].notna().any() else None})
    return pd.DataFrame(rows, columns=cols)


def render_clv(st: Any, snaps: Optional[pd.DataFrame], date_col: str = "GAME_DATE", today: Optional[str] = None) -> None:
    """Snapshots-tab panel: does the market move toward the app's picks before the game?"""
    st.markdown("#### 📈 Closing-line value (did the market move toward the pick?)")
    t = clv_summary(snaps, date_col, today)
    if t.empty:
        st.caption("No closing prices yet. A close is the last DraftKings price your app pulled before the game started, so pull the board "
                   "(Step 3 or a Best Picks refresh) again close to start time. Props show up here the day after their game.")
        return
    allr = t.iloc[0]
    good = allr["BEAT_CLOSE_%"] > 50 and (allr["AVG_CLV_EV_%"] is None or allr["AVG_CLV_EV_%"] > 0)
    (st.success if good else st.warning)(
        f"The app's picks beat the close {allr['BEAT_CLOSE_%']:.0f}% of the time over {int(allr['PROPS'])} props"
        + (f", worth {allr['AVG_CLV_EV_%']:+.1f}% on average at the close" if allr["AVG_CLV_EV_%"] is not None else "")
        + (". Beating the close more than half the time is the main sign of a real edge." if good else
           ". The market tends to move away from these picks, which points to the model's edges not being real.")
        + (" Under ~100 props this is still noisy." if allr["PROPS"] < 100 else ""))
    st.dataframe(t, use_container_width=True, hide_index=True)
    st.caption("BEAT_CLOSE_%: same line = the close made the app's price +EV; line moved = the line moved toward the pick. "
               "AVG_CLV_PP: change in the market's no-vig chance for the pick (same line only). AVG_MIN_BEFORE: how close to start your last pull was; "
               "the smaller, the more this is a true close.")


# ====================================================================================================
# 3. DAILY RESULTS (win % by game day)
# ====================================================================================================

BREAK_EVEN = 100.0 * 110.0 / 210.0
FILTERS = ["All props", "Favorites", "SHARP and LEAN", "SHARP only", "Best Picks 5/6+ & matchup", "Best Picks 5/6+ & matchup, +EV", "Best Picks 5/6+ & matchup, negative EV", "Pinnacle agrees, gap up to 3.5", "Pinnacle agrees, gap over 3.5", "Best Picks: Pinnacle agrees", "Best Picks: Pinnacle line differs", "Best Picks top 10", "Best Picks ✓ BOTH", "Placed bets", "Placed parlay legs"]


def _regrade(p: pd.DataFrame, side: pd.Series, price: Any) -> pd.DataFrame:
    """`p` re-graded on `side` ('Over'/'Under'/'Yes'/'No') from ACTUAL vs LINE, priced at `price` (NaN = no units)."""
    p = p.copy()
    over = side.astype(str).str.lower().isin(["over", "yes"])
    act, line = pd.to_numeric(p["ACTUAL"], errors="coerce"), pd.to_numeric(p["LINE"], errors="coerce")
    graded = (p["STATUS"] == "Graded") & act.notna() & line.notna()
    won = np.where(over, act > line, act < line)
    p["SHOWN_RESULT"] = np.where(graded, np.where(act == line, "Push", np.where(won, "Hit", "Miss")), "")
    p["SHOWN_SIDE"] = np.where(over, "Over", "Under")
    p["OVER_ODDS"], p["UNDER_ODDS"] = price, price
    return p


def as_pinnacle_picks(d: pd.DataFrame, gap: bool) -> pd.DataFrame:
    """Best Picks 'Pinnacle agrees' props (graded on the side ranked, BP_SIDE) or 'Pinnacle line differs' props (graded on the side
    Pinnacle's line favors, BP_GAPSIDE), each at that side's saved DraftKings price."""
    col = "BP_GAPSIDE" if gap else "BP_PIN"
    if col not in d.columns:
        return d.iloc[0:0]
    flag = d[col].fillna("").astype(str)
    p = d[flag != ""] if gap else d[flag == "Y"]
    if p.empty:
        return p
    side = p["BP_GAPSIDE"] if gap else p["BP_SIDE"]
    over = side.astype(str).str.lower().isin(["over", "yes"])
    price = np.where(over, pd.to_numeric(p["OVER_ODDS"], errors="coerce"), pd.to_numeric(p["UNDER_ODDS"], errors="coerce"))
    return _regrade(p, side, price)


def as_best_picks(d: pd.DataFrame, both_only: bool, top: int = 10, sharp_only: bool = False, ev: str = "") -> pd.DataFrame:
    """Props from the latest saved Best Picks list (BP_RANK <= top, or ✓ BOTH), graded on the side Best Picks ranked (BP_SIDE)
    at that side's saved DraftKings price."""
    if "BP_RANK" not in d.columns:
        return d.iloc[0:0]
    rank = pd.to_numeric(d["BP_RANK"], errors="coerce")
    if sharp_only:
        flag = d["BP_SHARP"] if "BP_SHARP" in d.columns else pd.Series("", index=d.index)
        p = d[(flag.fillna("").astype(str) == "Y") & rank.notna()]
        if ev:
            bev = pd.to_numeric(p["BP_EV"], errors="coerce") if "BP_EV" in p.columns else pd.Series(np.nan, index=p.index)
            p = p[bev > 0] if ev == "+" else p[bev <= 0]
    else:
        p = d[(d["BP_BOTH"].fillna("").astype(str) == "Y") & rank.notna()] if both_only else d[rank.notna() & (rank <= top)]
    if p.empty:
        return p
    over = p["BP_SIDE"].astype(str).str.lower().isin(["over", "yes"])
    price = np.where(over, pd.to_numeric(p["OVER_ODDS"], errors="coerce"), pd.to_numeric(p["UNDER_ODDS"], errors="coerce"))
    return _regrade(p, p["BP_SIDE"], price)


def as_placed(d: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Rows with a placed bet of `kind` ('Solo' / 'Parlay leg'), re-graded on the side YOU bet (PLACED_SIDE) from ACTUAL vs LINE.
    Solo bets are priced at PLACED_ODDS; parlay legs get no price (one leg's 'profit' isn't the ticket's), so they show win % only."""
    if "PLACED_KIND" not in d.columns:
        return d.iloc[0:0]
    p = d[d["PLACED_KIND"].fillna("").astype(str) == kind]
    if p.empty:
        return p
    return _regrade(p, p["PLACED_SIDE"], pd.to_numeric(p["PLACED_ODDS"], errors="coerce") if kind == "Solo" else np.nan)


def _filter_picks(d: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "Best Picks 5/6+ & matchup":
        return as_best_picks(d, False, sharp_only=True)
    if how == "Best Picks 5/6+ & matchup, +EV":
        return as_best_picks(d, False, sharp_only=True, ev="+")
    if how == "Best Picks 5/6+ & matchup, negative EV":
        return as_best_picks(d, False, sharp_only=True, ev="-")
    if how in ("Pinnacle agrees, gap up to 3.5", "Pinnacle agrees, gap over 3.5"):
        if not {"PIN_FLAG", "PIN_P_SIDE", "DK_P_SIDE"} <= set(d.columns):
            return d.iloc[0:0]
        gap = (pd.to_numeric(d["PIN_P_SIDE"], errors="coerce") - pd.to_numeric(d["DK_P_SIDE"], errors="coerce")).abs()
        agree = d["PIN_FLAG"].fillna("").astype(str) == "Agrees"
        return d[agree & (gap > 3.5 if how.endswith("over 3.5") else gap <= 3.5)]
    if how == "Best Picks: Pinnacle agrees":
        return as_pinnacle_picks(d, False)
    if how == "Best Picks: Pinnacle line differs":
        return as_pinnacle_picks(d, True)
    if how == "Best Picks top 10":
        return as_best_picks(d, False)
    if how == "Best Picks ✓ BOTH":
        return as_best_picks(d, True)
    if how == "Placed bets":
        return as_placed(d, "Solo")
    if how == "Placed parlay legs":
        return as_placed(d, "Parlay leg")
    tier = d.get("TIER_SHOWN", pd.Series("", index=d.index)).astype(str).str.upper()
    if how == "Favorites":
        return d[d.get("FAV", pd.Series("", index=d.index)).astype(str) == "Y"]
    if how == "SHARP and LEAN":
        return d[tier.isin(["SHARP", "LEAN"])]
    if how == "SHARP only":
        return d[tier == "SHARP"]
    return d


def graded_days(snaps: Optional[pd.DataFrame], date_col: str, today: str) -> List[str]:
    """Game days before `today` that have at least one graded Hit/Miss/Push, newest first."""
    if snaps is None or snaps.empty or date_col not in snaps.columns:
        return []
    g = snaps[(snaps["STATUS"] == "Graded") & snaps["SHOWN_RESULT"].isin(["Hit", "Miss", "Push"])]
    days = g[date_col].astype(str)
    return sorted({x for x in days if x and x != "nan" and x < today}, reverse=True)


def pick_units(d: pd.DataFrame) -> pd.DataFrame:
    """Per graded pick: PRICE (American, side shown), DEC (decimal), UNITS (1-unit stake; NaN when the price is missing)."""
    side = d.get("SHOWN_SIDE", pd.Series("", index=d.index)).astype(str).str.lower()
    over = pd.to_numeric(d.get("OVER_ODDS", pd.Series(np.nan, index=d.index)), errors="coerce")
    under = pd.to_numeric(d.get("UNDER_ODDS", pd.Series(np.nan, index=d.index)), errors="coerce")
    price = pd.Series(np.where(side.isin(["over", "yes"]), over, np.where(side.isin(["under", "no"]), under, np.nan)), index=d.index)
    price = price.where(price.abs() >= 100)
    dec = pd.Series(np.where(price > 0, 1.0 + price / 100.0, 1.0 + 100.0 / -price), index=d.index).where(price.notna())
    res = d["SHOWN_RESULT"].astype(str)
    units = pd.Series(np.select([res == "Hit", res == "Miss", res == "Push"], [dec - 1.0, -1.0, 0.0], np.nan), index=d.index).where(dec.notna())
    return pd.DataFrame({"PRICE": price, "DEC": dec, "UNITS": units}, index=d.index)


def _money(d: pd.DataFrame) -> dict:
    """units, roi_pct, price_be_pct and how many decided picks had no price, for graded picks `d`."""
    u = pick_units(d)
    staked = u["UNITS"].notna()
    decided = d["SHOWN_RESULT"].isin(["Hit", "Miss"])
    be = 100.0 * (1.0 / u.loc[decided & staked, "DEC"]).mean() if (decided & staked).any() else None
    total = float(u["UNITS"].sum()) if staked.any() else None
    return {"units": total, "roi_pct": 100.0 * total / int(staked.sum()) if staked.any() else None, "price_be_pct": be,
            "no_price": int((decided & ~staked).sum()), "staked": int(staked.sum())}


def day_counts(snaps: Optional[pd.DataFrame], day: str, date_col: str, how: str = "All props") -> dict:
    """{'hits','misses','pushes','pending','win_pct','units','roi_pct','price_be_pct','no_price','staked'} for one day and filter."""
    out = {"hits": 0, "misses": 0, "pushes": 0, "pending": 0, "win_pct": None, "units": None, "roi_pct": None, "price_be_pct": None, "no_price": 0, "staked": 0}
    if snaps is None or snaps.empty:
        return out
    d = _filter_picks(snaps[snaps[date_col].astype(str) == str(day)], how)
    res = d["SHOWN_RESULT"].astype(str)
    out.update(hits=int((res == "Hit").sum()), misses=int((res == "Miss").sum()), pushes=int((res == "Push").sum()),
               pending=int((d["STATUS"] == "Pending").sum()))
    dec = out["hits"] + out["misses"]
    out["win_pct"] = 100.0 * out["hits"] / dec if dec else None
    out.update(_money(d[res.isin(["Hit", "Miss", "Push"])]))
    return out


def by_stat(snaps: pd.DataFrame, day: str, date_col: str, how: str = "All props") -> pd.DataFrame:
    """Win % per stat for one day."""
    cols = ["STAT", "PICKS", "HITS", "WIN_%", "UNITS", "ROI_%"]
    d = _filter_picks(snaps[snaps[date_col].astype(str) == str(day)], how)
    d = d[d["SHOWN_RESULT"].isin(["Hit", "Miss"])]
    if d.empty:
        return pd.DataFrame(columns=cols)
    d = d.assign(_U=pick_units(d)["UNITS"])
    g = d.groupby("STAT").agg(PICKS=("SHOWN_RESULT", "size"), HITS=("SHOWN_RESULT", lambda s: int((s == "Hit").sum())),
                              UNITS=("_U", "sum"), _N=("_U", "count")).reset_index()
    g["WIN_%"] = (100.0 * g["HITS"] / g["PICKS"]).round(1)
    g["ROI_%"] = (100.0 * g["UNITS"] / g["_N"].replace(0, np.nan)).round(1)
    g["UNITS"] = g["UNITS"].where(g["_N"] > 0).round(2)
    return g.sort_values("PICKS", ascending=False).reset_index(drop=True)[cols]


def recent(snaps: pd.DataFrame, date_col: str, today: str, how: str = "All props", n_days: int = 7) -> pd.DataFrame:
    """Win % for each of the last `n_days` graded game days, plus a running total row."""
    cols = ["DAY", "PICKS", "HITS", "WIN_%", "UNITS", "ROI_%"]
    rows, staked = [], []
    for day in graded_days(snaps, date_col, today)[:n_days]:
        c = day_counts(snaps, day, date_col, how)
        if c["hits"] + c["misses"]:
            rows.append({"DAY": day, "PICKS": c["hits"] + c["misses"], "HITS": c["hits"], "WIN_%": round(c["win_pct"], 1),
                         "UNITS": None if c["units"] is None else round(c["units"], 2), "ROI_%": None if c["roi_pct"] is None else round(c["roi_pct"], 1)})
            staked.append(c["staked"])
    t = pd.DataFrame(rows, columns=cols)
    if len(t) > 1:
        u = pd.to_numeric(t["UNITS"], errors="coerce")
        n = float(sum(staked))
        tot = pd.DataFrame([{"DAY": f"Last {len(t)} days", "PICKS": int(t["PICKS"].sum()), "HITS": int(t["HITS"].sum()),
                             "WIN_%": round(100.0 * t["HITS"].sum() / t["PICKS"].sum(), 1),
                             "UNITS": round(float(u.sum()), 2) if u.notna().any() else None,
                             "ROI_%": round(100.0 * float(u.sum()) / n, 1) if n > 0 else None}])
        t = pd.concat([t, tot], ignore_index=True)
    return t


def needs_grading(snaps: Optional[pd.DataFrame], date_col: str, today: str) -> bool:
    """True when props from a day before `today` are still pending."""
    if snaps is None or snaps.empty:
        return False
    days = snaps[date_col].astype(str)
    return bool(((snaps["STATUS"] == "Pending") & (days != "") & (days != "nan") & (days < today)).any())


def _daily_donut(c: dict, title: str) -> Any:
    import plotly.graph_objects as go
    fig = go.Figure(go.Pie(labels=["Hit", "Miss", "Push"], values=[c["hits"], c["misses"], c["pushes"]], hole=0.6, sort=False,
                           marker={"colors": ["#2e9e5b", "#d1495b", "#9aa0a6"]}, textinfo="value"))
    fig.update_layout(height=260, margin={"l": 10, "r": 10, "t": 30, "b": 10}, title={"text": title, "font": {"size": 14}},
                      annotations=[{"text": f"{c['win_pct']:.0f}%" if c["win_pct"] is not None else "—", "showarrow": False, "font": {"size": 26}}],
                      legend={"orientation": "h", "y": -0.05})
    return fig


def render_daily(st: Any, snaps: Optional[pd.DataFrame], date_col: str, today: str,
           grade: Optional[Callable[[], Optional[pd.DataFrame]]] = None) -> None:
    """📅 Daily results panel. `grade()` grades finished games and returns the updated snapshots (run once per session)."""
    ss = st.session_state
    st.markdown("### 📅 Daily results")
    if grade is not None and not ss.get("daily_auto_graded") and needs_grading(snaps, date_col, today):
        ss["daily_auto_graded"] = True
        try:
            with st.spinner("Grading finished games..."):
                new = grade()
            if new is not None:
                snaps = new
        except Exception as e:
            st.warning(f"Automatic grading failed ({str(e)[:100]}). Use the Grade finished games button.")
    days = graded_days(snaps, date_col, today)
    if not days:
        st.caption("No graded game days yet. Results appear here automatically the day after the games, once box scores are available.")
        return
    c1, c2 = st.columns(2)
    day = c1.selectbox("Game day", days, index=0, key="daily_day")
    how = c2.selectbox("Picks", FILTERS, key="daily_filter")
    c = day_counts(snaps, day, date_col, how)
    if c["win_pct"] is None:
        st.info(f"No decided {how.lower()} on {day}." + (" Record bets with the 🎯 Place Bet buttons (or ✅ I placed this ticket in the Parlay tab) "
                                                        "and they show up here after the games are graded." if how.startswith("Placed") else ""))
    else:
        left, right = st.columns([1, 1])
        left.plotly_chart(_daily_donut(c, f"{how} · {day}"), use_container_width=True)
        be = c["price_be_pct"] if c["price_be_pct"] is not None else BREAK_EVEN
        right.metric("Win %", f"{c['win_pct']:.1f}%", delta=f"{c['win_pct'] - be:+.1f} vs the {be:.1f}% these prices needed")
        if c["units"] is not None:
            right.metric("Profit (1 unit per pick)", f"{c['units']:+.2f} u", delta=f"ROI {c['roi_pct']:+.1f}%")
        right.write(f"**{c['hits']}** hit · **{c['misses']}** missed · {c['pushes']} push" + (f" · {c['pending']} not graded yet" if c["pending"] else ""))
        right.caption({"Best Picks 5/6+ & matchup": "Every 5/6 or 6/6 prop with a top-third matchup in the last Best Picks scan before the games, "
                                                    "graded on that side, 1 unit each at the saved DraftKings price.",
                       "Best Picks 5/6+ & matchup, +EV": "The 5/6 and 6/6 matchup picks Best Picks showed by default (EV above 0 at the scan), "
                                                         "graded on that side, 1 unit each at the saved price.",
                       "Best Picks 5/6+ & matchup, negative EV": "The 5/6 and 6/6 matchup picks hidden for negative EV: if these lose money, hiding "
                                                                 "them is right; if they win, the EV numbers are off.",
                       "Pinnacle agrees, gap up to 3.5": "Every prop where Pinnacle agreed with the side the app showed by 3.5 points or "
                                                         "less, graded on that side, 1 unit each at the saved DraftKings price.",
                       "Pinnacle agrees, gap over 3.5": "The big-gap Pinnacle-agrees props Best Picks hides by default. If these keep losing, "
                                                        "hiding them is right.",
                       "Best Picks: Pinnacle agrees": "Props where Pinnacle rated the ranked side more likely than DraftKings in the last Best Picks "
                                                      "scan before the games, 1 unit each at the saved DraftKings price.",
                       "Best Picks: Pinnacle line differs": "Props whose Pinnacle line was half a point or more from DraftKings', graded on the side "
                                                            "Pinnacle's line favored, 1 unit each at that side's saved DraftKings price.",
                       "Best Picks top 10": "The top 10 of the last Best Picks scan before the games, graded on the side it ranked, 1 unit each at "
                                            "the saved DraftKings price.",
                       "Best Picks ✓ BOTH": "Every ✓ BOTH pick of the last Best Picks scan before the games (still +EV and a top-third matchup), "
                                            "graded on the side it ranked, 1 unit each at the saved price.",
                       "Placed bets": "Your recorded solo bets, graded on the side you bet, 1 unit each at the price you recorded. Pushes are refunds.",
                       "Placed parlay legs": "Each leg of the tickets you recorded, graded on the side you took. Win % only: a leg's result isn't "
                                             "your ticket's profit."}.get(how, "Side the app showed for each prop, 1 unit on each at the DraftKings price saved "
                                                                                "with it. Pushes are refunds.")
                      + (f" {c['no_price']} pick(s) had no saved price and are left out of units." if c["no_price"] and how != "Placed parlay legs" else ""))
        t = by_stat(snaps, day, date_col, how)
        if len(t) > 1:
            st.dataframe(t, use_container_width=True, hide_index=True)
    trend = recent(snaps, date_col, today, how)
    if len(trend) > 1:
        st.markdown("**Recent game days**")
        st.dataframe(trend, use_container_width=True, hide_index=True)
    st.caption("One day is a small sample: 10 picks at a true 55% will often show anywhere from 30% to 80%, and props from the same games win and lose "
               "together. Judge the model on the running units and ROI over weeks, and on the CLV panel. Units are what flat 1-unit bets would have "
               "made at the saved prices, not your actual bets.")
