"""
Pinnacle cross-check: does the sharpest book agree with the side your model likes?  (Information only: it never changes the projection.)

Math
----
Per book, two-way multiplicative devig of the SAME line:   q_over = imp(over) / (imp(over) + imp(under)),   imp(american) = implied win probability.
For a side S:
    DK_P_SIDE  = DraftKings no-vig probability of S           PIN_P_SIDE = Pinnacle no-vig probability of S
    PIN_EV_%   = 100 * (PIN_P_SIDE * decimal(DraftKings price of S) - 1)   (what the DraftKings bet is worth if Pinnacle is right)
    GAP_PP     = 100 * (PIN_P_SIDE - DK_P_SIDE)
FLAG for the side the app shows:  "Agrees" if GAP_PP >= FLAG_PP, "Disagrees" if GAP_PP <= -FLAG_PP, else "Flat".
Props Pinnacle does not price get "No Pinnacle".

Different lines ("Converted line"): Pinnacle's no-vig Over chance p at ITS line Lp is moved to DraftKings' line Ld:
    count stats with a small line (Lp <= 7.5: threes, assists, receptions, TDs...):  X ~ Poisson(lam); lam solves P(X > Lp) = p
                                                                                      (bisection), then P_over(Ld) = P(X > Ld)
    everything else:  X ~ Normal(mu, sigma),  sigma = CV(stat) * Lp  (typical spread of that stat; CV table below),
                      mu = Lp + sigma * Phi^-1(p),  then  P_over(Ld) = 1 - Phi((Ld - mu) / sigma)
Then it is scored like a same line, with a wider dead zone (CONVERTED_FLAG_PP = 2 points) because the conversion is approximate.
Lines further apart than max(2, 15% of the line) are not converted ("Different line": too far to trust).
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

FLAG_PP = 1.0
CONVERTED_FLAG_PP = 2.0
BIG_GAP_PP = 3.5             # Pinnacle vs DraftKings gaps this big lost money in the graded props: usually news, a stale line or a rough conversion
BIG_GAP_NOTE = "⚠️ big gap: check news"
POISSON_MAX_LINE = 7.5
# Typical spread (standard deviation / mean) of one game's result, by stat; matched on words in the stat name, first match wins.
STAT_CV = (("PASS YARDS", 0.28), ("PASSING YARDS", 0.28), ("RUSH YARDS", 0.45), ("RUSHING YARDS", 0.45), ("RECEIVING YARDS", 0.55),
           ("REC YARDS", 0.55), ("YARDS", 0.45), ("ATTEMPTS", 0.25), ("COMPLETIONS", 0.25), ("RECEPTIONS", 0.40), ("KICKING", 0.45),
           ("POINTS + REBOUNDS + ASSISTS", 0.25), ("PTS+REB+AST", 0.25), ("PRA", 0.25), ("POINTS + REBOUNDS", 0.27), ("POINTS + ASSISTS", 0.27),
           ("REBOUNDS + ASSISTS", 0.33), ("POINTS", 0.30), ("REBOUNDS", 0.38), ("ASSISTS", 0.45), ("3-POINTERS", 0.55), ("THREES", 0.55))
DEFAULT_CV = 0.35          # Pinnacle must differ from DraftKings by at least this many points to count as agreeing / disagreeing
PIN_COLUMNS = ["Pin Line", "Pin Over %", "Pin EV Over", "Pin EV Under"]


def norm(name: Any) -> str:
    """Lower-case, accent-free, letters-only key so 'Luka Dončić' and 'Luka Doncic' match."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower().replace(" jr.", "").replace(" jr", "").replace(" iii", "").replace(" ii", ""))


def _imp(american: Any) -> float:
    """Implied win probability of an American price (nan if invalid)."""
    try:
        o = float(american)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(o) or abs(o) < 100:
        return float("nan")
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def _dec(american: Any) -> float:
    """Decimal odds of an American price (nan if invalid)."""
    p = _imp(american)
    return 1.0 / p if np.isfinite(p) and p > 0 else float("nan")


def novig_over(over: Any, under: Any) -> float:
    """Over probability after removing the vig (nan if either price is missing)."""
    a, b = _imp(over), _imp(under)
    return a / (a + b) if np.isfinite(a) and np.isfinite(b) and a + b > 0 else float("nan")


def split(board: pd.DataFrame, book_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(DraftKings rows, Pinnacle rows) of a raw prop board. Without a book column everything is treated as DraftKings."""
    if board is None or board.empty or book_col not in board.columns:
        return board, pd.DataFrame(columns=[] if board is None else board.columns)
    b = board[book_col].astype(str).str.lower()
    pin = board[b.str.contains("pinnacle")].reset_index(drop=True)
    dk = board[~b.str.contains("pinnacle")].reset_index(drop=True)
    return dk, pin


def stat_cv(stat: Optional[str]) -> float:
    """Typical spread (sd / mean) of one game's result for this stat (DEFAULT_CV when unknown)."""
    s = str(stat or "").upper()
    return next((cv for word, cv in STAT_CV if word in s), DEFAULT_CV)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _phi(mid) < p else (lo, mid)
    return 0.5 * (lo + hi)


def _poisson_over(lam: float, line: float) -> float:
    """P(X > line) for X ~ Poisson(lam)."""
    k_max = int(math.floor(line))
    term, cdf = math.exp(-lam), 0.0
    for k in range(k_max + 1):
        cdf += term
        term *= lam / (k + 1)
    return max(0.0, 1.0 - cdf)


def convert_over(p_over: float, pin_line: float, dk_line: float, stat: Optional[str] = None) -> float:
    """Pinnacle's Over chance moved from its line to DraftKings' line (nan when the lines are too far apart to trust)."""
    if not (0.0 < p_over < 1.0) or abs(dk_line - pin_line) > max(2.0, 0.15 * abs(pin_line)):
        return float("nan")
    if pin_line <= POISSON_MAX_LINE and not any(w in str(stat or "").upper() for w in ("YARDS", "POINTS")):
        lo, hi = 1e-4, 60.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if _poisson_over(mid, pin_line) < p_over else (lo, mid)
        return _poisson_over(0.5 * (lo + hi), dk_line)
    sigma = max(stat_cv(stat) * max(pin_line, 1.0), 0.5)
    mu = pin_line + sigma * _phi_inv(p_over)
    return 1.0 - _phi((dk_line - mu) / sigma)


def cross(dk_line: Any, dk_over: Any, dk_under: Any, pin_line: Any, pin_over: Any, pin_under: Any, stat: Optional[str] = None) -> Dict[str, Any]:
    """Compare one DraftKings prop with Pinnacle's price on the same player and stat (a different Pinnacle line is converted, see above)."""
    out: Dict[str, Any] = {"STATUS": "No Pinnacle", "PIN_LINE": None, "DK_LINE": None, "PIN_P_OVER": None, "DK_P_OVER": None,
                           "PIN_EV_OVER": None, "PIN_EV_UNDER": None}
    pq = novig_over(pin_over, pin_under)
    try:
        pl = float(pin_line)
    except (TypeError, ValueError):
        return out
    if not np.isfinite(pl) or not np.isfinite(pq):
        return out
    out["PIN_LINE"] = pl
    try:
        dl = float(dk_line)
    except (TypeError, ValueError):
        dl = float("nan")
    out["DK_LINE"] = dl if np.isfinite(dl) else None
    same = np.isfinite(dl) and abs(pl - dl) < 1e-9
    status = "Same line"
    if not same:
        moved = convert_over(pq, pl, dl, stat) if np.isfinite(dl) else float("nan")
        if not np.isfinite(moved):
            out["STATUS"] = "Different line"
            return out
        pq, status = moved, "Converted line"
    dq = novig_over(dk_over, dk_under)
    out.update(STATUS=status, PIN_P_OVER=100.0 * pq, DK_P_OVER=100.0 * dq if np.isfinite(dq) else None)
    for side, p, price in (("OVER", pq, dk_over), ("UNDER", 1.0 - pq, dk_under)):
        d = _dec(price)
        out[f"PIN_EV_{side}"] = 100.0 * (p * d - 1.0) if np.isfinite(d) else None
    return out


def table(summary: pd.DataFrame, pin_summary: Optional[pd.DataFrame], stat: Optional[str] = None) -> pd.DataFrame:
    """The board summary (Player, Line, Over, Under) plus the PIN_COLUMNS for each player ('' where there is no usable Pinnacle line)."""
    out = summary.copy()
    pins = {} if pin_summary is None or pin_summary.empty else {norm(r["Player"]): r for _, r in pin_summary.iterrows()}
    rows = []
    for _, r in summary.iterrows():
        p = pins.get(norm(r["Player"]))
        c = cross(r["Line"], r["Over"], r["Under"], p["Line"], p["Over"], p["Under"], stat) if p is not None else cross(0, 0, 0, None, None, None)
        if c["STATUS"] == "No Pinnacle":
            rows.append(["—", "", "", ""])
        elif c["STATUS"] == "Different line":
            rows.append([f"{c['PIN_LINE']:g} (differs)", "", "", ""])
        else:
            where = f"{c['PIN_LINE']:g}" + (f" → {c['DK_LINE']:g} (converted)" if c["STATUS"] == "Converted line" else "")
            if c.get("DK_P_OVER") is not None and abs(c["PIN_P_OVER"] - c["DK_P_OVER"]) > BIG_GAP_PP:
                where += f" {BIG_GAP_NOTE}"
            rows.append([where, _r(c["PIN_P_OVER"]), _r(c["PIN_EV_OVER"]), _r(c["PIN_EV_UNDER"])])
    for i, col in enumerate(PIN_COLUMNS):
        out[col] = [x[i] for x in rows]
    return out


def _r(x: Any) -> str:
    """One decimal as text ('' when missing), so the board never shows 54.500000."""
    return "" if x is None else f"{float(x):.1f}"


def assess(side: str, c: Mapping[str, Any]) -> Dict[str, Any]:
    """FLAG, probabilities and a plain sentence for one side ('Over' / 'Under'; Yes / No count as Over / Under)."""
    s = "UNDER" if str(side).strip().lower() in ("under", "no") else "OVER"
    base: Dict[str, Any] = {"FLAG": c.get("STATUS", "No Pinnacle"), "PIN_LINE": c.get("PIN_LINE"), "PIN_P_SIDE": None, "DK_P_SIDE": None, "PIN_EV": None, "GAP_PP": None, "TEXT": ""}
    if c.get("STATUS") == "No Pinnacle":
        base["TEXT"] = "Pinnacle has no price for this prop, so there is nothing to cross-check."
        return base
    if c.get("STATUS") == "Different line":
        base["TEXT"] = f"Pinnacle's line is {c['PIN_LINE']:g}, too far from DraftKings' to convert reliably, so the prices aren't compared."
        return base
    po, do = c["PIN_P_OVER"], c.get("DK_P_OVER")
    pin_p = po if s == "OVER" else 100.0 - po
    dk_p = None if do is None else (do if s == "OVER" else 100.0 - do)
    gap = None if dk_p is None else pin_p - dk_p
    conv = c.get("STATUS") == "Converted line"
    dead = CONVERTED_FLAG_PP if conv else FLAG_PP
    flag = "Flat" if gap is None or abs(gap) < dead else ("Agrees" if gap > 0 else "Disagrees")
    ev = c.get(f"PIN_EV_{s}")
    base.update(FLAG=flag, PIN_P_SIDE=pin_p, DK_P_SIDE=dk_p, GAP_PP=gap, PIN_EV=ev)
    words = {"Agrees": "Pinnacle rates this side MORE likely than DraftKings does", "Disagrees": "Pinnacle rates this side LESS likely than DraftKings does",
             "Flat": "Pinnacle and DraftKings price this side about the same"}[flag]
    base["TEXT"] = (f"Pinnacle's line is {c['PIN_LINE']:g}; converted to DraftKings' {c['DK_LINE']:g} (approximate). " if conv else "") + \
        (f"{words}: Pinnacle {pin_p:.1f}% vs DraftKings {dk_p:.1f}% (no-vig)" if dk_p is not None else f"Pinnacle {pin_p:.1f}% (no-vig)") + \
        (f"; at DraftKings' price that side is worth {ev:+.1f}% if Pinnacle is right." if ev is not None else ".") + \
        (f" ⚠️ A gap over {BIG_GAP_PP:g} points usually means news, a stale line or a rough conversion, and these lost money in your graded "
         "props: check for news before betting." if gap is not None and abs(gap) > BIG_GAP_PP else "")
    return base


def for_player(pin_summary: Optional[pd.DataFrame], name: str, line: float, over: float, under: float, side: str,
               stat: Optional[str] = None) -> Dict[str, Any]:
    """assess() for a named player against a Pinnacle summary (Player, Line, Over, Under); handles a missing summary."""
    if pin_summary is None or pin_summary.empty:
        return assess(side, cross(line, over, under, None, None, None))
    hit = pin_summary[pin_summary["Player"].map(norm) == norm(name)]
    if hit.empty:
        return assess(side, cross(line, over, under, None, None, None))
    p = hit.iloc[0]
    return assess(side, cross(line, over, under, p["Line"], p["Over"], p["Under"], stat))
