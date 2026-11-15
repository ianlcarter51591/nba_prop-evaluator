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
Only the SAME line is compared (a different Pinnacle line is reported but not scored). Props Pinnacle does not price get "No Pinnacle".
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

FLAG_PP = 1.0          # Pinnacle must differ from DraftKings by at least this many points to count as agreeing / disagreeing
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


def cross(dk_line: Any, dk_over: Any, dk_under: Any, pin_line: Any, pin_over: Any, pin_under: Any) -> Dict[str, Any]:
    """Compare one DraftKings prop with Pinnacle's price on the same player and stat."""
    out: Dict[str, Any] = {"STATUS": "No Pinnacle", "PIN_LINE": None, "PIN_P_OVER": None, "DK_P_OVER": None, "PIN_EV_OVER": None, "PIN_EV_UNDER": None}
    pq = novig_over(pin_over, pin_under)
    try:
        pl = float(pin_line)
    except (TypeError, ValueError):
        return out
    if not np.isfinite(pl) or not np.isfinite(pq):
        return out
    out["PIN_LINE"] = pl
    try:
        same = abs(pl - float(dk_line)) < 1e-9
    except (TypeError, ValueError):
        same = False
    if not same:
        out["STATUS"] = "Different line"
        return out
    dq = novig_over(dk_over, dk_under)
    out.update(STATUS="Same line", PIN_P_OVER=100.0 * pq, DK_P_OVER=100.0 * dq if np.isfinite(dq) else None)
    for side, p, price in (("OVER", pq, dk_over), ("UNDER", 1.0 - pq, dk_under)):
        d = _dec(price)
        out[f"PIN_EV_{side}"] = 100.0 * (p * d - 1.0) if np.isfinite(d) else None
    return out


def table(summary: pd.DataFrame, pin_summary: Optional[pd.DataFrame]) -> pd.DataFrame:
    """The board summary (Player, Line, Over, Under) plus the PIN_COLUMNS for each player ('' where there is no usable Pinnacle line)."""
    out = summary.copy()
    pins = {} if pin_summary is None or pin_summary.empty else {norm(r["Player"]): r for _, r in pin_summary.iterrows()}
    rows = []
    for _, r in summary.iterrows():
        p = pins.get(norm(r["Player"]))
        c = cross(r["Line"], r["Over"], r["Under"], p["Line"], p["Over"], p["Under"]) if p is not None else cross(0, 0, 0, None, None, None)
        if c["STATUS"] == "No Pinnacle":
            rows.append(["—", "", "", ""])
        elif c["STATUS"] == "Different line":
            rows.append([f"{c['PIN_LINE']:g} (differs)", "", "", ""])
        else:
            rows.append([f"{c['PIN_LINE']:g}", round(c["PIN_P_OVER"], 1), _r(c["PIN_EV_OVER"]), _r(c["PIN_EV_UNDER"])])
    for i, col in enumerate(PIN_COLUMNS):
        out[col] = [x[i] for x in rows]
    return out


def _r(x: Any) -> Any:
    return "" if x is None else round(float(x), 1)


def assess(side: str, c: Mapping[str, Any]) -> Dict[str, Any]:
    """FLAG, probabilities and a plain sentence for one side ('Over' / 'Under'; Yes / No count as Over / Under)."""
    s = "UNDER" if str(side).strip().lower() in ("under", "no") else "OVER"
    base: Dict[str, Any] = {"FLAG": c.get("STATUS", "No Pinnacle"), "PIN_LINE": c.get("PIN_LINE"), "PIN_P_SIDE": None, "DK_P_SIDE": None, "PIN_EV": None, "GAP_PP": None, "TEXT": ""}
    if c.get("STATUS") == "No Pinnacle":
        base["TEXT"] = "Pinnacle has no price for this prop, so there is nothing to cross-check."
        return base
    if c.get("STATUS") == "Different line":
        base["TEXT"] = f"Pinnacle's line is {c['PIN_LINE']:g}, not the same as DraftKings, so the prices can't be compared directly."
        return base
    po, do = c["PIN_P_OVER"], c.get("DK_P_OVER")
    pin_p = po if s == "OVER" else 100.0 - po
    dk_p = None if do is None else (do if s == "OVER" else 100.0 - do)
    gap = None if dk_p is None else pin_p - dk_p
    flag = "Flat" if gap is None or abs(gap) < FLAG_PP else ("Agrees" if gap > 0 else "Disagrees")
    ev = c.get(f"PIN_EV_{s}")
    base.update(FLAG=flag, PIN_P_SIDE=pin_p, DK_P_SIDE=dk_p, GAP_PP=gap, PIN_EV=ev)
    words = {"Agrees": "Pinnacle rates this side MORE likely than DraftKings does", "Disagrees": "Pinnacle rates this side LESS likely than DraftKings does",
             "Flat": "Pinnacle and DraftKings price this side about the same"}[flag]
    base["TEXT"] = (f"{words}: Pinnacle {pin_p:.1f}% vs DraftKings {dk_p:.1f}% (no-vig)" if dk_p is not None else f"Pinnacle {pin_p:.1f}% (no-vig)") + \
        (f"; at DraftKings' price that side is worth {ev:+.1f}% if Pinnacle is right." if ev is not None else ".")
    return base


def for_player(pin_summary: Optional[pd.DataFrame], name: str, line: float, over: float, under: float, side: str) -> Dict[str, Any]:
    """assess() for a named player against a Pinnacle summary (Player, Line, Over, Under); handles a missing summary."""
    if pin_summary is None or pin_summary.empty:
        return assess(side, cross(line, over, under, None, None, None))
    hit = pin_summary[pin_summary["Player"].map(norm) == norm(name)]
    if hit.empty:
        return assess(side, cross(line, over, under, None, None, None))
    p = hit.iloc[0]
    return assess(side, cross(line, over, under, p["Line"], p["Over"], p["Under"]))
