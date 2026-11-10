"""Six-point prop criteria for the NBA Prop Evaluator.

The rules, tiers, side selection and odds-table tagging live in six_point_core.py (shared with the NFL app, so both
score identically). This file holds only NBA-specific data: thresholds, who counts as a relevant injury, the recent
opponent table, and the spread parser. 6/6 -> SHARP, 5/6 -> LEAN, <= 4/6 -> PASS; an unverifiable check counts as failed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

import six_point_core as core
from six_point_core import (  # noqa: F401  (re-exported: the app and tests import these from here)
    SHARP_SCORE, LEAN_SCORE, TAG_COL, Check, SixPointResult, add_six_point_column, nickname, opponent_for, style_tags,
    tag_text, tier_for,
)

_best = core.shown

# --- Thresholds (edit freely) -------------------------------------------------
MIN_LINE: Dict[str, float] = {
    "POINTS": 14.5, "REBOUNDS": 4.5, "ASSISTS": 3.5, "3-POINTERS MADE": 1.5,
    "PTS + REB + AST": 20.5, "PTS + AST": 16.5, "PTS + REB": 17.5, "REB + AST": 8.5,
    "STEALS": 1.5, "BLOCKS": 1.5,
}
MIN_EDGE_ABS: Dict[str, float] = {
    "POINTS": 1.5, "REBOUNDS": 1.0, "ASSISTS": 1.0, "3-POINTERS MADE": 0.5,
    "PTS + REB + AST": 2.5, "PTS + AST": 2.0, "PTS + REB": 2.0, "REB + AST": 1.5,
    "STEALS": 0.4, "BLOCKS": 0.4,
}
MIN_EDGE_PCT = 0.06
IMPLIED_MIN = 0.50
RANK_CUTOFF = 15          # half of a 30-team league
RECENT_GAMES = 10         # "recent" opponent window, in that team's games
BLOWOUT_SPREAD = 10.0

NBA_SPEC = core.SportSpec("NBA", MIN_LINE, MIN_EDGE_ABS, RANK_CUTOFF, RECENT_GAMES, MIN_EDGE_PCT, IMPLIED_MIN, BLOWOUT_SPREAD)


# --- Individual checks (thin wrappers over the shared rules) --------------------------
def check_line(stat_col: str, line: float) -> Check:
    """1. Baby line."""
    return core.check_line(NBA_SPEC, stat_col, line)


def check_edge(side: str, stat_col: str, line: float, proj: float) -> Check:
    """2. AI edge."""
    return core.check_edge(NBA_SPEC, side, stat_col, line, proj)


def check_implied(side: str, p_market: float) -> Check:
    """3. Implied chance."""
    return core.check_implied(NBA_SPEC, side, p_market)


def rank_supports(side: str, rank: int, n_teams: int, cutoff: int = RANK_CUTOFF) -> bool:
    """Rank 1 = softest defense. Over wants a soft defense, Under a tough one."""
    return core.rank_supports(side, rank, n_teams, cutoff)


def check_defense(side: str, season_rank: Optional[int], recent_rank: Optional[int], n_teams: Optional[int]) -> Check:
    """4. Defensive rank: season-long and recent opponent ranks (30 teams) both support the side."""
    return core.check_defense(NBA_SPEC, side, season_rank, n_teams, recent_rank, n_teams,
                              hint="pick an opponent and enable stat-specific opponent factors")


def flagged_players(inj: Mapping[str, Any], rotation_min: float) -> List[str]:
    """Names with an active flag: the player himself and rotation teammates."""
    out: List[str] = []
    if inj.get("self_status"):
        out.append(f"the player ({inj['self_status']})")
    out += [f"{x['name']} ({x['status']})" for x in inj.get("listed", [])
            if x.get("p_out", 0.0) > 0 and x.get("mpg", rotation_min) >= rotation_min]
    return out


def check_injury(inj: Mapping[str, Any], rotation_min: float) -> Check:
    """5. Injury: nobody in the rotation carries an active flag (feed must have been read)."""
    if not inj.get("checked") or not inj.get("feed_ok", False):
        return core.check_injury(None)
    return core.check_injury(flagged_players(inj, rotation_min))


def check_game_script(spread: Optional[float]) -> Check:
    """6. Game script: no blowout risk; the NBA is indoors, so weather cannot interfere."""
    return core.check_game_script(NBA_SPEC, spread, True, False, "indoors, no weather risk",
                                  hint="spread unavailable (load the game in Steps 1-3 to fetch it)")


def score_prop(side: str, stat_col: str, line: float, proj: float, p_market: float,
               season_rank: Optional[int], recent_rank: Optional[int], n_teams: Optional[int],
               inj: Mapping[str, Any], spread: Optional[float], rotation_min: float = 10.0) -> SixPointResult:
    """Run all six checks for one side and return the score and tier."""
    return core.result_from_checks(side, (
        check_line(stat_col, line), check_edge(side, stat_col, line, proj), check_implied(side, p_market),
        check_defense(side, season_rank, recent_rank, n_teams), check_injury(inj, rotation_min), check_game_script(spread),
    ))


def assess(stat_col: str, line: float, proj: float, p_market: Mapping[str, float], season_rank: Optional[int],
           recent_rank: Optional[int], n_teams: Optional[int], inj: Mapping[str, Any], spread: Optional[float],
           ev_side: str, rotation_min: float = 10.0) -> Dict[str, Any]:
    """Both sides scored; `best` = the side meeting more criteria (ties -> `ev_side`). p_market: {"Over": p, "Under": p} in 0-1."""
    both = {s: score_prop(s, stat_col, line, proj, float(p_market[s]), season_rank, recent_rank, n_teams, inj, spread, rotation_min)
            for s in core.SIDES}
    return core.assemble(both["Over"], both["Under"], ev_side)


# --- Data helpers (pure, testable) -------------------------------------------
def compute_opp_allowed_recent(games: pd.DataFrame, stat_cols: List[str],
                               n_recent: int = RECENT_GAMES, min_games: int = 3) -> pd.DataFrame:
    """Like compute_opp_allowed, but only each opponent's last `n_recent` games.

    Production allowed relative to each opposing player's own full-sample average
    (1.00 = normal). Expects columns PID, DATE, OPP plus `stat_cols`.
    """
    g = games[games.groupby("PID")["DATE"].transform("size") >= min_games]
    if g.empty:
        return pd.DataFrame()
    base = g.groupby("PID")[stat_cols].transform("mean")
    day = pd.to_datetime(g["DATE"])
    newest_first = day.groupby(g["OPP"]).rank(method="dense", ascending=False)
    mask = (newest_first <= n_recent).to_numpy()
    recent, base_recent = g[mask], base[mask]
    num = recent.groupby("OPP")[stat_cols].sum()
    den = base_recent.groupby(recent["OPP"]).sum()
    out = (num / den.replace(0, np.nan)).fillna(1.0)
    out["G"] = recent.groupby("OPP")["DATE"].nunique()
    return out


def extract_spread(payload: Mapping[str, Any]) -> Optional[float]:
    """Absolute point spread from a The Odds API event-odds payload (first book)."""
    for book in payload.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") == "spreads":
                points = [abs(float(o["point"])) for o in market.get("outcomes", [])
                          if o.get("point") is not None]
                if points:
                    return min(points)
    return None
