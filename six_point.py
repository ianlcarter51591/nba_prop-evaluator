"""Six-point prop criteria for the NBA Prop Evaluator.

Score = number of passed checks for the recommended side:
    6/6 -> SHARP     5/6 -> LEAN     <= 4/6 -> PASS

Checks (a check that cannot be verified counts as NOT passed):
    1. Baby line      line >= MIN_LINE[stat]
    2. AI edge        side-direction margin = proj - line (Over) or line - proj (Under)
                      must be >= max(MIN_EDGE_ABS[stat], MIN_EDGE_PCT * line)
    3. Implied chance devigged market probability of the side > IMPLIED_MIN
    4. Defensive rank season-long AND recent opponent rank both support the side
                      (rank 1 = softest; Over needs rank <= RANK_CUTOFF,
                       Under needs rank > n_teams - RANK_CUTOFF)
    5. Injury         player not flagged, and no rotation teammate flagged
                      (injury feed must have been read)
    6. Game script    |spread| < BLOWOUT_SPREAD (NBA is indoors: no weather risk)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

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
SHARP_SCORE, LEAN_SCORE = 6, 5


@dataclass(frozen=True)
class Check:
    """One criterion's outcome. `verified=False` means the data was unavailable."""

    key: str
    label: str
    passed: bool
    detail: str
    verified: bool = True


@dataclass(frozen=True)
class SixPointResult:
    """All six checks for one side plus the tier."""

    side: str
    score: int
    tier: str
    checks: Tuple[Check, ...]

    def as_dict(self) -> Dict[str, Any]:
        """Plain-dict form (safe for st.session_state / JSON)."""
        return {"side": self.side, "score": self.score, "tier": self.tier,
                "checks": [c.__dict__.copy() for c in self.checks]}


def tier_for(score: int) -> str:
    """6 -> SHARP, 5 -> LEAN, otherwise PASS."""
    return "SHARP" if score >= SHARP_SCORE else "LEAN" if score == LEAN_SCORE else "PASS"


# --- Individual checks --------------------------------------------------------
def check_line(stat_col: str, line: float) -> Check:
    """1. Baby line: the line must be at or above the stat's minimum."""
    need = MIN_LINE.get(stat_col, 0.0)
    ok = bool(line >= need)
    return Check("baby_line", "1. Line High Enough (Baby Line)", ok,
                 f"Line {line:g} vs minimum {need:g} for this stat"
                 + ("" if ok else ": too low, too risky"))


def check_edge(side: str, stat_col: str, line: float, proj: float) -> Check:
    """2. AI edge: projection far enough past the line in the side's direction."""
    margin = proj - line if side == "Over" else line - proj
    need = max(MIN_EDGE_ABS.get(stat_col, 0.0), MIN_EDGE_PCT * abs(line))
    ok = bool(margin >= need)
    return Check("ai_edge", "2. AI Edge", ok,
                 f"Projection {proj:.1f} is {margin:+.1f} {'above' if side == 'Over' else 'below'} "
                 f"the line in your direction; needs at least {need:.1f}")


def check_implied(side: str, p_market: float) -> Check:
    """3. Implied chance: the devigged market favors the selected side."""
    ok = bool(p_market > IMPLIED_MIN)
    return Check("implied_chance", "3. Implied Chance", ok,
                 f"Devigged DraftKings price gives {side} {p_market:.1%} "
                 f"(needs more than {IMPLIED_MIN:.0%}); single book, no multi-book vote")


def rank_supports(side: str, rank: int, n_teams: int, cutoff: int = RANK_CUTOFF) -> bool:
    """Rank 1 = softest defense. Over wants a soft defense, Under a tough one."""
    return rank <= cutoff if side == "Over" else rank > n_teams - cutoff


def check_defense(side: str, season_rank: Optional[int], recent_rank: Optional[int],
                  n_teams: Optional[int]) -> Check:
    """4. Defensive rank: season-long and recent opponent ranks both support the side."""
    label = "4. Defensive Rank"
    if season_rank is None or recent_rank is None or not n_teams:
        have = "season" if season_rank is not None else "recent" if recent_rank is not None else "neither"
        return Check("defensive_rank", label, False,
                     f"Not verified ({have} rank available): pick an opponent and enable "
                     "stat-specific opponent factors", verified=False)
    ok = rank_supports(side, season_rank, n_teams) and rank_supports(side, recent_rank, n_teams)
    want = f"top {RANK_CUTOFF} softest" if side == "Over" else f"top {RANK_CUTOFF} toughest"
    return Check("defensive_rank", label, bool(ok),
                 f"Season #{season_rank}, last {RECENT_GAMES} games #{recent_rank} of {n_teams} "
                 f"(rank 1 = softest); {side} needs both in the {want}")


def flagged_players(inj: Mapping[str, Any], rotation_min: float) -> List[str]:
    """Names with an active flag: the player himself and rotation teammates."""
    out: List[str] = []
    if inj.get("self_status"):
        out.append(f"the player ({inj['self_status']})")
    out += [f"{x['name']} ({x['status']})" for x in inj.get("listed", [])
            if x.get("p_out", 0.0) > 0 and x.get("mpg", rotation_min) >= rotation_min]
    return out


def check_injury(inj: Mapping[str, Any], rotation_min: float) -> Check:
    """5. Injury: nobody relevant on the team carries an active flag."""
    label = "5. Injury"
    if not inj.get("checked") or not inj.get("feed_ok", False):
        return Check("injury", label, False,
                     "Not verified: injury feed unavailable or injury adjustment turned off", verified=False)
    flagged = flagged_players(inj, rotation_min)
    return Check("injury", label, not flagged,
                 "No active injury flags on the team" if not flagged else "Flagged: " + ", ".join(flagged))


def check_game_script(spread: Optional[float]) -> Check:
    """6. Game script: no blowout risk; NBA is indoors so weather cannot interfere."""
    label = "6. Game Script"
    if spread is None:
        return Check("game_script", label, False,
                     "Not verified: spread unavailable (load the game in Steps 1-3 to fetch it)", verified=False)
    ok = bool(abs(spread) < BLOWOUT_SPREAD)
    return Check("game_script", label, ok,
                 f"Spread {abs(spread):g} (blowout risk at {BLOWOUT_SPREAD:g}+); indoors, no weather risk")


def score_prop(side: str, stat_col: str, line: float, proj: float, p_market: float,
               season_rank: Optional[int], recent_rank: Optional[int], n_teams: Optional[int],
               inj: Mapping[str, Any], spread: Optional[float], rotation_min: float = 10.0) -> SixPointResult:
    """Run all six checks for one side and return the score and tier."""
    checks = (
        check_line(stat_col, line),
        check_edge(side, stat_col, line, proj),
        check_implied(side, p_market),
        check_defense(side, season_rank, recent_rank, n_teams),
        check_injury(inj, rotation_min),
        check_game_script(spread),
    )
    score = int(np.sum([c.passed for c in checks]))
    return SixPointResult(side=side, score=score, tier=tier_for(score), checks=checks)


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
