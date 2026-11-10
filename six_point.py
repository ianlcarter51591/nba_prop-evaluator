"""Six-point prop criteria for the NBA Prop Evaluator.

Score = number of passed checks for the recommended side:
    6/6 -> SHARP     5/6 -> LEAN     <= 4/6 -> PASS
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

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
RANK_CUTOFF = 15
RECENT_GAMES = 10
BLOWOUT_SPREAD = 10.0
SHARP_SCORE, LEAN_SCORE = 6, 5


@dataclass(frozen=True)
class Check:
    key: str
    label: str
    passed: bool
    detail: str
    verified: bool = True


@dataclass(frozen=True)
class SixPointResult:
    side: str
    score: int
    tier: str
    checks: Tuple[Check, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "side": self.side,
            "score": self.score,
            "tier": self.tier,
            "checks": [c.__dict__.copy() for c in self.checks],
        }


def tier_for(score: int) -> str:
    return "SHARP" if score >= SHARP_SCORE else "LEAN" if score == LEAN_SCORE else "PASS"


def check_line(stat_col: str, line: float) -> Check:
    need = MIN_LINE.get(stat_col, 0.0)
    ok = bool(line >= need)
    return Check(
        "baby_line",
        "1. Line High Enough (Baby Line)",
        ok,
        f"Line {line:g} vs minimum {need:g} for this stat" + ("" if ok else ": too low, too risky"),
    )


def check_edge(side: str, stat_col: str, line: float, proj: float) -> Check:
    margin = proj - line if side == "Over" else line - proj
    need = max(MIN_EDGE_ABS.get(stat_col, 0.0), MIN_EDGE_PCT * abs(line))
    ok = bool(margin >= need)
    return Check(
        "ai_edge",
        "2. AI Edge",
        ok,
        f"Projection {proj:.1f} is {margin:+.1f} {'above' if side == 'Over' else 'below'} "
        f"the line in your direction; needs at least {need:.1f}",
    )


def check_implied(side: str, p_market: float) -> Check:
    ok = bool(p_market > IMPLIED_MIN)
    return Check(
        "implied_chance",
        "3. Implied Chance",
        ok,
        f"Devigged market price gives {side} {p_market:.1%} "
        f"(needs more than {IMPLIED_MIN:.0%}); single book, no multi-book vote",
    )


def rank_supports(side: str, rank: int, n_teams: int, cutoff: int = RANK_CUTOFF) -> bool:
    return rank <= cutoff if side == "Over" else rank > n_teams - cutoff


def check_defense(side: str, season_rank: Optional[int], recent_rank: Optional[int], n_teams: Optional[int]) -> Check:
    label = "4. Defensive Rank"
    if season_rank is None or recent_rank is None or not n_teams:
        have = "season" if season_rank is not None else "recent" if recent_rank is not None else "neither"
        return Check(
            "defensive_rank",
            label,
            False,
            f"Not verified ({have} rank available): pick an opponent and enable stat-specific opponent factors",
            verified=False,
        )
    ok = rank_supports(side, season_rank, n_teams) and rank_supports(side, recent_rank, n_teams)
    want = f"top {RANK_CUTOFF} softest" if side == "Over" else f"top {RANK_CUTOFF} toughest"
    return Check(
        "defensive_rank",
        label,
        bool(ok),
        f"Season #{season_rank}, last {RECENT_GAMES} games #{recent_rank} of {n_teams} "
        f"(rank 1 = softest); {side} needs both in the {want}",
    )


def flagged_players(inj: Mapping[str, Any], rotation_min: float) -> List[str]:
    out: List[str] = []
    if inj.get("self_status"):
        out.append(f"the player ({inj['self_status']})")
    out += [
        f"{x['name']} ({x['status']})"
        for x in inj.get("listed", [])
        if x.get("p_out", 0.0) > 0 and x.get("mpg", rotation_min) >= rotation_min
    ]
    return out


def check_injury(inj: Mapping[str, Any], rotation_min: float) -> Check:
    label = "5. Injury"
    if not inj.get("checked") or not inj.get("feed_ok", False):
        return Check(
            "injury", label, False, "Not verified: injury feed unavailable or injury adjustment turned off", verified=False
        )
    flagged = flagged_players(inj, rotation_min)
    return Check("injury", label, not flagged, "No active injury flags on the team" if not flagged else "Flagged: " + ", ".join(flagged))


def check_game_script(spread: Optional[float]) -> Check:
    label = "6. Game Script"
    if spread is None:
        return Check(
            "game_script", label, False, "Not verified: spread unavailable (load the game in Steps 1-3 to fetch it)", verified=False
        )
    ok = bool(abs(spread) < BLOWOUT_SPREAD)
    return Check("game_script", label, ok, f"Spread {abs(spread):g} (blowout risk at {BLOWOUT_SPREAD:g}+); indoors, no weather risk")


def score_prop(
    side: str,
    stat_col: str,
    line: float,
    proj: float,
    p_market: float,
    season_rank: Optional[int],
    recent_rank: Optional[int],
    n_teams: Optional[int],
    inj: Mapping[str, Any],
    spread: Optional[float],
    rotation_min: float = 10.0,
) -> SixPointResult:
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


def compute_opp_allowed_recent(
    games: pd.DataFrame, stat_cols: List[str], n_recent: int = RECENT_GAMES, min_games: int = 3
) -> pd.DataFrame:
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
    for book in payload.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") == "spreads":
                points = [abs(float(o["point"])) for o in market.get("outcomes", []) if o.get("point") is not None]
                if points:
                    return min(points)
    return None


TAG_COL = "Six-Point"


def nickname(team: Any) -> str:
    parts = str(team or "").split()
    return parts[-1].lower() if parts else ""


def opponent_for(player_team: Optional[str], game_teams: List[str]) -> Optional[str]:
    if not player_team or len(game_teams) != 2:
        return None
    mine = [i for i, t in enumerate(game_teams) if nickname(t) == nickname(player_team)]
    return game_teams[1 - mine[0]] if len(mine) == 1 else None


def _best(six: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    return six[six["best"]] if six and "best" in six else None


def tag_text(six: Optional[Mapping[str, Any]]) -> str:
    best = _best(six)
    if best is None:
        return "—"
    if best["tier"] == "SHARP":
        return f"🔥 SHARP {best['score']}/6 · {best['side']}"
    if best["tier"] == "LEAN":
        return f"⚡ LEAN {best['score']}/6 · {best['side']}"
    return f"{best['score']}/6 · {best['side']}"


def add_six_point_column(summary: pd.DataFrame, six_by_player: Mapping[str, Optional[Mapping[str, Any]]]) -> pd.DataFrame:
    out = summary.copy()
    out[TAG_COL] = out["Player"].map(lambda p: tag_text(six_by_player.get(p)))
    rank = out["Player"].map(lambda p: (_best(six_by_player.get(p)) or {"score": -1})["score"])
    return out.assign(_rank=rank).sort_values("_rank", ascending=False, kind="stable").drop(columns="_rank").reset_index(drop=True)


def style_tags(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def row_style(row: pd.Series) -> List[str]:
        tag = str(row.get(TAG_COL, ""))
        css = (
            "background-color:#166534;color:#ffffff;font-weight:700"
            if tag.startswith("🔥")
            else "background-color:#a16207;color:#ffffff;font-weight:700"
            if tag.startswith("⚡")
            else ""
        )
        return [css] * len(row)

    return df.style.apply(row_style, axis=1)
