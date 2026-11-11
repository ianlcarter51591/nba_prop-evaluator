"""
Vegas game environment: turn the market's total and spread into a projection multiplier (shared by the NBA and NFL apps).

Math
----
With game total T and a team's spread s (negative when that team is favored), the market's implied points are
    implied_team = T/2 - s/2            implied_opponent = T/2 + s/2
The environment ratio compares tonight with what the player's own averages already contain:
    scoring stats :  ratio = implied_team / team's average points per game
    volume  stats :  ratio = T / team's average combined points per game       (pace of the whole game)
    multiplier    =  clip(1 + k * shrink * (ratio - 1), lo, hi),   shrink = G / (G + G0)   (few games = less trust)
When the favorite can't be read, only the total is known, so the volume form is used for every stat.
k is how much of the environment change reaches the stat; it is an assumption unless a backtest calibrated it.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from six_point_core import spread_from_espn_odds

SHRINK_G0 = 10.0


def lines_from_espn_odds(odds: Any, home_abbr: str = "", away_abbr: str = "") -> Dict[str, Optional[float]]:
    """{'total', 'home_spread'} from an ESPN competition `odds` list. home_spread < 0 when the home team is favored;
    None when the favorite can't be determined (the size alone is not enough)."""
    out: Dict[str, Optional[float]] = {"total": None, "home_spread": None}
    for o in odds or []:
        if not isinstance(o, Mapping):
            continue
        ou = o.get("overUnder")
        if out["total"] is None and isinstance(ou, (int, float)) and not isinstance(ou, bool) and ou > 0:
            out["total"] = float(ou)
        size = spread_from_espn_odds([o])
        if out["home_spread"] is None and size is not None:
            if size == 0:
                out["home_spread"] = 0.0
                continue
            home_fav = (o.get("homeTeamOdds") or {}).get("favorite")
            away_fav = (o.get("awayTeamOdds") or {}).get("favorite")
            if home_fav is True and away_fav is not True:
                out["home_spread"] = -size
            elif away_fav is True and home_fav is not True:
                out["home_spread"] = size
            else:
                m = re.match(r"\s*([A-Za-z]{2,4})\s+[+-]?\d", str(o.get("details") or ""))   # "KC -3.5" names the favorite
                fav = m[1].upper() if m else ""
                if fav and fav == str(home_abbr).upper():
                    out["home_spread"] = -size
                elif fav and fav == str(away_abbr).upper():
                    out["home_spread"] = size
    return out


def find_espn_lines(payload: Any, teams: Sequence[str], norm: Callable[[Any], str]) -> Optional[Dict[str, Any]]:
    """Lines for the game between `teams` in an ESPN scoreboard payload. A competitor matches when `norm` of its
    displayName, name or abbreviation equals `norm(team)`. Returns total, home_spread, home, away (normalised), or None."""
    want = {norm(t) for t in teams if norm(t)}
    for ev in payload.get("events", []) if isinstance(payload, Mapping) else []:
        comp = (ev.get("competitions") or [{}])[0]
        sides: Dict[str, Dict[str, Any]] = {}
        for c in comp.get("competitors", []):
            team = c.get("team") or {}
            names = {norm(team.get(k)) for k in ("displayName", "name", "abbreviation") if team.get(k)}
            hit = next((w for w in want if w in names), None)
            sides[str(c.get("homeAway"))] = {"match": hit, "abbr": str(team.get("abbreviation") or "")}
        got = {s["match"] for s in sides.values() if s["match"]}
        if want and want <= got and {"home", "away"} <= set(sides):
            lines = lines_from_espn_odds(comp.get("odds"), sides["home"]["abbr"], sides["away"]["abbr"])
            return {**lines, "home": sides["home"]["match"], "away": sides["away"]["match"]}
    return None


def implied_team_total(total: float, team_spread: float) -> float:
    """Market-implied points for a team whose spread is `team_spread` (negative = favored)."""
    return float(total) / 2.0 - float(team_spread) / 2.0


def env_multiplier(ratio: float, k: float, games: float = 1e9, lo: float = 0.90, hi: float = 1.10, g0: float = SHRINK_G0) -> float:
    """clip(1 + k * G/(G+g0) * (ratio - 1), lo, hi); 1.0 for a missing or non-positive ratio."""
    if ratio is None or not np.isfinite(ratio) or ratio <= 0:
        return 1.0
    shrink = games / (games + g0) if np.isfinite(games) and games >= 0 else 0.0
    return float(np.clip(1.0 + k * shrink * (ratio - 1.0), lo, hi))


def team_environment(total: Optional[float], home_spread: Optional[float], is_home: Optional[bool], team_ppg: Optional[float],
                     team_total_ppg: Optional[float], games: float, k: float, scoring_stat: bool,
                     lo: float = 0.90, hi: float = 1.10) -> Optional[Dict[str, Any]]:
    """Everything the engine needs for one player: implied points, the ratio used, the multiplier and a plain note.
    None when there is no total or no baseline to compare it with."""
    if total is None or not np.isfinite(total) or total <= 0:
        return None
    implied: Optional[float] = None
    if home_spread is not None and is_home is not None:
        implied = implied_team_total(total, home_spread if is_home else -home_spread)
    if scoring_stat and implied is not None and team_ppg and team_ppg > 0:
        kind, ratio, base = "team points", implied / team_ppg, team_ppg
    elif team_total_ppg and team_total_ppg > 0:
        kind, ratio, base = "game total", total / team_total_ppg, team_total_ppg
    else:
        return None
    mult = env_multiplier(ratio, k, games, lo, hi)
    shown = implied if kind == "team points" else total
    return {"total": float(total), "implied": implied, "kind": kind, "ratio": float(ratio), "baseline": float(base), "mult": mult, "k": float(k),
            "note": f"Vegas {kind}: {shown:.1f} vs season {base:.1f} per game (x{ratio:.3f}) -> projection x{mult:.3f}"}
