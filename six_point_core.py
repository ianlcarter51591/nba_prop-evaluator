"""Six-point prop criteria, shared by the NBA and NFL apps (one copy of the rules, so both score identically).

Score = number of passed checks for a side:  6/6 -> SHARP,  5/6 -> LEAN,  <= 4/6 -> PASS.
A check that cannot be verified counts as NOT passed (shown as unverified).

    1. Baby line      line >= spec.min_line[stat]
    2. AI edge        margin (proj - line for Over, line - proj for Under) >= max(min_edge_abs[stat], min_edge_pct * line)
    3. Implied chance devigged market probability of the side > spec.implied_min
    4. Defensive rank season-long AND recent opponent ranks both support the side
                      (rank 1 = softest: Over needs rank <= cutoff, Under needs rank > n_teams - cutoff)
    5. Injury         nobody relevant flagged (the sport module decides who is relevant)
    6. Game script    |spread| < spec.blowout_spread, and no bad weather where the sport has any

Which side is shown: whichever side meets MORE criteria; a tie goes to the model's best-EV side (choose_best).
The odds-table tag and the Final Assessment both read the same result, so they cannot disagree.

Only sport-specific data (thresholds, injury rules, weather, rank source) lives in six_point.py / nfl_six_point.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SHARP_SCORE, LEAN_SCORE = 6, 5
SIDES: Tuple[str, str] = ("Over", "Under")


@dataclass(frozen=True)
class SportSpec:
    """Per-sport thresholds. Everything else in this file is identical for every sport."""

    name: str
    min_line: Mapping[str, float]
    min_edge_abs: Mapping[str, float]
    rank_cutoff: int                 # half of the league (15 of 30 NBA teams, 16 of 32 NFL teams)
    recent_games: int                # "recent" defense window, in that team's games
    min_edge_pct: float = 0.06
    implied_min: float = 0.50
    blowout_spread: float = 10.0


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


# --- The six checks -------------------------------------------------------------
def check_line(spec: SportSpec, stat: str, line: float) -> Check:
    """1. Baby line: the line must be at or above the stat's minimum."""
    need = spec.min_line.get(stat, 0.0)
    ok = bool(line >= need)
    return Check("baby_line", "1. Line High Enough (Baby Line)", ok,
                 f"Line {line:g} vs minimum {need:g} for this stat" + ("" if ok else ": too low, too risky"))


def check_edge(spec: SportSpec, side: str, stat: str, line: float, proj: float) -> Check:
    """2. AI edge: projection far enough past the line in the side's direction."""
    margin = proj - line if side == "Over" else line - proj
    need = max(spec.min_edge_abs.get(stat, 0.0), spec.min_edge_pct * abs(line))
    ok = bool(margin >= need)
    return Check("ai_edge", "2. AI Edge", ok,
                 f"Projection {proj:.1f} is {margin:+.1f} {'above' if side == 'Over' else 'below'} "
                 f"the line in your direction; needs at least {need:.1f}")


def check_implied(spec: SportSpec, side: str, p_market: float) -> Check:
    """3. Implied chance: the devigged market favors the selected side."""
    ok = bool(p_market > spec.implied_min)
    return Check("implied_chance", "3. Implied Chance", ok,
                 f"Devigged DraftKings price gives {side} {p_market:.1%} (needs more than {spec.implied_min:.0%}); "
                 "single book, no multi-book vote")


def rank_supports(side: str, rank: int, n_teams: int, cutoff: int) -> bool:
    """Rank 1 = softest defense. Over wants a soft defense, Under a tough one."""
    return rank <= cutoff if side == "Over" else rank > n_teams - cutoff


def check_defense(spec: SportSpec, side: str, season_rank: Optional[int], season_n: Optional[int],
                  recent_rank: Optional[int], recent_n: Optional[int], who: str = "", hint: str = "") -> Check:
    """4. Defensive rank: season-long and recent opponent ranks both support the side."""
    label = "4. Defensive Rank"
    if None in (season_rank, season_n, recent_rank, recent_n) or not season_n or not recent_n:
        have = "season" if season_rank is not None else "recent" if recent_rank is not None else "neither"
        return Check("defensive_rank", label, False,
                     f"Not verified ({have} rank available)" + (f": {hint}" if hint else ""), verified=False)
    ok = (rank_supports(side, int(season_rank), int(season_n), spec.rank_cutoff)
          and rank_supports(side, int(recent_rank), int(recent_n), spec.rank_cutoff))
    want = f"{spec.rank_cutoff} softest" if side == "Over" else f"{spec.rank_cutoff} toughest"
    return Check("defensive_rank", label, bool(ok),
                 f"{who}season #{season_rank} of {season_n}, last {spec.recent_games} games #{recent_rank} of {recent_n} "
                 f"(1 = softest); {side} needs both in the {want}")


def check_injury(flagged: Optional[Sequence[str]], hint: str = "") -> Check:
    """5. Injury: `flagged` lists everyone relevant with an active flag; None = the feed could not be read."""
    label = "5. Injury"
    if flagged is None:
        return Check("injury", label, False, "Not verified: " + (hint or "injury feed unavailable or injury adjustment turned off"),
                     verified=False)
    return Check("injury", label, not flagged,
                 "No active injury flags on the team" if not flagged else "Flagged: " + ", ".join(flagged))


def check_game_script(spec: SportSpec, spread: Optional[float], env_verified: bool = True, env_bad: bool = False,
                      env_detail: str = "", hint: str = "") -> Check:
    """6. Game script: no blowout risk and no bad weather (`env_*` describe weather; indoor sports pass verified/not bad)."""
    label = "6. Game Script"
    if spread is None or not env_verified:
        parts = ([hint or "spread unavailable"] if spread is None else []) + ([env_detail or "weather unavailable"] if not env_verified else [])
        return Check("game_script", label, False, "Not verified: " + "; ".join(parts), verified=False)
    ok = bool(abs(spread) < spec.blowout_spread) and not env_bad
    return Check("game_script", label, ok,
                 f"Spread {abs(spread):g} (blowout risk at {spec.blowout_spread:g}+)" + (f"; {env_detail}" if env_detail else ""))


def result_from_checks(side: str, checks: Sequence[Check]) -> SixPointResult:
    """Count the passed checks for one side and attach the tier."""
    score = int(np.sum([c.passed for c in checks]))
    return SixPointResult(side=side, score=score, tier=tier_for(score), checks=tuple(checks))


# --- Which side is shown ----------------------------------------------------------
def choose_best(six: Mapping[str, Mapping[str, Any]], ev_side: str) -> str:
    """Side with the higher score; a tie goes to the model's best-EV side."""
    return max(SIDES, key=lambda s: (six[s]["score"], s == ev_side))


def assemble(over: SixPointResult, under: SixPointResult, ev_side: str) -> Dict[str, Any]:
    """The dict both apps store: {"Over": {...}, "Under": {...}, "best": side shown, "ev_side": model's EV side}."""
    out: Dict[str, Any] = {"Over": over.as_dict(), "Under": under.as_dict(), "ev_side": ev_side}
    out["best"] = choose_best(out, ev_side)
    return out


def side_p_market(res: Mapping[str, Any], ev_side: str) -> Dict[str, float]:
    """Devigged market probability (0-1) of each side from an evaluation result (per-side numbers, else derived)."""
    sides = res.get("sides")
    if sides:
        return {s: float(sides[s]["p_market"]) / 100.0 for s in SIDES}
    p = float(res["p_market"]) / 100.0
    return {ev_side: p, ("Under" if ev_side == "Over" else "Over"): 1.0 - p}


# --- Free spread from ESPN's scoreboard (saves an Odds API credit) -----------------------------
def spread_from_espn_odds(odds: Any) -> Optional[float]:
    """Absolute spread from an ESPN `odds` list (numeric field, else the 'KC -3.5' text); 0 for pick'em; None if absent."""
    for o in odds or []:
        sp = o.get("spread")
        if isinstance(sp, (int, float)) and not isinstance(sp, bool):
            return abs(float(sp))
        det = str(o.get("details") or "")
        if re.search(r"\b(EVEN|PK|PICK)\b", det, re.IGNORECASE):
            return 0.0
        m = re.search(r"([+-]\d+(?:\.\d+)?)\s*$", det)
        if m:
            return abs(float(m[1]))
    return None


def find_espn_spread(payload: Mapping[str, Any], teams: Sequence[str]) -> Optional[float]:
    """Absolute spread for the game between `teams` in an ESPN scoreboard payload (teams matched by nickname), else None."""
    want = {nickname(t) for t in teams if nickname(t)}
    for ev in payload.get("events", []) if isinstance(payload, Mapping) else []:
        comp = (ev.get("competitions") or [{}])[0]
        got = {nickname((c.get("team") or {}).get("displayName") or (c.get("team") or {}).get("name") or "")
               for c in comp.get("competitors", [])}
        if want and want <= got:
            return spread_from_espn_odds(comp.get("odds"))
    return None


# --- Odds-board tagging (same cell text, order and colors in both apps) -----------------
TAG_COL = "Six-Point"


def nickname(team: Any) -> str:
    """Last word of a team name, lower-cased ("LA Clippers" and "Los Angeles Clippers" both -> "clippers")."""
    parts = str(team or "").split()
    return parts[-1].lower() if parts else ""


def opponent_for(player_team: Optional[str], game_teams: List[str]) -> Optional[str]:
    """The other team in a two-team game, or None if the player's team isn't one of them."""
    if not player_team or len(game_teams) != 2:
        return None
    mine = [i for i, t in enumerate(game_teams) if nickname(t) == nickname(player_team)]
    return game_teams[1 - mine[0]] if len(mine) == 1 else None


def shown(six: Any) -> Optional[Mapping[str, Any]]:
    """The scored result for the side that is shown, or None (also None for a 'not scored' reason string)."""
    return six[six["best"]] if isinstance(six, Mapping) and six else None


def tag_text(six: Any) -> str:
    """Cell text that always names the side the score belongs to.

    "🔥 SHARP UNDER 6/6" and "⚡ LEAN OVER 5/6" for 6/6 and 5/6; "4/6 Under" otherwise; a dash (plus the reason) if not scored.
    """
    if isinstance(six, str):
        return f"— {six}"
    best = shown(six)
    if best is None:
        return "—"
    if best["tier"] == "SHARP":
        return f"🔥 SHARP {best['side'].upper()} {best['score']}/6"
    if best["tier"] == "LEAN":
        return f"⚡ LEAN {best['side'].upper()} {best['score']}/6"
    return f"{best['score']}/6 {best['side']}"


def add_six_point_column(summary: pd.DataFrame, six_by_player: Mapping[str, Any]) -> pd.DataFrame:
    """Add the tag column to the odds-board table and list the best-scoring props first (stable)."""
    out = summary.copy()
    out[TAG_COL] = out["Player"].map(lambda p: tag_text(six_by_player.get(p)))
    rank = out["Player"].map(lambda p: (shown(six_by_player.get(p)) or {"score": -1})["score"])
    return out.assign(_rank=rank).sort_values("_rank", ascending=False, kind="stable").drop(columns="_rank").reset_index(drop=True)


# Row colors: Over = green (SHARP) / gold (LEAN); Under = navy (SHARP) / purple (LEAN). White text on all of them.
ROW_COLORS: Dict[Tuple[str, str], str] = {
    ("SHARP", "OVER"): "#166534", ("LEAN", "OVER"): "#a16207",
    ("SHARP", "UNDER"): "#1e3a8a", ("LEAN", "UNDER"): "#6b21a8",
}
LEGEND = ("🔥 SHARP = 6/6 criteria · ⚡ LEAN = 5/6 · the tag names the side the criteria support "
          "(green/gold rows = Over, navy/purple rows = Under) · other rows show their best side's score · — = couldn't be scored")


def style_tags(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Row colors by tier and side (see ROW_COLORS)."""
    def row_style(row: pd.Series) -> List[str]:
        tag = str(row.get(TAG_COL, ""))
        tier = "SHARP" if tag.startswith("🔥") else "LEAN" if tag.startswith("⚡") else ""
        side = "UNDER" if " UNDER " in tag else "OVER" if " OVER " in tag else ""
        color = ROW_COLORS.get((tier, side))
        return [f"background-color:{color};color:#ffffff;font-weight:700" if color else ""] * len(row)

    return df.style.apply(row_style, axis=1)


def assessment_summary(res: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Everything the Final Assessment panel shows, from one result: the same numbers the odds table tags use.

    Returns None when the result has no six-point block. Keys: best, other, ev_side, side_differs, side_ev_pct.
    """
    six = res.get("six_point")
    if not six:
        return None
    best = six[six["best"]]
    other = six["Under" if six["best"] == "Over" else "Over"]
    ev_side = six.get("ev_side", six["best"])
    side_ev = (res.get("sides") or {}).get(best["side"], {}).get("ev_pct", res.get("ev_pct", 0.0))
    return {"best": best, "other": other, "ev_side": ev_side, "side_differs": best["side"] != ev_side, "side_ev_pct": float(side_ev),
            "tag": tag_text(six)}
