"""Six-point prop criteria for the NFL Prop Evaluator.

Score = number of passed checks for a side:  6/6 -> SHARP,  5/6 -> LEAN,  <= 4/6 -> PASS.
A check that cannot be verified counts as NOT passed (and is shown as unverified).

    1. Baby line      line >= MIN_LINE[stat]
    2. AI edge        side-direction margin (proj - line for Over, line - proj for Under)
                      >= max(MIN_EDGE_ABS[stat], MIN_EDGE_PCT * line)
    3. Implied chance devigged market probability of the side > IMPLIED_MIN
    4. Defensive rank season-long AND last-RECENT_GAMES opponent ranks (vs the player's position
                      group, 1 = softest) both support the side: Over needs rank <= RANK_CUTOFF,
                      Under needs rank > n_teams - RANK_CUTOFF
    5. Injury         player not flagged, no flagged skill teammate, QB1 not flagged (receiving props)
    6. Game script    |spread| < BLOWOUT_SPREAD and no bad weather at an outdoor stadium
                      (wind, rain, snow or extreme cold in the 3 hours from kickoff)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from nfl_matchups import _defense_games

# --- Thresholds (edit freely) -------------------------------------------------
MIN_LINE: Dict[str, float] = {
    "PASSING YARDS": 175.5, "PASS COMPLETIONS": 14.5, "PASS ATTEMPTS": 24.5, "PASSING TDS": 1.5,
    "RUSHING YARDS": 35.5, "RUSHING ATTEMPTS": 8.5, "RECEIVING YARDS": 35.5, "RECEPTIONS": 3.5,
    "INTERCEPTIONS": 0.5,
}
MIN_EDGE_ABS: Dict[str, float] = {
    "PASSING YARDS": 12.0, "PASS COMPLETIONS": 1.5, "PASS ATTEMPTS": 2.0, "PASSING TDS": 0.25,
    "RUSHING YARDS": 6.0, "RUSHING ATTEMPTS": 1.5, "RECEIVING YARDS": 6.0, "RECEPTIONS": 0.75,
    "INTERCEPTIONS": 0.15,
}
MIN_EDGE_PCT = 0.06
IMPLIED_MIN = 0.50
RANK_CUTOFF = 16            # half of a 32-team league
RECENT_GAMES = 4            # "recent" = each defense's last 4 games this season
RECENT_MIN_GAMES = 2        # fewer than this and the recent rank is unverified
BLOWOUT_SPREAD = 10.0
WIND_MPH, GUST_MPH, PRECIP_MM, SNOW_CM, COLD_F = 15.0, 25.0, 3.0, 0.5, 20.0
WEATHER_HOURS = 3
SHARP_SCORE, LEAN_SCORE = 6, 5

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
# ESPN code -> nflverse code (applied to both sides before comparing).
ABBR_CANON: Dict[str, str] = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LA"}
# Home stadium (lat, lon, roof). roof: outdoor | dome | retractable | canopy. Approximate coordinates.
STADIUMS: Dict[str, Tuple[float, float, str]] = {
    "ARI": (33.5276, -112.2626, "retractable"), "ATL": (33.7554, -84.4010, "dome"),
    "BAL": (39.2780, -76.6227, "outdoor"), "BUF": (42.7738, -78.7870, "outdoor"),
    "CAR": (35.2258, -80.8528, "outdoor"), "CHI": (41.8623, -87.6167, "outdoor"),
    "CIN": (39.0955, -84.5161, "outdoor"), "CLE": (41.5061, -81.6995, "outdoor"),
    "DAL": (32.7473, -97.0945, "retractable"), "DEN": (39.7439, -105.0201, "outdoor"),
    "DET": (42.3400, -83.0456, "dome"), "GB": (44.5013, -88.0622, "outdoor"),
    "HOU": (29.6847, -95.4107, "retractable"), "IND": (39.7601, -86.1639, "retractable"),
    "JAX": (30.3239, -81.6373, "outdoor"), "KC": (39.0489, -94.4839, "outdoor"),
    "LA": (33.9535, -118.3392, "canopy"), "LAC": (33.9535, -118.3392, "canopy"),
    "LV": (36.0909, -115.1833, "dome"), "MIA": (25.9580, -80.2389, "outdoor"),
    "MIN": (44.9735, -93.2575, "dome"), "NE": (42.0909, -71.2643, "outdoor"),
    "NO": (29.9511, -90.0812, "dome"), "NYG": (40.8135, -74.0745, "outdoor"),
    "NYJ": (40.8135, -74.0745, "outdoor"), "PHI": (39.9008, -75.1675, "outdoor"),
    "PIT": (40.4468, -80.0158, "outdoor"), "SEA": (47.5952, -122.3316, "outdoor"),
    "SF": (37.4030, -121.9700, "outdoor"), "TB": (27.9759, -82.5033, "outdoor"),
    "TEN": (36.1665, -86.7713, "outdoor"), "WAS": (38.9076, -76.8645, "outdoor"),
}


def canon(abbr: Any) -> str:
    """Upper-case team code mapped to the nflverse convention."""
    a = str(abbr or "").upper().strip()
    return ABBR_CANON.get(a, a)


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
        """Plain-dict form (safe for st.session_state)."""
        return {"side": self.side, "score": self.score, "tier": self.tier,
                "checks": [c.__dict__.copy() for c in self.checks]}


@dataclass(frozen=True)
class GameInfo:
    """What the ESPN scoreboard says about one game."""

    home: str
    away: str
    kickoff_iso: str
    spread: Optional[float]      # absolute points, None if no line posted
    neutral: bool
    indoor: Optional[bool]


@dataclass(frozen=True)
class Weather:
    """Kickoff-window weather verdict for the game script check."""

    verified: bool
    bad: bool
    detail: str


def tier_for(score: int) -> str:
    """6 -> SHARP, 5 -> LEAN, otherwise PASS."""
    return "SHARP" if score >= SHARP_SCORE else "LEAN" if score == LEAN_SCORE else "PASS"


# --- Individual checks --------------------------------------------------------
def check_line(stat: str, line: float) -> Check:
    """1. Baby line: the line must be at or above the stat's minimum."""
    need = MIN_LINE.get(stat, 0.0)
    ok = bool(line >= need)
    return Check("baby_line", "1. Line High Enough (Baby Line)", ok,
                 f"Line {line:g} vs minimum {need:g} for this stat" + ("" if ok else ": too low, too risky"))


def check_edge(side: str, stat: str, line: float, proj: float) -> Check:
    """2. AI edge: projection far enough past the line in the side's direction."""
    margin = proj - line if side == "Over" else line - proj
    need = max(MIN_EDGE_ABS.get(stat, 0.0), MIN_EDGE_PCT * abs(line))
    ok = bool(margin >= need)
    return Check("ai_edge", "2. AI Edge", ok,
                 f"Projection {proj:.1f} is {margin:+.1f} {'above' if side == 'Over' else 'below'} "
                 f"the line in your direction; needs at least {need:.1f}")


def check_implied(side: str, p_market: float) -> Check:
    """3. Implied chance: the devigged market favors the selected side."""
    ok = bool(p_market > IMPLIED_MIN)
    return Check("implied_chance", "3. Implied Chance", ok,
                 f"Devigged DraftKings price gives {side} {p_market:.1%} (needs more than {IMPLIED_MIN:.0%}); single book, no multi-book vote")


def rank_supports(side: str, rank: int, n_teams: int, cutoff: int = RANK_CUTOFF) -> bool:
    """Rank 1 = softest defense. Over wants a soft defense, Under a tough one."""
    return rank <= cutoff if side == "Over" else rank > n_teams - cutoff


def check_defense(side: str, season_rank: Optional[int], season_n: Optional[int],
                  recent_rank: Optional[int], recent_n: Optional[int], pos: str = "") -> Check:
    """4. Defensive rank: season-long and recent ranks (vs this position group) both support the side."""
    label = "4. Defensive Rank"
    if None in (season_rank, season_n, recent_rank, recent_n):
        have = "season" if season_rank is not None else "recent" if recent_rank is not None else "neither"
        return Check("defensive_rank", label, False,
                     f"Not verified ({have} rank available): pick an opponent (or Auto) and make sure the weekly data loaded", verified=False)
    ok = rank_supports(side, season_rank, season_n) and rank_supports(side, recent_rank, recent_n)
    want = f"{RANK_CUTOFF} softest" if side == "Over" else f"{RANK_CUTOFF} toughest"
    return Check("defensive_rank", label, bool(ok),
                 f"Vs {pos or 'position'}s: season #{season_rank} of {season_n}, last {RECENT_GAMES} games #{recent_rank} of {recent_n} "
                 f"(1 = softest); {side} needs both in the {want}")


def flagged_players(inj: Mapping[str, Any]) -> List[str]:
    """Everyone with an active flag: the player, listed skill teammates, QB1 (receiving props)."""
    out: List[str] = []
    if inj.get("self_status"):
        out.append(f"the player ({inj['self_status']})")
    out += [f"{x['name']} ({x['status']})" for x in inj.get("listed", []) if x.get("p_out", 0.0) > 0]
    if inj.get("qb_flag"):
        out.append(str(inj["qb_flag"]))
    return out


def check_injury(inj: Optional[Mapping[str, Any]]) -> Check:
    """5. Injury: nobody relevant carries an active flag."""
    label = "5. Injury"
    if not inj or not inj.get("checked") or not inj.get("feed_ok", False):
        return Check("injury", label, False,
                     "Not verified: injury feed unavailable, injury adjustment turned off, or the player has no regular share", verified=False)
    flagged = flagged_players(inj)
    return Check("injury", label, not flagged,
                 "No active injury flags on the team" if not flagged else "Flagged: " + ", ".join(flagged))


def check_game_script(spread: Optional[float], weather: Optional[Weather]) -> Check:
    """6. Game script: no blowout risk and no bad weather."""
    label = "6. Game Script"
    if spread is None or weather is None or not weather.verified:
        parts = []
        if spread is None:
            parts.append("spread unavailable")
        if weather is None or not weather.verified:
            parts.append(weather.detail if weather else "weather unavailable")
        return Check("game_script", label, False, "Not verified: " + "; ".join(parts), verified=False)
    blow = abs(spread) >= BLOWOUT_SPREAD
    ok = (not blow) and (not weather.bad)
    spread_txt = f"Spread {abs(spread):g}" + (f" (blowout risk at {BLOWOUT_SPREAD:g}+)" if blow else "")
    return Check("game_script", label, bool(ok), f"{spread_txt}; {weather.detail}")


def score_prop(side: str, stat: str, line: float, proj: float, p_market: float,
               season_rank: Optional[int], season_n: Optional[int], recent_rank: Optional[int],
               recent_n: Optional[int], pos: str, inj: Optional[Mapping[str, Any]],
               spread: Optional[float], weather: Optional[Weather]) -> SixPointResult:
    """Run all six checks for one side and return the score and tier."""
    checks = (
        check_line(stat, line),
        check_edge(side, stat, line, proj),
        check_implied(side, p_market),
        check_defense(side, season_rank, season_n, recent_rank, recent_n, pos),
        check_injury(inj),
        check_game_script(spread, weather),
    )
    score = int(np.sum([c.passed for c in checks]))
    return SixPointResult(side=side, score=score, tier=tier_for(score), checks=checks)


# --- Data helpers (pure) --------------------------------------------------------
def recent_defense_rank(weekly: pd.DataFrame, stat: str, pos: str, season: int, opponent: str,
                        n_recent: int = RECENT_GAMES, min_games: int = RECENT_MIN_GAMES
                        ) -> Tuple[Optional[int], Optional[int]]:
    """Opponent's rank (1 = softest) by what it allowed `pos` in its last `n_recent` games this season.

    Returns (rank, n_teams) or (None, None) when the opponent has too few games.
    """
    g = _defense_games(weekly, stat, pos)
    g = g[g["SEASON"] == season].sort_values(["OPP", "WEEK"])
    g = g[g.groupby("OPP").cumcount(ascending=False) < n_recent]
    agg = g.groupby("OPP")["V"].agg(["mean", "size"])
    agg = agg[agg["size"] >= min_games]
    opp = canon(opponent)
    if opp not in agg.index:
        return None, None
    rank = agg["mean"].rank(ascending=False, method="min")
    return int(rank[opp]), int(len(agg))


def _spread_from_odds(odds: Any) -> Optional[float]:
    """Absolute spread from an ESPN `odds` list (numeric field, else the 'KC -3.5' text)."""
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


def parse_scoreboard_game(payload: Mapping[str, Any], team: str, opponent: Optional[str] = None) -> Optional[GameInfo]:
    """Find `team`'s game (vs `opponent` if given) in an ESPN scoreboard payload."""
    t, o = canon(team), canon(opponent) if opponent else None
    for ev in payload.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): canon((c.get("team") or {}).get("abbreviation")) for c in comp.get("competitors", [])}
        if t not in teams.values() or (o and o not in teams.values()):
            continue
        return GameInfo(home=teams.get("home", ""), away=teams.get("away", ""),
                        kickoff_iso=str(comp.get("date") or ev.get("date") or ""),
                        spread=_spread_from_odds(comp.get("odds")), neutral=bool(comp.get("neutralSite")),
                        indoor=(comp.get("venue") or {}).get("indoor"))
    return None


def summarize_forecast(payload: Mapping[str, Any], kickoff_iso: str, hours: int = WEATHER_HOURS) -> Optional[Dict[str, float]]:
    """Worst conditions in the `hours` hours from kickoff, from an Open-Meteo hourly payload (UTC)."""
    h = payload.get("hourly") or {}
    if not h.get("time"):
        return None
    start = pd.Timestamp(kickoff_iso).tz_convert("UTC").floor("h")
    times = pd.to_datetime(pd.Series(h["time"]), utc=True)
    mask = ((times >= start) & (times < start + pd.Timedelta(hours=hours))).to_numpy()
    if not mask.any():
        return None
    col = lambda k: np.nan_to_num(np.asarray(h.get(k) or [0.0] * len(times), dtype=float)[mask])
    return {"wind": float(col("wind_speed_10m").max()), "gust": float(col("wind_gusts_10m").max()),
            "precip": float(col("precipitation").sum()), "snow": float(col("snowfall").sum()),
            "temp": float(col("temperature_2m").min())}


def judge_weather(w: Mapping[str, float]) -> Weather:
    """Apply the thresholds to a forecast summary."""
    problems = [txt for bad, txt in (
        (w["wind"] >= WIND_MPH or w["gust"] >= GUST_MPH, f"wind {w['wind']:.0f} mph (gusts {w['gust']:.0f})"),
        (w["precip"] >= PRECIP_MM, f"rain {w['precip']:.1f} mm"),
        (w["snow"] >= SNOW_CM, f"snow {w['snow']:.1f} cm"),
        (w["temp"] <= COLD_F, f"{w['temp']:.0f}°F"),
    ) if bad]
    base = f"forecast wind {w['wind']:.0f} mph, {w['temp']:.0f}°F, precip {w['precip']:.1f} mm"
    return Weather(True, bool(problems), ("bad weather: " + ", ".join(problems)) if problems else "weather " + base)


def game_weather(info: Optional[GameInfo], fetch_forecast: Callable[[float, float], Mapping[str, Any]]) -> Weather:
    """Weather verdict for a game: indoor stadiums pass; outdoor ones use the forecast."""
    if info is None:
        return Weather(False, False, "game not found on the ESPN scoreboard, weather not checked")
    if info.neutral:
        return Weather(False, False, "neutral-site game, weather not checked")
    lat_lon_roof = STADIUMS.get(canon(info.home))
    if lat_lon_roof is None:
        return Weather(False, False, f"unknown stadium for {info.home}, weather not checked")
    lat, lon, roof = lat_lon_roof
    if roof != "outdoor" or info.indoor:
        return Weather(True, False, f"{roof if roof != 'outdoor' else 'indoor'} stadium, no weather risk")
    try:
        summary = summarize_forecast(fetch_forecast(lat, lon), info.kickoff_iso)
    except Exception as e:
        return Weather(False, False, f"forecast unavailable ({e})")
    if summary is None:
        return Weather(False, False, "no forecast hours for kickoff (beyond the 16-day window?)")
    return judge_weather(summary)


# --- Network fetchers (wrapped with st.cache_data in the app) --------------------
def fetch_game_info(team: str, opponent: Optional[str], gameday: Optional[str]) -> Optional[GameInfo]:
    """ESPN scoreboard lookup: gameday +/- 1 day if known (time zones shift dates), else the next 14 days."""
    import requests
    if gameday:
        day = pd.Timestamp(gameday)
        start, end = day - pd.Timedelta(days=1), day + pd.Timedelta(days=1)
    else:
        start = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        end = start + pd.Timedelta(days=14)
    resp = requests.get(ESPN_SCOREBOARD, params={"dates": f"{start:%Y%m%d}-{end:%Y%m%d}", "limit": 100}, timeout=8)
    resp.raise_for_status()
    return parse_scoreboard_game(resp.json(), team, opponent)


def fetch_forecast(lat: float, lon: float) -> Dict[str, Any]:
    """Open-Meteo hourly forecast (UTC, mph, mm, °F) for the next 16 days."""
    import requests
    params = {"latitude": lat, "longitude": lon, "forecast_days": 16, "timezone": "UTC",
              "hourly": "temperature_2m,precipitation,snowfall,wind_speed_10m,wind_gusts_10m",
              "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "mm"}
    resp = requests.get(OPEN_METEO, params=params, timeout=8)
    resp.raise_for_status()
    return resp.json()


# --- Small TTL cache (no Streamlit dependency, so the engine can use it from anywhere) ------
_CACHE: Dict[Tuple[Any, ...], Tuple[float, Any]] = {}


def _cached(ttl: float, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Memoize successful calls of `fn` for `ttl` seconds (exceptions are never cached)."""
    def wrapper(*args: Any) -> Any:
        key = (fn.__name__,) + args
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        val = fn(*args)
        _CACHE[key] = (time.time(), val)
        return val
    wrapper.__name__ = f"cached_{fn.__name__}"
    return wrapper


cached_game_info = _cached(300, fetch_game_info)
cached_forecast = _cached(1800, fetch_forecast)


# --- Orchestration -------------------------------------------------------------
def assess(res: Mapping[str, Any], stat_category: str, line: float, weekly: Optional[pd.DataFrame], season: int,
           get_game: Optional[Callable[[str, Optional[str], Optional[str]], Optional[GameInfo]]] = None,
           get_forecast: Optional[Callable[[float, float], Mapping[str, Any]]] = None,
           gameday: Optional[str] = None) -> Dict[str, Any]:
    """Six-point assessment for BOTH sides of an evaluate_prop result.

    The tag follows the better-scoring side (ties go to the model's EV side), so a 5/6 or 6/6 on either
    side is never hidden. Returns {"Over": {...}, "Under": {...}, "best": side, "ev_side": side}.
    """
    get_game = get_game or cached_game_info          # looked up at call time so tests can swap them
    get_forecast = get_forecast or cached_forecast
    stat = stat_category.upper().strip()
    ev_side = "Over" if str(res["best_side"]).startswith("Over") else "Under"
    sides = res.get("sides")
    if sides:
        p_mkt = {s_: float(sides[s_]["p_market"]) / 100.0 for s_ in ("Over", "Under")}
    else:
        p_best = float(res["p_market"]) / 100.0
        p_mkt = {ev_side: p_best, ("Under" if ev_side == "Over" else "Over"): 1.0 - p_best}
    m = res.get("matchup") or {}
    recent_rank = recent_n = None
    if m and weekly is not None:
        recent_rank, recent_n = recent_defense_rank(weekly, stat, m["pos"], season, m["opponent"])
    info: Optional[GameInfo] = None
    if m.get("opponent"):                 # without a known opponent we cannot tell which game is meant
        try:
            info = get_game(str(res["team"]), m["opponent"], gameday or m.get("gameday"))
        except Exception:
            info = None
    weather = game_weather(info, get_forecast)
    out: Dict[str, Any] = {"ev_side": ev_side}
    for side in ("Over", "Under"):
        out[side] = score_prop(side, stat, float(line), float(res["mu"]), p_mkt[side], m.get("rank"), m.get("n_teams"),
                               recent_rank, recent_n, str(m.get("pos", "")), res.get("inj"),
                               info.spread if info else None, weather).as_dict()
    out["best"] = max(("Over", "Under"), key=lambda sd: (out[sd]["score"], sd == ev_side))
    return out


# --- Odds-board tagging --------------------------------------------------------
TAG_COL = "Six-Point"
NICKNAMES: Dict[str, str] = {
    "ARI": "cardinals", "ATL": "falcons", "BAL": "ravens", "BUF": "bills", "CAR": "panthers", "CHI": "bears",
    "CIN": "bengals", "CLE": "browns", "DAL": "cowboys", "DEN": "broncos", "DET": "lions", "GB": "packers",
    "HOU": "texans", "IND": "colts", "JAX": "jaguars", "KC": "chiefs", "LA": "rams", "LAC": "chargers",
    "LV": "raiders", "MIA": "dolphins", "MIN": "vikings", "NE": "patriots", "NO": "saints", "NYG": "giants",
    "NYJ": "jets", "PHI": "eagles", "PIT": "steelers", "SEA": "seahawks", "SF": "49ers", "TB": "buccaneers",
    "TEN": "titans", "WAS": "commanders",
}


def game_matches(label: str, team: Optional[str], opponent: Optional[str]) -> bool:
    """True if both teams' nicknames appear in the Odds API game label ("Kansas City Chiefs @ Denver Broncos ...")."""
    nicks = [NICKNAMES.get(canon(t)) for t in (team, opponent)]
    return all(nicks) and all(n in str(label).lower() for n in nicks)


def teams_in_label(label: str) -> List[str]:
    """Team codes whose nicknames appear in an Odds API game label."""
    low = str(label).lower()
    return [code for code, nick in NICKNAMES.items() if nick in low]


def opponent_from_label(label: str, team: Optional[str]) -> Optional[str]:
    """The other team's code when `team` is one of exactly two teams in the label, else None."""
    teams = teams_in_label(label)
    t = canon(team)
    return next((c for c in teams if c != t), None) if len(teams) == 2 and t in teams else None


def label_gameday(label: str) -> Optional[str]:
    """Game date (YYYY-MM-DD, Eastern) from the end of an event label.

    Handles "(2026-11-09 8:20 PM EST)" (what the app shows now, already Eastern), and the older
    "(2026-11-09T02:12:00Z)" / "(2026-11-09 02:12 UTC)" UTC forms (converted to Eastern).
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2}) \d{1,2}:\d{2} [AP]M E[SD]T", str(label))
    if m:
        return m[1]
    m = re.search(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})(?::\d{2})?\s*(?:Z|UTC)", str(label))
    if not m:
        return None
    return pd.Timestamp(f"{m[1]} {m[2]}", tz="UTC").tz_convert("America/New_York").strftime("%Y-%m-%d")


def _best(six: Any) -> Optional[Mapping[str, Any]]:
    """The scored result for the recommended side, or None (also None for a not-scored reason string)."""
    return six[six["best"]] if isinstance(six, Mapping) else None


def tag_text(six: Any) -> str:
    """Cell text: tier tag for 5/6 and 6/6, plain score otherwise; a dash plus the reason when not scored."""
    if isinstance(six, str):
        return f"— {six}"
    best = _best(six)
    if best is None:
        return "—"
    if best["tier"] == "SHARP":
        return f"🔥 SHARP {best['score']}/6 · {best['side']}"
    if best["tier"] == "LEAN":
        return f"⚡ LEAN {best['score']}/6 · {best['side']}"
    return f"{best['score']}/6"


def add_six_point_column(summary: pd.DataFrame, six_by_player: Mapping[str, Any]) -> pd.DataFrame:
    """Add the tag column to the odds-board table and list the best-scoring props first (stable)."""
    out = summary.copy()
    out[TAG_COL] = out["Player"].map(lambda p: tag_text(six_by_player.get(p)))
    rank = out["Player"].map(lambda p: (_best(six_by_player.get(p)) or {"score": -1})["score"])
    return out.assign(_rank=rank).sort_values("_rank", ascending=False, kind="stable").drop(columns="_rank").reset_index(drop=True)


def style_tags(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Row colors: green for SHARP, gold for LEAN (white-on-dark so they read on the dark table)."""
    def row_style(row: pd.Series) -> List[str]:
        tag = str(row.get(TAG_COL, ""))
        css = ("background-color:#166534;color:#ffffff;font-weight:700" if tag.startswith("🔥")
               else "background-color:#a16207;color:#ffffff;font-weight:700" if tag.startswith("⚡") else "")
        return [css] * len(row)

    return df.style.apply(row_style, axis=1)
