"""NBA Prop Evaluator: Standalone App with Robust Odds Parsing & Python 3.9+ Compatibility."""
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

st.set_page_config(
    page_title="NBA Prop Evaluator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }
    .vision-card, [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"], .vision-header {
        background: #b9e6ff !important; border: 2px solid #7dd3fc !important; border-radius: 16px !important; box-shadow: 0 4px 14px rgba(0, 117, 255, 0.08) !important;
    }
    .vision-card { padding: 20px !important; margin-bottom: 20px !important; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: #0f172a !important; font-weight: 800 !important; font-size: 16px !important; }
    .vision-title { color: #0369a1 !important; font-size: 26px; font-weight: 900; margin: 0; }
    .vision-sub { color: #0f172a !important; font-size: 13px; font-weight: 600; margin-top: 4px; }
    [data-testid="stMetric"], .metric-box {
        background: #b9e6ff !important; border: 2px solid #7dd3fc !important; border-radius: 12px !important; padding: 14px 18px !important; text-align: center;
    }
    [data-testid="stMetricValue"], .metric-value { color: #0369a1 !important; font-size: 24px !important; font-weight: 900 !important; }
    [data-testid="stMetricLabel"], .metric-label { color: #0f172a !important; font-size: 11px !important; font-weight: 800 !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
    .metric-value-green { color: #15803d !important; font-size: 24px !important; font-weight: 900 !important; }
    label, p, span, div[data-testid="stMarkdownContainer"] p { color: #0f172a !important; font-weight: 700 !important; }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff !important; color: #0f172a !important; border-radius: 10px !important; border: 2px solid #0284c7 !important; font-weight: 700 !important;
    }
    div[data-baseweb="select"] * { color: #0f172a !important; }
    .stButton>button, [data-testid="stFormSubmitButton"]>button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important; color: #ffffff !important; font-weight: 900 !important; border-radius: 10px !important; border: None !important; padding: 12px 20px !important; width: 100% !important;
    }
    .stButton>button *, [data-testid="stFormSubmitButton"]>button * { color: #ffffff !important; font-weight: 900 !important; }
    [data-testid="stDataFrame"] { background-color: #ffffff !important; border: 2px solid #7dd3fc !important; border-radius: 10px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

STATS = ["Points", "Rebounds", "Assists", "3-Pointers Made", "Pts + Reb + Ast", "Steals", "Blocks"]
STAT_COL: Dict[str, str] = {
    "Points": "POINTS", "Rebounds": "REBOUNDS", "Assists": "ASSISTS", "3-Pointers Made": "3-POINTERS MADE",
    "Pts + Reb + Ast": "PTS + REB + AST", "Steals": "STEALS", "Blocks": "BLOCKS",
}
ODDS_API_MARKETS: Dict[str, str] = {
    "Points": "player_points", "Rebounds": "player_rebounds", "Assists": "player_assists",
    "3-Pointers Made": "player_threes", "Pts + Reb + Ast": "player_points_rebounds_assists",
    "Steals": "player_steals", "Blocks": "player_blocks",
}
CONTINUOUS_STATS = {"POINTS", "PTS + REB + AST"}  # Normal; everything else is a low count -> NegBin/Poisson
# stat -> (weight k, which opponent factor): "def" = points allowed vs league, "pace" = total points vs league
STAT_OPP: Dict[str, Tuple[float, str]] = {
    "POINTS": (0.5, "def"), "PTS + REB + AST": (0.3, "def"), "REBOUNDS": (0.3, "pace"),
    "ASSISTS": (0.3, "pace"), "3-POINTERS MADE": (0.3, "pace"), "STEALS": (0.3, "pace"), "BLOCKS": (0.3, "pace"),
}

CURRENT_SEASON_START = "2026-10-01"   # logs dated on/after this are "current season"
PRIOR_ESPN_SEASON = 2026              # ESPN season param = ending year (2025-26)
CURRENT_NBA_SEASON, PRIOR_NBA_SEASON = "2026-27", "2025-26"
N_PRIOR = 8.0          # pseudo-games of weight given to last season's rates
WINDOW = 10            # recent games used for the current-season estimate
DECAY = 0.8            # EWMA decay per game of age
N_SIMS = 50_000
DEFAULT_CV = 0.35
KELLY_FRACTION, KELLY_CAP = 0.25, 0.05


def _secret(name: str) -> str:
    """Read a secret from Streamlit secrets, falling back to environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def parse_val(v: Any) -> float:
    """Best-effort float parse (handles '-', None, '3/8', thousands separators)."""
    try:
        s = str(v).strip()
        if not s or s in ["-", "None", "null", "--"]:
            return 0.0
        if "/" in s:
            s = s.split("/")[0].strip()
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


def _clean_name(name: str) -> str:
    """Normalize a player name for matching (case, punctuation, suffixes)."""
    s = re.sub(r"[^\w\s]", "", str(name).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.split())


# --------------------------------------------------------------------------------------
# Odds API
# --------------------------------------------------------------------------------------
def list_nba_events(api_key: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Upcoming NBA events (id, label, home, away)."""
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    try:
        resp = requests.get(url, params={"apiKey": api_key}, timeout=8)
        if resp.status_code != 200:
            return pd.DataFrame(), f"Odds API Error ({resp.status_code}): {resp.text}"
        rows = []
        for ev in resp.json():
            home, away = ev.get("home_team", ""), ev.get("away_team", "")
            commence = ev.get("commence_time", "")[:16].replace("T", " ")
            rows.append({"id": ev.get("id"), "label": f"{away} @ {home} ({commence} UTC)", "home": home, "away": away})
        return pd.DataFrame(rows), None
    except Exception as e:
        return pd.DataFrame(), f"Error fetching NBA events: {e}"


def fetch_nba_prop_board(event_id: str, market: str, api_key: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """DraftKings prop board for one event/market, American odds."""
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
    params = {"apiKey": api_key, "regions": "us", "markets": market, "bookmakers": "draftkings", "oddsFormat": "american"}
    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code != 200:
            return pd.DataFrame(), f"Odds API Error ({resp.status_code}): {resp.text}"
        bookmakers = resp.json().get("bookmakers", [])
        if not bookmakers:
            return pd.DataFrame(), "No DraftKings odds found for this event/market."
        markets = bookmakers[0].get("markets", [])
        if not markets:
            return pd.DataFrame(), "No market data available."
        rows = []
        for o in markets[0].get("outcomes", []):
            try:
                price = int(round(float(o.get("price", -110))))
            except Exception:
                price = -110
            rows.append({"Player": o.get("description"), "Line": o.get("point", 0.5), "Price": price, "Outcome": o.get("name")})
        return pd.DataFrame(rows), None
    except Exception as e:
        return pd.DataFrame(), f"Error fetching prop board: {e}"


def summarize_board(board_df: pd.DataFrame) -> pd.DataFrame:
    """One row per player: line, Over price, Under price."""
    if board_df.empty:
        return pd.DataFrame()
    rows = []
    for player, group in board_df.groupby("Player"):
        over, under = -110, -110
        for outcome, price in zip(group["Outcome"].astype(str).str.lower(), group["Price"]):
            if "over" in outcome:
                over = int(price)
            elif "under" in outcome:
                under = int(price)
        rows.append({"Player": player, "Line": group["Line"].iloc[0], "Over": over, "Under": under})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Game logs: ESPN (cloud-safe) first, nba_api fallback
# --------------------------------------------------------------------------------------
ESPN_HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
}
ESPN_SEARCH_URLS: List[str] = [
    "https://site.web.api.espn.com/apis/common/v3/search",
    "https://site.web.api.espn.com/apis/search/v2",
]
ESPN_GAMELOG_URL = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{pid}/gamelog"
TEAM_LIST_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
TEAM_SCHEDULE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{tid}/schedule"


def _walk_json(obj: Any):
    """Yield every dict nested anywhere inside a JSON-like object."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v)


def _espn_find_player_id(name: str) -> Optional[str]:
    """Resolve a player name to an ESPN athlete id (keyless, cloud-friendly)."""
    target = _clean_name(name)
    candidates: List[Tuple[str, str]] = []
    for url in ESPN_SEARCH_URLS:
        try:
            resp = requests.get(
                url,
                params={"query": name, "limit": 10, "type": "player", "sport": "basketball", "league": "nba"},
                headers=ESPN_HEADERS, timeout=10,
            )
            if resp.status_code != 200:
                continue
            for d in _walk_json(resp.json()):
                uid = str(d.get("uid", ""))
                label = d.get("displayName") or d.get("fullName") or d.get("name")
                is_player = str(d.get("type", "")).lower() == "player" or "~a:" in uid
                is_nba = "l:46" in uid or str(d.get("league", "")).lower() == "nba" or str(d.get("defaultLeagueSlug", "")).lower() == "nba"
                if not (label and is_player and is_nba):
                    continue
                m = re.search(r"~a:(\d+)", uid)
                pid = str(d.get("id") or (m.group(1) if m else ""))
                if pid.isdigit():
                    candidates.append((pid, str(label)))
        except Exception:
            continue
        if candidates:
            break
    for pid, label in candidates:
        if _clean_name(label) == target:
            return pid
    return candidates[0][0] if candidates else None


def _split_made(v: Any) -> float:
    """Parse ESPN 'made-attempted' strings like '3-8' into made (3.0)."""
    s = str(v).strip()
    return parse_val(s.split("-")[0]) if "-" in s else parse_val(s)


def _espn_fetch_logs(pid: str, season: Optional[int] = None) -> pd.DataFrame:
    """Regular-season game logs for one ESPN season (None = current), including minutes."""
    resp = requests.get(ESPN_GAMELOG_URL.format(pid=pid), params={"season": season} if season else {}, headers=ESPN_HEADERS, timeout=12)
    resp.raise_for_status()
    data = resp.json()
    labels = [str(x).upper() for x in data.get("labels", [])]
    meta = data.get("events", {}) or {}
    rows: List[Dict[str, Any]] = []
    for require_regular in (True, False):
        for stype in data.get("seasonTypes", []):
            if require_regular and "regular" not in str(stype.get("displayName", "")).lower():
                continue
            for cat in stype.get("categories", []):
                for ev in cat.get("events", []):
                    stats = ev.get("stats", [])
                    if len(stats) != len(labels):
                        continue
                    rec = dict(zip(labels, stats))
                    info = meta.get(str(ev.get("eventId")), {}) or {}
                    opp = (info.get("opponent") or {}).get("abbreviation", "OPP")
                    sep = "vs." if str(info.get("atVs", "vs")).lower() == "vs" else "@"
                    pts, reb, ast = parse_val(rec.get("PTS")), parse_val(rec.get("REB")), parse_val(rec.get("AST"))
                    rows.append({
                        "POINTS": pts, "REBOUNDS": reb, "ASSISTS": ast,
                        "3-POINTERS MADE": _split_made(rec.get("3PT", 0)), "PTS + REB + AST": pts + reb + ast,
                        "STEALS": parse_val(rec.get("STL")), "BLOCKS": parse_val(rec.get("BLK")),
                        "MINUTES": parse_val(rec.get("MIN")),
                        "OPPONENT": f"{sep} {opp}", "DATE": str(info.get("gameDate", ""))[:10],
                    })
        if rows:
            break
    return pd.DataFrame(rows)


def _nba_api_fetch_logs(raw_input: str, season: str) -> pd.DataFrame:
    """Fallback: stats.nba.com via nba_api (works locally, usually blocked on cloud IPs)."""
    matched = nba_players.find_players_by_full_name(raw_input)
    if not matched:
        raise ValueError(f"No NBA player found matching '{raw_input}'.")
    gamelog = playergamelog.PlayerGameLog(
        player_id=matched[0]["id"], season=season,
        headers={"User-Agent": ESPN_HEADERS["User-Agent"], "Referer": "https://www.nba.com/"}, timeout=10,
    )
    df = gamelog.get_data_frames()[0]
    if df is None or df.empty:
        return pd.DataFrame()
    num = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return pd.DataFrame({
        "POINTS": num("PTS"), "REBOUNDS": num("REB"), "ASSISTS": num("AST"), "3-POINTERS MADE": num("FG3M"),
        "PTS + REB + AST": num("PTS") + num("REB") + num("AST"), "STEALS": num("STL"), "BLOCKS": num("BLK"),
        "MINUTES": num("MIN"), "OPPONENT": df["MATCHUP"].astype(str),
        "DATE": pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.strftime("%Y-%m-%d"),
    })


def _finalize_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe, sort oldest -> newest, and tag each game as current/prior season."""
    if df.empty:
        return df
    df = df.dropna(subset=["DATE"]).drop_duplicates(subset=["DATE", "OPPONENT"]).sort_values("DATE").reset_index(drop=True)
    df["SEASON"] = np.where(df["DATE"] >= CURRENT_SEASON_START, "current", "prior")
    return df


@st.cache_data(ttl=900, show_spinner=False)
def _load_logs_cached(player_name: str) -> pd.DataFrame:
    """Prior + current season logs. Raises on total failure so errors are never cached."""
    errors: List[str] = []
    try:
        pid = _espn_find_player_id(player_name)
        if not pid:
            raise ValueError(f"ESPN: no NBA player found for '{player_name}'.")
        df = _finalize_logs(pd.concat([_espn_fetch_logs(pid, None), _espn_fetch_logs(pid, PRIOR_ESPN_SEASON)], ignore_index=True))
        if not df.empty:
            return df
        errors.append("ESPN returned no games.")
    except Exception as e:
        errors.append(f"ESPN error: {e}")
    frames: List[pd.DataFrame] = []
    for season in (CURRENT_NBA_SEASON, PRIOR_NBA_SEASON):
        try:
            frames.append(_nba_api_fetch_logs(player_name, season))
        except Exception as e:
            errors.append(f"nba_api {season}: {e}")
    if frames:
        df = _finalize_logs(pd.concat(frames, ignore_index=True))
        if not df.empty:
            return df
    raise RuntimeError(" | ".join(errors))


def fetch_player_logs(player_name: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Public wrapper: (logs oldest -> newest, error)."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."
    try:
        return _load_logs_cached(player_name.strip()), None
    except Exception as e:
        return pd.DataFrame(), f"Could not fetch game logs for '{player_name.strip()}'. {e}"


# --------------------------------------------------------------------------------------
# Opponent environment (points allowed + scoring pace, from ESPN team schedules)
# --------------------------------------------------------------------------------------
@st.cache_data(ttl=21600, show_spinner=False)
def load_team_meta() -> List[Tuple[str, str]]:
    """[(espn_team_id, display_name)] for all NBA teams."""
    resp = requests.get(TEAM_LIST_URL, headers=ESPN_HEADERS, timeout=12)
    resp.raise_for_status()
    teams = resp.json()["sports"][0]["leagues"][0]["teams"]
    return [(str(t["team"]["id"]), str(t["team"]["displayName"])) for t in teams]


def _score(competitor: Dict[str, Any]) -> Optional[float]:
    """Competitor score from either {'value': x} or plain-string form."""
    s = competitor.get("score")
    if isinstance(s, dict):
        s = s.get("value", s.get("displayValue"))
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _team_totals(tid: str) -> Tuple[float, float, int]:
    """(points for, points against, games) over completed regular-season games."""
    resp = requests.get(TEAM_SCHEDULE_URL.format(tid=tid), params={"seasontype": 2}, headers=ESPN_HEADERS, timeout=12)
    resp.raise_for_status()
    pf = pa = 0.0
    n = 0
    for ev in resp.json().get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        if ((comp.get("status") or {}).get("type") or {}).get("completed") is False:
            continue
        comps = comp.get("competitors", [])
        cid = lambda c: str(c.get("id") or (c.get("team") or {}).get("id"))
        mine = [c for c in comps if cid(c) == tid]
        other = [c for c in comps if cid(c) != tid]
        if not mine or not other:
            continue
        a, b = _score(mine[0]), _score(other[0])
        if a is None or b is None or a + b <= 0:
            continue
        pf, pa, n = pf + a, pa + b, n + 1
    return pf, pa, n


@st.cache_data(ttl=21600, show_spinner=False)
def load_team_scoring() -> pd.DataFrame:
    """Per-team PF/PA/GAMES table. Raises if no completed games exist yet."""
    meta = load_team_meta()
    with ThreadPoolExecutor(max_workers=8) as ex:
        totals = list(ex.map(lambda m: _team_totals(m[0]), meta))
    df = pd.DataFrame({
        "TEAM": [m[1] for m in meta], "PF": [t[0] for t in totals],
        "PA": [t[1] for t in totals], "GAMES": [t[2] for t in totals],
    }).set_index("TEAM")
    if df["GAMES"].sum() == 0:
        raise ValueError("no completed games found yet")
    return df


def opp_multiplier(team: str, stat_col: str, scoring: pd.DataFrame) -> Tuple[float, str]:
    """Shrunk opponent multiplier: 1 + k * n/(n+10) * (factor - 1), clipped to [0.9, 1.1]."""
    if team not in scoring.index or scoring.loc[team, "GAMES"] <= 0:
        return 1.0, f"No scoring data for {team} yet; no opponent adjustment."
    row = scoring.loc[team]
    games = scoring["GAMES"].sum()
    league_pa = scoring["PA"].sum() / games
    league_total = (scoring["PF"].sum() + scoring["PA"].sum()) / games
    def_f = (row["PA"] / row["GAMES"]) / league_pa
    pace_f = ((row["PF"] + row["PA"]) / row["GAMES"]) / league_total
    k, kind = STAT_OPP.get(stat_col, (0.3, "pace"))
    factor = def_f if kind == "def" else pace_f
    shrink = row["GAMES"] / (row["GAMES"] + 10.0)
    mult = float(np.clip(1.0 + k * shrink * (factor - 1.0), 0.9, 1.1))
    return mult, f"{team}: pts-allowed idx {def_f:.2f}, pace idx {pace_f:.2f} ({int(row['GAMES'])} g) -> x{mult:.3f}"


# --------------------------------------------------------------------------------------
# Model math
# --------------------------------------------------------------------------------------
@dataclass
class Projection:
    """Posterior per-game projection for one stat."""
    mu: float
    sigma: float
    minutes: float
    n_current: int
    n_prior: int
    opp_mult: float


def ewma_weights(n: int, decay: float = DECAY) -> np.ndarray:
    """Normalized weights for n games ordered oldest -> newest (newest heaviest)."""
    w = decay ** np.arange(n)[::-1]
    return w / w.sum()


def _cv(x: np.ndarray) -> Optional[float]:
    """Coefficient of variation, or None if the sample is too small/degenerate."""
    if len(x) < 3 or float(np.mean(x)) <= 0:
        return None
    return float(np.std(x, ddof=1) / np.mean(x))


def build_projection(cur_x: np.ndarray, cur_m: np.ndarray, pri_x: np.ndarray, pri_m: np.ndarray, opp_mult: float = 1.0) -> Projection:
    """
    mu = shrunk(rate/min) * shrunk(minutes) * opp_mult, where
    shrunk(v) = (n*v_cur + n0*v_prior)/(n + n0), v_cur = EWMA over last WINDOW games.
    sigma = pooled CV * mu, inflated by (1 + 1/(n+n0)) for parameter uncertainty.
    """
    n = len(cur_x)
    n0 = N_PRIOR if len(pri_x) > 0 else 0.0
    if n == 0 and n0 == 0:
        raise ValueError("No usable games (current or prior season).")
    pairs_rate: List[Tuple[float, float]] = []
    pairs_min: List[Tuple[float, float]] = []
    if n > 0:
        w = ewma_weights(n)
        pairs_rate.append((float(n), float(np.sum(w * cur_x) / np.sum(w * cur_m))))
        pairs_min.append((float(n), float(np.sum(w * cur_m))))
    if n0 > 0:
        pairs_rate.append((n0, float(np.sum(pri_x) / np.sum(pri_m))))
        pairs_min.append((n0, float(np.mean(pri_m[-20:]))))
    blend = lambda pairs: sum(w * v for w, v in pairs) / sum(w for w, _ in pairs)
    rate, minutes = blend(pairs_rate), blend(pairs_min)
    mu = max(rate * minutes * opp_mult, 0.05)
    cvs = [(w, c) for w, c in ((float(n), _cv(cur_x)), (n0, _cv(pri_x[-40:]))) if c is not None and w > 0]
    cv = sum(w * c for w, c in cvs) / sum(w for w, _ in cvs) if cvs else DEFAULT_CV
    cv = max(cv, 0.15)  # floor: prop stats are never this stable; avoids overconfidence on tiny/flat samples
    sigma = max(0.5, cv * mu) * float(np.sqrt(1.0 + 1.0 / (n + n0)))
    return Projection(mu=mu, sigma=sigma, minutes=minutes, n_current=n, n_prior=len(pri_x), opp_mult=opp_mult)


def simulate(mu: float, sigma: float, count_stat: bool, n_sims: int = N_SIMS, seed: int = 42) -> np.ndarray:
    """Vectorized draws: Normal (rounded, >=0) for points/PRA; NegBin (Poisson if under-dispersed) for counts."""
    rng = np.random.default_rng(seed)
    var = sigma ** 2
    if count_stat:
        if var > mu * 1.0001:
            return rng.negative_binomial(mu ** 2 / (var - mu), mu / var, n_sims).astype(float)
        return rng.poisson(mu, n_sims).astype(float)
    return np.maximum(np.rint(rng.normal(mu, sigma, n_sims)), 0.0)


def implied_prob(odds: float) -> float:
    """American odds -> implied probability (with vig)."""
    o = float(odds)
    return -o / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def payout(odds: float) -> float:
    """Profit per 1 unit staked at American odds."""
    o = float(odds)
    return 100.0 / abs(o) if o < 0 else o / 100.0


def devig(over_odds: float, under_odds: float) -> Tuple[float, float]:
    """Multiplicative devig of a two-way market."""
    po, pu = implied_prob(over_odds), implied_prob(under_odds)
    return po / (po + pu), pu / (po + pu)


def prob_to_american(p: float) -> int:
    """Probability -> American odds (clipped to [1%, 99%])."""
    p = float(np.clip(p, 0.01, 0.99))
    return int(round(-100.0 * p / (1.0 - p))) if p >= 0.5 else int(round(100.0 * (1.0 - p) / p))


def kelly_fraction(p_win: float, p_lose: float, odds: float) -> float:
    """Fractional Kelly stake (share of bankroll), capped; 0 when EV <= 0."""
    b = payout(odds)
    full = (b * p_win - p_lose) / b
    return float(np.clip(full * KELLY_FRACTION, 0.0, KELLY_CAP))


def evaluate_nba_prop(player_name: str, stat_category: str, line: float, over_odds: float, under_odds: float,
                      opponent: Optional[str] = None) -> Dict[str, Any]:
    """Full evaluation: projection -> simulation -> devigged edge, EV, Kelly, hit rates."""
    try:
        for o in (over_odds, under_odds):
            if abs(float(o)) < 100:
                return {"error": f"Odds {o:g} aren't valid American odds (must be <= -100 or >= +100)."}
        stat_col = STAT_COL.get(stat_category)
        if stat_col is None:
            return {"error": f"Unknown stat category '{stat_category}'."}
        logs, err = fetch_player_logs(player_name)
        if err:
            return {"error": err}
        logs = logs.assign(**{stat_col: pd.to_numeric(logs[stat_col], errors="coerce")}).dropna(subset=[stat_col])
        played = logs[logs["MINUTES"] > 0]
        cur = played[played["SEASON"] == "current"].tail(WINDOW)
        pri = played[played["SEASON"] == "prior"]

        opp_mult, opp_note = 1.0, None
        if opponent:
            try:
                opp_mult, opp_note = opp_multiplier(opponent, stat_col, load_team_scoring())
            except Exception as e:
                opp_note = f"Opponent data unavailable ({e}); no adjustment applied."

        proj = build_projection(cur[stat_col].to_numpy(float), cur["MINUTES"].to_numpy(float),
                                pri[stat_col].to_numpy(float), pri["MINUTES"].to_numpy(float), opp_mult)
        sims = simulate(proj.mu, proj.sigma, stat_col not in CONTINUOUS_STATS)
        p_over, p_under = float(np.mean(sims > line)), float(np.mean(sims < line))
        decided = max(p_over + p_under, 1e-9)
        dv_over, dv_under = devig(over_odds, under_odds)
        ev_over = p_over * payout(over_odds) - p_under
        ev_under = p_under * payout(under_odds) - p_over
        is_over = ev_over >= ev_under
        p_w, p_l = (p_over, p_under) if is_over else (p_under, p_over)
        odds_side = over_odds if is_over else under_odds
        ev = ev_over if is_over else ev_under
        side_txt = f"{'Over' if is_over else 'Under'} {line:g}"
        p_cond = p_w / decided
        dv_side = dv_over if is_over else dv_under

        sides = {}
        for nm, pw_, pl_, od_, dv_, ev_ in (("Over", p_over, p_under, over_odds, dv_over, ev_over),
                                            ("Under", p_under, p_over, under_odds, dv_under, ev_under)):
            pc = pw_ / decided
            sides[nm] = {"odds": float(od_), "p_model": pc * 100.0, "p_market": dv_ * 100.0,
                         "edge_pp": (pc - dv_) * 100.0, "ev_pct": ev_ * 100.0}

        recent = played.tail(WINDOW)
        vals = recent[stat_col].to_numpy(float)
        warnings: List[str] = []
        if proj.n_current < 8:
            warnings.append(f"Small sample: {proj.n_current} games this season; projection leans on last season ({proj.n_prior} games).")
        if proj.n_prior == 0:
            warnings.append("No prior-season games found; no shrinkage applied.")
        if opp_note:
            warnings.append(opp_note)
        try:
            days_rest: Optional[int] = (date.today() - date.fromisoformat(str(logs["DATE"].iloc[-1]))).days
        except Exception:
            days_rest = None
        return {
            "player_name": player_name, "mean": float(np.mean(sims)), "median": float(np.median(sims)),
            "verdict": "PLAY" if ev > 0 else "PASS", "best_side": side_txt, "ev_pct": ev * 100.0,
            "edge_pp": (p_cond - dv_side) * 100.0, "p_model": p_cond * 100.0, "p_market": dv_side * 100.0,
            "fair_odds": prob_to_american(p_cond), "kelly_pct": kelly_fraction(p_w, p_l, odds_side) * 100.0,
            "proj_minutes": proj.minutes, "opp_mult": proj.opp_mult, "raw_avg": float(np.mean(vals)) if len(vals) else 0.0,
            "hit10": int(np.sum(vals > line)), "n10": len(vals),
            "hit5": int(np.sum(vals[-5:] > line)), "n5": len(vals[-5:]),
            "days_rest": days_rest, "warnings": warnings, "mu": proj.mu, "sides": sides,
            "gamelog_details": [
                {"stat_value": float(v), "season": s, "x_label": f"{d}<br>{o}"}
                for v, s, d, o in zip(vals, recent["SEASON"], recent["DATE"], recent["OPPONENT"])
            ],
        }
    except Exception as e:
        return {"error": f"Evaluation error: {e}"}


# --------------------------------------------------------------------------------------
# Bet log + closing line value
# --------------------------------------------------------------------------------------
LOG_PATH = "bet_log.csv"
LOG_COLS: List[str] = [
    "ID", "TS", "GAME_DATE", "PLAYER", "STAT", "SIDE", "LINE", "ODDS", "STAKE", "OPPONENT",
    "P_MODEL", "P_MARKET", "EDGE_PP", "EV_PCT", "CLOSE_LINE", "CLOSE_OVER", "CLOSE_UNDER",
    "LINE_MOVE", "CLV_PP", "RESULT", "ACTUAL", "PROFIT",
]
NUM_COLS: List[str] = [
    "LINE", "ODDS", "STAKE", "P_MODEL", "P_MARKET", "EDGE_PP", "EV_PCT", "CLOSE_LINE", "CLOSE_OVER",
    "CLOSE_UNDER", "LINE_MOVE", "CLV_PP", "ACTUAL", "PROFIT",
]
EDITABLE_COLS: List[str] = ["STAKE", "CLOSE_LINE", "CLOSE_OVER", "CLOSE_UNDER", "RESULT", "ACTUAL"]
RESULT_OPTIONS: List[str] = ["Pending", "Win", "Loss", "Push"]


def _today_et() -> date:
    """Today's date in US Eastern time (falls back to UTC)."""
    try:
        return datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return datetime.now(timezone.utc).date()


def _implied_vec(odds: np.ndarray) -> np.ndarray:
    """Vectorized American odds -> implied probability (NaN-safe)."""
    o = np.asarray(odds, dtype=float)
    return np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))


def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived columns.
    LINE_MOVE = (close_line - bet_line) * (+1 Over / -1 Under): positive = line moved your way.
    CLV_PP    = devigged closing prob of your side - devigged prob at bet time (same line only).
    PROFIT    = stake * payout (Win), -stake (Loss), 0 (Push), NaN (Pending).
    """
    df = df.copy()
    is_over = df["SIDE"].eq("Over").to_numpy()
    df["LINE_MOVE"] = (df["CLOSE_LINE"] - df["LINE"]) * np.where(is_over, 1.0, -1.0)
    co, cu = df["CLOSE_OVER"].to_numpy(float), df["CLOSE_UNDER"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = (np.abs(co) >= 100) & (np.abs(cu) >= 100) & (df["CLOSE_LINE"] == df["LINE"]).to_numpy()
        io_, iu_ = _implied_vec(co), _implied_vec(cu)
        q_close = np.where(is_over, io_, iu_) / (io_ + iu_)
        df["CLV_PP"] = np.where(valid, q_close * 100.0 - df["P_MARKET"].to_numpy(float), np.nan)
        odds, stake = df["ODDS"].to_numpy(float), df["STAKE"].to_numpy(float)
        b = np.where(odds < 0, 100.0 / np.abs(odds), odds / 100.0)
        res = df["RESULT"]
        df["PROFIT"] = np.select([res.eq("Win"), res.eq("Loss"), res.eq("Push")], [stake * b, -stake, 0.0], default=np.nan)
    return df


def normalize_log(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce any frame (CSV import, editor output) into the canonical log schema."""
    df = df.copy()
    for c in LOG_COLS:
        if c not in df.columns:
            df[c] = np.nan if c in NUM_COLS else ""
    df = df[LOG_COLS]
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in [c for c in LOG_COLS if c not in NUM_COLS]:
        df[c] = df[c].fillna("").astype(str)
    df["RESULT"] = np.where(df["RESULT"].isin(RESULT_OPTIONS), df["RESULT"], "Pending")
    df = df[df["PLAYER"].str.strip() != ""].copy()
    missing = df["ID"].str.strip() == ""
    df.loc[missing, "ID"] = [uuid.uuid4().hex[:8] for _ in range(int(missing.sum()))]
    df = df.drop_duplicates("ID", keep="last").reset_index(drop=True)
    return recompute_derived(df)


def read_log_file() -> pd.DataFrame:
    """Load the on-disk log if present (best effort; ephemeral on Streamlit Cloud)."""
    try:
        if os.path.exists(LOG_PATH):
            return normalize_log(pd.read_csv(LOG_PATH, dtype=str))
    except Exception:
        pass
    return normalize_log(pd.DataFrame(columns=LOG_COLS))


def save_log_file(df: pd.DataFrame) -> None:
    """Persist the log to disk (best effort)."""
    try:
        df.to_csv(LOG_PATH, index=False)
    except Exception:
        pass


def make_bet_row(res: Dict[str, Any], ctx: Dict[str, Any], side: str, stake: float, game_date: date) -> Dict[str, Any]:
    """One log row from an evaluation result, for the chosen side."""
    s = res["sides"][side]
    return {
        "ID": uuid.uuid4().hex[:8], "TS": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "GAME_DATE": game_date.isoformat(), "PLAYER": ctx["player"], "STAT": ctx["stat"], "SIDE": side,
        "LINE": float(ctx["line"]), "ODDS": s["odds"], "STAKE": float(stake), "OPPONENT": ctx.get("opponent") or "",
        "P_MODEL": s["p_model"], "P_MARKET": s["p_market"], "EDGE_PP": s["edge_pp"], "EV_PCT": s["ev_pct"],
        "RESULT": "Pending",
    }


def add_bet(log: pd.DataFrame, row: Dict[str, Any]) -> Tuple[pd.DataFrame, bool]:
    """Append a bet unless the same player/stat/side/line/date is already logged."""
    dup = ((log["PLAYER"] == row["PLAYER"]) & (log["STAT"] == row["STAT"]) & (log["SIDE"] == row["SIDE"])
           & (log["LINE"] == row["LINE"]) & (log["GAME_DATE"] == row["GAME_DATE"])).any()
    if dup:
        return log, False
    return normalize_log(pd.concat([log, pd.DataFrame([row])], ignore_index=True)), True


def apply_board_as_closing(log: pd.DataFrame, board_summary: pd.DataFrame, stat_label: str) -> Tuple[pd.DataFrame, int]:
    """Fill closing line/odds on pending bets from a freshly fetched board (same stat market)."""
    if board_summary.empty:
        return log, 0
    summ = board_summary.set_index("Player")
    mask = (log["RESULT"] == "Pending") & (log["STAT"] == stat_label) & log["PLAYER"].isin(summ.index)
    if not mask.any():
        return log, 0
    log = log.copy()
    for col, src in (("CLOSE_LINE", "Line"), ("CLOSE_OVER", "Over"), ("CLOSE_UNDER", "Under")):
        log.loc[mask, col] = log.loc[mask, "PLAYER"].map(summ[src]).astype(float)
    return recompute_derived(log), int(mask.sum())


def auto_grade(log: pd.DataFrame) -> Tuple[pd.DataFrame, int, List[str]]:
    """Grade pending bets whose game date has passed using ESPN logs (UTC date match, +1 day tolerance)."""
    log = log.copy()
    notes: List[str] = []
    graded = 0
    today = _today_et().isoformat()
    for idx in log.index[(log["RESULT"] == "Pending") & (log["GAME_DATE"] <= today)]:
        row = log.loc[idx]
        col = STAT_COL.get(row["STAT"])
        try:
            gd = str(row["GAME_DATE"])
            nd = (date.fromisoformat(gd) + timedelta(days=1)).isoformat()
            logs, err = fetch_player_logs(row["PLAYER"])
            if err or col is None:
                notes.append(f"{row['PLAYER']}: {err or 'unknown stat'}")
                continue
            hit = logs[logs["DATE"] == gd]
            if hit.empty:
                hit = logs[logs["DATE"] == nd]
            if hit.empty:
                continue  # not played / not in logs yet (or DNP: void it manually by deleting the row)
            actual = float(hit.iloc[0][col])
            line = float(row["LINE"])
            if actual == line:
                result = "Push"
            elif (actual > line) == (row["SIDE"] == "Over"):
                result = "Win"
            else:
                result = "Loss"
            log.at[idx, "ACTUAL"], log.at[idx, "RESULT"] = actual, result
            graded += 1
        except Exception as e:
            notes.append(f"{row['PLAYER']}: {e}")
    return recompute_derived(log), graded, notes


def summarize_log(log: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Headline stats, per-stat table, and model-calibration table."""
    wl = log[log["RESULT"].isin(["Win", "Loss"])]
    won = wl["RESULT"].eq("Win").to_numpy(float)
    staked = float(wl["STAKE"].sum())
    clv = log["CLV_PP"].dropna()
    pm, pk = wl["P_MODEL"].to_numpy(float) / 100.0, wl["P_MARKET"].to_numpy(float) / 100.0
    stats: Dict[str, float] = {
        "bets": float(len(log)), "graded": float(log["RESULT"].isin(RESULT_OPTIONS[1:]).sum()),
        "win_pct": float(won.mean() * 100.0) if len(wl) else float("nan"),
        "units": float(log["PROFIT"].sum()),
        "roi": float(log["PROFIT"].sum() / staked * 100.0) if staked > 0 else float("nan"),
        "avg_clv": float(clv.mean()) if len(clv) else float("nan"), "pos_clv": float((clv > 0).mean() * 100.0) if len(clv) else float("nan"),
        "n_clv": float(len(clv)),
        "brier_model": float(np.mean((pm - won) ** 2)) if len(wl) else float("nan"),
        "brier_market": float(np.mean((pk - won) ** 2)) if len(wl) else float("nan"),
    }
    tmp = log.assign(_staked=np.where(log["RESULT"].isin(["Win", "Loss"]), log["STAKE"], 0.0))
    by_stat = tmp.groupby("STAT").agg(BETS=("ID", "count"), UNITS=("PROFIT", "sum"), STAKED=("_staked", "sum"),
                                      AVG_EDGE_PP=("EDGE_PP", "mean"), AVG_CLV_PP=("CLV_PP", "mean"))
    by_stat["ROI_%"] = np.where(by_stat["STAKED"] > 0, by_stat["UNITS"] / by_stat["STAKED"].replace(0, np.nan) * 100.0, np.nan)
    by_stat = by_stat.drop(columns="STAKED").round(2).reset_index()
    if len(wl):
        bins = pd.cut(wl["P_MODEL"], [0, 45, 50, 55, 60, 65, 101], right=False)
        calib = wl.assign(_bin=bins, _w=won * 100.0).groupby("_bin", observed=True).agg(
            BETS=("ID", "count"), AVG_MODEL_PCT=("P_MODEL", "mean"), ACTUAL_WIN_PCT=("_w", "mean")).round(1).reset_index()
        calib["_bin"] = calib["_bin"].astype(str)
        calib = calib.rename(columns={"_bin": "MODEL_PROB_BUCKET"})
    else:
        calib = pd.DataFrame(columns=["MODEL_PROB_BUCKET", "BETS", "AVG_MODEL_PCT", "ACTUAL_WIN_PCT"])
    return stats, by_stat, calib


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
def _metric(col: Any, label: str, value: str, tone: str = "neutral") -> None:
    """Render one metric card; tone in {'neutral','good','bad'}."""
    color = {"neutral": "#0369a1", "good": "#15803d", "bad": "#b91c1c"}[tone]
    col.markdown(
        f'<div class="metric-box"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color} !important;">{value}</div></div>',
        unsafe_allow_html=True,
    )


def _fmt(x: float, spec: str = ".1f", suffix: str = "") -> str:
    """Format a number, or an em dash for NaN."""
    return "—" if x is None or not np.isfinite(x) else f"{x:{spec}}{suffix}"


def _tone(x: float) -> str:
    """Metric tone from sign (NaN -> neutral)."""
    return "neutral" if x is None or not np.isfinite(x) else ("good" if x > 0 else "bad")


def _opp_options(game_teams: List[str]) -> List[str]:
    """Opponent dropdown: None, the selected game's two teams first, then everyone else."""
    try:
        names = [n for _, n in load_team_meta()]
    except Exception:
        names = []
    gt = [t for t in game_teams if t]
    return ["None"] + gt + sorted(n for n in names if n not in gt)


def render_log_form(res: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    """'Log this bet' controls under the results."""
    ss = st.session_state
    if ss.get("_log_uid") != res["uid"]:
        ss["log_side"] = "Over" if res["best_side"].startswith("Over") else "Under"
        ss["_log_uid"] = res["uid"]
    st.markdown("### 📝 Log this bet")
    c1, c2, c3, c4 = st.columns(4)
    side = c1.selectbox("Side", ["Over", "Under"], key="log_side")
    stake = c2.number_input("Stake (units)", min_value=0.01, value=1.0, step=0.25, key="log_stake")
    gdate = c3.date_input("Game date (ET)", value=_today_et(), key="log_gdate")
    c4.write("")
    if c4.button("Log bet", key="log_btn"):
        row = make_bet_row(res, ctx, side, stake, gdate)
        ss["bet_log"], added = add_bet(ss["bet_log"], row)
        if added:
            save_log_file(ss["bet_log"])
            ss["editor_ver"] += 1
            st.toast(f"Logged {side} {ctx['line']:g} — {ctx['player']}")
        else:
            st.warning("That exact bet (player, stat, side, line, date) is already in your log.")


def render_results(res: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    """Metric cards, notes, chart, and the log form."""
    stat_category, line_val = ctx["stat"], ctx["line"]
    st.markdown(f"## {res['player_name']} | {stat_category}")
    c = st.columns(6)
    _metric(c[0], "MEAN", f"{res['mean']:.2f}")
    _metric(c[1], "MEDIAN", f"{res['median']:.1f}")
    _metric(c[2], "BEST SIDE" if res["verdict"] == "PLAY" else "LEAN (PASS)", res["best_side"], _tone(res["ev_pct"]))
    _metric(c[3], "FAIR ODDS", f"{res['fair_odds']:+d}")
    _metric(c[4], "EDGE VS DEVIG", f"{res['edge_pp']:+.1f} pp", _tone(res["edge_pp"]))
    _metric(c[5], "EV", f"{res['ev_pct']:+.1f}%", _tone(res["ev_pct"]))
    st.write("")
    d = st.columns(6)
    _metric(d[0], "PROJ MINUTES", f"{res['proj_minutes']:.1f}")
    _metric(d[1], "RAW L10 AVG", f"{res['raw_avg']:.2f}")
    _metric(d[2], "OVER HIT L5 / L10", f"{res['hit5']}/{res['n5']} · {res['hit10']}/{res['n10']}")
    _metric(d[3], "OPP ADJ", f"x{res['opp_mult']:.3f}")
    _metric(d[4], "MODEL vs MARKET", f"{res['p_model']:.1f}% / {res['p_market']:.1f}%")
    _metric(d[5], "¼ KELLY", f"{res['kelly_pct']:.1f}% BR")
    if res["days_rest"] is not None:
        st.caption(f"Days since last logged game: {res['days_rest']}")
    for w in res["warnings"]:
        st.caption(f"⚠️ {w}")
    st.markdown("---")
    render_log_form(res, ctx)
    st.markdown("---")
    st.markdown("### 📊 Game Log Performance")
    st.caption("Last 10 games (faded bars = last season). Dashed blue = line, dotted orange = model projection.")
    gl = pd.DataFrame(res["gamelog_details"])
    if gl.empty:
        return
    colors = np.where(gl["stat_value"] > line_val, "#22c55e", np.where(gl["stat_value"] < line_val, "#ef4444", "#94a3b8"))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=gl["x_label"], y=gl["stat_value"], marker_color=colors,
        marker_opacity=np.where(gl["season"] == "current", 1.0, 0.5),
        text=[f"{v:.1f}" if v % 1 else f"{int(v)}" for v in gl["stat_value"]],
        textposition="outside", textfont=dict(color="#0f172a", size=12, family="sans-serif"),
    ))
    fig.add_shape(type="line", x0=-0.5, x1=len(gl) - 0.5, y0=line_val, y1=line_val, line=dict(color="#2563eb", width=2, dash="dash"))
    fig.add_shape(type="line", x0=-0.5, x1=len(gl) - 0.5, y0=res["mu"], y1=res["mu"], line=dict(color="#f97316", width=2, dash="dot"))
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color="#0f172a"), height=380,
        margin=dict(l=20, r=20, t=30, b=40), showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(color="#0f172a")),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", tickfont=dict(color="#0f172a")),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_bet_log() -> None:
    """Bet Log tab: editable table, auto-grading, closing-line capture, performance + CLV summary."""
    ss = st.session_state
    st.caption("On Streamlit Cloud the server disk resets on reboot/redeploy — download your CSV regularly and re-import it here.")
    a, b, c = st.columns(3)
    if a.button("🏁 Auto-grade finished games", key="grade_btn"):
        ss["bet_log"], n, notes = auto_grade(ss["bet_log"])
        save_log_file(ss["bet_log"])
        ss["editor_ver"] += 1
        ss["grade_notes"] = notes[:5]
        st.toast(f"Graded {n} bet(s)")
        st.rerun()
    board = ss.get("board_df")
    if b.button("📸 Use current board as closing lines", key="close_btn"):
        if board is None or board.empty or not ss.get("board_stat"):
            st.warning("Fetch a prop board first (Evaluate tab, Step 3) — ideally right before tip-off.")
        else:
            ss["bet_log"], n = apply_board_as_closing(ss["bet_log"], summarize_board(board), ss["board_stat"])
            save_log_file(ss["bet_log"])
            ss["editor_ver"] += 1
            st.toast(f"Closing lines applied to {n} pending bet(s) ({ss['board_stat']} market)")
            st.rerun()
    c.download_button("⬇️ Download log (CSV)", data=ss["bet_log"].to_csv(index=False).encode("utf-8"),
                      file_name="bet_log.csv", mime="text/csv", key="dl_log")
    for note in ss.pop("grade_notes", []):
        st.warning(note)
    up = st.file_uploader("Import a saved bet log (CSV)", type="csv", key="log_upload")
    if up is not None:
        uid = getattr(up, "file_id", f"{up.name}-{up.size}")
        if ss.get("_up_id") != uid:
            try:
                merged = pd.concat([ss["bet_log"], normalize_log(pd.read_csv(up, dtype=str))], ignore_index=True)
                ss["bet_log"] = normalize_log(merged)
                ss["_up_id"] = uid
                save_log_file(ss["bet_log"])
                ss["editor_ver"] += 1
                st.rerun()
            except Exception as e:
                st.error(f"Could not import that file: {e}")

    log: pd.DataFrame = ss["bet_log"]
    if log.empty:
        st.info("No bets logged yet. Evaluate a prop, then use “Log this bet”.")
        return
    cols = [c for c in LOG_COLS if c != "ID"] + ["ID"]
    key = f"bet_editor_{ss['editor_ver']}"
    fmt = lambda label: st.column_config.NumberColumn(label, format="%.1f")
    edited = st.data_editor(
        log[cols], key=key, num_rows="dynamic", hide_index=True, use_container_width=True,
        disabled=[c for c in cols if c not in EDITABLE_COLS],
        column_config={
            "ID": None, "TS": None,
            "RESULT": st.column_config.SelectboxColumn("RESULT", options=RESULT_OPTIONS, required=True),
            "P_MODEL": fmt("P_MODEL %"), "P_MARKET": fmt("P_MKT %"), "EDGE_PP": fmt("EDGE pp"), "EV_PCT": fmt("EV %"),
            "LINE_MOVE": fmt("LINE MOVE"), "CLV_PP": fmt("CLV pp"), "PROFIT": st.column_config.NumberColumn("PROFIT (u)", format="%.2f"),
        },
    )
    state = ss.get(key) or {}
    if state.get("edited_rows") or state.get("added_rows") or state.get("deleted_rows"):
        ss["bet_log"] = normalize_log(edited)
        save_log_file(ss["bet_log"])
        ss["editor_ver"] += 1
        st.rerun()

    stats, by_stat, calib = summarize_log(ss["bet_log"])
    st.markdown("### 📈 Performance")
    m = st.columns(6)
    _metric(m[0], "BETS / GRADED", f"{int(stats['bets'])} / {int(stats['graded'])}")
    _metric(m[1], "WIN %", _fmt(stats["win_pct"], ".1f", "%"))
    _metric(m[2], "UNITS", _fmt(stats["units"], "+.2f"), _tone(stats["units"]))
    _metric(m[3], "ROI", _fmt(stats["roi"], "+.1f", "%"), _tone(stats["roi"]))
    _metric(m[4], "AVG CLV", _fmt(stats["avg_clv"], "+.2f", " pp"), _tone(stats["avg_clv"]))
    _metric(m[5], "+CLV RATE", _fmt(stats["pos_clv"], ".0f", "%"))
    st.caption(
        f"CLV is measured on {int(stats['n_clv'])} bet(s) with closing odds at the same line. "
        f"Brier (lower is better): model {_fmt(stats['brier_model'], '.4f')} vs market {_fmt(stats['brier_market'], '.4f')}. "
        "Treat everything as noise until you have 50+ graded bets; CLV converges much faster than win/loss."
    )
    graded = ss["bet_log"][ss["bet_log"]["RESULT"].isin(["Win", "Loss", "Push"])].sort_values("TS")
    if len(graded) >= 2:
        fig = go.Figure(go.Scatter(x=list(range(1, len(graded) + 1)), y=graded["PROFIT"].cumsum(), mode="lines+markers",
                                   line=dict(color="#0369a1", width=3)))
        fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color="#0f172a"), height=300,
                          margin=dict(l=20, r=20, t=30, b=40), title="Cumulative units",
                          xaxis=dict(title="Graded bet #", gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**By stat category**")
    st.dataframe(by_stat, use_container_width=True, hide_index=True)
    st.markdown("**Calibration (graded bets)**")
    st.dataframe(calib, use_container_width=True, hide_index=True)


def main() -> None:
    """Streamlit entry point."""
    st.markdown(
        """
        <div class="vision-header" style="padding: 20px;">
            <div class="vision-title">⚡ NBA Prop Evaluator</div>
            <div class="vision-sub">Quantitative Monte Carlo NBA Player Prop Analytics Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ss = st.session_state
    api_key = _secret("ODDS_API_KEY")
    for key, default in (("player", "Jayson Tatum"), ("stat", STATS[0]), ("line", 26.5), ("over", -110),
                         ("under", -110), ("opp", "None"), ("board_teams", []), ("editor_ver", 0)):
        ss.setdefault(key, default)
    if "bet_log" not in ss:
        ss["bet_log"] = read_log_file()

    tab_eval, tab_log = st.tabs(["⚡ Evaluate", "📒 Bet Log"])
    with tab_eval:
        with st.expander("📡 Fetch Live DraftKings Lines (The Odds API)", expanded=True):
            if not api_key:
                st.warning("Please configure `ODDS_API_KEY` in Streamlit Secrets.")
            else:
                selected_stat_label = st.selectbox("Select Prop Market", list(ODDS_API_MARKETS.keys()), key="live_stat_market")
                market_key = ODDS_API_MARKETS[selected_stat_label]
                if st.button("Step 1: Load Upcoming NBA Games", key="load_nba_games"):
                    events_df, err = list_nba_events(api_key)
                    if err:
                        st.error(err)
                    else:
                        ss["events_df"] = events_df
                events_df = ss.get("events_df")
                if events_df is not None and not events_df.empty:
                    game_idx = st.selectbox("Step 2: Select Game", range(len(events_df)), format_func=lambda i: str(events_df.loc[i, "label"]), key="live_game_idx")
                    if st.button(f"Step 3: Fetch {selected_stat_label} Odds", key="fetch_board_odds"):
                        board_df, err = fetch_nba_prop_board(str(events_df.loc[game_idx, "id"]), market_key, api_key)
                        if err:
                            st.error(err)
                        else:
                            ss["board_df"] = board_df
                            ss["board_stat"] = selected_stat_label
                            ss["board_teams"] = [str(events_df.loc[game_idx, "away"]), str(events_df.loc[game_idx, "home"])]
                board_df = ss.get("board_df")
                if board_df is not None and not board_df.empty:
                    summary = summarize_board(board_df)
                    st.dataframe(summary, use_container_width=True)
                    player_pick = st.selectbox("Step 4: Choose Player to Evaluate", summary["Player"].tolist(), key="pick_player_dropdown")
                    matched_row = summary[summary["Player"] == player_pick]
                    if not matched_row.empty:
                        r = matched_row.iloc[0]
                        if st.button(f"⚡ Populate {player_pick} ({r['Line']} | Over {r['Over']} / Under {r['Under']})", key="populate_btn_odds"):
                            ss.update(player=player_pick, stat=ss.get("board_stat", selected_stat_label), line=float(r["Line"]),
                                      over=int(r["Over"]), under=int(r["Under"]), opp="None")
                            st.rerun()

        opp_opts = _opp_options(ss["board_teams"])
        with st.form("nba_form"):
            c1, c2, c3 = st.columns([2, 1, 1])
            player_name = c1.text_input("Player Name", value=ss["player"])
            stat_category = c2.selectbox("Stat Category", STATS, index=STATS.index(ss["stat"]) if ss["stat"] in STATS else 0)
            line_val = c3.number_input("Line", step=0.5, value=float(ss["line"]))
            c4, c5, c6 = st.columns(3)
            over_odds = c4.number_input("Over Odds", value=int(ss["over"]))
            under_odds = c5.number_input("Under Odds", value=int(ss["under"]))
            opponent = c6.selectbox("Opponent (optional)", opp_opts, index=opp_opts.index(ss["opp"]) if ss["opp"] in opp_opts else 0)
            submitted = st.form_submit_button("⚡ Evaluate Prop")

        if submitted:
            ss.update(player=player_name, stat=stat_category, line=line_val, over=over_odds, under=under_odds, opp=opponent)
            res = evaluate_nba_prop(player_name, stat_category, line_val, over_odds, under_odds, None if opponent == "None" else opponent)
            if "error" in res:
                ss["last_res"], ss["last_err"] = None, res["error"]
            else:
                res["uid"] = uuid.uuid4().hex
                ss["last_res"], ss["last_err"] = res, None
                ss["last_ctx"] = {"player": player_name, "stat": stat_category, "line": float(line_val),
                                  "opponent": None if opponent == "None" else opponent}
        if ss.get("last_err"):
            st.error(ss["last_err"])
        elif ss.get("last_res"):
            render_results(ss["last_res"], ss["last_ctx"])

        with st.expander("ℹ️ How the model works"):
            st.markdown(
                "- **Projection** = shrunk per-minute rate × shrunk minutes × opponent multiplier. Current-season games "
                "(last 10, EWMA decay 0.8) are blended with last season's rates using ~8 pseudo-games of prior weight.\n"
                "- **Spread** = pooled coefficient of variation × projection, inflated for estimation uncertainty.\n"
                "- **Distribution**: Normal for points/PRA; Negative Binomial (Poisson if under-dispersed) for rebounds, assists, 3PM, steals, blocks. 50,000 seeded draws.\n"
                "- **Market**: two-way multiplicative devig. Edge = model % − devigged market %. EV uses the model probability at the offered price (pushes refunded).\n"
                "- **Opponent**: points-allowed (points, PRA) or scoring-pace (other stats) index from ESPN results, shrunk by games played, capped ±10%. It is a proxy, not true positional defense.\n"
                "- **Kelly**: ¼ Kelly capped at 5% of bankroll; 0 when EV ≤ 0.\n"
                "- **CLV**: devigged closing probability of your side minus the devigged probability when you bet, same line only. "
                "If the line moved, LINE MOVE shows how far (positive = in your favor)."
            )
    with tab_log:
        render_bet_log()


if __name__ == "__main__":
    main()
