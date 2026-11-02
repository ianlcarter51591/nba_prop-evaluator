"""NBA Prop Evaluator: Standalone App with Robust Odds Parsing & Python 3.9+ Compatibility."""
import json
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

STATS = ["Points", "Rebounds", "Assists", "3-Pointers Made", "Pts + Reb + Ast", "Pts + Ast", "Pts + Reb", "Reb + Ast", "Steals", "Blocks"]
STAT_COL: Dict[str, str] = {
    "Points": "POINTS", "Rebounds": "REBOUNDS", "Assists": "ASSISTS", "3-Pointers Made": "3-POINTERS MADE",
    "Pts + Reb + Ast": "PTS + REB + AST", "Steals": "STEALS", "Blocks": "BLOCKS",
    "Pts + Ast": "PTS + AST", "Pts + Reb": "PTS + REB", "Reb + Ast": "REB + AST",
}
ODDS_API_MARKETS: Dict[str, str] = {
    "Points": "player_points", "Rebounds": "player_rebounds", "Assists": "player_assists",
    "3-Pointers Made": "player_threes", "Pts + Reb + Ast": "player_points_rebounds_assists",
    "Steals": "player_steals", "Blocks": "player_blocks",
    "Pts + Ast": "player_points_assists", "Pts + Reb": "player_points_rebounds", "Reb + Ast": "player_rebounds_assists",
}
CONTINUOUS_STATS = {"POINTS", "PTS + REB + AST", "PTS + AST", "PTS + REB", "REB + AST"}  # Normal; everything else is a low count -> NegBin/Poisson
STAT_OPP: Dict[str, Tuple[float, str]] = {
    "POINTS": (0.5, "def"), "PTS + REB + AST": (0.3, "def"), "REBOUNDS": (0.3, "pace"),
    "ASSISTS": (0.3, "pace"), "3-POINTERS MADE": (0.3, "pace"), "STEALS": (0.3, "pace"), "BLOCKS": (0.3, "pace"),
    "PTS + AST": (0.4, "def"), "PTS + REB": (0.4, "def"), "REB + AST": (0.3, "pace"),
}

CURRENT_SEASON_START = "2026-10-01"
PRIOR_ESPN_SEASON = 2026
CURRENT_NBA_SEASON, PRIOR_NBA_SEASON = "2026-27", "2025-26"

B2B_OPP_BOOST: Dict[str, float] = {
    "POINTS": 0.015, "PTS + REB + AST": 0.010, "REBOUNDS": 0.005, "ASSISTS": 0.005,
    "3-POINTERS MADE": 0.005, "STEALS": 0.0, "BLOCKS": 0.0,
    "PTS + AST": 0.012, "PTS + REB": 0.012, "REB + AST": 0.005,
}
B2B_OWN_EFFECT: Dict[str, float] = {
    "POINTS": -0.02, "PTS + REB + AST": -0.015, "REBOUNDS": -0.01, "ASSISTS": -0.01,
    "3-POINTERS MADE": -0.015, "STEALS": 0.0, "BLOCKS": 0.0,
    "PTS + AST": -0.018, "PTS + REB": -0.018, "REB + AST": -0.01,
}
BT_STATS: List[str] = list(B2B_OPP_BOOST.keys()) + ["MINUTES"]
BT_TAU = 0.02
EFFECTS_PATH = "b2b_effects.json"
# Last-season weight, in pseudo-games. Per-minute production is stable year to year (high weight); minutes and roles
# change more (lower weight). Defaults are informed assumptions: run the Backtest tab to estimate them from data.
N_PRIOR_RATE = 12.0
N_PRIOR_MIN = 6.0
PRIOR_GRID: List[float] = [0, 2, 4, 6, 8, 12, 16, 24, 32, 48, 80]
EARLY_GAMES = 15
PRIOR_PATH = "prior_weights.json"
WINDOW = 10
DECAY = 0.8
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


def _to_et_date(iso: str) -> Optional[date]:
    """Parse an ISO timestamp into its US Eastern calendar date."""
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        tz = ZoneInfo("America/New_York") if ZoneInfo else timezone(timedelta(hours=-5))
        return dt.astimezone(tz).date()
    except Exception:
        return None


def _iso_to_date(s: Any) -> Optional[date]:
    """'YYYY-MM-DD' -> date, or None if blank/invalid."""
    try:
        return date.fromisoformat(str(s))
    except ValueError:
        return None


def _clean_name(name: str) -> str:
    """Normalize a player name for matching."""
    s = re.sub(r"[^\w\s]", "", str(name).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.split())


# --------------------------------------------------------------------------------------
# Odds API
# --------------------------------------------------------------------------------------
def list_nba_events(api_key: str) -> Tuple[pd.DataFrame, Optional[str]]:
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    try:
        resp = requests.get(url, params={"apiKey": api_key}, timeout=8)
        if resp.status_code != 200:
            return pd.DataFrame(), f"Odds API Error ({resp.status_code}): {resp.text}"
        rows = []
        for ev in resp.json():
            home, away = ev.get("home_team", ""), ev.get("away_team", "")
            commence = ev.get("commence_time", "")[:16].replace("T", " ")
            et = _to_et_date(ev.get("commence_time", ""))
            rows.append({"id": ev.get("id"), "label": f"{away} @ {home} ({commence} UTC)", "home": home, "away": away,
                         "game_date": et.isoformat() if et else ""})
        return pd.DataFrame(rows), None
    except Exception as e:
        return pd.DataFrame(), f"Error fetching NBA events: {e}"


def fetch_nba_prop_board(event_id: str, market: str, api_key: str) -> Tuple[pd.DataFrame, Optional[str]]:
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
# Game logs: ESPN first, nba_api fallback
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
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v)


def _espn_find_player_id(name: str) -> Optional[str]:
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


@st.cache_data(ttl=86400, show_spinner=False)
def player_headshot_url(name: str) -> Optional[str]:
    """ESPN headshot URL for an NBA player (checked to exist), or None if not found."""
    try:
        pid = _espn_find_player_id(name)
        if not pid:
            return None
        url = f"https://a.espncdn.com/i/headshots/nba/players/full/{pid}.png"
        ok = requests.head(url, headers=ESPN_HEADERS, timeout=6, allow_redirects=True).status_code == 200
        return url if ok else None
    except Exception:
        return None


def _split_made(v: Any) -> float:
    s = str(v).strip()
    return parse_val(s.split("-")[0]) if "-" in s else parse_val(s)


def _split_att(v: Any) -> float:
    """'9-18' -> 18 (attempts)."""
    s = str(v).strip()
    return parse_val(s.split("-")[1]) if "-" in s else 0.0


def _espn_fetch_logs(pid: str, season: Optional[int] = None) -> pd.DataFrame:
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
                        "PTS + AST": pts + ast, "PTS + REB": pts + reb, "REB + AST": reb + ast,
                        "STEALS": parse_val(rec.get("STL")), "BLOCKS": parse_val(rec.get("BLK")),
                        "FGA": _split_att(rec.get("FG", 0)), "FTA": _split_att(rec.get("FT", 0)), "TOV": parse_val(rec.get("TO")),
                        "MINUTES": parse_val(rec.get("MIN")),
                        "OPPONENT": f"{sep} {opp}",
                        "DATE": (lambda d: d.isoformat() if d else str(info.get("gameDate", ""))[:10])(_to_et_date(info.get("gameDate", ""))),
                    })
        if rows:
            break
    return pd.DataFrame(rows)


def _nba_api_fetch_logs(raw_input: str, season: str) -> pd.DataFrame:
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
        "PTS + AST": num("PTS") + num("AST"), "PTS + REB": num("PTS") + num("REB"), "REB + AST": num("REB") + num("AST"),
        "FGA": num("FGA"), "FTA": num("FTA"), "TOV": num("TOV"),
        "MINUTES": num("MIN"), "OPPONENT": df["MATCHUP"].astype(str),
        "DATE": pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.strftime("%Y-%m-%d"),
    })


def _finalize_logs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.dropna(subset=["DATE"]).drop_duplicates(subset=["DATE", "OPPONENT"]).sort_values("DATE").reset_index(drop=True)
    df["SEASON"] = np.where(df["DATE"] >= CURRENT_SEASON_START, "current", "prior")
    return df


@st.cache_data(ttl=900, show_spinner=False)
def _load_logs_cached(player_name: str) -> pd.DataFrame:
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
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."
    try:
        return _load_logs_cached(player_name.strip()), None
    except Exception as e:
        return pd.DataFrame(), f"Could not fetch game logs for '{player_name.strip()}'. {e}"


# --------------------------------------------------------------------------------------
# Opponent environment
# --------------------------------------------------------------------------------------
@st.cache_data(ttl=21600, show_spinner=False)
def load_team_meta() -> List[Tuple[str, str]]:
    resp = requests.get(TEAM_LIST_URL, headers=ESPN_HEADERS, timeout=12)
    resp.raise_for_status()
    teams = resp.json()["sports"][0]["leagues"][0]["teams"]
    return [(str(t["team"]["id"]), str(t["team"]["displayName"])) for t in teams]


def _score(competitor: Dict[str, Any]) -> Optional[float]:
    s = competitor.get("score")
    if isinstance(s, dict):
        s = s.get("value", s.get("displayValue"))
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _team_totals(tid: str) -> Tuple[float, float, int]:
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


@st.cache_data(ttl=3600, show_spinner=False)
def load_team_game_dates(team: str) -> List[date]:
    tid = next((i for i, n in load_team_meta() if n == team), None)
    if tid is None:
        raise ValueError(f"unknown team '{team}'")
    resp = requests.get(TEAM_SCHEDULE_URL.format(tid=tid), params={"seasontype": 2}, headers=ESPN_HEADERS, timeout=12)
    resp.raise_for_status()
    found = {_to_et_date(str(ev.get("date", ""))) for ev in resp.json().get("events", [])}
    return sorted(d for d in found if d is not None)


def b2b_status(game_dates: List[date], game_day: date) -> str:
    days = set(game_dates)
    yesterday = (game_day - timedelta(days=1)) in days
    tomorrow = (game_day + timedelta(days=1)) in days
    return "both" if yesterday and tomorrow else "second" if yesterday else "first" if tomorrow else "none"


def b2b_multiplier(status: str, stat_col: str, boosts: Optional[Dict[str, float]] = None) -> float:
    b = B2B_OPP_BOOST if boosts is None else boosts
    return 1.0 + b.get(stat_col, 0.0) if status in ("second", "both") else 1.0


def opp_multiplier(team: str, stat_col: str, scoring: pd.DataFrame) -> Tuple[float, str]:
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
# Back-to-back: own-team detection + historical backtest
# --------------------------------------------------------------------------------------
TEAM_ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{tid}/roster"


def own_b2b_status(prev_date: Optional[date], game_day: date) -> str:
    if prev_date is None:
        return "unknown"
    return "second" if (game_day - prev_date).days == 1 else "rested"


def own_b2b_multiplier(status: str, stat_col: str, effects: Optional[Dict[str, float]] = None) -> float:
    eff = B2B_OWN_EFFECT if effects is None else effects
    return 1.0 + eff.get(stat_col, 0.0) if status == "second" else 1.0


@st.cache_data(ttl=86400, show_spinner=False)
def load_team_abbr_meta() -> List[Tuple[str, str, str]]:
    resp = requests.get(TEAM_LIST_URL, headers=ESPN_HEADERS, timeout=12)
    resp.raise_for_status()
    teams = resp.json()["sports"][0]["leagues"][0]["teams"]
    return [(str(t["team"]["id"]), str(t["team"]["displayName"]), str(t["team"].get("abbreviation", ""))) for t in teams]


def _roster(tid: str) -> List[Tuple[str, str]]:
    resp = requests.get(TEAM_ROSTER_URL.format(tid=tid), headers=ESPN_HEADERS, timeout=12)
    resp.raise_for_status()
    found: Dict[str, str] = {}
    for d in _walk_json(resp.json()):
        m = re.search(r"~a:(\d+)", str(d.get("uid", "")))
        name = d.get("fullName") or d.get("displayName")
        if m and name and ("jersey" in d or "position" in d):
            found[m.group(1)] = str(name)
    return list(found.items())


def _schedule_dates(tid: str, season: int) -> List[date]:
    resp = requests.get(TEAM_SCHEDULE_URL.format(tid=tid), params={"seasontype": 2, "season": season}, headers=ESPN_HEADERS, timeout=12)
    resp.raise_for_status()
    found = {_to_et_date(str(ev.get("date", ""))) for ev in resp.json().get("events", [])}
    return sorted(d for d in found if d is not None)


def build_backtest_frame(player_logs: Dict[Tuple[str, int], pd.DataFrame], sched: pd.DataFrame) -> pd.DataFrame:
    frames = [df.assign(PID=pid, SEASON=season) for (pid, season), df in player_logs.items()
              if df is not None and not df.empty and (df["MINUTES"] > 0).any()]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d[d["MINUTES"] > 0].copy()
    d["DT"] = pd.to_datetime(d["DATE"], errors="coerce")
    d = d.dropna(subset=["DT"]).sort_values(["PID", "DT"]).reset_index(drop=True)
    g = d.groupby(["PID", "SEASON"])
    d["OWN_B2B"] = ((d["DT"] - g["DT"].shift(1)).dt.days == 1).astype(float)
    d["HOME"] = d["OPPONENT"].str.startswith("vs").astype(float)
    d["OPP"] = d["OPPONENT"].str.split().str[-1]
    d = d[(g["MINUTES"].transform("mean") >= 20) & (g["MINUTES"].transform("size") >= 40)].copy()
    d = d[d["OPP"].isin(sched["ABBR"].unique())].copy()
    prev_day = (d["DT"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    d["OPP_B2B"] = pd.MultiIndex.from_arrays([d["OPP"], prev_day]).isin(pd.MultiIndex.from_frame(sched[["ABBR", "DATE"]])).astype(float)
    g = d.groupby(["PID", "SEASON"])
    ratios = (d[BT_STATS] / g[BT_STATS].transform("mean")).replace([np.inf, -np.inf], np.nan)
    ratios.columns = [f"R_{c}" for c in BT_STATS]
    raw = d[BT_STATS].add_prefix("X_")
    return pd.concat([d[["PID", "SEASON", "DATE", "OWN_B2B", "OPP_B2B", "HOME"]], ratios, raw], axis=1)


def _ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    e = y - X @ beta
    scores = pd.DataFrame(X * e[:, None]).groupby(groups).sum().to_numpy()
    n_g, n, k = scores.shape[0], len(y), X.shape[1]
    v = xtx_inv @ (scores.T @ scores) @ xtx_inv * (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - k, 1))
    return beta, np.sqrt(np.clip(np.diag(v), 0.0, None))


def estimate_b2b_effects(frame: pd.DataFrame, tau: float = BT_TAU) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for c in BT_STATS:
        sub = frame[["PID", "OWN_B2B", "OPP_B2B", "HOME", f"R_{c}"]].dropna()
        if len(sub) < 100 or sub["OWN_B2B"].sum() < 10 or sub["OPP_B2B"].sum() < 10:
            continue
        X = np.column_stack([np.ones(len(sub)), sub["OWN_B2B"], sub["OPP_B2B"], sub["HOME"]])
        beta, se = _ols_cluster(X, sub[f"R_{c}"].to_numpy(float), sub["PID"].to_numpy())
        shrink = tau ** 2 / (tau ** 2 + se ** 2)
        rows.append({
            "STAT": c, "OWN_B2B_%": beta[1] * 100, "OWN_SE": se[1] * 100, "OWN_SHRUNK_%": beta[1] * shrink[1] * 100,
            "OPP_B2B_%": beta[2] * 100, "OPP_SE": se[2] * 100, "OPP_SHRUNK_%": beta[2] * shrink[2] * 100,
            "HOME_%": beta[3] * 100, "N": len(sub), "N_OWN_B2B": int(sub["OWN_B2B"].sum()), "N_OPP_B2B": int(sub["OPP_B2B"].sum()),
        })
    return pd.DataFrame(rows)


def effects_from_table(table: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    t = table[table["STAT"].isin(B2B_OPP_BOOST.keys())].set_index("STAT")
    return {"own": (t["OWN_SHRUNK_%"] / 100.0).to_dict(), "opp": (t["OPP_SHRUNK_%"] / 100.0).to_dict()}


def _early_season_arrays(frame: pd.DataFrame, stat_col: str) -> Optional[Tuple[np.ndarray, ...]]:
    """Player-seasons with a prior season: first EARLY_GAMES games as padded (P, G) arrays plus prior-season summaries."""
    f = frame[["PID", "SEASON", f"X_{stat_col}", "X_MINUTES"]].rename(columns={f"X_{stat_col}": "X", "X_MINUTES": "M"}).copy()
    g = f.groupby(["PID", "SEASON"])
    f["G"], f["REV"] = g.cumcount(), g.cumcount(ascending=False)
    prior = pd.DataFrame({"X_SUM": g["X"].sum(), "M_SUM": g["M"].sum(), "X_MEAN": g["X"].mean()})
    prior["M20"] = f[f["REV"] < 20].groupby(["PID", "SEASON"])["M"].mean()
    prior = prior.reset_index()
    prior["SEASON"] = prior["SEASON"] + 1                      # last season's summary, keyed to the season it predicts
    cur = f[f["G"] < EARLY_GAMES].merge(prior, on=["PID", "SEASON"], how="inner")
    cur = cur[(cur["X_MEAN"] > 0) & (cur["M20"] > 0) & (cur["M_SUM"] > 0)]
    if cur["PID"].nunique() < 15:
        return None
    xs = cur.pivot(index=["PID", "SEASON"], columns="G", values="X").reindex(columns=range(EARLY_GAMES))
    ms = cur.pivot(index=["PID", "SEASON"], columns="G", values="M").reindex(columns=range(EARLY_GAMES))
    pri = cur.drop_duplicates(["PID", "SEASON"]).set_index(["PID", "SEASON"]).loc[xs.index]
    return (xs.to_numpy(float), ms.to_numpy(float), (pri["X_SUM"] / pri["M_SUM"]).to_numpy(float),
            pri["M20"].to_numpy(float), pri["X_MEAN"].to_numpy(float))


def _prior_loss_curves(X: np.ndarray, M: np.ndarray, rate0: np.ndarray, min0: np.ndarray, pm: np.ndarray,
                       grid: List[float]) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Out-of-sample squared error of the model's one-game-ahead prediction for each prior weight n0 on the grid.
    rate_hat = (n*EWMA_rate + n0*last_season_rate)/(n + n0) scored as (stat - rate_hat*minutes)/last_season_mean;
    min_hat is the same blend for minutes scored as (minutes - min_hat)/last_season_minutes.
    """
    n0 = np.asarray(grid, float)[:, None]
    rate_sse, min_sse, cnt = np.zeros(len(grid)), np.zeros(len(grid)), 0
    for g in range(X.shape[1]):
        ok = ~np.isnan(X[:, g]) & ~np.isnan(M[:, g])
        if not ok.any():
            continue
        k = min(g, WINDOW)
        if k > 0:
            w = DECAY ** np.arange(k)[::-1]
            xp, mp = X[ok, g - k:g], M[ok, g - k:g]
            cur_rate, cur_min = (xp * w).sum(1) / (mp * w).sum(1), (mp * w).sum(1) / w.sum()
        else:
            cur_rate = cur_min = np.zeros(int(ok.sum()))
        denom = float(k) + n0
        with np.errstate(invalid="ignore", divide="ignore"):
            rate_hat = np.where(denom > 0, (k * cur_rate[None, :] + n0 * rate0[ok][None, :]) / denom, rate0[ok][None, :])
            min_hat = np.where(denom > 0, (k * cur_min[None, :] + n0 * min0[ok][None, :]) / denom, min0[ok][None, :])
        rate_sse += (((X[ok, g][None, :] - rate_hat * M[ok, g][None, :]) / pm[ok][None, :]) ** 2).sum(1)
        min_sse += (((M[ok, g][None, :] - min_hat) / min0[ok][None, :]) ** 2).sum(1)
        cnt += int(ok.sum())
    return rate_sse / max(cnt, 1), min_sse / max(cnt, 1), cnt


def estimate_prior_weights(frame: pd.DataFrame, grid: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Data-driven last-season weight: for the first EARLY_GAMES games of each season, find the n0 that minimizes
    one-game-ahead prediction error (per stat for the per-minute rate, plus minutes). N0_RANGE = every grid value
    within 1% of the best loss (how flat the optimum is).
    """
    grid = list(PRIOR_GRID if grid is None else grid)
    rows: List[Dict[str, Any]] = []
    min_curve: Optional[np.ndarray] = None
    for stat in BT_STATS[:-1]:
        arrays = _early_season_arrays(frame, stat) if f"X_{stat}" in frame.columns else None
        if arrays is None:
            continue
        r_curve, m_curve, cnt = _prior_loss_curves(*arrays, grid)
        if min_curve is None:
            min_curve, min_cnt = m_curve, cnt
        rows.append({"STAT": stat, "TARGET": "per-minute rate", "curve": r_curve, "N": cnt, "default": N_PRIOR_RATE})
    if min_curve is not None:
        rows.append({"STAT": "MINUTES", "TARGET": "minutes", "curve": min_curve, "N": min_cnt, "default": N_PRIOR_MIN})
    out: List[Dict[str, Any]] = []
    for r in rows:
        c = r["curve"]
        best = int(np.argmin(c))
        near = [grid[i] for i in range(len(grid)) if c[i] <= c[best] * 1.01]
        base = float(np.interp(r["default"], grid, c))
        out.append({"STAT": r["STAT"], "TARGET": r["TARGET"], "N0_BEST": float(grid[best]), "N0_RANGE": f"{min(near):g}-{max(near):g}",
                    "LOSS_BEST": float(c[best]), "LOSS_AT_DEFAULT": base, "GAIN_VS_DEFAULT_%": (base - float(c[best])) / base * 100.0,
                    "PLAYER_GAMES": r["N"]})
    return pd.DataFrame(out)


def load_prior_file() -> Optional[Dict[str, float]]:
    """Saved last-season weights (best effort; ephemeral on Streamlit Cloud)."""
    try:
        if os.path.exists(PRIOR_PATH):
            with open(PRIOR_PATH) as fh:
                return json.load(fh)
    except Exception:
        pass
    return None


def save_prior_file(weights: Optional[Dict[str, float]]) -> None:
    """Persist (or clear) the last-season weights."""
    try:
        if weights is None:
            if os.path.exists(PRIOR_PATH):
                os.remove(PRIOR_PATH)
        else:
            with open(PRIOR_PATH, "w") as fh:
                json.dump(weights, fh)
    except Exception:
        pass


def load_effects_file() -> Optional[Dict[str, Dict[str, float]]]:
    try:
        if os.path.exists(EFFECTS_PATH):
            with open(EFFECTS_PATH) as fh:
                return json.load(fh)
    except Exception:
        pass
    return None


def save_effects_file(effects: Optional[Dict[str, Dict[str, float]]]) -> None:
    try:
        if effects is None:
            if os.path.exists(EFFECTS_PATH):
                os.remove(EFFECTS_PATH)
        else:
            with open(EFFECTS_PATH, "w") as fh:
                json.dump(effects, fh)
    except Exception:
        pass


@st.cache_data(ttl=86400, show_spinner=False)
def load_backtest_dataset(n_per_team: int, seasons: Tuple[int, ...]) -> pd.DataFrame:
    meta = load_team_abbr_meta()

    def safe(fn: Any, *args: Any) -> Any:
        try:
            return fn(*args)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        rosters = list(ex.map(lambda m: safe(_roster, m[0]) or [], meta))
        players = [p for r in rosters for p in r[:n_per_team]]
        log_jobs = [(pid, s) for pid, _ in players for s in seasons]
        logs = list(ex.map(lambda j: safe(_espn_fetch_logs, j[0], j[1]), log_jobs))
        sched_jobs = [(m[0], m[2], s) for m in meta for s in seasons]
        sdates = list(ex.map(lambda j: safe(_schedule_dates, j[0], j[2]) or [], sched_jobs))
    player_logs = {job: df for job, df in zip(log_jobs, logs) if df is not None and not df.empty}
    sched = pd.DataFrame([(j[1], d.isoformat()) for j, ds in zip(sched_jobs, sdates) for d in ds], columns=["ABBR", "DATE"])
    if sched.empty or not player_logs:
        raise ValueError("ESPN returned no usable logs/schedules")
    frame = build_backtest_frame(player_logs, sched)
    if frame.empty:
        raise ValueError("no rotation-player games after filtering")
    return frame


# --------------------------------------------------------------------------------------
# Injuries + usage redistribution, and stat-specific opponent factors
# --------------------------------------------------------------------------------------
INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
USAGE_STATS: List[str] = ["POINTS", "REBOUNDS", "ASSISTS", "3-POINTERS MADE", "STEALS", "BLOCKS"]
USAGE_BASED = {"POINTS"}
ABSORB: Dict[str, float] = {"POINTS": 0.85, "REBOUNDS": 0.90, "ASSISTS": 0.75, "3-POINTERS MADE": 0.75,
                            "STEALS": 0.50, "BLOCKS": 0.50}
COMBO_PARTS: Dict[str, List[str]] = {
    "PTS + REB + AST": ["POINTS", "REBOUNDS", "ASSISTS"], "PTS + AST": ["POINTS", "ASSISTS"],
    "PTS + REB": ["POINTS", "REBOUNDS"], "REB + AST": ["REBOUNDS", "ASSISTS"],
}
ROTATION_MIN, MIN_CAP, HEADROOM_FLOOR, MAX_MIN_SHIFT = 10.0, 40.0, 4.0, 10.0
USAGE_MULT_BOUNDS: Tuple[float, float] = (0.80, 1.40)
MAX_CELL_LIFT = 0.60
ABBR_ALIAS: Dict[str, str] = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX", "BKLN": "BKN"}
OPP_SHRINK_G, OPP_STAT_BOUNDS, OPP_BLEND = 12.0, (0.88, 1.12), 0.5
DEFAULT_ADJ: Dict[str, Any] = {"injuries": True, "opp_stat": True, "absorb_scale": 1.0,
                               "manual_out": [], "manual_q": [], "manual_in": []}


def _status_to_p_out(status: str, text: str) -> float:
    """Probability a listed player misses the game, from the ESPN status/comment text."""
    s, t = status.lower(), text.lower()
    if "out" in s or "suspend" in s:
        return 1.0
    if "doubtful" in s or "doubtful" in t:
        return 0.85
    if "probable" in s or "probable" in t:
        return 0.10
    if "questionable" in s or "questionable" in t or "day" in s:
        return 0.50
    return 0.0


@st.cache_data(ttl=600, show_spinner=False)
def load_injuries() -> pd.DataFrame:
    """ESPN NBA injury report as PID, NAME, STATUS, P_OUT (empty frame if unavailable)."""
    cols = ["PID", "NAME", "STATUS", "P_OUT"]
    try:
        resp = requests.get(INJURY_URL, headers=ESPN_HEADERS, timeout=10)
        resp.raise_for_status()
        rows: List[Dict[str, Any]] = []
        for team in resp.json().get("injuries", []):
            for inj in team.get("injuries", []):
                ath = inj.get("athlete", {}) or {}
                pid, name = str(ath.get("id", "")), ath.get("displayName") or ath.get("fullName")
                if not pid or not name:
                    continue
                status = str(inj.get("status", ""))
                text = f"{inj.get('shortComment', '')} {inj.get('longComment', '')}"
                rows.append({"PID": pid, "NAME": str(name), "STATUS": status, "P_OUT": _status_to_p_out(status, text)})
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame(columns=cols)


@st.cache_data(ttl=21600, show_spinner=False)
def load_all_rosters() -> Dict[str, List[Tuple[str, str]]]:
    meta = load_team_meta()

    def safe(tid: str) -> List[Tuple[str, str]]:
        try:
            return _roster(tid)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=8) as ex:
        rosters = list(ex.map(lambda m: safe(m[0]), meta))
    return {m[0]: r for m, r in zip(meta, rosters)}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_player_id(name: str) -> Optional[str]:
    return _espn_find_player_id(name)


def _blend(pairs: List[Tuple[float, float]]) -> float:
    w = sum(p[0] for p in pairs)
    return sum(p[0] * p[1] for p in pairs) / w if w > 0 else 0.0


def profile_row(pid: str, name: str, logs: pd.DataFrame, n0_rate: float = N_PRIOR_RATE, n0_min: float = N_PRIOR_MIN) -> Optional[Dict[str, Any]]:
    played = logs[pd.to_numeric(logs["MINUTES"], errors="coerce") > 0].copy()
    if played.empty:
        return None
    for c in ("FGA", "FTA", "TOV"):
        if c not in played:
            played[c] = 0.0
    played["POSS"] = played["FGA"] + 0.44 * played["FTA"] + played["TOV"]
    cols = USAGE_STATS + ["POSS"]
    sums = played.groupby("SEASON")[cols + ["MINUTES"]].sum()
    cnt = played.groupby("SEASON").size()
    n = int(cnt.get("current", 0))
    has_c, has_p = "current" in sums.index, "prior" in sums.index
    r0, m0 = (n0_rate if has_p else 0.0), (n0_min if has_p else 0.0)
    rates = {}
    for c in cols:
        pairs = []
        if has_c:
            pairs.append((float(n), float(sums.loc["current", c] / sums.loc["current", "MINUTES"])))
        if has_p:
            pairs.append((r0 if has_c else 1.0, float(sums.loc["prior", c] / sums.loc["prior", "MINUTES"])))
        rates["R_" + c] = _blend(pairs)
    mins: List[Tuple[float, float]] = []
    if has_c:
        mins.append((float(n), float(played.loc[played["SEASON"] == "current", "MINUTES"].mean())))
    if has_p:
        mins.append((m0 if has_c else 1.0, float(played.loc[played["SEASON"] == "prior", "MINUTES"].tail(20).mean())))
    return {"PID": pid, "NAME": name, "N_CUR": n, "MIN": _blend(mins), **rates,
            "DATES": tuple(played.loc[played["SEASON"] == "current", "DATE"])}


@st.cache_data(ttl=3600, show_spinner=False)
def load_team_profile(tid: str) -> pd.DataFrame:
    roster = _roster(tid)

    def fetch(item: Tuple[str, str]) -> Optional[pd.DataFrame]:
        try:
            return _finalize_logs(pd.concat([_espn_fetch_logs(item[0], None), _espn_fetch_logs(item[0], PRIOR_ESPN_SEASON)], ignore_index=True))
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        logs = list(ex.map(fetch, roster))
    rows = [r for (pid, nm), df in zip(roster, logs) if df is not None and not df.empty
            for r in [profile_row(pid, nm, df)] if r is not None]
    prof = pd.DataFrame(rows)
    if prof.empty:
        return prof
    team_poss = float((prof["R_POSS"] * prof["MIN"]).sum())
    prof["USG"] = 100.0 * prof["R_POSS"] * 240.0 / max(team_poss, 1e-9)
    return prof


def absence_effects(M: np.ndarray, R: np.ndarray, W: np.ndarray, a: np.ndarray, absorb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    K = len(M)
    off = 1.0 - np.eye(K)
    wmin = a * M * np.maximum(MIN_CAP - M, HEADROOM_FLOOR)
    omega = wmin[None, :] * off
    omega = omega / np.maximum(omega.sum(1, keepdims=True), 1e-9)
    dM = np.minimum(M[:, None] * omega, np.maximum(MIN_CAP - M, 0.0)[None, :])
    leftover = np.maximum(R * M[:, None] - dM @ R, 0.0)
    wshare = (a[:, None] * W)[None, :, :] * off[:, :, None]
    share = wshare / np.maximum(wshare.sum(1, keepdims=True), 1e-9)
    dX = absorb[None, None, :] * leftover[:, None, :] * share
    dR = dX / np.maximum((M[None, :] + dM)[:, :, None] * np.maximum(R, 1e-4)[None, :, :], 1e-9)
    return dM, np.minimum(dR, MAX_CELL_LIFT)


def absent_fractions(cur_dates: np.ndarray, player_dates: List[Tuple[str, ...]], weights: np.ndarray) -> np.ndarray:
    K = len(player_dates)
    if len(cur_dates) == 0 or K == 0:
        return np.zeros(K)
    idx = np.repeat(np.arange(K), [len(d) for d in player_dates])
    dts = np.concatenate([np.asarray(d, dtype=object) for d in player_dates]) if idx.size else np.array([], dtype=object)
    present = np.zeros((K, len(cur_dates)), dtype=bool)
    if idx.size:
        col = {d: c for c, d in enumerate(cur_dates)}
        hit = np.fromiter((x in col for x in dts), dtype=bool, count=len(dts))
        present[idx[hit], np.fromiter((col[x] for x in dts[hit]), dtype=int, count=int(hit.sum()))] = True
    return ((~present) * weights[None, :]).sum(1)


def injury_adjustment(player_name: str, stat_col: str, cur: pd.DataFrame, cfg: Dict[str, Any],
                      n0: Tuple[float, float], has_prior: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {"minutes_extra": 0.0, "usage_mult": 1.0, "table": [], "notes": [], "usg": None, "team": None}
    pid = cached_player_id(player_name)
    if not pid:
        out["notes"].append("Injury adjustment skipped: could not resolve the player on ESPN.")
        return out
    tid = next((t for t, r in load_all_rosters().items() if any(p == pid for p, _ in r)), None)
    if tid is None:
        out["notes"].append("Injury adjustment skipped: player not found on a current roster.")
        return out
    prof = load_team_profile(tid).reset_index(drop=True)
    if prof.empty or pid not in set(prof["PID"]):
        out["notes"].append("Injury adjustment skipped: teammate profile unavailable.")
        return out
    i = int(np.flatnonzero(prof["PID"].to_numpy() == pid)[0])
    K = len(prof)
    out["team"] = next((nm for t, nm in load_team_meta() if t == tid), None)
    
    inj = load_injuries()
    p_by_pid = dict(zip(inj["PID"], inj["P_OUT"])) if not inj.empty else {}
    s_by_pid = dict(zip(inj["PID"], inj["STATUS"])) if not inj.empty else {}
    p_by_name = {_clean_name(row["NAME"]): row["P_OUT"] for _, row in inj.iterrows()} if not inj.empty else {}
    s_by_name = {_clean_name(row["NAME"]): row["STATUS"] for _, row in inj.iterrows()} if not inj.empty else {}

    q_list = []
    status_list = []
    for pid_val, name_val in zip(prof["PID"], prof["NAME"]):
        cleaned_nm = _clean_name(name_val)
        if pid_val in p_by_pid:
            q_list.append(float(p_by_pid[pid_val]))
            status_list.append(str(s_by_pid.get(pid_val, "")))
        elif cleaned_nm in p_by_name:
            q_list.append(float(p_by_name[cleaned_nm]))
            status_list.append(str(s_by_name.get(cleaned_nm, "")))
        else:
            q_list.append(0.0)
            status_list.append("")
    q = np.array(q_list)
    status = status_list

    names = prof["NAME"].map(_clean_name)
    for key, val, tag in (("manual_q", 0.5, "QUESTIONABLE (manual)"), ("manual_out", 1.0, "OUT (manual)"), ("manual_in", 0.0, "ACTIVE (manual)")):
        hit = names.isin({_clean_name(x) for x in cfg.get(key, [])}).to_numpy()
        q = np.where(hit, val, q)
        status = [tag if h else s for h, s in zip(hit, status)]
    if q[i] > 0:
        out["notes"].append(f"ESPN lists {player_name} as {status[i] or 'injured'}; the projection assumes he plays.")
    q[i] = 0.0
    a = 1.0 - q
    M = prof["MIN"].to_numpy(float)
    R = prof[["R_" + s for s in USAGE_STATS]].to_numpy(float)
    poss = (prof["R_POSS"].to_numpy(float) * M)[:, None]
    W = np.where(np.array([s in USAGE_BASED for s in USAGE_STATS])[None, :], poss, R * M[:, None])
    absorb = np.minimum(np.array([ABSORB[s] for s in USAGE_STATS]) * float(cfg.get("absorb_scale", 1.0)), 1.0)
    dM, dR = absence_effects(M, R, W, a, absorb)

    n = len(cur)
    w = ewma_weights(n) if n else np.array([])
    f_core = absent_fractions(cur["DATE"].to_numpy(dtype=object), list(prof["DATES"]), w)
    r0, m0 = (n0[0] if has_prior else 0.0), (n0[1] if has_prior else 0.0)
    f_rate, f_min = f_core * n / max(n + r0, 1e-9), f_core * n / max(n + m0, 1e-9)
    rot = (M >= ROTATION_MIN).astype(float)
    rot[i] = 0.0
    net_rate, net_min = (q - f_rate) * rot, (q - f_min) * rot

    parts = COMBO_PARTS.get(stat_col, [stat_col])
    pidx = [USAGE_STATS.index(s) for s in parts if s in USAGE_STATS]
    if not pidx:
        return out
    base = R[i, pidx] / max(R[i, pidx].sum(), 1e-9)
    per_j = (dR[:, i, :][:, pidx] * base[None, :]).sum(1)
    lift = float((net_rate * per_j).sum())
    minutes_extra = float(np.clip((net_min * dM[:, i]).sum(), -MAX_MIN_SHIFT, MAX_MIN_SHIFT))
    out.update(minutes_extra=minutes_extra, usage_mult=float(np.clip(1.0 + lift, *USAGE_MULT_BOUNDS)),
               usg=float(prof.loc[i, "USG"]))
    show = np.flatnonzero((np.abs(net_rate) > 0.02) | (np.abs(net_min) > 0.02))
    for j in show[np.argsort(-np.abs(net_rate[show] * per_j[show]))]:
        label = ("OUT" if q[j] >= 0.99 else f"{q[j]:.0%} to miss" if q[j] > 0 else "Active (missed recent games)")
        out["table"].append({"Teammate": prof.loc[j, "NAME"], "Status": label,
                             "USG%": round(float(prof.loc[j, "USG"]), 1), "MPG": round(float(M[j]), 1),
                             "Recent games missed": f"{f_core[j]:.0%}",
                             "Weight": round(float(net_rate[j]), 2),
                             "Minutes effect": round(float(net_min[j] * dM[j, i]), 2),
                             "Stat effect %": round(float(net_rate[j] * per_j[j]) * 100.0, 1)})
    return out


def compute_opp_allowed(games: pd.DataFrame, stat_cols: List[str], min_games: int = 3) -> pd.DataFrame:
    g = games[games.groupby("PID")["DATE"].transform("size") >= min_games]
    if g.empty:
        return pd.DataFrame()
    base = g.groupby("PID")[stat_cols].transform("mean")
    num, den = g.groupby("OPP")[stat_cols].sum(), base.groupby(g["OPP"]).sum()
    out = (num / den.replace(0, np.nan)).fillna(1.0)
    out["G"] = g.groupby("OPP")["DATE"].nunique()
    return out


def _opp_abbr(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split().str[-1].str.upper().replace(ABBR_ALIAS)


@st.cache_data(ttl=21600, show_spinner="Loading league game logs for matchup splits (one-time, about 30-60s)...")
def load_opp_allowed() -> pd.DataFrame:
    pids = [pid for r in load_all_rosters().values() for pid, _ in r]

    def fetch(pid: str) -> Optional[pd.DataFrame]:
        try:
            df = _finalize_logs(_espn_fetch_logs(pid, None))
            return df.assign(PID=pid) if not df.empty else None
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = [f for f in ex.map(fetch, pids) if f is not None]
    if not frames:
        raise ValueError("no league game logs available")
    games = pd.concat(frames, ignore_index=True)
    games = games[games["MINUTES"] >= ROTATION_MIN].assign(OPP=lambda d: _opp_abbr(d["OPPONENT"]))
    return compute_opp_allowed(games, list(STAT_COL.values()))


def opp_stat_detail(opponent: str, stat_col: str, allowed: pd.DataFrame, played_cur: pd.DataFrame) -> Optional[Dict[str, Any]]:
    abbr = next((_opp_abbr(pd.Series([ab])).iloc[0] for _, nm, ab in load_team_abbr_meta() if nm == opponent), None)
    if abbr is None or abbr not in allowed.index or stat_col not in allowed:
        return None
    f, g = float(allowed.loc[abbr, stat_col]), float(allowed.loc[abbr, "G"])
    mult = float(np.clip(1.0 + g / (g + OPP_SHRINK_G) * (f - 1.0), *OPP_STAT_BOUNDS))
    rank = allowed[stat_col].rank(ascending=False, method="min")
    n_teams = len(allowed)
    r = _opp_abbr(played_cur["OPPONENT"]).map(rank) if len(played_cur) else pd.Series(dtype=float)
    soft, tough = played_cur.loc[r <= 10, stat_col], played_cur.loc[r > n_teams - 10, stat_col]
    return {"abbr": abbr, "factor": f, "games": int(g), "mult": mult, "rank": int(rank[abbr]), "n_teams": n_teams,
            "soft_avg": float(soft.mean()) if len(soft) else None, "soft_n": int(len(soft)),
            "tough_avg": float(tough.mean()) if len(tough) else None, "tough_n": int(len(tough))}


def _split_names(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


# --------------------------------------------------------------------------------------
# Model math & Automated Contextual Checks
# --------------------------------------------------------------------------------------
@dataclass
class Projection:
    mu: float
    sigma: float
    minutes: float
    n_current: int
    n_prior: int
    opp_mult: float
    share_rate: float = 0.0
    share_min: float = 0.0
    usage_mult: float = 1.0
    extra_min: float = 0.0


def ewma_weights(n: int, decay: float = DECAY) -> np.ndarray:
    w = decay ** np.arange(n)[::-1]
    return w / w.sum()


def _cv(x: np.ndarray) -> Optional[float]:
    if len(x) < 3 or float(np.mean(x)) <= 0:
        return None
    return float(np.std(x, ddof=1) / np.mean(x))


def build_projection(cur_x: np.ndarray, cur_m: np.ndarray, pri_x: np.ndarray, pri_m: np.ndarray, opp_mult: float = 1.0,
                     n0_rate: float = N_PRIOR_RATE, n0_min: float = N_PRIOR_MIN,
                     extra_min: float = 0.0, usage_mult: float = 1.0) -> Projection:
    n = len(cur_x)
    has_prior = len(pri_x) > 0
    if n == 0 and not has_prior:
        raise ValueError("No usable games (current or prior season).")
    r0 = float(n0_rate) if has_prior else 0.0
    m0 = float(n0_min) if has_prior else 0.0
    if n == 0:
        r0, m0 = max(r0, 1.0), max(m0, 1.0)
    pairs_rate: List[Tuple[float, float]] = []
    pairs_min: List[Tuple[float, float]] = []
    if n > 0:
        w = ewma_weights(n)
        pairs_rate.append((float(n), float(np.sum(w * cur_x) / np.sum(w * cur_m))))
        pairs_min.append((float(n), float(np.sum(w * cur_m))))
    if r0 > 0:
        pairs_rate.append((r0, float(np.sum(pri_x) / np.sum(pri_m))))
    if m0 > 0:
        pairs_min.append((m0, float(np.mean(pri_m[-20:]))))
    blend = lambda pairs: sum(w * v for w, v in pairs) / sum(w for w, _ in pairs)
    rate, minutes = blend(pairs_rate), blend(pairs_min)
    minutes = float(np.clip(minutes + extra_min, 1.0, MIN_CAP + 2.0)) if extra_min else minutes
    mu = max(rate * minutes * opp_mult * usage_mult, 0.05)
    cvs = [(w, c) for w, c in ((float(n), _cv(cur_x)), (r0, _cv(pri_x[-40:]))) if c is not None and w > 0]
    cv = sum(w * c for w, c in cvs) / sum(w for w, _ in cvs) if cvs else DEFAULT_CV
    cv = max(cv, 0.15)
    sigma = max(0.5, cv * mu) * float(np.sqrt(1.0 + 1.0 / (n + r0)))
    return Projection(mu=mu, sigma=sigma, minutes=minutes, n_current=n, n_prior=len(pri_x), opp_mult=opp_mult,
                      share_rate=r0 / (n + r0) if n + r0 > 0 else 0.0, share_min=m0 / (n + m0) if n + m0 > 0 else 0.0,
                      usage_mult=usage_mult, extra_min=extra_min)


def simulate(mu: float, sigma: float, count_stat: bool, n_sims: int = N_SIMS, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    var = sigma ** 2
    if count_stat:
        if var > mu * 1.0001:
            return rng.negative_binomial(mu ** 2 / (var - mu), mu / var, n_sims).astype(float)
        return rng.poisson(mu, n_sims).astype(float)
    return np.maximum(np.rint(rng.normal(mu, sigma, n_sims)), 0.0)


def implied_prob(odds: float) -> float:
    o = float(odds)
    return -o / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def payout(odds: float) -> float:
    o = float(odds)
    return 100.0 / abs(o) if o < 0 else o / 100.0


def devig(over_odds: float, under_odds: float) -> Tuple[float, float]:
    po, pu = implied_prob(over_odds), implied_prob(under_odds)
    return po / (po + pu), pu / (po + pu)


def prob_to_american(p: float) -> int:
    p = float(np.clip(p, 0.01, 0.99))
    return int(round(-100.0 * p / (1.0 - p))) if p >= 0.5 else int(round(100.0 * (1.0 - p) / p))


def kelly_fraction(p_win: float, p_lose: float, odds: float) -> float:
    b = payout(odds)
    full = (b * p_win - p_lose) / b
    return float(np.clip(full * KELLY_FRACTION, 0.0, KELLY_CAP))


def run_automated_context_filters(proj_minutes: float, raw_avg: float, b2b_status_str: str) -> Dict[str, Any]:
    secure_volume = bool(proj_minutes >= 28.0 or raw_avg >= 15.0)
    clean_schedule = bool(b2b_status_str not in ("second", "both"))
    sharp_movement = bool(proj_minutes >= 24.0)
    return {
        "secure_volume": secure_volume,
        "clean_schedule": clean_schedule,
        "sharp_movement": sharp_movement,
        "all_passed": secure_volume and clean_schedule and sharp_movement
    }


def evaluate_nba_prop(player_name: str, stat_category: str, line: float, over_odds: float, under_odds: float,
                      opponent: Optional[str] = None, game_date: Optional[date] = None,
                      b2b_effects: Optional[Dict[str, Dict[str, float]]] = None,
                      prior_weights: Optional[Tuple[float, float]] = None,
                      adj_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        cfg = {**DEFAULT_ADJ, **(adj_cfg or {})}
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
        opp_detail: Optional[Dict[str, Any]] = None
        if opponent and cfg["opp_stat"]:
            try:
                opp_detail = opp_stat_detail(opponent, stat_col, load_opp_allowed(), cur)
                if opp_detail:
                    opp_detail["mult_old"] = opp_mult
                    opp_mult = float(np.clip(opp_mult ** (1.0 - OPP_BLEND) * opp_detail["mult"] ** OPP_BLEND, 0.88, 1.12))
                    opp_detail["mult_final"] = opp_mult
            except Exception as e:
                opp_note = f"{opp_note or ''} Stat-specific matchup data unavailable ({e}).".strip()
        b2b_state, b2b_mult, b2b_note = "unknown", 1.0, None
        game_day = game_date or _today_et()
        if opponent:
            try:
                b2b_state = b2b_status(load_team_game_dates(opponent), game_day)
                b2b_mult = b2b_multiplier(b2b_state, stat_col, (b2b_effects or {}).get("opp"))
            except Exception as e:
                b2b_note = f"Back-to-back check unavailable ({e}); no adjustment applied."
        else:
            b2b_note = "No opponent selected: opponent back-to-back check skipped."
        day_iso = game_day.isoformat()
        before = played[played["DATE"] < day_iso]
        prev_date = _iso_to_date(before["DATE"].iloc[-1]) if len(before) else None
        own_state = own_b2b_status(prev_date, game_day)
        own_mult = own_b2b_multiplier(own_state, stat_col, (b2b_effects or {}).get("own"))

        inj_out: Dict[str, Any] = {"minutes_extra": 0.0, "usage_mult": 1.0, "table": [], "notes": [], "usg": None, "team": None}
        if cfg["injuries"]:
            try:
                inj_out = injury_adjustment(player_name, stat_col, cur, cfg, tuple(prior_weights or (N_PRIOR_RATE, N_PRIOR_MIN)), len(pri) > 0)
            except Exception as e:
                inj_out["notes"].append(f"Injury adjustment unavailable ({e}); none applied.")
        proj = build_projection(cur[stat_col].to_numpy(float), cur["MINUTES"].to_numpy(float),
                                pri[stat_col].to_numpy(float), pri["MINUTES"].to_numpy(float), opp_mult * b2b_mult * own_mult,
                                *(prior_weights or (N_PRIOR_RATE, N_PRIOR_MIN)),
                                extra_min=inj_out["minutes_extra"], usage_mult=inj_out["usage_mult"])
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
                                            ("Under", p_under, p_over, under_odds, dv_under, ev_under
