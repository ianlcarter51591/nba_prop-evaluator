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


MARKET_LABEL: Dict[str, str] = {v: k for k, v in ODDS_API_MARKETS.items()}


def fetch_nba_all_boards(event_id: str, api_key: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """One request for every market (the API bills one credit per market). Long frame: Market, Player, Line, Price, Outcome."""
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
    params = {"apiKey": api_key, "regions": "us", "markets": ",".join(ODDS_API_MARKETS.values()),
              "bookmakers": "draftkings", "oddsFormat": "american"}
    try:
        resp = requests.get(url, params=params, timeout=12)
        if resp.status_code != 200:
            return pd.DataFrame(), f"Odds API Error ({resp.status_code}): {resp.text}"
        bookmakers = resp.json().get("bookmakers", [])
        if not bookmakers:
            return pd.DataFrame(), "No DraftKings odds found for this event."
        rows: List[Dict[str, Any]] = []
        for mk in bookmakers[0].get("markets", []):
            label = MARKET_LABEL.get(mk.get("key", ""))
            if label is None:
                continue
            for o in mk.get("outcomes", []):
                try:
                    price = int(round(float(o.get("price", -110))))
                except Exception:
                    price = -110
                rows.append({"Market": label, "Player": o.get("description"), "Line": o.get("point", 0.5),
                             "Price": price, "Outcome": o.get("name")})
        if not rows:
            return pd.DataFrame(), "No market data available."
        return pd.DataFrame(rows), None
    except Exception as e:
        return pd.DataFrame(), f"Error fetching prop board: {e}"


def player_lines(all_df: pd.DataFrame, player: str) -> pd.DataFrame:
    """Every market DraftKings offers for one player: Market, Line, Over, Under (in STATS order)."""
    cols = ["Market", "Line", "Over", "Under"]
    sub = all_df[all_df["Player"] == player]
    rows: List[Dict[str, Any]] = []
    for label in STATS:
        grp = sub[sub["Market"] == label]
        if grp.empty:
            continue
        r = summarize_board(grp).iloc[0]
        rows.append({"Market": label, "Line": float(r["Line"]), "Over": int(r["Over"]), "Under": int(r["Under"])})
    return pd.DataFrame(rows, columns=cols)


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


def _split_made(v: Any) -> float:
    s = str(v).strip()
    return parse_val(s.split("-")[0]) if "-" in s else parse_val(s)


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
    share_rate: float = 0.0     # fraction of the per-minute-rate estimate that comes from last season
    share_min: float = 0.0      # fraction of the minutes estimate that comes from last season


def ewma_weights(n: int, decay: float = DECAY) -> np.ndarray:
    w = decay ** np.arange(n)[::-1]
    return w / w.sum()


def _cv(x: np.ndarray) -> Optional[float]:
    if len(x) < 3 or float(np.mean(x)) <= 0:
        return None
    return float(np.std(x, ddof=1) / np.mean(x))


def build_projection(cur_x: np.ndarray, cur_m: np.ndarray, pri_x: np.ndarray, pri_m: np.ndarray, opp_mult: float = 1.0,
                     n0_rate: float = N_PRIOR_RATE, n0_min: float = N_PRIOR_MIN) -> Projection:
    """
    rate    = (n*EWMA_rate + n0_rate*last_season_rate) / (n + n0_rate)
    minutes = (n*EWMA_min  + n0_min *last_season_min ) / (n + n0_min)
    mu      = rate * minutes * opp_mult;  spread = pooled CV * mu, widened for estimation uncertainty.
    """
    n = len(cur_x)
    has_prior = len(pri_x) > 0
    if n == 0 and not has_prior:
        raise ValueError("No usable games (current or prior season).")
    r0 = float(n0_rate) if has_prior else 0.0
    m0 = float(n0_min) if has_prior else 0.0
    if n == 0:                                   # nothing but last season to go on
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
    mu = max(rate * minutes * opp_mult, 0.05)
    cvs = [(w, c) for w, c in ((float(n), _cv(cur_x)), (r0, _cv(pri_x[-40:]))) if c is not None and w > 0]
    cv = sum(w * c for w, c in cvs) / sum(w for w, _ in cvs) if cvs else DEFAULT_CV
    cv = max(cv, 0.15)
    sigma = max(0.5, cv * mu) * float(np.sqrt(1.0 + 1.0 / (n + r0)))
    return Projection(mu=mu, sigma=sigma, minutes=minutes, n_current=n, n_prior=len(pri_x), opp_mult=opp_mult,
                      share_rate=r0 / (n + r0) if n + r0 > 0 else 0.0, share_min=m0 / (n + m0) if n + m0 > 0 else 0.0)


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
    """
    Automated pre-flight checks:
    1. Secure Volume & Usage: Checked automatically if projected minutes >= 28.0 or raw average output is robust.
    2. Clean Schedule Spot: Checked automatically if the team is NOT on the second night of a back-to-back.
    3. Rotation Minutes: checked automatically as projected minutes >= 24 (this does not measure sportsbook line movement).
    """
    secure_volume = bool(proj_minutes >= 28.0 or raw_avg >= 15.0)
    clean_schedule = bool(b2b_status_str not in ("second", "both"))
    sharp_movement = bool(proj_minutes >= 24.0)  # NOTE: a rotation-minutes check; it does not look at line movement
    
    return {
        "secure_volume": secure_volume,
        "clean_schedule": clean_schedule,
        "sharp_movement": sharp_movement,
        "all_passed": secure_volume and clean_schedule and sharp_movement
    }


def evaluate_nba_prop(player_name: str, stat_category: str, line: float, over_odds: float, under_odds: float,
                      opponent: Optional[str] = None, game_date: Optional[date] = None,
                      b2b_effects: Optional[Dict[str, Dict[str, float]]] = None,
                      prior_weights: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
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

        proj = build_projection(cur[stat_col].to_numpy(float), cur["MINUTES"].to_numpy(float),
                                pri[stat_col].to_numpy(float), pri["MINUTES"].to_numpy(float), opp_mult * b2b_mult * own_mult,
                                *(prior_weights or (N_PRIOR_RATE, N_PRIOR_MIN)))
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

        cur_vals = played[played["SEASON"] == "current"][stat_col].to_numpy(float)
        this_season = {"n": len(cur_vals), "avg": float(np.mean(cur_vals)) if len(cur_vals) else None,
                       "median": float(np.median(cur_vals)) if len(cur_vals) else None,
                       "over": int(np.sum(cur_vals > line)), "under": int(np.sum(cur_vals < line)),
                       "push": int(np.sum(cur_vals == line))}
        recent = played.tail(WINDOW)
        vals = recent[stat_col].to_numpy(float)
        raw_avg = float(np.mean(vals)) if len(vals) else 0.0
        
        # Automatically run contextual checks
        context_checks = run_automated_context_filters(proj.minutes, raw_avg, own_state)

        warnings: List[str] = []
        if proj.n_current < 8:
            warnings.append(f"Small sample: {proj.n_current} games this season; last season ({proj.n_prior} games) carries "
                            f"{proj.share_rate:.0%} of the per-minute rate and {proj.share_min:.0%} of minutes (see Model settings).")
        if proj.n_prior == 0:
            warnings.append("No prior-season games found; no shrinkage applied.")
        if opp_note:
            warnings.append(opp_note)
        if b2b_note:
            warnings.append(b2b_note)
        days_rest: Optional[int] = (game_day - prev_date).days if prev_date else None
        return {
            "player_name": player_name, "mean": float(np.mean(sims)), "median": float(np.median(sims)),
            "verdict": "PLAY" if ev > 0 else "PASS", "best_side": side_txt, "ev_pct": ev * 100.0,
            "edge_pp": (p_cond - dv_side) * 100.0, "p_model": p_cond * 100.0, "p_market": dv_side * 100.0,
            "fair_odds": prob_to_american(p_cond), "kelly_pct": kelly_fraction(p_w, p_l, odds_side) * 100.0,
            "proj_minutes": proj.minutes, "opp_mult": proj.opp_mult, "raw_avg": raw_avg,
            "hit10": int(np.sum(vals > line)), "n10": len(vals),
            "hit5": int(np.sum(vals[-5:] > line)), "n5": len(vals[-5:]),
            "days_rest": days_rest, "warnings": warnings, "mu": proj.mu, "sides": sides,
            "context_checks": context_checks, "this_season": this_season,
            "prior_share": {"rate": proj.share_rate, "minutes": proj.share_min},
            "own_b2b": {"status": own_state, "mult": own_mult, "calibrated": bool(b2b_effects)},
            "b2b_calibrated": bool(b2b_effects),
            "b2b": {"status": b2b_state, "mult": b2b_mult, "opponent": opponent or "", "game_date": game_day.isoformat()},
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
    "ID", "TS", "GAME_DATE", "PLAYER", "STAT", "SIDE", "LINE", "ODDS", "STAKE", "OPPONENT", "OPP_B2B", "OWN_B2B",
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
    try:
        return datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return datetime.now(timezone.utc).date()


def _implied_vec(odds: np.ndarray) -> np.ndarray:
    o = np.asarray(odds, dtype=float)
    return np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))


def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
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
    try:
        if os.path.exists(LOG_PATH):
            return normalize_log(pd.read_csv(LOG_PATH, dtype=str))
    except Exception:
        pass
    return normalize_log(pd.DataFrame(columns=LOG_COLS))


def save_log_file(df: pd.DataFrame) -> None:
    try:
        df.to_csv(LOG_PATH, index=False)
    except Exception:
        pass


def make_bet_row(res: Dict[str, Any], ctx: Dict[str, Any], side: str, stake: float, game_date: date) -> Dict[str, Any]:
    s = res["sides"][side]
    return {
        "ID": uuid.uuid4().hex[:8], "TS": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "GAME_DATE": game_date.isoformat(), "PLAYER": ctx["player"], "STAT": ctx["stat"], "SIDE": side,
        "LINE": float(ctx["line"]), "ODDS": s["odds"], "STAKE": float(stake), "OPPONENT": ctx.get("opponent") or "",
        "OPP_B2B": res.get("b2b", {}).get("status", ""),
        "OWN_B2B": res.get("own_b2b", {}).get("status", ""), "P_MODEL": s["p_model"], "P_MARKET": s["p_market"], "EDGE_PP": s["edge_pp"], "EV_PCT": s["ev_pct"],
        "RESULT": "Pending",
    }


def add_bet(log: pd.DataFrame, row: Dict[str, Any]) -> Tuple[pd.DataFrame, bool]:
    dup = ((log["PLAYER"] == row["PLAYER"]) & (log["STAT"] == row["STAT"]) & (log["SIDE"] == row["SIDE"])
           & (log["LINE"] == row["LINE"]) & (log["GAME_DATE"] == row["GAME_DATE"])).any()
    if dup:
        return log, False
    return normalize_log(pd.concat([log, pd.DataFrame([row])], ignore_index=True)), True


def apply_board_as_closing(log: pd.DataFrame, board_summary: pd.DataFrame, stat_label: str) -> Tuple[pd.DataFrame, int]:
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
                continue
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
    color = {"neutral": "#0369a1", "good": "#15803d", "bad": "#b91c1c"}[tone]
    col.markdown(
        f'<div class="metric-box"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color} !important;">{value}</div></div>',
        unsafe_allow_html=True,
    )


def _fmt(x: float, spec: str = ".1f", suffix: str = "") -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:{spec}}{suffix}"


def _tone(x: float) -> str:
    return "neutral" if x is None or not np.isfinite(x) else ("good" if x > 0 else "bad")


def _opp_options(game_teams: List[str]) -> List[str]:
    try:
        names = [n for _, n in load_team_meta()]
    except Exception:
        names = []
    gt = [t for t in game_teams if t]
    return ["None"] + gt + sorted(n for n in names if n not in gt)


def render_log_form(res: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    ss = st.session_state
    if ss.get("_log_uid") != res["uid"]:
        ss["log_side"] = "Over" if res["best_side"].startswith("Over") else "Under"
        ss["log_gdate"] = ctx.get("game_date") or _today_et()
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


def render_b2b_alert(res: Dict[str, Any]) -> None:
    own = res.get("own_b2b") or {}
    if own.get("status") == "second":
        src = "backtested" if own.get("calibrated") else "default assumption: calibrate in the Backtest tab"
        st.warning(f"🛌 **{res['player_name']} is on the SECOND night of a back-to-back** (played yesterday). "
                   f"Projection adjusted x{own['mult']:.3f} ({src}).")
    b = res.get("b2b") or {}
    team, status, mult = b.get("opponent", ""), b.get("status", "unknown"), b.get("mult", 1.0)
    if status == "second":
        st.warning(f"🚨 **{team} is on the SECOND night of a back-to-back** (played yesterday). Projection adjusted x{mult:.3f}.")
    elif status == "both":
        st.warning(f"🚨 **{team} is on the second night of a back-to-back AND plays again tomorrow** (3-in-3 stretch). Projection adjusted x{mult:.3f}.")
    elif status == "first":
        st.info(f"⏭️ **{team} plays again tomorrow** (first game of a back-to-back). Informational only, no adjustment applied.")
    elif status == "none":
        st.caption(f"✅ {team} is rested: no game yesterday or tomorrow.")


def render_automated_checks(res: Dict[str, Any]) -> None:
    """Render automated pre-flight contextual check results as verified checkboxes."""
    checks = res.get("context_checks", {})
    v1 = checks.get("secure_volume", True)
    v2 = checks.get("clean_schedule", True)
    v3 = checks.get("sharp_movement", True)
    
    with st.expander("🔍 Step 3: Automated Contextual Reality Check (Pre-Flight Filter)", expanded=True):
        st.markdown("Automated filters running against player volume, schedule fatigue, and rotation minutes:")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.checkbox("1. Secure Volume & Usage", value=v1, disabled=True, help="Automatically verified via projected minutes and raw average volume.")
        with c2:
            st.checkbox("2. Clean Schedule Spot", value=v2, disabled=True, help="Automatically verified via team back-to-back status.")
        with c3:
            st.checkbox("3. Rotation Minutes (≥ 24 proj)", value=v3, disabled=True, help="Verified via projected minutes of at least 24. This does not measure sportsbook line movement.")
        st.caption("💡 *Note: High-volume players are less exposed to single-game variance. Back-to-back effects are estimated in the B2B Backtest tab.*")


def render_results(res: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    stat_category, line_val = ctx["stat"], ctx["line"]
    st.markdown(f"## {res['player_name']} | {stat_category}")
    render_b2b_alert(res)
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
    st.write("")
    ts = res["this_season"]
    e = st.columns(4)
    pct = lambda k: f"{ts[k]}/{ts['n']} · {ts[k] / ts['n'] * 100:.0f}%" if ts["n"] else "—"
    _metric(e[0], "THIS SEASON AVG / MEDIAN", f"{ts['avg']:.1f} / {ts['median']:.1f}" if ts["n"] else "—")
    _metric(e[1], "THIS SEASON OVER HIT", pct("over"))
    _metric(e[2], "THIS SEASON UNDER HIT", pct("under"))
    _metric(e[3], "LAST-SEASON WEIGHT", f"{res['prior_share']['rate']:.0%} rate · {res['prior_share']['minutes']:.0%} min")
    if ts["n"] and ts["push"]:
        st.caption(f"{ts['push']} of {ts['n']} games this season landed exactly on the line (pushes).")
    if res["days_rest"] is not None:
        st.caption(f"Days since the player's previous game: {res['days_rest']} (1 = back-to-back)")
    for w in res["warnings"]:
        st.caption(f"⚠️ {w}")
    st.markdown("---")
    render_automated_checks(res)
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
    board = ss.get("board_all")
    if b.button("📸 Use current board as closing lines", key="close_btn"):
        if board is None or board.empty:
            st.warning("Fetch a prop board first (Evaluate tab, Step 3) — ideally right before tip-off.")
        else:
            n = 0
            for label, grp in board.groupby("Market"):
                ss["bet_log"], k = apply_board_as_closing(ss["bet_log"], summarize_board(grp), str(label))
                n += k
            save_log_file(ss["bet_log"])
            ss["editor_ver"] += 1
            st.toast(f"Closing lines applied to {n} pending bet(s)")
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


def render_backtest() -> None:
    ss = st.session_state
    st.markdown(
        "Estimates how much a player's stat changes when **he** is on the second night of a back-to-back and when "
        "**the opponent** is, using within-player regressions (each game as a % of that player's own season average, "
        "controlling for home/away, standard errors clustered by player). Results are shrunk toward 0 before use."
    )
    c1, c2, c3 = st.columns(3)
    n_per_team = c1.slider("Players per team (roster order)", 3, 12, 8)
    seasons = c2.multiselect("Seasons (ESPN ending year)", [2026, 2025, 2024], default=[2026, 2025, 2024])
    c3.write("")
    if c3.button("🔬 Run backtest", key="run_bt") and seasons:
        with st.spinner("Downloading rosters, logs and schedules from ESPN (first run takes a minute or two)…"):
            try:
                frame = load_backtest_dataset(int(n_per_team), tuple(sorted(seasons)))
                ss["bt_table"] = estimate_b2b_effects(frame)
                ss["bt_meta"] = {"games": len(frame), "players": int(frame["PID"].nunique()), "own": int(frame["OWN_B2B"].sum()),
                                 "opp": int(frame["OPP_B2B"].sum())}
            except Exception as e:
                ss["bt_table"] = None
                st.error(f"Backtest failed: {e}")
    table = ss.get("bt_table")
    if table is not None and not table.empty:
        meta = ss["bt_meta"]
        st.caption(f"{meta['games']:,} player-games · {meta['players']} players · {meta['own']:,} own-B2B games · {meta['opp']:,} opponent-B2B games")
        show = table.round(2)
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("Effects are % of the player's own average (negative OWN = fatigue; positive OPP = tired opponent helps). "
                   "|effect| < 2 × SE is not distinguishable from zero. SHRUNK columns are what the model uses.")
        a, b = st.columns(2)
        if a.button("✅ Apply to model", key="apply_bt"):
            ss["b2b_effects"] = effects_from_table(table)
            save_effects_file(ss["b2b_effects"])
            st.toast("Backtested back-to-back effects applied")
        if b.button("↩️ Reset to defaults", key="reset_bt"):
            ss["b2b_effects"] = None
            save_effects_file(None)
            st.toast("Back to default assumptions")
    elif table is not None:
        st.warning("Not enough back-to-back games in the sample. Try more players per team or more seasons.")
    st.markdown("---")
    st.markdown("### ⚖️ Last-season weight calibration")
    st.markdown(
        "For the first 15 games of each season, finds the last-season weight (in pseudo-games) that predicts the next game best, "
        "separately for each stat's per-minute rate and for minutes. Needs at least two completed seasons in the selection above "
        "(e.g. 2025 and 2024)."
    )
    if st.button("⚖️ Calibrate last-season weight", key="run_pw") and seasons:
        with st.spinner("Loading logs (cached after the first run) and scoring predictions…"):
            try:
                ss["pw_table"] = estimate_prior_weights(load_backtest_dataset(int(n_per_team), tuple(sorted(seasons))))
            except Exception as e:
                ss["pw_table"] = None
                st.error(f"Calibration failed: {e}")
    pw = ss.get("pw_table")
    if pw is not None and not pw.empty:
        st.dataframe(pw.round(4), use_container_width=True, hide_index=True)
        st.caption("N0_BEST = best last-season weight on the grid; N0_RANGE = every weight within 1% of the best loss (a wide range means the "
                   "choice barely matters). LOSS is squared error relative to last season's average; GAIN_VS_DEFAULT compares to your current setting.")
        rate_rows, min_rows = pw[pw["TARGET"] == "per-minute rate"], pw[pw["TARGET"] == "minutes"]
        if not rate_rows.empty and not min_rows.empty:
            rate_pick, min_pick = int(round(float(rate_rows["N0_BEST"].median()))), int(round(float(min_rows["N0_BEST"].iloc[0])))
            if st.button(f"✅ Apply: rate {rate_pick} / minutes {min_pick} pseudo-games", key="apply_pw"):
                ss["pw_pending"] = (rate_pick, min_pick)
                save_prior_file({"rate": rate_pick, "minutes": min_pick})
                st.rerun()
    elif pw is not None:
        st.warning("Not enough player-seasons with a prior season. Select at least two completed seasons (e.g. 2025 and 2024) and more players per team.")
    active = ss.get("b2b_effects")
    st.markdown("**Active parameters**" + (" (backtested)" if active else " (default assumptions)"))
    own, opp = (active or {}).get("own", B2B_OWN_EFFECT), (active or {}).get("opp", B2B_OPP_BOOST)
    st.dataframe(pd.DataFrame({"STAT": list(B2B_OPP_BOOST.keys()),
                               "OWN_B2B_%": [own.get(k, 0.0) * 100 for k in B2B_OPP_BOOST],
                               "OPP_B2B_%": [opp.get(k, 0.0) * 100 for k in B2B_OPP_BOOST]}).round(2),
                 use_container_width=True, hide_index=True)
    if active:
        st.caption("To make these permanent, paste into the constants at the top of the file:")
        st.code(f"B2B_OWN_EFFECT = {json.dumps({k: round(v, 4) for k, v in active['own'].items()})}\n"
                f"B2B_OPP_BOOST = {json.dumps({k: round(v, 4) for k, v in active['opp'].items()})}", language="python")


def main() -> None:
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
                         ("under", -110), ("opp", "None"), ("board_teams", []), ("editor_ver", 0), ("game_date", _today_et())):
        ss.setdefault(key, default)
    if "b2b_effects" not in ss:
        ss["b2b_effects"] = load_effects_file()
    if "n0_rate" not in ss:
        saved = load_prior_file() or {}
        ss["n0_rate"], ss["n0_min"] = int(saved.get("rate", N_PRIOR_RATE)), int(saved.get("minutes", N_PRIOR_MIN))
    if ss.get("pw_pending"):                       # applied from the backtest tab; must be set before the sliders exist
        ss["n0_rate"], ss["n0_min"] = ss.pop("pw_pending")
    if "bet_log" not in ss:
        ss["bet_log"] = read_log_file()

    tab_eval, tab_log, tab_bt = st.tabs(["⚡ Evaluate", "📒 Bet Log", "🔬 B2B Backtest"])
    with tab_eval:
        with st.expander("📡 Fetch Live DraftKings Lines (The Odds API)", expanded=True):
            if not api_key:
                st.warning("Please configure `ODDS_API_KEY` in Streamlit Secrets.")
            else:
                if st.button("Step 1: Load Upcoming NBA Games", key="load_nba_games"):
                    events_df, err = list_nba_events(api_key)
                    if err:
                        st.error(err)
                    else:
                        ss["events_df"] = events_df
                events_df = ss.get("events_df")
                if events_df is not None and not events_df.empty:
                    game_idx = st.selectbox("Step 2: Select Game", range(len(events_df)), format_func=lambda i: str(events_df.loc[i, "label"]), key="live_game_idx")
                    if st.button(f"Step 3: Fetch ALL Player Props for This Game ({len(ODDS_API_MARKETS)} API credits)", key="fetch_board_odds"):
                        board_all, err = fetch_nba_all_boards(str(events_df.loc[game_idx, "id"]), api_key)
                        if err:
                            st.error(err)
                        else:
                            ss["board_all"] = board_all
                            ss["board_teams"] = [str(events_df.loc[game_idx, "away"]), str(events_df.loc[game_idx, "home"])]
                            ss["board_date"] = _iso_to_date(events_df.loc[game_idx].get("game_date", ""))
                board_all = ss.get("board_all")
                if board_all is not None and not board_all.empty:
                    players = sorted(board_all["Player"].dropna().unique().tolist())
                    player_pick = st.selectbox("Step 4: Choose Player to Evaluate", players, key="pick_player_dropdown")
                    lines = player_lines(board_all, player_pick)
                    st.markdown(f"**{player_pick}: all DraftKings lines**")
                    head = st.columns([2, 1, 1, 1, 2])
                    for col, title in zip(head, ["Market", "Line", "Over", "Under", ""]):
                        col.markdown(f"**{title}**")
                    for _, r in lines.iterrows():
                        row_cols = st.columns([2, 1, 1, 1, 2])
                        row_cols[0].write(r["Market"])
                        row_cols[1].write(f"{r['Line']:g}")
                        row_cols[2].write(f"{int(r['Over']):+d}")
                        row_cols[3].write(f"{int(r['Under']):+d}")
                        if row_cols[4].button("⚡ Populate", key=f"populate_{player_pick}_{r['Market']}"):
                            ss.update(player=player_pick, stat=r["Market"], line=float(r["Line"]), over=int(r["Over"]),
                                      under=int(r["Under"]), opp="None", game_date=ss.get("board_date") or _today_et())
                            st.rerun()
                    with st.expander("📋 Full board for one market"):
                        avail = [m for m in STATS if m in set(board_all["Market"])]
                        mk = st.selectbox("Market", avail, key="full_board_market")
                        st.dataframe(summarize_board(board_all[board_all["Market"] == mk]), use_container_width=True)

        with st.expander("⚙️ Model settings: how much weight does last season get?", expanded=False):
            s1, s2 = st.columns(2)
            s1.slider("Per-minute rate: last-season weight (pseudo-games)", 0, 40, key="n0_rate")
            s2.slider("Minutes: last-season weight (pseudo-games)", 0, 40, key="n0_min")
            st.caption("Weight w means last season counts like w extra games next to this season's n games, so its share is w / (n + w). "
                       "0 = trust this season only. Rates are stable year to year (higher weight); minutes and roles change more (lower). "
                       "Press Evaluate to apply. Use the B2B Backtest tab to estimate these from data.")
        opp_opts = _opp_options(ss["board_teams"])
        with st.form("nba_form"):
            c1, c2, c3 = st.columns([2, 1, 1])
            player_name = c1.text_input("Player Name", value=ss["player"])
            stat_category = c2.selectbox("Stat Category", STATS, index=STATS.index(ss["stat"]) if ss["stat"] in STATS else 0)
            line_val = c3.number_input("Line", step=0.5, value=float(ss["line"]))
            c4, c5, c6, c7 = st.columns(4)
            over_odds = c4.number_input("Over Odds", value=int(ss["over"]))
            under_odds = c5.number_input("Under Odds", value=int(ss["under"]))
            opponent = c6.selectbox("Opponent (optional)", opp_opts, index=opp_opts.index(ss["opp"]) if ss["opp"] in opp_opts else 0)
            game_day = c7.date_input("Game date (ET)", value=ss["game_date"])
            submitted = st.form_submit_button("⚡ Evaluate Prop")

        if submitted:
            ss.update(player=player_name, stat=stat_category, line=line_val, over=over_odds, under=under_odds, opp=opponent, game_date=game_day)
            res = evaluate_nba_prop(player_name, stat_category, line_val, over_odds, under_odds,
                                    None if opponent == "None" else opponent, game_day, ss.get("b2b_effects"),
                                    (float(ss["n0_rate"]), float(ss["n0_min"])))
            if "error" in res:
                ss["last_res"], ss["last_err"] = None, res["error"]
            else:
                res["uid"] = uuid.uuid4().hex
                ss["last_res"], ss["last_err"] = res, None
                ss["last_ctx"] = {"player": player_name, "stat": stat_category, "line": float(line_val),
                                  "opponent": None if opponent == "None" else opponent, "game_date": game_day}
        if ss.get("last_err"):
            st.error(ss["last_err"])
        elif ss.get("last_res"):
            render_results(ss["last_res"], ss["last_ctx"])

        with st.expander("ℹ️ How the model works"):
            st.markdown(
                "- **Projection** = shrunk per-minute rate × shrunk minutes × opponent multiplier. Current-season games "
                "(last 10, EWMA decay 0.8) are blended with last season's rates; the last-season weights (per-minute rate and minutes) are set in Model settings.\n"
                "- **Spread** = pooled coefficient of variation × projection, inflated for estimation uncertainty.\n"
                "- **Distribution**: Normal for points, PRA and the points/rebounds/assists combos; Negative Binomial (Poisson if under-dispersed) for rebounds, assists, 3PM, steals, blocks. 50,000 seeded draws.\n"
                "- **Market**: two-way multiplicative devig. Edge = model % − devigged market %. EV uses the model probability at the offered price (pushes refunded).\n"
                "- **Opponent**: points-allowed (points, PRA, any combo with points) or scoring-pace (other stats) index from ESPN results, shrunk by games played, capped ±10%. It is a proxy, not true positional defense.\n"
                "- **Kelly**: ¼ Kelly capped at 5% of bankroll; 0 when EV ≤ 0.\n"
                "- **CLV**: devigged closing probability of your side minus the devigged probability when you bet, same line only. "
                "If the line moved, LINE MOVE shows how far (positive = in your favor)."
            )
    with tab_log:
        render_bet_log()
    with tab_bt:
        render_backtest()


if __name__ == "__main__":
    main()
