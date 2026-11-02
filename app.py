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
CONTINUOUS_STATS = {"POINTS", "PTS + REB + AST", "PTS + AST", "PTS + REB", "REB + AST"}
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
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def parse_val(v: Any) -> float:
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
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        tz = ZoneInfo("America/New_York") if ZoneInfo else timezone(timedelta(hours=-5))
        return dt.astimezone(tz).date()
    except Exception:
        return None


def _iso_to_date(s: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(s))
    except ValueError:
        return None


def _clean_name(name: str) -> str:
    s = re.sub(r"[^\w\s]", "", str(name).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.split())


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
            return pd.DataFrame(), "No DraftKings odds found."
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


ESPN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}
ESPN_SEARCH_URLS = ["https://site.web.api.espn.com/apis/common/v3/search", "https://site.web.api.espn.com/apis/search/v2"]
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
    candidates = []
    for url in ESPN_SEARCH_URLS:
        try:
            resp = requests.get(url, params={"query": name, "limit": 10, "type": "player", "sport": "basketball", "league": "nba"}, headers=ESPN_HEADERS, timeout=10)
            if resp.status_code != 200:
                continue
            for d in _walk_json(resp.json()):
                uid = str(d.get("uid", ""))
                label = d.get("displayName") or d.get("fullName") or d.get("name")
                if label and str(d.get("type", "")).lower() == "player":
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


def _espn_fetch_logs(pid: str, season: Optional[int] = None) -> pd.DataFrame:
    resp = requests.get(ESPN_GAMELOG_URL.format(pid=pid), params={"season": season} if season else {}, headers=ESPN_HEADERS, timeout=12)
    resp.raise_for_status()
    data = resp.json()
    labels = [str(x).upper() for x in data.get("labels", [])]
    meta = data.get("events", {}) or {}
    rows = []
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
    gamelog = playergamelog.PlayerGameLog(player_id=matched[0]["id"], season=season, timeout=10)
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
    try:
        pid = _espn_find_player_id(player_name)
        if pid:
            df = _finalize_logs(pd.concat([_espn_fetch_logs(pid, None), _espn_fetch_logs(pid, PRIOR_ESPN_SEASON)], ignore_index=True))
            if not df.empty:
                return df
    except Exception:
        pass
    for season in (CURRENT_NBA_SEASON, PRIOR_NBA_SEASON):
        try:
            df = _nba_api_fetch_logs(player_name, season)
            if not df.empty:
                return _finalize_logs(df)
        except Exception:
            continue
    raise RuntimeError(f"Could not load logs for '{player_name}'.")


def fetch_player_logs(player_name: str) -> Tuple[pd.DataFrame, Optional[str]]:
    if not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."
    try:
        return _load_logs_cached(player_name.strip()), None
    except Exception as e:
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=21600, show_spinner=False)
def load_team_meta() -> List[Tuple[str, str]]:
    resp = requests.get(TEAM_LIST_URL, headers=ESPN_HEADERS, timeout=12)
    resp.raise_for_status()
    teams = resp.json()["sports"][0]["leagues"][0]["teams"]
    return [(str(t["team"]["id"]), str(t["team"]["displayName"])) for t in teams]


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
        a, b = float(mine[0].get("score", 0)), float(other[0].get("score", 0))
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
    return df


def opp_multiplier(team: str, stat_col: str, scoring: pd.DataFrame) -> Tuple[float, str]:
    if team not in scoring.index or scoring.loc[team, "GAMES"] <= 0:
        return 1.0, f"No scoring data for {team}."
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
    return mult, f"{team}: pts-allowed idx {def_f:.2f} -> x{mult:.3f}"


def own_b2b_status(prev_date: Optional[date], game_day: date) -> str:
    if prev_date is None:
        return "unknown"
    return "second" if (game_day - prev_date).days == 1 else "rested"


def own_b2b_multiplier(status: str, stat_col: str, effects: Optional[Dict[str, float]] = None) -> float:
    eff = B2B_OWN_EFFECT if effects is None else effects
    return 1.0 + eff.get(stat_col, 0.0) if status == "second" else 1.0


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


def ewma_weights(n: int, decay: float = DECAY) -> np.ndarray:
    w = decay ** np.arange(n)[::-1]
    return w / w.sum()


def _cv(x: np.ndarray) -> Optional[float]:
    if len(x) < 3 or float(np.mean(x)) <= 0:
        return None
    return float(np.std(x, ddof=1) / np.mean(x))


def build_projection(cur_x: np.ndarray, cur_m: np.ndarray, pri_x: np.ndarray, pri_m: np.ndarray, opp_mult: float = 1.0,
                     n0_rate: float = N_PRIOR_RATE, n0_min: float = N_PRIOR_MIN) -> Projection:
    n = len(cur_x)
    has_prior = len(pri_x) > 0
    if n == 0 and not has_prior:
        raise ValueError("No usable games found.")
    r0, m0 = (float(n0_rate) if has_prior else 0.0), (float(n0_min) if has_prior else 0.0)
    if n == 0:
        r0, m0 = max(r0, 1.0), max(m0, 1.0)
    pairs_rate, pairs_min = [], []
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
                      prior_weights: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    try:
        stat_col = STAT_COL.get(stat_category)
        if not stat_col:
            return {"error": f"Unknown stat '{stat_category}'."}
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
                opp_note = f"Opponent data error: {e}"

        game_day = game_date or datetime.now().date()
        day_iso = game_day.isoformat()
        before = played[played["DATE"] < day_iso]
        prev_date = datetime.fromisoformat(before["DATE"].iloc[-1]).date() if len(before) else None
        own_state = own_b2b_status(prev_date, game_day)
        own_mult = own_b2b_multiplier(own_state, stat_col, (b2b_effects or {}).get("own"))

        proj = build_projection(cur[stat_col].to_numpy(float), cur["MINUTES"].to_numpy(float),
                                pri[stat_col].to_numpy(float), pri["MINUTES"].to_numpy(float), opp_mult * own_mult,
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

        context_checks = run_automated_context_filters(proj.minutes, raw_avg, own_state)

        warnings = []
        if proj.n_current < 8:
            warnings.append(f"Small sample: {proj.n_current} games this season.")
        if opp_note:
            warnings.append(opp_note)

        return {
            "player_name": player_name, "mean": float(np.mean(sims)), "median": float(np.median(sims)),
            "verdict": "PLAY" if ev > 0 else "PASS", "best_side": side_txt, "ev_pct": ev * 100.0,
            "edge_pp": (p_cond - dv_side) * 100.0, "p_model": p_cond * 100.0, "p_market": dv_side * 100.0,
            "fair_odds": prob_to_american(p_cond), "kelly_pct": kelly_fraction(p_w, p_l, odds_side) * 100.0,
            "proj_minutes": proj.minutes, "opp_mult": proj.opp_mult, "raw_avg": raw_avg,
            "hit10": int(np.sum(vals > line)), "n10": len(vals),
            "hit5": int(np.sum(vals[-5:] > line)), "n5": len(vals[-5:]),
            "days_rest": (game_day - prev_date).days if prev_date else None, "warnings": warnings, "mu": proj.mu, "sides": sides,
            "context_checks": context_checks, "this_season": this_season,
            "prior_share": {"rate": proj.share_rate, "minutes": proj.share_min},
            "own_b2b": {"status": own_state, "mult": own_mult},
            "gamelog_details": [
                {"stat_value": float(v), "season": s, "x_label": f"{d}<br>{o}"}
                for v, s, d, o in zip(vals, recent["SEASON"], recent["DATE"], recent["OPPONENT"])
            ],
        }
    except Exception as e:
        return {"error": f"Evaluation error: {e}"}


def render_automated_checks(res: Dict[str, Any]) -> None:
    checks = res.get("context_checks", {})
    with st.expander("🔍 Step 3: Automated Contextual Reality Check (Pre-Flight Filter)", expanded=True):
        st.markdown("Automated filters running against player volume, schedule fatigue, and rotation minutes:")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.checkbox("1. Secure Volume & Usage", value=checks.get("secure_volume", True), disabled=True)
        with c2:
            st.checkbox("2. Clean Schedule Spot", value=checks.get("clean_schedule", True), disabled=True)
        with c3:
            st.checkbox("3. Rotation Minutes (≥ 24 proj)", value=checks.get("sharp_movement", True), disabled=True)
        st.caption("💡 *Note: High-volume players are less exposed to single-game variance.*")


def render_results(res: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    stat_category, line_val = ctx["stat"], ctx["line"]
    photo_col, title_col = st.columns([1, 6])
    photo = player_headshot_url(res["player_name"])
    if photo:
        photo_col.image(photo, width=120)
    title_col.markdown(f"## {res['player_name']} | {stat_category}")
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
    st.markdown("---")
    render_automated_checks(res)
    st.markdown("---")
    st.markdown("### 📊 Game Log Performance")
    gl = pd.DataFrame(res["gamelog_details"])
    if gl.empty:
        return
    colors = np.where(gl["stat_value"] > line_val, "#22c55e", np.where(gl["stat_value"] < line_val, "#ef4444", "#94a3b8"))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=gl["x_label"], y=gl["stat_value"], marker_color=colors, marker_opacity=np.where(gl["season"] == "current", 1.0, 0.5),
        text=[f"{v:.1f}" if v % 1 else f"{int(v)}" for v in gl["stat_value"]],
        textposition="outside", textfont=dict(color="#0f172a", size=12, family="sans-serif"),
    ))
    fig.add_shape(type="line", x0=-0.5, x1=len(gl) - 0.5, y0=line_val, y1=line_val, line=dict(color="#2563eb", width=2, dash="dash"))
    fig.add_shape(type="line", x0=-0.5, x1=len(gl) - 0.5, y0=res["mu"], y1=res["mu"], line=dict(color="#f97316", width=2, dash="dot"))
    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color="#0f172a"), height=380, margin=dict(l=20, r=20, t=30, b=40), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


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
                         ("under", -110), ("opp", "None"), ("board_teams", []), ("game_date", datetime.now().date())):
        ss.setdefault(key, default)

    tab_eval, tab_log, tab_bt = st.tabs(["⚡ Evaluate", "📒 Bet Log", "🔬 B2B Backtest"])
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

        with st.form("nba_form"):
            c1, c2, c3 = st.columns([2, 1, 1])
            player_name = c1.text_input("Player Name", value=ss["player"])
            stat_category = c2.selectbox("Stat Category", STATS, index=STATS.index(ss["stat"]) if ss["stat"] in STATS else 0)
            line_val = c3.number_input("Line", step=0.5, value=float(ss["line"]))
            c4, c5, c6, c7 = st.columns(4)
            over_odds = c4.number_input("Over Odds", value=int(ss["over"]))
            under_odds = c5.number_input("Under Odds", value=int(ss["under"]))
            opponent = c6.selectbox("Opponent (optional)", ["None"], index=0)
            game_day = c7.date_input("Game date (ET)", value=ss["game_date"])
            submitted = st.form_submit_button("⚡ Evaluate Prop")

        if submitted:
            ss.update(player=player_name, stat=stat_category, line=line_val, over=over_odds, under=under_odds, opp=opponent, game_date=game_day)
            res = evaluate_nba_prop(player_name, stat_category, line_val, over_odds, under_odds,
                                    None if opponent == "None" else opponent, game_day)
            if "error" in res:
                ss["last_res"], ss["last_err"] = None, res["error"]
            else:
                ss["last_res"], ss["last_err"] = res, None
                ss["last_ctx"] = {"player": player_name, "stat": stat_category, "line": float(line_val), "game_date": game_day}
        if ss.get("last_err"):
            st.error(ss["last_err"])
        elif ss.get("last_res"):
            render_results(ss["last_res"], ss["last_ctx"])


if __name__ == "__main__":
    main()
