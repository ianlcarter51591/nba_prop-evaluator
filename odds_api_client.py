"""Client for fetching live player props from The Odds API."""
from __future__ import annotations

import requests
import pandas as pd
from typing import Dict, Any, Tuple, List

BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_BOOKS = "draftkings"

SPORT_KEYS: Dict[str, str] = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
}

# Mapping user-friendly names to official market keys required by The Odds API
MARKETS: Dict[str, Dict[str, str]] = {
    "NFL": {
        "Passing Yards": "player_pass_yds",
        "Pass Completions": "player_pass_completions",
        "Pass Attempts": "player_pass_attempts",
        "Passing TDs": "player_pass_tds",
        "Rushing Yards": "player_rush_yds",
        "Rushing Attempts": "player_rush_attempts",
        "Receiving Yards": "player_reception_yds",
        "Receptions": "player_receptions",
        "Interceptions": "player_pass_interceptions",
    },
    "NBA": {
        "Points": "player_points",
        "Rebounds": "player_rebounds",
        "Assists": "player_assists",
        "3-Pointers Made": "player_threes",
        "Pts + Reb + Ast": "player_points_rebounds_assists",
        "Steals": "player_steals",
        "Blocks": "player_blocks",
    },
}

class OddsApiError(Exception):
    pass

def parse_books(books_str: str) -> Tuple[str, ...]:
    return tuple(b.strip().lower() for b in books_str.split(",") if b.strip())

def list_events(sport: str, api_key: str) -> Tuple[pd.DataFrame, Any]:
    sport_key = SPORT_KEYS.get(sport.upper(), "americanfootball_nfl")
    url = f"{BASE_URL}/sports/{sport_key}/events"
    params = {"apiKey": api_key}
    
    res = requests.get(url, params=params, timeout=10)
    if res.status_code != 200:
        raise OddsApiError(f"API Error ({res.status_code}): {res.text}")
    
    events = res.json()
    rows = []
    for e in events:
        home = e.get("home_team", "Home")
        away = e.get("away_team", "Away")
        rows.append({
            "id": e.get("id"),
            "label": f"{away} @ {home}",
            "commence_time": e.get("commence_time")
        })
    return pd.DataFrame(rows), None

def fetch_prop_board(
    sport: str, event_id: str, market_label: str, books: Tuple[str, ...], api_key: str
) -> Tuple[pd.DataFrame, Any]:
    sport_key = SPORT_KEYS.get(sport.upper(), "americanfootball_nfl")
    market_key = MARKETS.get(sport.upper(), {}).get(market_label, "player_pass_yds")
    book_param = ",".join(books) if books else "draftkings"
    
    url = f"{BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": market_key,
        "bookmakers": book_param,
        "oddsFormat": "american",
    }
    
    res = requests.get(url, params=params, timeout=10)
    if res.status_code != 200:
        raise OddsApiError(f"API Error ({res.status_code}): {res.text}")
    
    data = res.json()
    rows = []
    for b in data.get("bookmakers", []):
        book_name = b.get("title", "DraftKings")
        for m in b.get("markets", []):
            for outcome in m.get("outcomes", []):
                rows.append({
                    "Player": outcome.get("description", outcome.get("name")),
                    "Book": book_name,
                    "Type": outcome.get("name"),  # Over or Under
                    "Line": outcome.get("point", 0.0),
                    "Price": outcome.get("price", -110),
                })
    return pd.DataFrame(rows), None

def summarize_board(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return pd.DataFrame()
    
    players = board["Player"].unique()
    rows = []
    for p in players:
        p_df = board[board["Player"] == p]
        line = p_df["Line"].iloc[0] if not p_df.empty else 0.0
        over_row = p_df[p_df["Type"] == "Over"]
        under_row = p_df[p_df["Type"] == "Under"]
        
        over_price = over_row["Price"].iloc[0] if not over_row.empty else -110
        under_price = under_row["Price"].iloc[0] if not under_row.empty else -110
        
        rows.append({
            "Player": p,
            "Line": line,
            "Over": over_price,
            "Under": under_price,
        })
    return pd.DataFrame(rows)

def player_offers(board: pd.DataFrame, player: str) -> pd.DataFrame:
    p_df = board[board["Player"] == player]
    if p_df.empty:
        return pd.DataFrame()
    return summarize_board(p_df)
