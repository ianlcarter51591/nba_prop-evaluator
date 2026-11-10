"""Odds API client for fetching live DraftKings player prop markets."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests

API_BASE = "https://api.the-odds-api.com/v4/sports"

SPORT_KEYS: Dict[str, str] = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
}

MARKETS: Dict[str, Dict[str, str]] = {
    "NFL": {
        "Passing Yards": "player_pass_yds",
        "Pass Completions": "player_pass_completions",
        "Pass Attempts": "player_pass_attempts",
        "Passing TDs": "player_pass_tds",
        "Interceptions": "player_pass_interceptions",
        "Rushing Yards": "player_rush_yds",
        "Rushing Attempts": "player_rush_attempts",
        "Receiving Yards": "player_reception_yds",
        "Receptions": "player_receptions",
        "Kicking Points": "player_kicking_points",
        "Field Goals Made": "player_field_goals",
        "Anytime Touchdown Scorer": "player_anytime_td",
    }
}


class OddsApiError(Exception):
    """Custom exception for Odds API request issues."""
    pass


def _resolve_sport_key(sport: str) -> str:
    s = str(sport or "").upper().strip()
    return SPORT_KEYS.get(s, s.lower())


def et_label(iso: str) -> str:
    """Odds API UTC timestamp -> US Eastern label like '2026-11-09 8:20 PM EST' (EDT in summer); the raw text if unparseable."""
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(ZoneInfo("America/New_York"))
        return f"{dt:%Y-%m-%d} {int(dt.strftime('%I'))}:{dt:%M} {dt:%p} {dt:%Z}"
    except Exception:
        return str(iso)


def list_events(sport: str, api_key: str) -> Tuple[pd.DataFrame, int]:
    sport_key = _resolve_sport_key(sport)
    url = f"{API_BASE}/{sport_key}/events"
    params = {"apiKey": api_key}
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        raise OddsApiError(f"API Error ({resp.status_code}): {resp.text}")
    
    data = resp.json()
    remaining_requests = int(resp.headers.get("x-requests-remaining", 0))
    
    rows = []
    for ev in data:
        rows.append({
            "id": ev["id"],
            "commence_time": ev["commence_time"],
            "home_team": ev["home_team"],
            "away_team": ev["away_team"],
            "label": f"{ev['away_team']} @ {ev['home_team']} ({et_label(ev['commence_time'])})"
        })
    return pd.DataFrame(rows), remaining_requests


def fetch_prop_board(
    sport: str,
    event_id: str,
    market_label: str,
    bookmakers: Tuple[str, ...],
    api_key: str
) -> Tuple[pd.DataFrame, int]:
    sport_map = MARKETS.get(sport.upper(), MARKETS.get("NFL", {}))
    market_key = sport_map.get(market_label)
    if not market_key:
        raise OddsApiError(f"Market label '{market_label}' is not supported for live API fetching.")

    sport_key = _resolve_sport_key(sport)
    url = f"{API_BASE}/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": market_key,
        "bookmakers": ",".join(bookmakers),
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        raise OddsApiError(f"API Error ({resp.status_code}): {resp.text}")

    remaining_requests = int(resp.headers.get("x-requests-remaining", 0))
    data = resp.json()
    
    rows = []
    for bmark in data.get("bookmakers", []):
        book_title = bmark.get("title", bmark.get("key"))
        for mkt in bmark.get("markets", []):
            for outcome in mkt.get("outcomes", []):
                rows.append({
                    "Bookmaker": book_title,
                    "Player": outcome.get("description", outcome.get("name")),
                    "Side": outcome.get("name"),
                    "Line": outcome.get("point", np.nan),
                    "Odds": outcome.get("price"),
                })
    return pd.DataFrame(rows), remaining_requests


def _synthetic_under(over_odds: float) -> float:
    """Generate fair synthetic opposite Under odds if the sportsbook only lists Over."""
    try:
        o = float(over_odds)
        p_over = 100.0 / (o + 100.0) if o > 0 else (-o) / (-o + 100.0)
        p_under = max(1.0 - p_over, 0.05)
        p_juiced = min(p_under * 1.05, 0.95)
        if p_juiced >= 0.5:
            return -round((p_juiced / (1.0 - p_juiced)) * 100.0)
        return round(((1.0 - p_juiced) / p_juiced) * 100.0)
    except Exception:
        return -110.0


def summarize_board(board: pd.DataFrame) -> pd.DataFrame:
    """Pivot raw outcomes table into a single row per player with Line, Over, and Under columns."""
    if board.empty:
        return pd.DataFrame(columns=["Player", "Line", "Over", "Under"])
    
    pivoted = []
    for player, grp in board.groupby("Player"):
        line_series = grp["Line"].dropna()
        line = line_series.iloc[0] if not line_series.empty else 0.5
        
        over_odds = grp[grp["Side"].isin(["Over", "Yes"])]["Odds"].values
        under_odds = grp[grp["Side"].isin(["Under", "No"])]["Odds"].values
        
        over_val = over_odds[0] if len(over_odds) > 0 else np.nan
        under_val = under_odds[0] if len(under_odds) > 0 else np.nan
        
        if pd.notna(over_val) and pd.isna(under_val):
            under_val = _synthetic_under(over_val)
        
        pivoted.append({
            "Player": player,
            "Line": float(line) if pd.notna(line) else 0.5,
            "Over": float(over_val) if pd.notna(over_val) else np.nan,
            "Under": float(under_val) if pd.notna(under_val) else np.nan,
        })
    return pd.DataFrame(pivoted)


def player_offers(board: pd.DataFrame, player_name: str) -> pd.DataFrame:
    summary = summarize_board(board)
    return summary[summary["Player"] == player_name]
