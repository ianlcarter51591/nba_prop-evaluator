"""NBA Prop Evaluator: Standalone App with Retry Logic, Live Odds API & Powder Blue Theme."""
import os
import re
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

st.set_page_config(
    page_title="NBA Prop Evaluator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# High-Contrast Powder Blue UI Styling
st.markdown(
    """
    <style>
    /* Pure White Main Background */
    .stApp {
        background-color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Powder Blue Container Boxes, Cards & Expanders */
    .vision-card, 
    [data-testid="stVerticalBlockBorderWrapper"], 
    [data-testid="stExpander"], 
    .vision-header {
        background: #b9e6ff !important;
        border: 2px solid #7dd3fc !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 14px rgba(0, 117, 255, 0.08) !important;
    }
    .vision-card { padding: 20px !important; margin-bottom: 20px !important; }

    /* Fix Expander Header Title Text */
    [data-testid="stExpander"] summary, 
    [data-testid="stExpander"] summary * {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 16px !important;
    }

    /* Header Text Styling */
    .vision-title {
        color: #0369a1 !important;
        font-size: 26px;
        font-weight: 900;
        margin: 0;
    }

    .vision-sub {
        color: #0f172a !important;
        font-size: 13px;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Powder Blue Metric Boxes */
    [data-testid="stMetric"], .metric-box {
        background: #b9e6ff !important;
        border: 2px solid #7dd3fc !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        text-align: center;
    }
    [data-testid="stMetricValue"], .metric-value {
        color: #0369a1 !important;
        font-size: 24px !important;
        font-weight: 900 !important;
    }
    [data-testid="stMetricLabel"], .metric-label {
        color: #0f172a !important;
        font-size: 11px !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    .metric-value-green {
        color: #15803d !important;
        font-size: 24px !important;
        font-weight: 900 !important;
    }

    /* Text & Label Overrides */
    label, p, span, div[data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] label,
    [data-testid="stExpander"] label,
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span,
    .vision-card label,
    [data-testid="stForm"] label {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Input Fields & Dropdowns */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: 2px solid #0284c7 !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] * {
        color: #ffffff !important;
    }

    /* Action Buttons */
    .stButton>button, [data-testid="stFormSubmitButton"]>button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        font-weight: 900 !important;
        border-radius: 10px !important;
        border: None !important;
        padding: 12px 20px !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
        width: 100% !important;
    }
    .stButton>button *, [data-testid="stFormSubmitButton"]>button * {
        color: #ffffff !important;
        font-weight: 900 !important;
    }

    /* Data Table Styling */
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 2px solid #7dd3fc !important;
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

STATS = [
    "Points",
    "Rebounds",
    "Assists",
    "3-Pointers Made",
    "Pts + Reb + Ast",
    "Steals",
    "Blocks",
]

STAT_COLUMN_MAP = {
    "POINTS": ["PTS", "POINTS"],
    "REBOUNDS": ["REB", "REBOUNDS", "TRB"],
    "ASSISTS": ["AST", "ASSISTS"],
    "3-POINTERS MADE": ["FG3M", "3PM", "THREE_POINTERS_MADE", "3-POINTERS MADE"],
    "PTS + REB + AST": ["PTS+REB+AST", "PRA", "PTS + REB + AST"],
    "STEALS": ["STL", "STEALS"],
    "BLOCKS": ["BLK", "BLOCKS"],
}

ODDS_API_MARKETS = {
    "Points": "player_points",
    "Rebounds": "player_rebounds",
    "Assists": "player_assists",
    "3-Pointers Made": "player_threes",
    "Steals": "player_steals",
    "Blocks": "player_blocks",
}

def _secret(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()

def parse_val(v) -> float:
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
    s = str(name).lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.split())

# --- THE ODDS API CLIENT FUNCTIONS ---
def list_nba_events(api_key: str) -> Tuple[pd.DataFrame, str | None]:
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    try:
        resp = requests.get(url, params={"apiKey": api_key}, timeout=8)
        if resp.status_code != 200:
            return pd.DataFrame(), f"Odds API Error ({resp.status_code}): {resp.text}"
        data = resp.json()
        rows = []
        for ev in data:
            home = ev.get("home_team", "")
            away = ev.get("away_team", "")
            commence = ev.get("commence_time", "")[:16].replace("T", " ")
            rows.append({
                "id": ev.get("id"),
                "label": f"{away} @ {home} ({commence} UTC)",
            })
        return pd.DataFrame(rows), None
    except Exception as e:
        return pd.DataFrame(), f"Error fetching NBA events: {e}"

def fetch_nba_prop_board(event_id: str, market: str, api_key: str) -> Tuple[pd.DataFrame, str | None]:
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
    try:
        resp = requests.get(url, params={"apiKey": api_key, "regions": "us", "markets": market, "bookmakers": "draftkings"}, timeout=8)
        if resp.status_code != 200:
            return pd.DataFrame(), f"Odds API Error ({resp.status_code}): {resp.text}"
        data = resp.json()
        bookmakers = data.get("bookmakers", [])
        if not bookmakers:
            return pd.DataFrame(), "No DraftKings odds found for this event/market."
        
        markets = bookmakers[0].get("markets", [])
        if not markets:
            return pd.DataFrame(), "No market data available."

        outcomes = markets[0].get("outcomes", [])
        rows = []
        for o in outcomes:
            rows.append({
                "Player": o.get("description"),
                "Line": o.get("point"),
                "Price": o.get("price"),
                "Outcome": o.get("name"),
            })
        return pd.DataFrame(rows), None
    except Exception as e:
        return pd.DataFrame(), f"Error fetching prop board: {e}"

def summarize_board(board_df: pd.DataFrame) -> pd.DataFrame:
    if board_df.empty:
        return pd.DataFrame()
    summary_rows = []
    for player, group in board_df.groupby("Player"):
        line = group["Line"].iloc[0] if "Line" in group.columns else 0.5
        over_price = -110
        under_price = -110
        for _, r in group.iterrows():
            outcome = str(r.get("Outcome", "")).lower()
            if "over" in outcome:
                over_price = int(r.get("Price", -110))
            elif "under" in outcome:
                under_price = int(r.get("Price", -110))
        summary_rows.append({
            "Player": player,
            "Line": line,
            "Over": over_price,
            "Under": under_price,
        })
    return pd.DataFrame(summary_rows)

# --- ROBUST NBA STATS FETCHING WITH RETRY LOGIC ---
def fetch_nba_player_logs(player_name: str, season: str = "2026-27") -> Tuple[pd.DataFrame, str | None]:
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."

    raw_input = player_name.strip()
    target_clean = _clean_name(raw_input)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            matched_players = nba_players.find_players_by_full_name(raw_input)
            if not matched_players:
                all_p = nba_players.get_players()
                parts = target_clean.split()
                last_name = parts[-1] if parts else target_clean
                matched_players = [p for p in all_p if last_name in p["full_name"].lower()]

            if not matched_players:
                return pd.DataFrame(), f"No NBA player found matching '{raw_input}'."

            player_id = matched_players[0]["id"]
            
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id, 
                season=season,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36', 'Referer': 'https://www.nba.com/'},
                timeout=45
            )
            df = gamelog.get_data_frames()[0]

            if df is not None and not df.empty:
                rows = []
                for _, r in df.iterrows():
                    pts = parse_val(r.get("PTS", 0))
                    reb = parse_val(r.get("REB", 0))
                    ast = parse_val(r.get("AST", 0))
                    matchup = str(r.get("MATCHUP", "OPP"))
                    g_date = str(r.get("GAME_DATE", ""))

                    rows.append({
                        "POINTS": pts,
                        "REBOUNDS": reb,
                        "ASSISTS": ast,
                        "3-POINTERS MADE": parse_val(r.get("FG3M", 0)),
                        "PTS + REB + AST": pts + reb + ast,
                        "STEALS": parse_val(r.get("STL", 0)),
                        "BLOCKS": parse_val(r.get("BLK", 0)),
                        "OPPONENT": matchup,
                        "DATE": g_date,
                    })

                if rows:
                    df_nba = pd.DataFrame(rows)
                    return df_nba.iloc[::-1].reset_index(drop=True), None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                return pd.DataFrame(), f"NBA API Error (after {max_retries} attempts): {e}"

    return pd.DataFrame(), f"No active NBA game logs found for '{raw_input}'."

def american_to_implied_prob(american_odds: float) -> float:
    try:
        odds = float(american_odds)
        if odds < 0:
            return round((-odds / (-odds + 100)) * 100, 1)
        return round((100 / (odds + 100)) * 100, 1)
    except Exception:
        return 52.4

def calculate_ewma(stats: np.ndarray, decay: float = 0.8) -> float:
    if len(stats) == 0:
        return 0.0
    weights = decay ** np.arange(len(stats))[::-1]
    weights /= weights.sum()
    return float(np.sum(stats * weights))

def evaluate_nba_prop(
    player_name: str,
    stat_category: str,
    line: float,
    over_odds: float,
    under_odds: float,
    season: str = "2026-27",
) -> Dict[str, Any]:
    try:
        df, err = fetch_nba_player_logs(player_name, season)
        if err or df is None or df.empty:
            return {"error": err or f"No game logs found for '{player_name}'."}

        df.columns = [c.upper().strip() for c in df.columns]
        stat_upper = stat_category.upper().strip()

        if stat_upper == "PTS + REB + AST" and "PTS+REB+AST" not in df.columns:
            if all(col in df.columns for col in ["PTS", "REB", "AST"]):
                df["PTS+REB+AST"] = df["PTS"].astype(float) + df["REB"].astype(float) + df["AST"].astype(float)

        target_col = None
        if stat_upper in df.columns:
            target_col = stat_upper
        else:
            possible_cols = STAT_COLUMN_MAP.get(stat_upper, [stat_upper])
            for col in possible_cols:
                if col in df.columns:
                    target_col = col
                    break

        if not target_col or target_col not in df.columns:
            return {"error": f"Stat category '{stat_category}' not found in logs for '{player_name}'."}

        stats_list = []
        gamelog_details = []

        for _, row in df.iterrows():
            try:
                val = float(row[target_col])
            except (ValueError, TypeError):
                continue
            
            opp = str(row.get("OPPONENT", "OPP"))
            gdate = str(row.get("DATE", ""))
            stats_list.append(val)
            gamelog_details.append({
                "stat_value": val,
                "opponent": opp,
                "date": gdate,
                "x_label": f"{gdate}<br>{opp}" if gdate else opp,
            })

        if not stats_list:
            return {"error": f"No valid numeric data found for stat '{stat_category}'."}

        stats_arr = np.array(stats_list[:10])
        gamelog_details = gamelog_details[:10]

        real_mean = float(np.mean(stats_arr))
        real_median = float(np.median(stats_arr))
        proj_mean = calculate_ewma(stats_arr)

        std_dev = float(np.std(stats_arr)) if len(stats_arr) > 1 else max(1.0, proj_mean * 0.15)
        simulations = np.random.normal(proj_mean, std_dev if std_dev > 0 else 1.0, 10000)

        over_hits = np.sum(simulations > line)
        over_model_pct = (over_hits / 10000) * 100
        under_model_pct = 100 - over_model_pct

        over_market_pct = american_to_implied_prob(over_odds)
        under_market_pct = american_to_implied_prob(under_odds)

        over_win_pct = (over_model_pct * 0.5) + (over_market_pct * 0.5)
        under_win_pct = (under_model_pct * 0.5) + (under_market_pct * 0.5)

        over_fair_odds = int(-100 * (over_win_pct / (100 - over_win_pct))) if over_win_pct > 50 else int(100 * ((100 - over_win_pct) / over_win_pct))
        under_fair_odds = int(-100 * (under_win_pct / (100 - under_win_pct))) if under_win_pct > 50 else int(100 * ((100 - under_win_pct) / under_win_pct))

        is_over = over_win_pct > under_win_pct
        best_side = f"Over {line}" if is_over else f"Under {line}"
        edge_vs_market = (over_win_pct - over_market_pct) if is_over else (under_win_pct - under_market_pct)
        
        chosen_odds = over_odds if is_over else under_odds
        chosen_win_pct = over_win_pct if is_over else under_win_pct
        
        if chosen_odds < 0:
            ev_pct = (chosen_win_pct * (100 / abs(chosen_odds))) - (100 - chosen_win_pct)
        else:
            ev_pct = (chosen_win_pct * (chosen_odds / 100)) - (100 - chosen_win_pct)

        return {
            "player_name": player_name,
            "projection_mean": real_mean,
            "projection_median": real_median,
            "best_side": best_side,
            "fair_odds": over_fair_odds if is_over else under_fair_odds,
            "edge_vs_market": edge_vs_market,
            "ev_pct": ev_pct,
            "gamelog_details": gamelog_details,
        }
    except Exception as e:
        return {"error": f"Evaluation error: {e}"}

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

    # --- LIVE ODDS API PANEL ---
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
                game_idx = st.selectbox(
                    "Step 2: Select Game",
                    range(len(events_df)),
                    format_func=lambda i: str(events_df.loc[i, "label"]),
                    key="live_game_idx",
                )
                if st.button(f"Step 3: Fetch {selected_stat_label} Odds", key="fetch_board_odds"):
                    event_id = str(events_df.loc[game_idx, "id"])
                    board_df, err = fetch_nba_prop_board(event_id, market_key, api_key)
                    if err:
                        st.error(err)
                    else:
                        ss["board_df"] = board_df

            board_df = ss.get("board_df")
            if board_df is not None and not board_df.empty:
                summary = summarize_board(board_df)
                st.dataframe(summary, use_container_width=True)
                player_pick = st.selectbox("Step 4: Choose Player to Evaluate", summary["Player"].tolist(), key="pick_player_dropdown")
                
                matched_row = summary[summary["Player"] == player_pick]
                if not matched_row.empty:
                    r = matched_row.iloc[0]
                    if st.button(f"⚡ Populate {player_pick} ({r['Line']} | Over {r['Over']} / Under {r['Under']})", key="populate_btn_odds"):
                        ss["player"] = player_pick
                        ss["stat"] = selected_stat_label
                        ss["line"] = float(r["Line"])
                        ss["over"] = int(r["Over"])
                        ss["under"] = int(r["Under"])
                        st.rerun()

    # --- EVALUATION FORM ---
    if "player" not in ss:
        ss["player"] = "Jayson Tatum"
    if "stat" not in ss:
        ss["stat"] = STATS[0]
    if "line" not in ss:
        ss["line"] = 26.5
    if "over" not in ss:
        ss["over"] = -110
    if "under" not in ss:
        ss["under"] = -110

    with st.form("nba_form"):
        c1, c2, c3 = st.columns([2, 1, 1])
        player_name = c1.text_input("Player Name", value=ss["player"])
        stat_category = c2.selectbox("Stat Category", STATS, index=STATS.index(ss["stat"]) if ss["stat"] in STATS else 0)
        line_val = c3.number_input("Line", step=0.5, value=float(ss["line"]))

        c4, c5 = st.columns(2)
        over_odds = c4.number_input("Over Odds", value=int(ss["over"]))
        under_odds = c5.number_input("Under Odds", value=int(ss["under"]))

        submitted = st.form_submit_button("⚡ Evaluate Prop")

    if submitted:
        ss["player"] = player_name
        ss["stat"] = stat_category
        ss["line"] = line_val
        ss["over"] = over_odds
        ss["under"] = under_odds

        res = evaluate_nba_prop(
            player_name=player_name,
            stat_category=stat_category,
            line=line_val,
            over_odds=over_odds,
            under_odds=under_odds,
        )

        if "error" in res:
            st.error(res["error"])
        else:
            st.markdown(f"## {res['player_name']} | {stat_category}")
            
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.markdown(f'<div class="metric-box"><div class="metric-label">MEAN</div><div class="metric-value">{res["projection_mean"]:.2f}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-box"><div class="metric-label">MEDIAN</div><div class="metric-value">{res["projection_median"]:.2f}</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-box"><div class="metric-label">BEST SIDE</div><div class="metric-value">{res["best_side"]}</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-box"><div class="metric-label">FAIR ODDS</div><div class="metric-value">{res["fair_odds"]:+d}</div></div>', unsafe_allow_html=True)
            m5.markdown(f'<div class="metric-box"><div class="metric-label">EDGE VS MKT</div><div class="metric-value-green">+{res["edge_vs_market"]:.1f}%</div></div>', unsafe_allow_html=True)
            m6.markdown(f'<div class="metric-box"><div class="metric-label">EV</div><div class="metric-value-green">+{res["ev_pct"]:.1f}%</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📊 Game Log Performance")

            gamelog_data = res.get("gamelog_details", [])
            if gamelog_data:
                df_chart = pd.DataFrame(gamelog_data)
                colors = ["#22c55e" if v > line_val else "#ef4444" for v in df_chart["stat_value"]]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_chart["x_label"],
                    y=df_chart["stat_value"],
                    text=[f"{v:.1f}" if v % 1 != 0 else f"{int(v)}" for v in df_chart["stat_value"]],
                    textposition="outside",
                    textfont=dict(color="#0f172a", size=12, family="sans-serif"),
                    marker_color=colors,
                ))
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=len(df_chart) - 0.5,
                    y0=line_val,
                    y1=line_val,
                    line=dict(color="#2563eb", width=2, dash="dash"),
                )
                fig.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#0f172a"),
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=40),
                    xaxis=dict(showgrid=False, tickfont=dict(color="#0f172a")),
                    yaxis=dict(showgrid=True, gridcolor="#e2e8f0", tickfont=dict(color="#0f172a")),
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
