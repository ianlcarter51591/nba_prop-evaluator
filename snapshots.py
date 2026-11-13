"""
Prediction snapshots: every evaluation is saved with what the model believed, then graded against the real result.

Math / logic
------------
Per snapshot we keep the model's view: projection (MU, mean), projected minutes, P(Over), P(Under), six-point scores and
failed criteria per side.  Grading joins the player's actual game (same ET date, or the next UTC date) and adds
    STAT_ERR = ACTUAL - MU            MIN_ERR = ACTUAL_MIN - PROJ_MIN       MIN_ERR_PCT = MIN_ERR / PROJ_MIN
    SHOWN_RESULT = Hit / Miss / Push  for the side the app displayed;  EV_RESULT likewise for the model's EV side.
Heuristic miss reason (a hint, not proof): a miss with |MIN_ERR_PCT| >= MIN_MISS_PCT is a minutes miss; otherwise
a miss with the minutes as projected is shooting / usage variance. Sports without minutes (NFL) just report
"actual vs projected". NFL rows are matched to the log by (SEASON, WEEK); NBA rows by game date.
One module serves both apps (identical copy in each repo).
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

SNAP_PATH = os.environ.get("SNAPSHOT_PATH", "snapshots.csv")
MIN_MISS_PCT = 0.25                 # minutes off by 25%+ of the projection counts as a minutes miss
KEY_COLS: List[str] = ["PLAYER", "STAT", "LINE", "GAME_DATE"]
TEXT_COLS: List[str] = ["SNAP_ID", "TS_ET", "GAME_DATE", "GAME", "PLAYER", "STAT", "OPPONENT", "SOURCE", "SHOWN_SIDE", "EV_SIDE",
                        "TIER_SHOWN", "FAILED_OVER", "FAILED_UNDER", "STATUS", "SHOWN_RESULT", "EV_RESULT", "WHY", "OVER_LINK", "UNDER_LINK"]
NUM_COLS: List[str] = ["SEASON", "WEEK", "LINE", "OVER_ODDS", "UNDER_ODDS", "MU", "PROJ_MEAN", "PROJ_MIN", "RAW_AVG", "OPP_MULT", "VEGAS_TOTAL", "MIN_EXTRA",
                       "P_OVER", "P_UNDER", "PM_OVER", "PM_UNDER", "SIX_OVER", "SIX_UNDER", "ACTUAL", "ACTUAL_MIN", "STAT_ERR", "MIN_ERR", "MIN_ERR_PCT"]
COLS: List[str] = TEXT_COLS + NUM_COLS


def _failed(six_side: Optional[Mapping[str, Any]]) -> str:
    """Labels of the criteria that did not pass, '; '-joined (unverified ones are marked)."""
    if not six_side:
        return ""
    return "; ".join(f"{c['label']}{'' if c.get('verified', True) else ' (unverified)'}"
                     for c in six_side.get("checks", []) if not c.get("passed"))


def make_row(res: Mapping[str, Any], ctx: Mapping[str, Any], stat_label: str, line: float, over: float, under: float,
             source: str, now_et: str, game_date: str, season: Optional[int] = None,
             week: Optional[int] = None, links: Tuple[str, str] = ("", "")) -> Dict[str, Any]:
    """One snapshot row from an evaluation result (a dict without 'error'). NFL passes season + week; NBA a game date."""
    six = res.get("six_point") or {}
    sides = res.get("sides") or {}
    shown = six.get("best", "")
    return {
        "SNAP_ID": uuid.uuid4().hex[:8], "TS_ET": now_et, "GAME_DATE": game_date, "PLAYER": res.get("player_name", ""),
        "STAT": stat_label, "SEASON": season, "WEEK": week, "LINE": float(line), "OVER_ODDS": float(over), "UNDER_ODDS": float(under),
        "OPPONENT": ctx.get("opponent") or "", "SOURCE": source,
        "GAME": ctx.get("game") or "", "VEGAS_TOTAL": (res.get("vegas") or {}).get("total"),
        "MU": res.get("mu"), "PROJ_MEAN": res.get("mean"), "PROJ_MIN": res.get("proj_minutes"), "RAW_AVG": res.get("raw_avg"),
        "OPP_MULT": res.get("opp_mult"), "MIN_EXTRA": (res.get("injury") or {}).get("minutes_extra"),
        "P_OVER": (sides.get("Over") or {}).get("p_model"), "P_UNDER": (sides.get("Under") or {}).get("p_model"),
        "PM_OVER": (sides.get("Over") or {}).get("p_market"), "PM_UNDER": (sides.get("Under") or {}).get("p_market"),
        "SHOWN_SIDE": shown, "EV_SIDE": six.get("ev_side", ""), "TIER_SHOWN": (six.get(shown) or {}).get("tier", ""),
        "SIX_OVER": (six.get("Over") or {}).get("score"), "SIX_UNDER": (six.get("Under") or {}).get("score"),
        "FAILED_OVER": _failed(six.get("Over")), "FAILED_UNDER": _failed(six.get("Under")), "STATUS": "Pending",
        "OVER_LINK": links[0], "UNDER_LINK": links[1],
    }


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every column exists with the right dtype; drop rows with no player."""
    d = df.copy()
    for c in COLS:
        if c not in d.columns:
            d[c] = np.nan if c in NUM_COLS else ""
    d = d[COLS]
    for c in NUM_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    for c in TEXT_COLS:
        d[c] = d[c].fillna("").astype(str)
    d["STATUS"] = np.where(d["STATUS"] == "Graded", "Graded", "Pending")
    return d[d["PLAYER"].str.strip() != ""].reset_index(drop=True)


def _row_key(d: pd.DataFrame) -> pd.Series:
    """Identity of a prop: player|stat|line|game date|season|week; with no date and no week, the day it was evaluated."""
    undated = (d["GAME_DATE"] == "") & d["WEEK"].isna()
    day = np.where(undated, "~" + d["TS_ET"].str[:10], "")
    num = lambda c: d[c].fillna(-1).round(3).astype(str)          # NaN-safe text (astype(str) keeps NaN on newer pandas)
    return (d["PLAYER"] + "|" + d["STAT"] + "|" + num("LINE") + "|" + d["GAME_DATE"] + "|" + num("SEASON") + "|" + num("WEEK")
            + "|" + pd.Series(day, index=d.index))


def upsert(existing: pd.DataFrame, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Add rows; a newer snapshot of the same prop replaces an older PENDING one. Graded rows stay."""
    new = normalize(pd.DataFrame(rows)) if rows else normalize(pd.DataFrame(columns=COLS))
    base = normalize(existing)
    allr = pd.concat([base, new], ignore_index=True)
    key = _row_key(allr)
    allr = allr[~(key.isin(set(key[allr["STATUS"] == "Graded"])) & (allr["STATUS"] == "Pending"))]
    return allr[~_row_key(allr).duplicated(keep="last")].reset_index(drop=True)


def _num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def _lookups(logs_by_player: Mapping[str, pd.DataFrame], stat_cols: Mapping[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(by_date, by_week) lookups of ACTUAL / ACTUAL_MIN per player and stat. Minutes are optional (NaN when the log has none)."""
    dates: List[pd.DataFrame] = []
    weeks: List[pd.DataFrame] = []
    for name, df in logs_by_player.items():
        if df is None or df.empty:
            continue
        mins = _num(df["MINUTES"]) if "MINUTES" in df.columns else pd.Series(np.nan, index=df.index)
        for label, col in stat_cols.items():
            if col not in df.columns:
                continue
            base = pd.DataFrame({"PLAYER": name, "STAT": label, "ACTUAL": _num(df[col]), "ACTUAL_MIN": mins})
            if "DATE" in df.columns:
                dates.append(base.assign(DATE=df["DATE"].astype(str)))
            if {"SEASON", "WEEK"} <= set(df.columns):
                weeks.append(base.assign(SEASON=_num(df["SEASON"]), WEEK=_num(df["WEEK"])).dropna(subset=["SEASON", "WEEK"]))
    empty_d = pd.DataFrame(columns=["PLAYER", "STAT", "DATE", "ACTUAL", "ACTUAL_MIN"])
    empty_w = pd.DataFrame(columns=["PLAYER", "STAT", "SEASON", "WEEK", "ACTUAL", "ACTUAL_MIN"])
    by_date = pd.concat(dates, ignore_index=True).drop_duplicates(["PLAYER", "STAT", "DATE"]) if dates else empty_d
    by_week = pd.concat(weeks, ignore_index=True).drop_duplicates(["PLAYER", "STAT", "SEASON", "WEEK"]) if weeks else empty_w
    return by_date, by_week


def grade(df: pd.DataFrame, fetch_logs: Callable[[str], Tuple[pd.DataFrame, Optional[str]]], stat_cols: Mapping[str, str],
          today: str) -> Tuple[pd.DataFrame, int, List[str]]:
    """Grade pending snapshots. `fetch_logs(name)` -> (game log; error). Dated rows grade once their date has arrived and the
    log shows that game (same ET date or the next UTC date); rows with a WEEK match the log's (SEASON, WEEK)."""
    d = normalize(df)
    dated = (d["GAME_DATE"] != "") & (d["GAME_DATE"] <= today)
    due = d[(d["STATUS"] == "Pending") & (dated | d["WEEK"].notna())]
    if due.empty:
        return d, 0, []
    notes: List[str] = []
    logs: Dict[str, pd.DataFrame] = {}
    for name in due["PLAYER"].unique():
        try:
            lg, err = fetch_logs(name)
        except Exception as e:  # network / parse failure must not stop grading the others
            lg, err = pd.DataFrame(), str(e)
        if err:
            notes.append(f"{name}: {err}")
        logs[name] = lg
    by_date, by_week = _lookups(logs, stat_cols)
    nxt = (pd.to_datetime(due["GAME_DATE"], errors="coerce") + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    k = due[["PLAYER", "STAT", "GAME_DATE", "SEASON", "WEEK"]].assign(NEXT=nxt).reset_index(drop=True)
    a = k.merge(by_date.rename(columns={"DATE": "GAME_DATE"}), on=["PLAYER", "STAT", "GAME_DATE"], how="left")
    b = k.merge(by_date.rename(columns={"DATE": "NEXT"}), on=["PLAYER", "STAT", "NEXT"], how="left")
    w = k.merge(by_week, on=["PLAYER", "STAT", "SEASON", "WEEK"], how="left")
    use_week = k["WEEK"].notna().to_numpy()
    na, nb, nw = _num(a["ACTUAL"]), _num(b["ACTUAL"]), _num(w["ACTUAL"])
    by_d = na.where(na.notna(), nb)
    act = np.where(use_week, nw, by_d).astype(float)
    mn_d = _num(a["ACTUAL_MIN"]).where(na.notna(), _num(b["ACTUAL_MIN"]))
    amin = np.where(use_week, _num(w["ACTUAL_MIN"]), mn_d).astype(float)
    found = ~np.isnan(act)
    idx = due.index[found]
    if len(idx):
        d.loc[idx, "ACTUAL"] = act[found]
        d.loc[idx, "ACTUAL_MIN"] = amin[found]
        d.loc[idx, "STATUS"] = "Graded"
        d = _derive(d)
    return d, int(len(idx)), notes


def _result(actual: pd.Series, line: pd.Series, side: pd.Series) -> pd.Series:
    """Hit / Miss / Push for a chosen side."""
    over_hit = actual > line
    hit = np.where(side == "Over", over_hit, np.where(side == "Under", actual < line, False))
    out = np.where(actual == line, "Push", np.where(side.isin(["Over", "Under"]), np.where(hit, "Hit", "Miss"), ""))
    return pd.Series(out, index=actual.index)


def _derive(d: pd.DataFrame) -> pd.DataFrame:
    """Error columns, side results and the heuristic WHY for graded rows."""
    d = d.copy()
    g = d["STATUS"] == "Graded"
    d.loc[g, "STAT_ERR"] = d.loc[g, "ACTUAL"] - d.loc[g, "MU"]
    d.loc[g, "MIN_ERR"] = d.loc[g, "ACTUAL_MIN"] - d.loc[g, "PROJ_MIN"]
    d.loc[g, "MIN_ERR_PCT"] = d.loc[g, "MIN_ERR"] / d.loc[g, "PROJ_MIN"].where(d.loc[g, "PROJ_MIN"] > 0)
    d.loc[g, "SHOWN_RESULT"] = _result(d.loc[g, "ACTUAL"], d.loc[g, "LINE"], d.loc[g, "SHOWN_SIDE"])
    d.loc[g, "EV_RESULT"] = _result(d.loc[g, "ACTUAL"], d.loc[g, "LINE"], d.loc[g, "EV_SIDE"])
    miss = g & (d["SHOWN_RESULT"] == "Miss")
    has_min = d["PROJ_MIN"].notna() & d["ACTUAL_MIN"].notna()
    big_min = has_min & (d["MIN_ERR_PCT"].abs() >= MIN_MISS_PCT)
    d.loc[g, "WHY"] = ""
    d.loc[g & (d["SHOWN_RESULT"] == "Hit"), "WHY"] = "Hit"
    d.loc[g & (d["SHOWN_RESULT"] == "Push"), "WHY"] = "Push"
    d.loc[miss & big_min & (d["MIN_ERR"] < 0), "WHY"] = "Minutes miss: played fewer than projected"
    d.loc[miss & big_min & (d["MIN_ERR"] > 0), "WHY"] = "Minutes miss: played more than projected"
    d.loc[miss & has_min & ~big_min, "WHY"] = "Minutes as projected: shooting / usage variance"
    no_min = "Missed: actual " + d["ACTUAL"].round(1).astype(str) + " vs projected " + d["MU"].round(1).astype(str)
    d.loc[miss & ~has_min, "WHY"] = no_min[miss & ~has_min]
    return d


def by_tier(d: pd.DataFrame) -> pd.DataFrame:
    """Graded results per displayed tier: picks, hit rate (pushes excluded), average minutes error, share of misses that were minutes misses."""
    g = d[(d["STATUS"] == "Graded") & d["SHOWN_RESULT"].isin(["Hit", "Miss"])]
    if g.empty:
        return pd.DataFrame(columns=["TIER", "PICKS", "HIT_%", "AVG_MIN_ERR", "MINUTES_MISSES_%_OF_MISSES"])
    rows = []
    for tier, s in g.groupby(g["TIER_SHOWN"].replace("", "n/a")):
        miss = s[s["SHOWN_RESULT"] == "Miss"]
        rows.append({"TIER": tier, "PICKS": len(s), "HIT_%": 100.0 * (s["SHOWN_RESULT"] == "Hit").mean(),
                     "AVG_MIN_ERR": s["MIN_ERR"].mean(),
                     "MINUTES_MISSES_%_OF_MISSES": 100.0 * miss["WHY"].str.startswith("Minutes miss").mean() if len(miss) else np.nan})
    return pd.DataFrame(rows).sort_values("TIER").reset_index(drop=True)


def load(path: Optional[str] = None) -> pd.DataFrame:
    """Read the snapshot file (empty frame if missing or unreadable)."""
    path = path or SNAP_PATH
    try:
        if os.path.exists(path):
            return normalize(pd.read_csv(path, dtype=str))
    except Exception:
        pass
    return normalize(pd.DataFrame(columns=COLS))


def save(df: pd.DataFrame, path: Optional[str] = None) -> None:
    """Write atomically; failures are swallowed (the in-session copy and the download button remain)."""
    path = path or SNAP_PATH
    try:
        tmp = f"{path}.tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception:
        pass
