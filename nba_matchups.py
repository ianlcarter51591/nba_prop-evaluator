"""
Position-vs-defense matchup model for NBA props, with a no-look-ahead backtest.  (Backtest only: NOT used by the engine.)

Math
----
For a player-game with stat x and the player's leave-one-out average m:      R = x / m           (1.00 = a normal game)
For a defense o facing position group g, using ONLY games before date d:
    I_pos(o, g, d) = 1 + sum_prev(R - 1) / (N_prev + m0)       shrunk toward 1 by m0 pseudo player-games
    I_all(o, d)    = same, pooled over every position         (what the app already does, per stat)
    D              = I_pos - I_all                             the *extra* information the position split adds
Backtest regression, clustered by player:     R - 1 = a + k_all * (I_all - 1) + k_inc * D + e
    k_inc is the weight the position split deserves ON TOP of the existing all-player factor (the thing we must prove).
Chronological hold-out: fit on the earlier half of dates, score squared error on the later half for
    baseline (no matchup) / all-player index / all-player + position split.
Live multiplier (only when the backtest earns it):   pos_mult = clip(1 + k_inc_eff * (I_pos - I_all), lo, hi)
    k_inc_eff = k_inc shrunk toward 0 by its standard error (see calibrated_k), so noise never gets full weight.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

M0_POS = 30.0                       # pseudo player-games shrinking a (defense, position) index toward 1
M0_ALL = 60.0                       # same for the pooled index
MIN_HISTORY = 15                    # need this many earlier player-games in the (defense, position) cell to score a row
MIN_PLAYER_GAMES = 10
K_TAU = 0.5                         # prior sd of k_inc around 0 (k_inc = 1 means "position index is fully right")
MULT_LO, MULT_HI = 0.94, 1.06       # the extra position factor is capped tighter than the main opponent factor

# Minimum per-game average for a player to enter the backtest (keeps ratios stable).
BT_MIN_MEAN: Dict[str, float] = {
    "POINTS": 8.0, "REBOUNDS": 3.0, "ASSISTS": 2.0, "3-POINTERS MADE": 0.8, "PTS + REB + AST": 14.0,
    "PTS + AST": 10.0, "PTS + REB": 12.0, "REB + AST": 5.0, "STEALS": 0.6, "BLOCKS": 0.5,
}
GROUPS: Tuple[str, ...] = ("G", "F", "C")
_ABBR_TO_GROUP: Dict[str, str] = {"PG": "G", "SG": "G", "G": "G", "GF": "F", "SF": "F", "PF": "F", "F": "F", "FC": "C", "C": "C"}


def position_group(abbr: Optional[str]) -> Optional[str]:
    """ESPN position abbreviation -> 'G' (guards), 'F' (wings/forwards) or 'C' (bigs); None if unknown."""
    return _ABBR_TO_GROUP.get(str(abbr or "").strip().upper())


def _walk(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def positions_from_roster_json(payload: Any) -> Dict[str, str]:
    """{ESPN player id -> position group} read from one team's roster JSON (position.abbreviation)."""
    import re
    out: Dict[str, str] = {}
    for d in _walk(payload):
        m = re.search(r"~a:(\d+)", str(d.get("uid", "")))
        pos = d.get("position")
        abbr = pos.get("abbreviation") if isinstance(pos, dict) else pos
        grp = position_group(abbr)
        if m and grp:
            out[m.group(1)] = grp
    return out


def position_abbrs_from_roster_json(payload: Any) -> Dict[str, str]:
    """{ESPN player id -> raw position abbreviation (PG, SG, SF, PF, C, G, F...)} from one team's roster JSON."""
    import re
    out: Dict[str, str] = {}
    for d in _walk(payload):
        m = re.search(r"~a:(\d+)", str(d.get("uid", "")))
        pos = d.get("position")
        raw = pos.get("abbreviation") if isinstance(pos, dict) else pos
        abbr = str(raw).strip().upper() if raw else ""
        if m and abbr:
            out[m.group(1)] = abbr
    return out


def build_ratio_frame(games: pd.DataFrame, groups: Mapping[str, str], stat_cols: List[str]) -> pd.DataFrame:
    """
    Long frame (PID, DATE, OPP, GROUP, STAT, R). R = stat / leave-one-out player mean, so a game never inflates its own
    baseline. `games` needs PID, DATE, OPP and the stat columns; `groups` maps PID -> 'G'/'F'/'C'. Vectorised.
    """
    g = games.assign(GROUP=games["PID"].astype(str).map(dict(groups))).dropna(subset=["GROUP", "OPP", "DATE"])
    g = g[g.groupby("PID")["DATE"].transform("size") >= MIN_PLAYER_GAMES]
    parts: List[pd.DataFrame] = []
    for s in stat_cols:
        if s not in g:
            continue
        x = pd.to_numeric(g[s], errors="coerce")
        n = x.groupby(g["PID"]).transform("count")
        loo = (x.groupby(g["PID"]).transform("sum") - x) / (n - 1).where(n > 1)
        ok = loo >= BT_MIN_MEAN.get(s, 0.5)
        parts.append(pd.DataFrame({"PID": g["PID"], "DATE": g["DATE"], "OPP": g["OPP"], "GROUP": g["GROUP"],
                                   "STAT": s, "R": (x / loo)})[ok & x.notna()])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["PID", "DATE", "OPP", "GROUP", "STAT", "R"])


def expanding_index(df: pd.DataFrame, keys: List[str], m0: float) -> pd.DataFrame:
    """Index per (keys, DATE) from STRICTLY EARLIER dates only: 1 + sum(R-1)/(n + m0). Returns keys+DATE+IDX+N."""
    d = df.assign(E=df["R"] - 1.0)
    daily = d.groupby(keys + ["DATE"], as_index=False).agg(S=("E", "sum"), N=("E", "size")).sort_values(keys + ["DATE"])
    grp = daily.groupby(keys)
    daily["CS"] = grp["S"].cumsum() - daily["S"]
    daily["CN"] = grp["N"].cumsum() - daily["N"]
    daily["IDX"] = 1.0 + daily["CS"] / (daily["CN"] + m0)
    return daily[keys + ["DATE", "IDX", "CN"]].rename(columns={"CN": "N"})


def _ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """OLS with cluster-robust (CR1) standard errors; returns (beta, se)."""
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    e = y - X @ beta
    scores = pd.DataFrame(X * e[:, None]).groupby(groups).sum().to_numpy()
    n_g, n, k = scores.shape[0], len(y), X.shape[1]
    v = xtx_inv @ (scores.T @ scores) @ xtx_inv * (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - k, 1))
    return beta, np.sqrt(np.clip(np.diag(v), 0.0, None))


def _scored_rows(frame: pd.DataFrame, stat: str, m0_pos: float, m0_all: float) -> pd.DataFrame:
    """Rows of one stat with I_all, I_pos, D, each computed without look-ahead."""
    sub = frame[frame["STAT"] == stat]
    if sub.empty:
        return sub.assign(I_ALL=np.nan, I_POS=np.nan, D=np.nan, N_POS=np.nan)
    ia = expanding_index(sub, ["OPP"], m0_all).rename(columns={"IDX": "I_ALL"}).drop(columns="N")
    ip = expanding_index(sub, ["OPP", "GROUP"], m0_pos).rename(columns={"IDX": "I_POS", "N": "N_POS"})
    d = sub.merge(ia, on=["OPP", "DATE"], how="left").merge(ip, on=["OPP", "GROUP", "DATE"], how="left")
    d = d[d["N_POS"] >= MIN_HISTORY].copy()
    d["D"] = d["I_POS"] - d["I_ALL"]
    return d


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]) -> float:
    """Hold-out mean squared error of E = R - 1 for a linear model on `cols` (empty = intercept only)."""
    Xtr = np.column_stack([np.ones(len(train))] + [train[c].to_numpy(float) for c in cols])
    Xte = np.column_stack([np.ones(len(test))] + [test[c].to_numpy(float) for c in cols])
    beta = np.linalg.lstsq(Xtr, train["E"].to_numpy(float), rcond=None)[0]
    return float(np.mean((test["E"].to_numpy(float) - Xte @ beta) ** 2))


def position_backtest(frame: pd.DataFrame, stat: str, m0_pos: float = M0_POS, m0_all: float = M0_ALL) -> Optional[Dict[str, Any]]:
    """Estimate k_all / k_inc and the chronological hold-out gain for one stat. None if the sample is too small."""
    d = _scored_rows(frame, stat, m0_pos, m0_all)
    if len(d) < 400 or d["PID"].nunique() < 30:
        return None
    d = d.assign(E=d["R"] - 1.0, XA=d["I_ALL"] - 1.0).sort_values("DATE")
    X = np.column_stack([np.ones(len(d)), d["XA"].to_numpy(float), d["D"].to_numpy(float)])
    beta, se = _ols_cluster(X, d["E"].to_numpy(float), d["PID"].to_numpy())
    cut = d["DATE"].iloc[len(d) // 2]
    tr, te = d[d["DATE"] < cut], d[d["DATE"] >= cut]
    out: Dict[str, Any] = {"STAT": stat, "N": int(len(d)), "PLAYERS": int(d["PID"].nunique()),
                           "K_ALL": float(beta[1]), "SE_ALL": float(se[1]), "K_INC": float(beta[2]), "SE_INC": float(se[2]),
                           "T_INC": float(beta[2] / se[2]) if se[2] > 0 else 0.0, "SD_D": float(d["D"].std())}
    if len(tr) >= 200 and len(te) >= 200:
        base = _fit_predict(tr, te, [])
        out["GAIN_ALL_%"] = float((1.0 - _fit_predict(tr, te, ["XA"]) / base) * 100.0)
        out["GAIN_BOTH_%"] = float((1.0 - _fit_predict(tr, te, ["XA", "D"]) / base) * 100.0)
        out["GAIN_POS_ONLY_%"] = float(out["GAIN_BOTH_%"] - out["GAIN_ALL_%"])
    else:
        out.update({"GAIN_ALL_%": np.nan, "GAIN_BOTH_%": np.nan, "GAIN_POS_ONLY_%": np.nan})
    return out


def calibrated_k(k_hat: float, se: float, k0: float = 0.0, tau: float = K_TAU) -> float:
    """Shrink k_hat toward k0: k0 + (k_hat - k0) * tau^2 / (tau^2 + se^2), clipped to [0, 1]."""
    return float(np.clip(k0 + (k_hat - k0) * tau ** 2 / (tau ** 2 + se ** 2), 0.0, 1.0))


def run_position_backtests(frame: pd.DataFrame, stats: Optional[List[str]] = None) -> pd.DataFrame:
    """Backtest every stat; adds shrunk weight K_EFF and a plain-English verdict (use only if t>=2 AND hold-out gain>0)."""
    rows: List[Dict[str, Any]] = []
    for s in (stats or list(BT_MIN_MEAN)):
        r = position_backtest(frame, s)
        if r is None:
            continue
        r["K_EFF"] = calibrated_k(r["K_INC"], r["SE_INC"])
        r["VERDICT"] = "USE" if (r["T_INC"] >= 2.0 and (r["GAIN_POS_ONLY_%"] > 0 if np.isfinite(r["GAIN_POS_ONLY_%"]) else False)) else "DON'T USE"
        rows.append(r)
    return pd.DataFrame(rows)


def position_index_now(frame: pd.DataFrame, stat: str, m0_pos: float = M0_POS, m0_all: float = M0_ALL) -> pd.DataFrame:
    """Latest (all-history) shrunk indices per defense and position group for live use: OPP, GROUP, I_POS, I_ALL, D, RANK (1 = softest)."""
    sub = frame[frame["STAT"] == stat].assign(E=lambda x: x["R"] - 1.0)
    if sub.empty:
        return pd.DataFrame(columns=["OPP", "GROUP", "I_POS", "I_ALL", "D", "RANK"])
    ia = (1.0 + sub.groupby("OPP")["E"].sum() / (sub.groupby("OPP")["E"].size() + m0_all)).rename("I_ALL")
    ip = (1.0 + sub.groupby(["OPP", "GROUP"])["E"].sum() / (sub.groupby(["OPP", "GROUP"])["E"].size() + m0_pos)).rename("I_POS")
    t = ip.reset_index().merge(ia.reset_index(), on="OPP")
    t["D"] = t["I_POS"] - t["I_ALL"]
    t["RANK"] = t.groupby("GROUP")["I_POS"].rank(ascending=False, method="min").astype(int)
    return t.sort_values(["GROUP", "RANK"]).reset_index(drop=True)


def position_rank_table(frame: pd.DataFrame, stat: str, m0_pos: float = M0_POS) -> pd.DataFrame:
    """Defense vs position group for one stat: OPP, GROUP, I_POS (shrunk toward 1), N (player-games), RANK (1 = softest in that group)."""
    cols = ["OPP", "GROUP", "I_POS", "N", "RANK"]
    sub = frame[frame["STAT"] == stat].assign(E=lambda x: x["R"] - 1.0)
    if sub.empty:
        return pd.DataFrame(columns=cols)
    g = sub.groupby(["OPP", "GROUP"])["E"].agg(S="sum", N="size").reset_index()
    g["I_POS"] = 1.0 + g["S"] / (g["N"] + m0_pos)
    g["RANK"] = g.groupby("GROUP")["I_POS"].rank(ascending=False, method="min").astype(int)
    return g[cols].sort_values(["GROUP", "RANK"]).reset_index(drop=True)


def position_multiplier(i_pos: float, i_all: float, k_inc: float) -> float:
    """Extra factor from the position split, applied on top of the existing all-player factor (never instead of it)."""
    return float(np.clip(1.0 + k_inc * (i_pos - i_all), MULT_LO, MULT_HI))
