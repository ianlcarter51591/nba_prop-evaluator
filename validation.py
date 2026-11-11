"""
Is the six-point system working?  Analytics over graded prediction snapshots (shared by the NBA and NFL apps).

Math
----
Hit rate h = hits / n (pushes excluded). 95% Wilson interval for a proportion:
    centre = (h + z^2/2n) / (1 + z^2/n),   half = z * sqrt(h(1-h)/n + z^2/4n^2) / (1 + z^2/n),   z = 1.96
Calibration: picks are bucketed by the model's probability for the displayed side; a calibrated model's actual hit rate sits
near the bucket's average model probability.
Brier score = mean((p - hit)^2), lower is better; reported for the model and for the devigged market on the same picks.
Criterion value: hit rate of picks where the criterion passed minus where it failed (displayed side). With a small n on
either side the difference is noise, so each row carries both counts and intervals.
Required columns: STATUS, SHOWN_SIDE, SHOWN_RESULT, SIX_OVER, SIX_UNDER, P_OVER, P_UNDER, PM_OVER, PM_UNDER, FAILED_OVER, FAILED_UNDER.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

Z95 = 1.959964
MIN_PICKS = 50                      # below this, every table is labelled "too early"
PROB_BINS: List[float] = [0.0, 50.0, 55.0, 60.0, 65.0, 70.0, 100.01]
PROB_LABELS: List[str] = ["under 50%", "50-55%", "55-60%", "60-65%", "65-70%", "70%+"]


def wilson(hits: np.ndarray, n: np.ndarray, z: float = Z95) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised 95% Wilson interval (as fractions). n == 0 gives (nan, nan)."""
    hits, n = np.asarray(hits, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = hits / n
        denom = 1.0 + z ** 2 / n
        centre = (h + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(h * (1 - h) / n + z ** 2 / (4 * n ** 2)) / denom
    return np.where(n > 0, centre - half, np.nan), np.where(n > 0, centre + half, np.nan)


def graded(d: pd.DataFrame) -> pd.DataFrame:
    """Graded, decided picks with HIT (0/1), SCORE, P and PM (model / market % for the displayed side), FAILED (text)."""
    need = ["STATUS", "SHOWN_SIDE", "SHOWN_RESULT"]
    if d is None or d.empty or any(c not in d.columns for c in need):
        return pd.DataFrame(columns=["HIT", "SCORE", "P", "PM", "FAILED"])
    g = d[(d["STATUS"] == "Graded") & d["SHOWN_RESULT"].isin(["Hit", "Miss"]) & d["SHOWN_SIDE"].isin(["Over", "Under"])].copy()
    over = g["SHOWN_SIDE"].eq("Over")
    pick = lambda a, b: pd.to_numeric(g[a], errors="coerce").where(over, pd.to_numeric(g[b], errors="coerce")) if a in g and b in g else np.nan
    g["HIT"] = g["SHOWN_RESULT"].eq("Hit").astype(float)
    g["SCORE"], g["P"], g["PM"] = pick("SIX_OVER", "SIX_UNDER"), pick("P_OVER", "P_UNDER"), pick("PM_OVER", "PM_UNDER")
    g["FAILED"] = (g["FAILED_OVER"].where(over, g["FAILED_UNDER"]) if "FAILED_OVER" in g else "")
    g["FAILED"] = g["FAILED"].fillna("").astype(str)
    return g


def _rate_table(g: pd.DataFrame, key: pd.Series, name: str) -> pd.DataFrame:
    """PICKS, HIT_%, 95% interval, average model and market % per group."""
    t = g.groupby(key, observed=True).agg(PICKS=("HIT", "size"), HITS=("HIT", "sum"), AVG_MODEL=("P", "mean"), AVG_MARKET=("PM", "mean"))
    lo, hi = wilson(t["HITS"].to_numpy(), t["PICKS"].to_numpy())
    out = pd.DataFrame({name: t.index.astype(str), "PICKS": t["PICKS"].to_numpy(), "HIT_%": 100.0 * t["HITS"].to_numpy() / t["PICKS"].to_numpy(),
                        "LOW_%": 100.0 * lo, "HIGH_%": 100.0 * hi, "AVG_MODEL_%": t["AVG_MODEL"].to_numpy(), "AVG_MARKET_%": t["AVG_MARKET"].to_numpy()})
    return out.reset_index(drop=True)


def by_score(d: pd.DataFrame) -> pd.DataFrame:
    """Hit rate by six-point score of the displayed side (6, 5, 4, 3 or less)."""
    g = graded(d).dropna(subset=["SCORE"])
    if g.empty:
        return pd.DataFrame(columns=["SCORE", "PICKS", "HIT_%", "LOW_%", "HIGH_%", "AVG_MODEL_%", "AVG_MARKET_%"])
    label = np.select([g["SCORE"] >= 6, g["SCORE"] == 5, g["SCORE"] == 4], ["6/6 SHARP", "5/6 LEAN", "4/6"], default="3/6 or less")
    t = _rate_table(g, pd.Series(label, index=g.index), "SCORE")
    order = {"6/6 SHARP": 0, "5/6 LEAN": 1, "4/6": 2, "3/6 or less": 3}
    return t.sort_values("SCORE", key=lambda s: s.map(order)).reset_index(drop=True)


def calibration(d: pd.DataFrame) -> pd.DataFrame:
    """Actual hit rate by model-probability bucket for the displayed side."""
    g = graded(d).dropna(subset=["P"])
    if g.empty:
        return pd.DataFrame(columns=["MODEL_PROB", "PICKS", "HIT_%", "LOW_%", "HIGH_%", "AVG_MODEL_%", "AVG_MARKET_%"])
    return _rate_table(g, pd.cut(g["P"], PROB_BINS, right=False, labels=PROB_LABELS), "MODEL_PROB")


def criteria_labels(g: pd.DataFrame) -> List[str]:
    """Every criterion label seen failing at least once (a criterion that never failed can't be evaluated)."""
    parts = g["FAILED"].str.split(";").explode().str.replace("(unverified)", "", regex=False).str.strip()
    return sorted(x for x in parts.dropna().unique() if x)


def by_criterion(d: pd.DataFrame) -> pd.DataFrame:
    """For each criterion: hit rate when it passed vs failed (displayed side), with counts and the difference in points."""
    g = graded(d)
    cols = ["CRITERION", "PASSED_PICKS", "PASSED_HIT_%", "FAILED_PICKS", "FAILED_HIT_%", "DIFF_PP", "READ"]
    if g.empty:
        return pd.DataFrame(columns=cols)
    rows: List[Dict[str, object]] = []
    hit = g["HIT"].to_numpy()
    tokens = g["FAILED"].str.split(";").explode().str.replace("(unverified)", "", regex=False).str.strip()   # exact labels, not substrings
    for label in criteria_labels(g):
        failed = g.index.isin(tokens.index[tokens == label])
        n_f, n_p = int(failed.sum()), int((~failed).sum())
        h_f = float(hit[failed].mean() * 100.0) if n_f else np.nan
        h_p = float(hit[~failed].mean() * 100.0) if n_p else np.nan
        if min(n_f, n_p) < 20:
            read = "too few picks on one side"
        else:
            (lo_p, hi_p), (lo_f, hi_f) = wilson(hit[~failed].sum(), n_p), wilson(hit[failed].sum(), n_f)
            read = "helps" if lo_p > hi_f else "hurts" if hi_p < lo_f else "no clear difference yet"
        rows.append({"CRITERION": label, "PASSED_PICKS": n_p, "PASSED_HIT_%": h_p, "FAILED_PICKS": n_f, "FAILED_HIT_%": h_f,
                     "DIFF_PP": h_p - h_f if n_f and n_p else np.nan, "READ": read})
    return pd.DataFrame(rows, columns=cols)


def brier(d: pd.DataFrame) -> Dict[str, float]:
    """Model vs market Brier score on picks that have both probabilities; plus counts and overall hit rate."""
    g = graded(d)
    both = g.dropna(subset=["P", "PM"])
    nan = float("nan")
    return {"picks": float(len(g)), "hit_pct": float(g["HIT"].mean() * 100.0) if len(g) else nan, "n_both": float(len(both)),
            "model": float(np.mean((both["P"] / 100.0 - both["HIT"]) ** 2)) if len(both) else nan,
            "market": float(np.mean((both["PM"] / 100.0 - both["HIT"]) ** 2)) if len(both) else nan}


def verdict(d: pd.DataFrame) -> List[str]:
    """Plain-English reading of the tables (cautious by design)."""
    g = graded(d)
    n = len(g)
    if n == 0:
        return ["No graded picks yet. Grade finished games on this tab and the tables fill in."]
    out: List[str] = []
    if n < MIN_PICKS:
        out.append(f"Too early to judge: {n} graded pick(s). Treat everything below as noise until about {MIN_PICKS}.")
    t = by_score(d).set_index("SCORE")
    top = [s for s in ("6/6 SHARP", "5/6 LEAN") if s in t.index]
    low = [s for s in ("4/6", "3/6 or less") if s in t.index]
    if top and low:
        n_top, n_low = t.loc[top, "PICKS"].sum(), t.loc[low, "PICKS"].sum()
        h_top = (t.loc[top, "HIT_%"] * t.loc[top, "PICKS"]).sum() / n_top
        h_low = (t.loc[low, "HIT_%"] * t.loc[low, "PICKS"]).sum() / n_low
        (lo_t, hi_t), (lo_l, hi_l) = wilson(h_top / 100 * n_top, n_top), wilson(h_low / 100 * n_low, n_low)
        if lo_t > hi_l:
            out.append(f"SHARP/LEAN picks hit {h_top:.0f}% vs {h_low:.0f}% for lower scores, and the gap is larger than chance would explain.")
        elif hi_t < lo_l:
            out.append(f"Warning: SHARP/LEAN picks hit {h_top:.0f}%, clearly below lower scores at {h_low:.0f}%. The scoring may be mis-weighted.")
        else:
            out.append(f"SHARP/LEAN picks hit {h_top:.0f}% ({int(n_top)} picks) vs {h_low:.0f}% ({int(n_low)}) for lower scores: not a clear difference yet.")
    b = brier(d)
    if b["n_both"] >= MIN_PICKS:
        better = "better than" if b["model"] < b["market"] else "worse than"
        out.append(f"Probability accuracy (Brier, lower is better): model {b['model']:.4f}, {better} the market's {b['market']:.4f} on {int(b['n_both'])} picks.")
    return out
