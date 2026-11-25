"""
Shadow models, run beside the current model (never shown as the pick) and saved with each snapshot so they can be compared
on graded props before switching. Both split scoring into attempts (fast-moving: they follow role) and make % (slow: shrunk
hard toward last season and the league).

Make % for each shot type:
    p0 = (makes last season + PRIOR_ATT * league %) / (attempts last season + PRIOR_ATT)
    p  = (makes this season + K * p0) / (attempts this season + K)          K by shot type (threes settle slowest)
    p_i ~ Beta(p * (A + K), (1 - p) * (A + K))                               uncertainty in his true %

3-pointers made:   makes_i = Binomial(3PA_i, p3_i),  3PA_i from the app's minutes-aware simulation.
Points:            minutes_i (shared with the main model) and a usage shock u_i ~ Gamma(mean 1, cv_u) scale all shots:
                   FGA_i ~ Poisson(mu_FGA * minutes_i / m0 * u_i);   3PA_i ~ Binomial(FGA_i, share of shots that are threes)
                   FTA_i ~ Poisson(mu_FTA * minutes_i / m0 * u_i)
                   points_i = 2 * Bin(FGA_i - 3PA_i, p2_i) + 3 * Bin(3PA_i, p3_i) + Bin(FTA_i, pft_i)
                   cv_u^2 = max(cv(FGA per minute)^2 - 1 / mean FGA, CV_U_MIN^2)   (spread beyond Poisson shot-count noise)
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

PRIOR_ATT = 100.0               # last season's % is pulled toward the league with 100 attempts' weight
SHOT: Dict[str, Tuple[float, float]] = {"3": (0.36, 250.0), "2": (0.54, 150.0), "FT": (0.78, 150.0)}   # (league %, K attempts)
CV_U_MIN = 0.05
THREES, POINTS = "3-POINTERS MADE", "POINTS"
MIN_COMPARE = 150               # graded props per stat before the comparison is worth acting on
COMPARE_STATS: Tuple[Tuple[str, str], ...] = (("Points", "POINTS"), ("3-Pointers Made", "3-POINT"))


def shrunk_pct(cur_made: float, cur_att: float, pri_made: float, pri_att: float, shot: str = "3") -> Tuple[float, float]:
    """(shrunk make %, effective attempts behind it) for shot type '3', '2' or 'FT'."""
    league, k = SHOT[shot]
    p0 = (float(pri_made) + PRIOR_ATT * league) / (float(pri_att) + PRIOR_ATT)
    p = (float(cur_made) + k * p0) / (float(cur_att) + k)
    return float(np.clip(p, 0.05, 0.98)), float(cur_att) + k


def makes(att: np.ndarray, p: float, n_eff: float, rng: np.random.Generator) -> np.ndarray:
    """Made shots for each simulated attempt count, with the make % itself uncertain (Beta around p)."""
    p_i = rng.beta(max(p * n_eff, 1e-3), max((1.0 - p) * n_eff, 1e-3), len(att))
    return rng.binomial(np.maximum(np.rint(att), 0).astype(int), p_i).astype(float)


def makes_from_attempts(att: np.ndarray, p: float, n_eff: float, seed: int = 7) -> np.ndarray:
    """Threes made from simulated 3PA."""
    return makes(att, p, n_eff, np.random.default_rng(seed))


def summarize(draws: np.ndarray, line: float, **extra: float) -> Dict[str, float]:
    """Mean and conditional over / under % (pushes left out, like the main model), plus any extra fields."""
    po, pu = float(np.mean(draws > line)), float(np.mean(draws < line))
    d = max(po + pu, 1e-9)
    return {"mu": float(np.mean(draws)), "p_over": po / d * 100.0, "p_under": pu / d * 100.0, **{k: float(v) for k, v in extra.items()}}


def has_cols(cur: pd.DataFrame, pri: pd.DataFrame, cols: Tuple[str, ...]) -> bool:
    """Logs carry these columns (older cached logs don't) with some attempts in them."""
    return all(c in cur.columns and c in pri.columns for c in cols) and float(cur[cols[0]].sum() + pri[cols[0]].sum()) > 0


def has_attempts(cur: pd.DataFrame, pri: pd.DataFrame) -> bool:
    """Threes shadow can run."""
    return has_cols(cur, pri, ("3PA",))


def has_shots(cur: pd.DataFrame, pri: pd.DataFrame) -> bool:
    """Points shadow can run."""
    return has_cols(cur, pri, ("FGA", "FGM", "3PA", "3-POINTERS MADE", "FTA", "FTM"))


def usage_cv(rate_cv_fga: float, mu_fga: float) -> float:
    """Night-to-night usage swing beyond the Poisson noise of the shot count itself."""
    return float(np.sqrt(max(rate_cv_fga ** 2 - 1.0 / max(mu_fga, 1e-6), CV_U_MIN ** 2)))


def simulate_points(mins: np.ndarray, m0: float, mu_fga: float, mu_3pa: float, mu_fta: float, cv_u: float,
                    pct: Mapping[str, Tuple[float, float]], seed: int = 11) -> Dict[str, np.ndarray]:
    """Points draws from shots (see module docstring). pct = {'2': (p, n_eff), '3': ..., 'FT': ...}."""
    rng = np.random.default_rng(seed)
    k = 1.0 / cv_u ** 2
    u = rng.gamma(k, 1.0 / k, len(mins))
    scale = np.asarray(mins, float) / max(float(m0), 1.0) * u
    fga = rng.poisson(np.maximum(mu_fga * scale, 0.0))
    three = rng.binomial(fga, float(np.clip(mu_3pa / max(mu_fga, 1e-6), 0.0, 1.0)))
    fta = rng.poisson(np.maximum(mu_fta * scale, 0.0))
    m2, m3, mft = (makes(a, *pct[s], rng) for a, s in ((fga - three, "2"), (three, "3"), (fta, "FT")))
    return {"points": 2.0 * m2 + 3.0 * m3 + mft, "fga": fga.astype(float), "three": three.astype(float), "fta": fta.astype(float)}


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 0.01, 0.99)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def compare(snaps: pd.DataFrame, stat: str = "3-POINT") -> Optional[pd.DataFrame]:
    """
    Graded props of one stat with both models' over %: log loss and Brier (lower = better probabilities), hit % of the side
    each model favours, and average miss of the projection. `stat`: 'POINTS' (exact) or '3-POINT' (contains)."""
    need = {"SHADOW_P_OVER", "P_OVER", "ACTUAL", "LINE", "STAT", "STATUS"}
    if snaps is None or snaps.empty or not need <= set(snaps.columns):
        return None
    label = snaps["STAT"].astype(str).str.upper().str.strip()
    pick = label.eq(stat) if stat == "POINTS" else label.str.contains(stat, regex=False)
    d = snaps[(snaps["STATUS"] == "Graded") & pick].copy()
    for c in ("SHADOW_P_OVER", "P_OVER", "PM_OVER", "ACTUAL", "LINE", "MU", "SHADOW_MU"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["SHADOW_P_OVER", "P_OVER", "ACTUAL", "LINE"])
    d = d[d["ACTUAL"] != d["LINE"]]
    if d.empty:
        return None
    y = (d["ACTUAL"] > d["LINE"]).to_numpy(float)
    rows = []
    for name, pcol, mcol in (("Current model", "P_OVER", "MU"), ("Shot-based shadow", "SHADOW_P_OVER", "SHADOW_MU"),
                             ("Market (no-vig)", "PM_OVER", None)):
        if pcol not in d.columns or d[pcol].isna().all():
            continue
        p = d[pcol].to_numpy(float) / 100.0
        ok = ~np.isnan(p)
        miss = float(np.mean(np.abs(d[mcol] - d["ACTUAL"]))) if mcol and mcol in d.columns and d[mcol].notna().any() else np.nan
        rows.append({"MODEL": name, "PROPS": int(ok.sum()), "LOG_LOSS": _logloss(p[ok], y[ok]),
                     "BRIER": float(np.mean((p[ok] - y[ok]) ** 2)),
                     "HIT_%_SIDE_IT_FAVOURS": float(np.mean((p[ok] > 0.5) == (y[ok] == 1)) * 100.0), "AVG_MISS": miss})
    return pd.DataFrame(rows)


def render(st: Any, snaps: pd.DataFrame) -> None:
    """Snapshot-tab panel: how each shadow model is doing against the current one."""
    st.markdown("#### 🧪 Shadow test: shot-based Points and 3-Pointers")
    shown = False
    for label, key in COMPARE_STATS:
        table = compare(snaps, key)
        if table is None:
            continue
        shown = True
        st.markdown(f"**{label}**")
        st.dataframe(table.round(3), use_container_width=True, hide_index=True)
        n = int(table["PROPS"].max())
        t = table.set_index("MODEL")
        if n < MIN_COMPARE:
            st.caption(f"{n} graded {label.lower()} props so far: wait for about {MIN_COMPARE} before trusting the difference. "
                       "Lower log loss = better probabilities.")
        elif "Shot-based shadow" in t.index:
            better = t.at["Shot-based shadow", "LOG_LOSS"] < t.at["Current model", "LOG_LOSS"]
            (st.success if better else st.info)(
                f"The shot-based model has the lower log loss on {label.lower()}: it's ready to take over." if better else
                f"On {label.lower()}, the current model is still as good or better: keep it.")
    if not shown:
        st.caption("Collecting: every Points and 3-Pointers Made evaluation now also saves a shot-based projection "
                   "(attempts × make %). The comparison appears once those props are graded.")
