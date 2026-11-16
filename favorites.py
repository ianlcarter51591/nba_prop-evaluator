"""
Favorite props: star a saved prop (a pending snapshot) and the parlay builder offers only those.

A favorite is the FAV == "Y" flag on a snapshot row, so it is saved and restored with the snapshots. Re-evaluating a favorited prop keeps the
star (see snapshots.upsert). Only PENDING rows can be starred or used as legs; once a prop is graded it drops out of the parlay list.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set

import numpy as np
import pandas as pd


def _with_fav(snaps: Optional[pd.DataFrame]) -> pd.DataFrame:
    d = pd.DataFrame() if snaps is None else snaps.copy()
    if "FAV" not in d.columns:
        d["FAV"] = ""
    d["FAV"] = d["FAV"].fillna("").astype(str)
    return d


def _where(row: Any) -> str:
    g = str(row.get("GAME", "") or "").strip()
    return g or str(row.get("GAME_DATE", "") or "").strip()


def option_table(snaps: Optional[pd.DataFrame], stat: Optional[str] = None) -> pd.DataFrame:
    """Pending props that can be starred: LABEL (unique), SNAP_ID, FAV. `stat` limits it to one stat label."""
    cols = ["LABEL", "SNAP_ID", "FAV"]
    d = _with_fav(snaps)
    need = {"STATUS", "PLAYER", "STAT", "LINE", "SNAP_ID"}
    if d.empty or not need <= set(d.columns):
        return pd.DataFrame(columns=cols)
    d = d[d["STATUS"] == "Pending"]
    if stat:
        d = d[d["STAT"].astype(str).str.upper() == str(stat).upper()]
    if d.empty:
        return pd.DataFrame(columns=cols)
    label = d["PLAYER"].astype(str) + " · " + d["STAT"].astype(str) + " " + pd.to_numeric(d["LINE"], errors="coerce").map(lambda x: f"{x:g}") \
        + d.apply(lambda r: (" · " + _where(r)) if _where(r) else "", axis=1)
    out = pd.DataFrame({"LABEL": label.values, "SNAP_ID": d["SNAP_ID"].values, "FAV": d["FAV"].values})
    dup = out["LABEL"].duplicated(keep=False)
    out.loc[dup, "LABEL"] = out.loc[dup, "LABEL"] + " [" + out.loc[dup, "SNAP_ID"].astype(str).str[:4] + "]"
    return out.sort_values("LABEL").reset_index(drop=True)[cols]


def set_flags(snaps: pd.DataFrame, favorite_ids: Iterable[str], scope_ids: Iterable[str]) -> pd.DataFrame:
    """Copy of `snaps` where rows in `scope_ids` are starred iff they are in `favorite_ids`; other rows are untouched."""
    d = _with_fav(snaps)
    scope, fav = set(scope_ids), set(favorite_ids)
    inside = d["SNAP_ID"].isin(scope)
    d.loc[inside, "FAV"] = np.where(d.loc[inside, "SNAP_ID"].isin(fav), "Y", "")
    return d


def find_id(snaps: Optional[pd.DataFrame], player: str, stat: str, line: float) -> Optional[str]:
    """SNAP_ID of the latest pending snapshot of this prop (None if it hasn't been saved)."""
    d = _with_fav(snaps)
    if d.empty or "SNAP_ID" not in d.columns:
        return None
    m = d[(d["STATUS"] == "Pending") & (d["PLAYER"].astype(str).str.lower() == str(player).lower())
          & (d["STAT"].astype(str).str.upper() == str(stat).upper()) & ((pd.to_numeric(d["LINE"], errors="coerce") - float(line)).abs() < 1e-6)]
    return None if m.empty else str(m.iloc[-1]["SNAP_ID"])


def is_favorite(snaps: Optional[pd.DataFrame], snap_id: Optional[str]) -> bool:
    d = _with_fav(snaps)
    return bool(snap_id) and bool(len(d)) and bool((d.loc[d["SNAP_ID"] == snap_id, "FAV"] == "Y").any())


def toggle(snaps: pd.DataFrame, snap_id: str) -> pd.DataFrame:
    """Flip the star on one snapshot."""
    d = _with_fav(snaps)
    on = d["SNAP_ID"] == snap_id
    d.loc[on, "FAV"] = np.where(d.loc[on, "FAV"] == "Y", "", "Y")
    return d


def favorite_ids(snaps: Optional[pd.DataFrame]) -> Set[str]:
    d = _with_fav(snaps)
    return set() if d.empty or "SNAP_ID" not in d.columns else set(d.loc[(d["FAV"] == "Y") & (d["STATUS"] == "Pending"), "SNAP_ID"])


def legs(snaps: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Parlay legs (Over and Under) of favorited pending props only. Adds EDGE_PP (model minus market, points) and APP_PICK (the side the app shows)."""
    import parlay as pl
    cols = ["LABEL", "GAME", "P_MODEL", "P_MARKET", "ODDS", "LINK", "EDGE_PP", "APP_PICK"]
    d = _with_fav(snaps)
    if d.empty or "STATUS" not in d.columns:
        return pd.DataFrame(columns=cols)
    f = d[(d["FAV"] == "Y") & (d["STATUS"] == "Pending")]
    out = pl.legs_from_snapshots(f)
    if out.empty:
        return pd.DataFrame(columns=cols)
    out = out.copy()
    out["EDGE_PP"] = (out["P_MODEL"] - out["P_MARKET"]) * 100.0
    shown = {f"{r['PLAYER']} {r['SHOWN_SIDE']} {float(r['LINE']):g} {r['STAT']}" for _, r in f.iterrows() if str(r.get("SHOWN_SIDE", "")) in ("Over", "Under")} \
        if "SHOWN_SIDE" in f.columns else set()
    out["APP_PICK"] = out["LABEL"].isin(shown)
    return out.sort_values(["APP_PICK", "LABEL"], ascending=[False, True]).reset_index(drop=True)[cols]
