"""Streamlit screens shared by the NBA and NFL apps: storage status, six-point validation, parlay evaluator.

Every function takes the `streamlit` module as `st`, so this file imports nothing from either app.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import cloud_store
import parlay as pl
import validation as val


def _num(x: Any, default: float) -> float:
    """float(x) or the default (widgets can return None)."""
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _odds_text(x: float) -> str:
    """American odds as text, e.g. +581 / -110."""
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.0f}"


def render_storage_status(st: Any) -> None:
    """One line saying where the logs are being saved."""
    s = cloud_store.status()
    (st.caption if s["level"] == "ok" else st.warning)(("💾 " if s["level"] == "ok" else "⚠️ ") + s["text"])


def render_validation(st: Any, snaps: pd.DataFrame) -> None:
    """'Is the six-point system working?' tables from graded snapshots."""
    st.markdown("### 🧪 Is the six-point system working?")
    for line in val.verdict(snaps):
        st.info(line)
    score = val.by_score(snaps)
    if score.empty:
        return
    st.markdown("**Hit rate by score** (LOW / HIGH = 95% range; wide ranges mean too few picks to judge)")
    st.dataframe(score.round(1), use_container_width=True, hide_index=True)
    crit = val.by_criterion(snaps)
    if not crit.empty:
        st.markdown("**Does each check matter?** Hit rate when the check passed vs failed, for the side shown")
        st.dataframe(crit.round(1), use_container_width=True, hide_index=True)
        st.caption("A check that never failed on a graded pick can't be judged and isn't listed. READ compares the two 95% ranges: "
                   "'helps' or 'hurts' only when they don't overlap.")
    calib = val.calibration(snaps)
    if not calib.empty:
        st.markdown("**Are the model's probabilities honest?** A calibrated model's HIT % sits near AVG MODEL %")
        st.dataframe(calib.round(1), use_container_width=True, hide_index=True)
    b = val.brier(snaps)
    if b["n_both"] > 0:
        st.caption(f"Probability accuracy on {int(b['n_both'])} picks (Brier, lower is better): model {b['model']:.4f}, market {b['market']:.4f}. "
                   "Snapshots saved before this update have no market probability and are left out of this line.")


def _legs(st: Any) -> List[Dict[str, Any]]:
    ss = st.session_state
    if "parlay_legs" not in ss:
        ss["parlay_legs"] = []
    return ss["parlay_legs"]


def add_leg(st: Any, label: str, p_model: float, odds: Optional[float], p_market: Optional[float], game: str, link: str = "") -> bool:
    """Append a leg unless the same label is already on the ticket."""
    legs = _legs(st)
    if not label or any(l["label"] == label for l in legs):
        return False
    legs.append({"label": label, "p_model": float(p_model), "odds": odds, "p_market": p_market, "game": str(game or "").strip(), "link": str(link or "")})
    return True


def _link_button(st: Any, label: str, url: str, primary: bool) -> None:
    """A button that opens `url` in a new tab (plain markdown link on Streamlit versions without link_button)."""
    if hasattr(st, "link_button"):
        st.link_button(label, url, type="primary" if primary else "secondary", use_container_width=True)
    else:
        st.markdown(f"[{label}]({url})")


def render_place_bet(st: Any, player: str, stat: str, line: float, odds: Dict[str, Optional[float]], links: Any, best_side: str = "") -> None:
    """Solo-bet buttons under an evaluation: one per side that has a saved DraftKings link. The app only opens DraftKings; you confirm the bet there."""
    over_link, under_link = (links or ("", ""))[:2]
    st.markdown("### 🎯 Place Bet")
    if not (over_link or under_link):
        st.caption("No DraftKings link is saved for this prop. Fetch the player's odds board (Step 3), then evaluate from the board so the link is kept.")
        return
    cols = st.columns(2)
    for col, side, url in ((cols[0], "Over", over_link), (cols[1], "Under", under_link)):
        price = odds.get(side)
        label = f"Place Bet: {side} {line:g} {stat}" + (f" ({price:+.0f})" if price is not None else "")
        with col:
            if url:
                _link_button(st, label, url, primary=(side == best_side))
            else:
                st.caption(f"{side}: no DraftKings link (that price is an estimate, not a listed bet).")
    st.caption(f"{player}: opens DraftKings with the pick added to your bet slip when DraftKings provides that link, otherwise the game page. "
               "Check the line and price on the slip before you confirm; the app never places the bet.")


def render_slip(st: Any, legs: List[Dict[str, Any]]) -> None:
    """DraftKings buttons for the ticket: one link per leg plus one that opens them all. The app never places the bet."""
    try:
        import bet_links as bl
    except ImportError:
        st.warning("bet_links.py isn't uploaded to the app's GitHub repo yet, so the DraftKings buttons are off.")
        return
    st.markdown("**Take it to DraftKings**")
    links = [(l.get("link") or bl.DK_HOME, bl.kind_of(l.get("link"))) for l in legs]
    for l, (url, kind) in zip(legs, links):
        st.markdown(f"- {l['label']}: [Bet on DraftKings ↗]({url}) ({bl.KIND_LABEL.get(kind, kind)})")
    try:
        from streamlit.components.v1 import html as _components_html
        _components_html(bl.open_all_html([u for u, _ in links], f"➕ Add slip: open all {len(links)} legs"), height=60)
    except Exception:
        pass       # no components support (tests): the per-leg links above still work
    st.caption(f"{bl.real_links(links)} of {len(links)} legs have a DraftKings link for that pick or game; the rest open the DraftKings sportsbook. "
               "Each link opens in its own tab (allow pop-ups for this site once). DraftKings decides what a link does, so check the slip: the "
               "price and line there are DraftKings' current ones, and you confirm the bet yourself. Legs typed in by hand have no link."
               " Links are saved when you evaluate a prop, so a leg saved before this update opens the sportsbook home page.")


def render_parlay(st: Any, snaps: Optional[pd.DataFrame]) -> None:
    """Build a ticket from saved predictions or by hand, then price it."""
    ss = st.session_state
    legs = _legs(st)
    st.markdown("Build a ticket from props you've already evaluated, or type legs in. The evaluator multiplies your legs' model "
                "probabilities, adjusts legs from the same game for correlation, applies any boost, and compares that with the price.")
    pool = pl.legs_from_snapshots(snaps) if snaps is not None else pd.DataFrame()
    if not pool.empty:
        c1, c2 = st.columns([4, 1])
        pick = c1.selectbox("Add a leg from your saved predictions (pending games)", pool["LABEL"].tolist(), key="parlay_pick")
        c2.write("")
        if c2.button("➕ Add leg", key="parlay_add") and pick in set(pool["LABEL"]):
            r = pool[pool["LABEL"] == pick].iloc[0]
            ok = add_leg(st, r["LABEL"], r["P_MODEL"], None if pd.isna(r["ODDS"]) else float(r["ODDS"]),
                         None if pd.isna(r["P_MARKET"]) else float(r["P_MARKET"]), r["GAME"], r.get("LINK", ""))
            (st.toast if ok else st.warning)("Leg added" if ok else "That leg is already on the ticket.")
    else:
        st.caption("No pending saved predictions yet: evaluate a prop or scan a board first, or add legs by hand below.")
    with st.expander("Add a leg by hand"):
        m1, m2, m3, m4 = st.columns([3, 1, 1, 2])
        label = m1.text_input("Leg", key="parlay_m_label", placeholder="Player Over 22.5 Points")
        pm = m2.number_input("Your chance %", min_value=1.0, max_value=99.0, value=55.0, step=1.0, key="parlay_m_p")
        od = m3.number_input("Leg odds (0 = none)", value=0, step=5, key="parlay_m_odds")
        game = m4.text_input("Game (same text = same game)", key="parlay_m_game", placeholder="MIA @ UTA")
        if st.button("➕ Add this leg", key="parlay_m_add"):
            o = _num(od, 0.0)
            ok = add_leg(st, str(label or "").strip(), _num(pm, 55.0) / 100.0, o if abs(o) >= 100 else None, None, str(game or ""))
            (st.toast if ok else st.warning)("Leg added" if ok else "Give the leg a name that isn't already on the ticket.")
    if not legs:
        st.info("Your ticket is empty.")
        return
    st.markdown("**Your ticket**")
    st.dataframe(pd.DataFrame([{"LEG": l["label"], "GAME": l["game"], "YOUR_CHANCE_%": round(l["p_model"] * 100.0, 1),
                                "MARKET_%": None if l["p_market"] is None else round(l["p_market"] * 100.0, 1),
                                "ODDS": l["odds"]} for l in legs]), use_container_width=True, hide_index=True)
    r1, r2 = st.columns([4, 1])
    drop = r1.selectbox("Remove a leg", [l["label"] for l in legs], key="parlay_drop")
    r2.write("")
    if r2.button("🗑️ Remove", key="parlay_remove"):
        ss["parlay_legs"] = [l for l in legs if l["label"] != drop]
        st.rerun()
    if st.button("Clear ticket", key="parlay_clear"):
        ss["parlay_legs"] = []
        st.rerun()
    render_slip(st, legs)
    p1, p2, p3 = st.columns(3)
    offered = _num(p1.number_input("Parlay odds offered, before any boost (0 = multiply the leg odds)", value=0, step=5, key="parlay_offered"), 0.0)
    boost = _num(p2.number_input("Boost %", min_value=0.0, max_value=200.0, value=0.0, step=5.0, key="parlay_boost"), 0.0)
    rho = _num(p3.slider("Same-game correlation", -0.5, 0.8, 0.0, 0.05, key="parlay_rho"), 0.0)
    res = pl.evaluate([pl.Leg(l["label"], l["p_model"], l["odds"], l["p_market"], l["game"]) for l in legs],
                      offered if abs(offered) >= 100 else None, boost, rho)
    if "error" in res:
        st.warning(res["error"])
        return
    m = st.columns(5)
    for col, lab, value in zip(m, ("ALL LEGS HIT", "BREAK-EVEN", "EV PER UNIT", "PRICE YOU GET", "FAIR PRICE"),
                               (f"{res['p_joint'] * 100:.1f}%", f"{res['breakeven'] * 100:.1f}%", f"{res['ev_pct']:+.1f}%",
                                _odds_text(res["boosted_american"]), _odds_text(res["fair_american"]))):
        col.metric(lab, value)
    verdict = (f"**{res['verdict']}**: your legs hit together {res['p_joint'] * 100:.1f}% of the time by the model; the price needs "
               f"{res['breakeven'] * 100:.1f}%. Suggested stake {res['kelly_pct']:.2f}% of bankroll (quarter Kelly, capped at 5%).")
    (st.success if res["verdict"] == "PLAY" else st.error)(verdict)
    notes = [f"Price used: {res['price_source']} ({_odds_text(res['base_american'])})" + (f", boosted {boost:g}% to {_odds_text(res['boosted_american'])}." if boost else ".")]
    if np.isfinite(res["p_market"]):
        notes.append(f"The market's own no-vig chance for these legs is {res['p_market'] * 100:.1f}%.")
    if res["same_game"]:
        notes.append("Some legs share a game, so the answer depends on the correlation you set. The table shows how much.")
    st.caption(" ".join(notes))
    if res["same_game"] and not res["sensitivity"].empty:
        st.dataframe(res["sensitivity"].round(2), use_container_width=True, hide_index=True)
    st.caption("Limits: each leg's chance comes from the single-prop model, so any error there multiplies across legs. The same-game "
               "correlation is your assumption, not something the app measures. Books price same-game parlays below the product of "
               "the leg odds, so enter the price the book actually offers.")
