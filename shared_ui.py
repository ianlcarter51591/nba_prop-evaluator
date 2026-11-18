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


def _now_text() -> str:
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M")


def render_tracked_place_bet(st: Any, snaps: Optional[pd.DataFrame], save: Any, player: str, stat: str, line: float,
                             odds: Dict[str, Optional[float]], links: Any, best_side: str = "") -> bool:
    """Two-tap Place Bet that records the bet: tap 1 saves 'placed' (side, price, time) on the prop's snapshot, then the DraftKings
    link appears for tap 2. Streamlit can't see clicks on links, so recording happens on the app's own button.
    Returns False when the prop isn't saved (the caller then shows plain links)."""
    import favorites as fv
    sid = fv.find_id(snaps, player, stat, line)
    if not sid or save is None:
        return False
    over_link, under_link = (links or ("", ""))[:2]
    done = fv.placed(snaps, sid)
    st.markdown("### 🎯 Place Bet")
    cols = st.columns(2)
    for col, side, url in ((cols[0], "Over", over_link), (cols[1], "Under", under_link)):
        price = odds.get(side)
        ptxt = f" ({price:+.0f})" if price is not None and not pd.isna(price) else ""
        with col:
            if done.get("side") == side and done.get("kind") == "Solo":
                st.success(f"✅ Placed: {side} {line:g}{ptxt}")
                if url:
                    _link_button(st, "Open on DraftKings ↗", url, primary=True)
                if st.button("Undo placed", key=f"unplace_{sid}_{side}"):
                    _apply_snapshots(st, save, fv.unmark_placed(snaps, sid))
                    st.rerun()
            elif st.button(f"🎯 Place Bet: {side} {line:g} {stat}{ptxt}", key=f"place_{sid}_{side}", type="primary" if side == best_side else "secondary",
                           use_container_width=True):
                _apply_snapshots(st, save, fv.mark_placed(snaps, sid, side, price, "Solo", _now_text()))
                st.rerun()
            elif not url:
                st.caption(f"{side}: no DraftKings link saved; the button still records the bet if you place it yourself.")
    st.caption(f"{player}: tap Place Bet to record it in your Snapshots (Placed bets), then tap Open on DraftKings to add it to your slip. "
               "Confirm the line and price on DraftKings; the app never places the bet. Undo removes the record.")
    return True


def render_place_bet(st: Any, player: str, stat: str, line: float, odds: Dict[str, Optional[float]], links: Any, best_side: str = "",
                     snaps: Optional[pd.DataFrame] = None, save: Any = None) -> None:
    """Solo-bet buttons under an evaluation. With the saved snapshots (and a save function) they also record placed bets;
    otherwise one link per side that has a saved DraftKings link. The app only opens DraftKings; you confirm the bet there."""
    if snaps is not None and render_tracked_place_bet(st, snaps, save, player, stat, line, odds, links, best_side):
        return
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


def render_mark_ticket(st: Any, legs: List[Dict[str, Any]], snaps: Optional[pd.DataFrame], save: Any) -> None:
    """'I placed this ticket' button: records every leg that comes from a saved prop as a placed parlay leg (side and leg price)."""
    import favorites as fv
    if snaps is None or save is None or not legs:
        return
    ids = [(fv.snap_id_for_leg(snaps, l["label"]), l) for l in legs]
    found = [(sid, l) for sid, l in ids if sid]
    if st.button(f"✅ I placed this ticket (record its {len(found)} saved leg(s))", key="parlay_mark_placed", disabled=not found):
        d, now = snaps, _now_text()
        for sid, l in found:
            side = "Under" if " Under " in f" {l['label']} " else "Over"
            d = fv.mark_placed(d, sid, side, l.get("odds"), "Parlay leg", now)
        _apply_snapshots(st, save, d)
        st.success(f"Recorded {len(found)} leg(s) under Placed parlay legs in the Snapshots tab."
                   + (f" {len(legs) - len(found)} typed-in leg(s) have no saved prop and weren't recorded." if len(found) < len(legs) else ""))


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


def _apply_snapshots(st: Any, save: Any, new: pd.DataFrame) -> None:
    """Save `new` snapshots (the app's save function returns what was stored) and remember them for this session."""
    st.session_state["snapshots"] = save(new)


def render_favorites(st: Any, snaps: Optional[pd.DataFrame], save: Any, stat: Optional[str] = None, title: str = "⭐ Favorite props") -> None:
    """Multiselect of saved (pending) props; whatever is selected becomes a favorite and is what the parlay builder offers."""
    import favorites as fv
    opts = fv.option_table(snaps, stat)
    if opts.empty:
        st.caption("No saved props to star yet: evaluate a prop or scan a board first (every evaluated prop is saved).")
        return
    current = opts.loc[opts["FAV"] == "Y", "LABEL"].tolist()
    picked = st.multiselect(title, opts["LABEL"].tolist(), default=current,
                            help="Starred props are the only ones the Parlay tab offers. Only props from games that haven't been graded yet are listed.")
    if picked is None or set(picked) == set(current):
        return
    _apply_snapshots(st, save, fv.set_flags(snaps, set(opts.loc[opts["LABEL"].isin(picked), "SNAP_ID"]), set(opts["SNAP_ID"])))
    st.rerun()


def render_star_button(st: Any, snaps: Optional[pd.DataFrame], player: str, stat: str, line: float, save: Any) -> None:
    """☆ / ★ button for the prop on screen (it must already be saved, which every evaluation does)."""
    import favorites as fv
    sid = fv.find_id(snaps, player, stat, line)
    if not sid:
        return
    on = fv.is_favorite(snaps, sid)
    if st.button("★ Favorite (click to remove)" if on else "☆ Favorite this prop", key=f"star_{sid}",
                 help="Favorites are the only props the Parlay tab offers."):
        _apply_snapshots(st, save, fv.toggle(snaps, sid))
        st.rerun()


def clickable_table(st: Any, shown: Any, payload: pd.DataFrame, open_fn: Any, key: str, hint: str = "", **kwargs: Any) -> None:
    """A st.dataframe whose rows can be clicked: the clicked row of `payload` (same order as `shown`) goes to `open_fn(row) -> message`,
    then the page reruns so the Evaluate tab picks it up. Each table remembers its own last click, so an old selection never re-opens."""
    import zlib
    ss = st.session_state
    try:
        ident = key + "_" + str(zlib.crc32(payload.to_csv(index=False).encode()))      # new rows -> fresh selection
    except Exception:
        ident = key
    if hint:
        st.caption(hint)
    try:
        event = st.dataframe(shown, on_select="rerun", selection_mode="single-row", key=ident, **kwargs)
    except TypeError:                                    # Streamlit older than 1.35: no row clicks, plain table
        st.dataframe(shown, **kwargs)
        return
    sel = getattr(event, "selection", None)
    if sel is None and isinstance(event, dict):
        sel = event.get("selection")
    rows = getattr(sel, "rows", None)
    if rows is None and isinstance(sel, dict):
        rows = sel.get("rows")
    if not rows or int(rows[0]) >= len(payload):
        return
    token = (ident, int(rows[0]))
    if ss.get("clicked_" + key) == token:
        if ss.get("clicked_msg_" + key):
            st.success(ss["clicked_msg_" + key])
        return
    ss["clicked_" + key] = token
    try:
        ss["clicked_msg_" + key] = open_fn(payload.iloc[int(rows[0])].to_dict())
    except Exception as e:
        ss["clicked_msg_" + key] = ""
        st.warning(f"Couldn't load that prop: {str(e)[:120]}")
        return
    st.rerun()


CLICK_HINT = "👆 Click a row (the box at its left edge) to load that prop into the **⚡ Evaluate** tab."


def render_parlay(st: Any, snaps: Optional[pd.DataFrame], save: Any = None, open_prop: Any = None) -> None:
    """Build a ticket from saved predictions or by hand, then price it."""
    ss = st.session_state
    legs = _legs(st)
    import favorites as fv
    st.markdown("Build a ticket from your ⭐ favorite props. The evaluator multiplies your legs' model "
                "probabilities, adjusts legs from the same game for correlation, applies any boost, and compares that with the price.")
    pool = fv.legs(snaps)

    def _add(r: Any) -> bool:
        return add_leg(st, r["LABEL"], r["P_MODEL"], None if pd.isna(r["ODDS"]) else float(r["ODDS"]),
                       None if pd.isna(r["P_MARKET"]) else float(r["P_MARKET"]), r["GAME"], r.get("LINK", ""))

    if not pool.empty:
        st.markdown("**Your favorites** (both sides of each starred prop; ✓ = the side the app shows)")
        fav_table = pd.DataFrame({"LEG": pool["LABEL"], "GAME": pool["GAME"], "YOUR_CHANCE_%": (pool["P_MODEL"] * 100).round(1),
                                  "MARKET_%": (pool["P_MARKET"] * 100).round(1), "EDGE_PP": pool["EDGE_PP"].round(1), "ODDS": pool["ODDS"],
                                  "APP_PICK": pool["APP_PICK"].map({True: "✓", False: ""})})
        if open_prop is not None and snaps is not None and "SNAP_ID" in snaps.columns:
            rows = snaps.drop_duplicates("SNAP_ID").set_index("SNAP_ID").reindex(pool["SNAP_ID"]).reset_index()
            clickable_table(st, fav_table, rows, open_prop, "parlay_favs", CLICK_HINT, use_container_width=True, hide_index=True)
        else:
            st.dataframe(fav_table, use_container_width=True, hide_index=True)
        picks = st.multiselect("Add legs to the ticket", pool["LABEL"].tolist(), key="parlay_picks")
        c1, c2 = st.columns(2)
        if c1.button("➕ Add selected legs", key="parlay_add") and picks:
            added = sum(_add(r) for _, r in pool[pool["LABEL"].isin(picks)].iterrows())
            (st.toast if added else st.warning)(f"{added} leg(s) added" if added else "Those legs are already on the ticket.")
        if c2.button("➕ Add the app's pick for every favorite", key="parlay_add_all"):
            sub = pool[pool["APP_PICK"]] if pool["APP_PICK"].any() else pool
            added = sum(_add(r) for _, r in sub.iterrows())
            (st.toast if added else st.warning)(f"{added} leg(s) added" if added else "Those legs are already on the ticket.")
    else:
        st.info("No favorite props yet. Star a prop with the ☆ Favorite button under its evaluation, or pick favorites below or under the board, and it shows up here.")
    if save is not None:
        with st.expander("⭐ Manage favorites"):
            render_favorites(st, snaps, save)
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
    render_mark_ticket(st, legs, snaps, save)
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
