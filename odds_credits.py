"""Odds API credit helpers shared by the NBA and NFL apps (no Streamlit dependency).

- Credits counter: the API reports `x-requests-remaining` on every call; we remember the latest value.
- Odds cache: a fetched prop board is reused for BOARD_TTL seconds, so re-pressing "Fetch odds" for the same game and
  market costs 0 credits. Only successful fetches are cached. Tick "refresh" to force a new fetch.
- Friendly errors: "OUT_OF_USAGE_CREDITS" and rate limits get a plain-English message.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Tuple

BOARD_TTL = 600.0          # seconds a fetched odds board is reused
LOW_CREDITS = 20           # warn at or below this many credits

CREDITS: Dict[str, Optional[float]] = {"remaining": None, "used": None, "last": None}


def _num(v: Any) -> Optional[float]:
    """Header value -> float, or None (ignores anything that isn't plain text/number)."""
    if isinstance(v, bool) or not isinstance(v, (str, int, float)):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def record(remaining: Any, used: Any = None, last: Any = None) -> None:
    """Remember the latest credit numbers (None values keep the previous number)."""
    for key, val in (("remaining", remaining), ("used", used), ("last", last)):
        n = _num(val)
        if n is not None:
            CREDITS[key] = n


def record_headers(headers: Optional[Mapping[str, Any]]) -> None:
    """Pull x-requests-remaining / -used / -last out of a response's headers."""
    try:
        get = headers.get  # type: ignore[union-attr]
        record(get("x-requests-remaining"), get("x-requests-used"), get("x-requests-last"))
    except Exception:
        pass


def credits_status() -> Tuple[str, str]:
    """(text to show, level) where level is 'unknown' | 'ok' | 'low' | 'out'."""
    rem = CREDITS["remaining"]
    if rem is None:
        return "", "unknown"
    n = int(rem)
    last = f" · last call used {int(CREDITS['last'])}" if CREDITS["last"] is not None else ""
    if n <= 0:
        return "Odds API credits left: 0. Fetching odds will fail until your plan renews or you add credits.", "out"
    if n <= LOW_CREDITS:
        return f"⚠️ Odds API credits left: {n}{last}. Fetch odds sparingly (scanning and evaluating are free).", "low"
    return f"Odds API credits left: {n}{last}", "ok"


def friendly_error(message: Any) -> str:
    """Plain-English version of common Odds API failures; any other message is returned unchanged."""
    text = str(message)
    up = text.upper()
    if "OUT_OF_USAGE_CREDITS" in up or "USAGE QUOTA HAS BEEN REACHED" in up:
        return ("Out of Odds API credits: this key's monthly quota is used up. Credits come back when your plan renews "
                "(see your dashboard at the-odds-api.com), or you can upgrade or use another key. You can still evaluate "
                "a player by typing the line and odds into the form below.")
    if "EXCEEDED_FREQ_LIMIT" in up or "(429)" in text:
        return "The Odds API says you're sending requests too fast. Wait a minute and try again."
    if "INVALID_KEY" in up or "(401)" in text:
        return "The Odds API rejected the key: check ODDS_API_KEY in Streamlit Secrets. Original message: " + text
    return text


class TTLStore:
    """Tiny in-memory cache with per-read expiry."""

    def __init__(self) -> None:
        self._d: Dict[Hashable, Tuple[float, Any]] = {}

    def get(self, key: Hashable, ttl: float) -> Optional[Tuple[Any, float]]:
        """(value, age in seconds) if present and younger than ttl, else None."""
        hit = self._d.get(key)
        if hit is None:
            return None
        age = time.time() - hit[0]
        return (hit[1], age) if age < ttl else None

    def set(self, key: Hashable, value: Any) -> None:
        self._d[key] = (time.time(), value)

    def clear(self) -> None:
        self._d.clear()


BOARDS = TTLStore()


def fetch_cached(key: Hashable, fetch: Callable[[], Any], is_ok: Callable[[Any], bool], refresh: bool = False,
                 ttl: float = BOARD_TTL, store: Optional[TTLStore] = None) -> Tuple[Any, Optional[float]]:
    """Return (value, cached_age): cached_age is None for a fresh fetch, else the saved copy's age in seconds.

    A saved copy younger than `ttl` is used (0 credits) unless `refresh`. Failed fetches are never saved.
    """
    store = BOARDS if store is None else store
    if not refresh:
        hit = store.get(key, ttl)
        if hit is not None:
            return hit[0], hit[1]
    value = fetch()
    if is_ok(value):
        store.set(key, value)
    return value, None


def cache_note(cached_age: Optional[float]) -> str:
    """One line telling the user whether odds were fetched fresh or reused."""
    if cached_age is None:
        return "Fetched fresh odds."
    mins = max(int(cached_age // 60), 0)
    return f"Reused odds saved {mins} min ago: no credit used. Tick “Refresh odds” for live prices."
