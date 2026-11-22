"""Emote names for chat features: Twitch emotes from the IRC 'emotes' tag, third-party emotes (7TV, BTTV, FFZ)
from their public APIs (no key). Cached per channel in data/emotes/<login>.json for 24 h. A failing service
never blocks: 5 s timeout, warning, empty result."""
from __future__ import annotations

import json
import time

import httpx

from vc import log
from vc.config import get_config

L = log.get("emotes")
TTL_S = 24 * 3600
TIMEOUT_S = 5.0


def parse_emotes_tag(tag: str, text: str) -> frozenset[str]:
    """'25:0-4,12-16/354:6-10' + text -> {'Kappa', 'Keepo'}. Positions are code points (= Python str index)."""
    names = set()
    for part in filter(None, tag.split("/")):
        _, _, spans = part.partition(":")
        for span in filter(None, spans.split(",")):
            a, _, b = span.partition("-")
            try:
                names.add(text[int(a):int(b) + 1])
            except ValueError:
                continue
    return frozenset(n for n in names if n)


def _names_7tv(data: dict) -> dict[str, bool]:
    emotes = (data.get("emote_set") or data).get("emotes") or []
    return {e["name"]: bool((e.get("data") or {}).get("animated")) for e in emotes if e.get("name")}


def _names_bttv(data) -> dict[str, bool]:
    items = data if isinstance(data, list) else (data.get("channelEmotes") or []) + (data.get("sharedEmotes") or [])
    return {e["code"]: e.get("imageType") == "gif" or bool(e.get("animated")) for e in items if e.get("code")}


def _names_ffz(data: dict) -> dict[str, bool]:
    return {e["name"]: bool(e.get("animated")) for s in (data.get("sets") or {}).values()
            for e in s.get("emoticons") or [] if e.get("name")}


def sources(twitch_id: str | None) -> list[tuple[str, str, object]]:
    out = [("7tv", "https://7tv.io/v3/emote-sets/global", _names_7tv),
           ("bttv", "https://api.betterttv.net/3/cached/emotes/global", _names_bttv),
           ("ffz", "https://api.frankerfacez.com/v1/set/global", _names_ffz)]
    if twitch_id:
        out += [("7tv", f"https://7tv.io/v3/users/twitch/{twitch_id}", _names_7tv),
                ("bttv", f"https://api.betterttv.net/3/cached/users/twitch/{twitch_id}", _names_bttv),
                ("ffz", f"https://api.frankerfacez.com/v1/room/id/{twitch_id}", _names_ffz)]
    return out


async def fetch(twitch_id: str | None) -> dict[str, bool]:
    """name -> animated, global + channel sets. Each failing service is skipped with a warning."""
    out: dict[str, bool] = {}
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as h:
        for name, url, parse in sources(twitch_id):
            try:
                r = await h.get(url)
                if r.status_code == 404:  # channel has no set there
                    continue
                r.raise_for_status()
                out.update(parse(r.json()))  # type: ignore[operator]
            except Exception as e:  # noqa: BLE001
                L.warning("Emotes %s nicht geladen (%s): %s", name, url.rsplit("/", 2)[-2], str(e)[:100])
    return out


async def channel_emotes(login: str, twitch_id: str | None = None) -> frozenset[str]:
    """Cached third-party emote names of a channel (refetched after 24 h; an empty fetch keeps the old cache)."""
    p = get_config().data / "emotes" / f"{login.lower()}.json"
    cached: dict = {}
    try:
        cached = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if cached and time.time() - cached.get("fetched", 0) < TTL_S:
        return frozenset(cached.get("emotes", {}))
    fresh = await fetch(twitch_id)
    if fresh:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"fetched": time.time(), "emotes": fresh}, ensure_ascii=False), encoding="utf-8")
        return frozenset(fresh)
    return frozenset(cached.get("emotes", {}))
