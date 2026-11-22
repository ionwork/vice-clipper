"""Kick chat via Kick's public Pusher websocket. Unofficial interface: optional, fails soft."""
from __future__ import annotations

import asyncio
import json
import time

from vc import db, log
from vc.config import get_config
from vc.sources.chat_twitch import ChatHub, ChatMessage

L = log.get("kick")


async def chatroom_id(slug: str) -> int:
    if cached := await db.kv_get(f"kick_chatroom:{slug}"):
        return int(cached)
    from curl_cffi.requests import AsyncSession  # browser impersonation for Cloudflare
    async with AsyncSession(impersonate="chrome") as s:
        r = await s.get(f"https://kick.com/api/v2/channels/{slug}", timeout=20)
        r.raise_for_status()
        cid = int(r.json()["chatroom"]["id"])
    await db.kv_set(f"kick_chatroom:{slug}", str(cid))
    return cid


def parse_event(raw: str) -> tuple[str, str] | None:
    """Pusher frame -> (user, text) for chat messages."""
    msg = json.loads(raw)
    if msg.get("event") != r"App\Events\ChatMessageEvent":
        return None
    data = json.loads(msg["data"]) if isinstance(msg.get("data"), str) else msg.get("data") or {}
    return (data.get("sender") or {}).get("username", "?").lower(), data.get("content", "")


class KickChat:
    def __init__(self, hub: ChatHub) -> None:
        self.hub = hub
        self.channels: dict[str, int] = {}  # slug -> chatroom id
        self._ws = None
        self._warned = False

    async def join(self, slug: str) -> None:
        slug = slug.lower()
        try:
            self.channels[slug] = await chatroom_id(slug)
            self.hub.subscribe(slug)
            if self._ws:
                await self._sub(self._ws, self.channels[slug])
        except Exception as e:  # noqa: BLE001
            self._warn(f"Kick-Chat {slug}: {e}")

    def _warn(self, msg: str) -> None:
        if not self._warned:
            L.warning("%s (Kick-Chat deaktiviert sich weich, Rest laeuft weiter)", msg)
            self._warned = True

    async def _sub(self, ws, cid: int) -> None:
        await ws.send(json.dumps({"event": "pusher:subscribe", "data": {"auth": "", "channel": f"chatrooms.{cid}.v2"}}))

    async def run(self) -> None:
        import websockets
        c = get_config().chat
        if not c.kick_enabled or not c.kick_pusher_key:
            return
        url = f"wss://ws-{c.kick_pusher_cluster}.pusher.com/app/{c.kick_pusher_key}?protocol=7&client=js&version=8.4.0&flash=false"
        by_id = lambda: {v: k for k, v in self.channels.items()}  # noqa: E731
        backoff = 2.0
        while True:
            try:
                async with websockets.connect(url, ping_interval=None) as ws:
                    self._ws = ws
                    for cid in self.channels.values():
                        await self._sub(ws, cid)
                    backoff = 2.0
                    async for raw in ws:
                        m = json.loads(raw)
                        if m.get("event") == "pusher:ping":
                            await ws.send(json.dumps({"event": "pusher:pong", "data": {}}))
                            continue
                        if (p := parse_event(raw)) and (slug := by_id().get(int(m.get("channel", "x.0").split(".")[1]))):
                            self.hub.publish(ChatMessage(time.time(), slug, *p))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self._warn(f"Kick-Websocket: {e}")
            self._ws = None
            await asyncio.sleep(backoff)
            backoff = min(120.0, backoff * 2)
