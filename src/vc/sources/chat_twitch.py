"""Anonymous Twitch IRC reader + ChatHub (per-channel ring buffers, optional jsonl log)."""
from __future__ import annotations

import asyncio
import json
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from vc import log
from vc.config import get_config

L = log.get("chat")
KEEP_S = 600


@dataclass(slots=True)
class ChatMessage:
    ts: float
    channel: str
    user: str
    text: str
    emotes: frozenset[str] = field(default_factory=frozenset)  # Twitch emote names; empty for Kick and old logs


class ChatHub:
    """Ring buffer per channel; listeners get every message (TriggerEngine feeds its stats from here)."""

    def __init__(self) -> None:
        self.buffers: dict[str, deque[ChatMessage]] = {}
        self.listeners: list[Callable[[ChatMessage], None]] = []
        self._queues: dict[str, set[asyncio.Queue]] = {}
        self._logs: dict[str, tuple[str, object]] = {}

    def subscribe(self, channel: str) -> deque[ChatMessage]:
        return self.buffers.setdefault(channel.lower(), deque())

    def recent(self, channel: str, seconds: float, now: float | None = None) -> list[ChatMessage]:
        cutoff = (now or time.time()) - seconds
        return [m for m in self.buffers.get(channel.lower(), ()) if m.ts >= cutoff]

    def publish(self, m: ChatMessage) -> None:
        buf = self.subscribe(m.channel)
        buf.append(m)
        cutoff = m.ts - KEEP_S
        while buf and buf[0].ts < cutoff:
            buf.popleft()
        for cb in self.listeners:
            cb(m)
        for q in self._queues.get(m.channel, ()):
            if not q.full():
                q.put_nowait(m)
        if get_config().chat.log:
            self._write_log(m)

    async def stream(self, channel: str):
        """Async iterator of messages for one channel."""
        q: asyncio.Queue = asyncio.Queue(maxsize=5000)
        self._queues.setdefault(channel.lower(), set()).add(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._queues[channel.lower()].discard(q)

    def _write_log(self, m: ChatMessage) -> None:
        day = date.today().isoformat()
        cur = self._logs.get(m.channel)
        if not cur or cur[0] != day:
            if cur:
                cur[1].close()  # type: ignore[attr-defined]
            d = get_config().data / "chatlogs"
            d.mkdir(parents=True, exist_ok=True)
            cur = (day, open(d / f"{m.channel}-{day}.jsonl", "a", encoding="utf-8", buffering=1))  # noqa: SIM115
            self._logs[m.channel] = cur
        row = {"ts": m.ts, "channel": m.channel, "user": m.user, "text": m.text}
        if m.emotes:
            row["emotes"] = sorted(m.emotes)
        cur[1].write(json.dumps(row, ensure_ascii=False) + "\n")  # type: ignore[attr-defined]


def parse_irc(line: str) -> tuple[str, str, str, frozenset[str]] | None:
    """PRIVMSG line -> (channel, user, text, Twitch emote names from the 'emotes' tag). Tags optional; else None."""
    tags: dict[str, str] = {}
    if line.startswith("@"):
        raw, _, line = line.partition(" ")
        tags = dict(kv.partition("=")[::2] for kv in raw[1:].split(";"))
    if not line.startswith(":"):
        return None
    prefix, _, rest = line[1:].partition(" ")
    cmd, _, rest = rest.partition(" ")
    if cmd != "PRIVMSG":
        return None
    chan, _, text = rest.partition(" :")
    user = tags.get("display-name") or prefix.split("!", 1)[0]
    if text.startswith("\x01ACTION ") and text.endswith("\x01"):
        text = text[8:-1]
    from vc.sources.emotes import parse_emotes_tag
    return chan.lstrip("#").lower(), user.lower(), text, parse_emotes_tag(tags.get("emotes", ""), text)


class TwitchChat:
    URL = "wss://irc-ws.chat.twitch.tv:443"

    def __init__(self, hub: ChatHub) -> None:
        self.hub = hub
        self.channels: set[str] = set()
        self._ws = None
        self._join_q: asyncio.Queue[str] = asyncio.Queue()

    def join(self, channel: str) -> None:
        ch = channel.lower()
        if ch not in self.channels:
            self.channels.add(ch)
            self.hub.subscribe(ch)
            self._join_q.put_nowait(ch)

    async def part(self, channel: str) -> None:
        ch = channel.lower()
        self.channels.discard(ch)
        if self._ws:
            try:
                await self._ws.send(f"PART #{ch}")
            except Exception:  # noqa: BLE001
                pass

    async def _joiner(self, ws) -> None:
        """Max 20 JOINs per 10 s."""
        while True:
            batch = [await self._join_q.get()]
            while len(batch) < 20 and not self._join_q.empty():
                batch.append(self._join_q.get_nowait())
            for ch in batch:
                if ch in self.channels:
                    await ws.send(f"JOIN #{ch}")
            await asyncio.sleep(10)

    async def run(self) -> None:
        import websockets
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.URL, ping_interval=60, max_size=2**22) as ws:
                    self._ws = ws
                    await ws.send("CAP REQ :twitch.tv/tags twitch.tv/commands")
                    await ws.send("PASS SCHMOOPIIE")
                    await ws.send(f"NICK justinfan{random.randint(10000, 99999)}")
                    while not self._join_q.empty():
                        self._join_q.get_nowait()
                    for ch in self.channels:
                        self._join_q.put_nowait(ch)
                    joiner = asyncio.create_task(self._joiner(ws))
                    L.info("twitch chat verbunden (%d Kanaele)", len(self.channels))
                    backoff = 1.0
                    try:
                        async for frame in ws:
                            for line in str(frame).split("\r\n"):
                                if not line:
                                    continue
                                if line.startswith("PING"):
                                    await ws.send("PONG" + line[4:])
                                elif " RECONNECT" in line[:80]:
                                    raise ConnectionError("RECONNECT vom Server")
                                elif (p := parse_irc(line)) and p[0] in self.channels:
                                    self.hub.publish(ChatMessage(time.time(), *p))
                    finally:
                        joiner.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                L.warning("twitch chat getrennt: %s, neu in %.0fs", e, backoff)
            self._ws = None
            await asyncio.sleep(backoff)
            backoff = min(60.0, backoff * 2)
