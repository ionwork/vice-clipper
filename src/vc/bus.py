"""In-process fan-out of events to live subscribers (dashboard SSE). No-op without subscribers."""
from __future__ import annotations

import asyncio

_subs: set[asyncio.Queue] = set()


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    _subs.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    _subs.discard(q)


def publish(msg: dict) -> None:
    for q in list(_subs):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass
