"""SQLite persistence (aiosqlite). One table clips, one table events, plus small helper tables."""
from __future__ import annotations

import json
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite

from vc.config import get_config
from vc.models import Candidate, Clip, ClipStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  streamer_id TEXT NOT NULL,
  platform TEXT, lang TEXT,
  t_trigger TEXT, trigger_offset_s REAL,
  signals TEXT, pre_score REAL,
  judge TEXT, hype INTEGER,
  status TEXT NOT NULL, reason TEXT DEFAULT '',
  raw_path TEXT, ass_path TEXT DEFAULT '', out_path TEXT DEFAULT '', text_path TEXT DEFAULT '',
  account TEXT, layout TEXT DEFAULT '',
  streamer TEXT, chat_context TEXT, origin TEXT DEFAULT '',
  tg_message_id INTEGER,
  created_at TEXT, delivered_at TEXT, posted_at TEXT
);
CREATE INDEX IF NOT EXISTS clips_status ON clips(status);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, kind TEXT NOT NULL, clip_id TEXT, streamer_id TEXT, data TEXT
);
CREATE INDEX IF NOT EXISTS events_kind_ts ON events(kind, ts);
CREATE TABLE IF NOT EXISTS vod_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  url TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'NEW', streamer_id TEXT,
  requested_at TEXT, error TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS harvest_clips (
  id TEXT PRIMARY KEY, broadcaster TEXT, broadcaster_id TEXT, title TEXT, url TEXT,
  created_at TEXT, duration REAL, language TEXT, processed INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS harvest_views (clip_id TEXT, ts TEXT, view_count INTEGER);
CREATE INDEX IF NOT EXISTS harvest_views_clip ON harvest_views(clip_id, ts);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""

JSON_COLS = ("signals", "judge", "streamer", "chat_context")
DT_COLS = ("t_trigger", "created_at", "delivered_at", "posted_at")


def now() -> datetime:
    return datetime.now(timezone.utc)


def db_path() -> Path:
    return get_config().data / "vc.sqlite"


@asynccontextmanager
async def connect() -> AsyncIterator[aiosqlite.Connection]:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(p, timeout=30) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        yield db


async def init_db() -> None:
    async with connect() as db:
        await db.executescript(SCHEMA)
        await db.commit()


def _enc(k: str, v: Any) -> Any:
    if k in JSON_COLS:
        return None if v is None else json.dumps(v, ensure_ascii=False, default=str)
    if isinstance(v, datetime):
        return v.isoformat()
    if hasattr(v, "value"):  # enums
        return v.value
    return v


def _row_to_clip(row: aiosqlite.Row) -> Clip:
    d = dict(row)
    for k in JSON_COLS:
        d[k] = json.loads(d[k]) if d.get(k) else ({} if k in ("signals", "streamer") else None if k == "judge" else [])
    for k in DT_COLS:
        d[k] = datetime.fromisoformat(d[k]) if d.get(k) else None
    return Clip.model_validate(d)


def new_clip_id(streamer_id: str, t: datetime | None = None) -> str:
    t = (t or now()).astimezone()
    prefix = "".join(ch for ch in streamer_id.lower() if ch.isalnum())[:2] or "xx"
    return f"{prefix}-{t:%m%d-%H%M}-{secrets.token_hex(1)}"


async def create_clip(c: Candidate, account: str | None = None) -> Clip:
    s = c.streamer
    clip = Clip(
        id=new_clip_id(s.id, c.t_trigger),
        source=c.source, streamer_id=s.id, platform=s.platform, lang=s.lang,
        t_trigger=c.t_trigger, trigger_offset_s=c.trigger_offset_s, signals=c.signals, pre_score=c.pre_score,
        raw_path=c.raw_path, account=account or s.account, streamer=s.model_dump(mode="json"),
        chat_context=c.chat_context, origin=c.origin, created_at=now(),
    )
    data = {k: _enc(k, v) for k, v in clip.model_dump().items()}
    cols = ", ".join(data)
    async with connect() as db:
        await db.execute(f"INSERT INTO clips ({cols}) VALUES ({', '.join('?' * len(data))})", list(data.values()))
        await db.commit()
    await add_event("candidate", clip_id=clip.id, streamer_id=s.id, source=c.source.value, pre_score=c.pre_score)
    return clip


async def update_clip(clip_id: str, **fields: Any) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    async with connect() as db:
        await db.execute(f"UPDATE clips SET {sets} WHERE id = ?", [_enc(k, v) for k, v in fields.items()] + [clip_id])
        await db.commit()


async def get_clip(clip_id: str) -> Clip | None:
    async with connect() as db:
        async with db.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)) as cur:
            row = await cur.fetchone()
    return _row_to_clip(row) if row else None


async def list_clips(status: ClipStatus | str | None = None, limit: int = 500, since: datetime | None = None) -> list[Clip]:
    q, args = "SELECT * FROM clips WHERE 1=1", []
    if status:
        q += " AND status = ?"
        args.append(str(status))
    if since:
        q += " AND created_at >= ?"
        args.append(since.isoformat())
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    async with connect() as db:
        async with db.execute(q, args) as cur:
            return [_row_to_clip(r) for r in await cur.fetchall()]


async def add_event(kind: str, clip_id: str | None = None, streamer_id: str | None = None, **data: Any) -> None:
    async with connect() as db:
        await db.execute(
            "INSERT INTO events (ts, kind, clip_id, streamer_id, data) VALUES (?, ?, ?, ?, ?)",
            (now().isoformat(), kind, clip_id, streamer_id, json.dumps(data, ensure_ascii=False, default=str)),
        )
        await db.commit()
    from vc import bus  # late import, bus is optional fan-out for the dashboard
    bus.publish({"kind": kind, "clip_id": clip_id, "streamer_id": streamer_id, **data})


async def list_events(kind: str | None = None, since: datetime | None = None, limit: int = 200) -> list[dict]:
    q, args = "SELECT * FROM events WHERE 1=1", []
    if kind:
        q += " AND kind = ?"
        args.append(kind)
    if since:
        q += " AND ts >= ?"
        args.append(since.isoformat())
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    async with connect() as db:
        async with db.execute(q, args) as cur:
            rows = await cur.fetchall()
    return [{**dict(r), "data": json.loads(r["data"] or "{}")} for r in rows]


def local_midnight_utc() -> datetime:
    return datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


async def cost_today() -> tuple[int, int, float]:
    """(prompt tokens, output tokens, estimated $) from gemini_usage events since local midnight."""
    p = o = 0
    usd = 0.0
    for e in await list_events("gemini_usage", since=local_midnight_utc(), limit=100000):
        p += e["data"].get("prompt", 0)
        o += e["data"].get("output", 0)
        usd += e["data"].get("usd", 0.0)
    return p, o, usd


async def kv_get(k: str, default: str | None = None) -> str | None:
    async with connect() as db:
        async with db.execute("SELECT v FROM kv WHERE k = ?", (k,)) as cur:
            row = await cur.fetchone()
    return row[0] if row else default


async def kv_set(k: str, v: str) -> None:
    async with connect() as db:
        await db.execute("INSERT INTO kv (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", (k, v))
        await db.commit()


async def add_vod_job(url: str, streamer_id: str | None = None) -> int:
    async with connect() as db:
        cur = await db.execute(
            "INSERT INTO vod_jobs (url, status, streamer_id, requested_at) VALUES (?, 'NEW', ?, ?)",
            (url, streamer_id, now().isoformat()),
        )
        await db.commit()
        return cur.lastrowid or 0
