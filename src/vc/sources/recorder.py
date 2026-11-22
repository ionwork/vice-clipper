"""Source A buffer: streamlink -> ffmpeg segmenter, 5-minute ring per channel, retroactive cut()."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable

from vc import db, log
from vc.config import get_config, get_secrets, get_streamers
from vc.models import Streamer
from vc.proc import run, tool

L = log.get("recorder")
SEG_FMT = "%Y%m%d-%H%M%S"
MIN_SEG_BYTES = 50_000


def buffer_dir(streamer_id: str) -> Path:
    return get_config().data / "buffer" / streamer_id


def to_local_naive(t: datetime) -> datetime:
    return t.astimezone().replace(tzinfo=None) if t.tzinfo else t


def seg_time(p: Path) -> datetime | None:
    try:
        return datetime.strptime(p.stem, SEG_FMT)
    except ValueError:
        return None


def list_segments(d: Path) -> list[tuple[datetime, Path]]:
    out = [(t, p) for p in d.glob("*.ts") if (t := seg_time(p))]
    return sorted(out)


def select_segments(segs: list[tuple[datetime, Path]], t0: datetime, t1: datetime, seg_len: float,
                    sizes: dict[Path, int] | None = None) -> list[tuple[datetime, Path]]:
    """Segments overlapping [t0 - 4 s, t1 + 4 s]; drops broken ones (< 50 KB). Pure."""
    lo, hi = t0 - timedelta(seconds=4), t1 + timedelta(seconds=4)
    picked = []
    for i, (t, p) in enumerate(segs):
        end = segs[i + 1][0] if i + 1 < len(segs) else t + timedelta(seconds=seg_len)
        if end > lo and t < hi:
            size = sizes[p] if sizes is not None else p.stat().st_size
            if size >= MIN_SEG_BYTES:
                picked.append((t, p))
    return picked


def expired(segs: list[tuple[datetime, Path]], now: datetime, keep_s: float) -> list[Path]:
    cutoff = now - timedelta(seconds=keep_s)
    return [p for t, p in segs if t < cutoff]


def stream_url(s: Streamer) -> str:
    return f"https://kick.com/{s.login}" if s.platform == "kick" else f"https://twitch.tv/{s.login}"


class Recorder:
    def __init__(self, streamer: Streamer) -> None:
        self.s = streamer
        self.dir = buffer_dir(streamer.id)
        self._task: asyncio.Task | None = None
        self._procs: list[asyncio.subprocess.Process] = []
        self.started_at: datetime | None = None
        self.on_segment: Callable[[Path], Awaitable[None]] | None = None  # audio monitor hook

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def last_segment_at(self) -> datetime | None:
        segs = list_segments(self.dir)
        return segs[-1][0] if segs else None

    def start(self) -> None:
        if not self.is_running:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._task = asyncio.create_task(self._supervise(), name=f"rec-{self.s.id}")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await self._kill()
        self._task = None
        self.started_at = None

    async def _kill(self) -> None:
        for p in self._procs:
            if p.returncode is None:
                try:
                    p.kill()
                except ProcessLookupError:
                    pass
                await p.wait()
        self._procs = []

    async def _spawn(self) -> None:
        cfg = get_config()
        sl_args = ["--stdout", "--hls-live-edge", "2", "--retry-streams", "10", "--retry-open", "3", "--loglevel", "warning"]
        if self.s.platform == "kick":
            sl_args.insert(1, "--kick-low-latency")
        else:
            sl_args.insert(1, "--twitch-low-latency")
            if tok := get_secrets().twitch_user_oauth:
                sl_args += ["--twitch-api-header", f"Authorization=OAuth {tok}"]
        sl_args += [stream_url(self.s), self.s.quality]
        ff_args = ["-hide_banner", "-loglevel", "error", "-fflags", "+genpts", "-i", "pipe:0", "-c", "copy",
                   "-f", "segment", "-segment_time", str(cfg.recorder.segment_seconds), "-segment_format", "mpegts",
                   "-reset_timestamps", "1", "-strftime", "1", str(self.dir / f"{SEG_FMT}.ts")]
        logf = open(cfg.data / "logs" / f"rec-{self.s.id}.log", "ab")  # noqa: SIM115  closed after spawn, children keep it
        r, w = os.pipe()  # streamlink stdout -> ffmpeg stdin, no shell
        try:
            sl = await asyncio.create_subprocess_exec(tool("streamlink"), *sl_args, stdout=w, stderr=logf, stdin=asyncio.subprocess.DEVNULL)
            ff = await asyncio.create_subprocess_exec(tool("ffmpeg"), *ff_args, stdin=r, stdout=asyncio.subprocess.DEVNULL, stderr=logf)
        finally:
            os.close(r)
            os.close(w)
            logf.close()
        self._procs = [sl, ff]

    async def _supervise(self) -> None:
        backoff = 2.0
        aux = [asyncio.create_task(self._janitor()), asyncio.create_task(self._snapshots())]
        try:
            while True:
                t_start = datetime.now()
                await self._spawn()
                self.started_at = self.started_at or datetime.now().astimezone()
                await db.add_event("recorder_started", streamer_id=self.s.id)
                reason = await self._watch(t_start)
                await self._kill()
                await db.add_event("recorder_error" if reason != "stopped" else "recorder_stopped", streamer_id=self.s.id, reason=reason)
                if (datetime.now() - t_start).total_seconds() > 120:
                    backoff = 2.0
                L.info("%s: neu starten in %.0fs (%s)", self.s.id, backoff, reason)
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2)
        finally:
            for t in aux:
                t.cancel()
            await self._kill()

    async def _watch(self, t_start: datetime) -> str:
        seen = None
        last_change = datetime.now()
        while True:
            await asyncio.sleep(2)
            for p, name in zip(self._procs, ("streamlink", "ffmpeg")):
                if p.returncode is not None:
                    return f"{name} exit {p.returncode}"
            newest = self.last_segment_at
            if newest != seen:
                if seen is not None and self.on_segment:
                    segs = list_segments(self.dir)
                    if len(segs) >= 2:  # second newest is complete
                        asyncio.create_task(self.on_segment(segs[-2][1]))
                seen, last_change = newest, datetime.now()
            # 20 s without a new segment = stalled (allow 40 s at start for ads / stream warmup)
            limit = 40 if (datetime.now() - t_start).total_seconds() < 60 else 20
            if (datetime.now() - last_change).total_seconds() > limit:
                return "kein neues Segment"

    async def _janitor(self) -> None:
        keep = get_config().recorder.buffer_seconds + 20
        while True:
            for p in expired(list_segments(self.dir), datetime.now(), keep):
                p.unlink(missing_ok=True)
            await asyncio.sleep(10)

    async def _snapshots(self) -> None:
        while True:
            await asyncio.sleep(10)
            segs = list_segments(self.dir)
            if len(segs) >= 2:
                await run("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", segs[-2][1], "-frames:v", "1",
                          "-vf", "scale=640:-2", "-q:v", "5", self.dir / "snapshot.jpg", timeout=20)


async def cut(streamer_id: str, t0: datetime, t1: datetime, out: Path) -> tuple[Path, datetime]:
    """Concat (no re-encode) the buffered segments covering [t0, t1]. Returns (path, wall-clock of first frame)."""
    d = buffer_dir(streamer_id)
    seg_len = get_config().recorder.segment_seconds
    t0, t1 = to_local_naive(t0), to_local_naive(t1)
    for _ in range(15):  # wait until the segment after t1 has started, i.e. the one with t1 is complete
        segs = list_segments(d)
        if segs and segs[-1][0] >= t1:
            break
        await asyncio.sleep(1)
    picked = select_segments(list_segments(d), t0, t1, seg_len)
    if not picked:
        raise RuntimeError(f"{streamer_id}: keine Segmente im Puffer fuer {t0:%H:%M:%S}-{t1:%H:%M:%S}")
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for _, p in picked), encoding="utf-8")
    r = await run("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                  "-c", "copy", "-bsf:a", "aac_adtstoasc", out, timeout=120)
    lst.unlink(missing_ok=True)
    if r.code != 0:
        raise RuntimeError(f"cut fehlgeschlagen: {r.tail(5)}")
    return out, picked[0][0].astimezone()


async def kick_livestream(slug: str) -> dict | None:
    """Kick channel API: the livestream object, or None when offline."""
    from curl_cffi.requests import AsyncSession  # browser impersonation for Cloudflare
    async with AsyncSession(impersonate="chrome") as s:
        r = await s.get(f"https://kick.com/api/v2/channels/{slug}", timeout=20)
        r.raise_for_status()
        return r.json().get("livestream")


class RecorderManager:
    """Starts/stops recorders to match live status, respecting max_parallel and priority."""

    def __init__(self, api=None) -> None:
        self.api = api
        self.recorders: dict[str, Recorder] = {}
        self.live_info: dict[str, dict] = {}   # streamer id -> {"game_name": ...} for everyone live right now
        self.listeners: list[Callable[[str, Recorder | None], Awaitable[None]]] = []  # (streamer_id, recorder|None on stop)
        self.live_listeners: list[Callable[[str, bool, dict], Awaitable[None]]] = []  # (streamer_id, now live?, info)

    async def _live_twitch(self, streamers: list[Streamer]) -> dict[str, dict]:
        tw = [s for s in streamers if s.platform == "twitch"]
        if not tw or not self.api:
            return {}
        by_login = {s.login: s.id for s in tw}
        streams = await self.api.get_streams(list(by_login))
        return {by_login[x["user_login"]]: x for x in streams if x["user_login"] in by_login}

    async def _live_kick(self, streamers: list[Streamer]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for s in (s for s in streamers if s.platform == "kick"):
            if (ls := await kick_livestream(s.login)) is not None:
                out[s.id] = {"game_name": ((ls.get("categories") or [{}])[0] or {}).get("name", "")}
        return out

    def desired(self, streamers: list[Streamer], live: set[str]) -> list[str]:
        """Pure: only streamers that are live (Twitch Helix or Kick livestream != null), best priority first, capped."""
        want = [s for s in streamers if s.id in live]
        want.sort(key=lambda s: (s.priority, s.id))
        return [s.id for s in want[: get_config().recorder.max_parallel]]

    async def reconcile(self) -> None:
        streamers = list(get_streamers().values())
        live_info = {**await self._live_twitch(streamers), **await self._live_kick(streamers)}
        for sid in set(live_info) ^ set(self.live_info):  # live <-> offline transitions
            for cb in self.live_listeners:
                await cb(sid, sid in live_info, live_info.get(sid) or self.live_info.get(sid) or {})
        self.live_info = live_info
        want = set(self.desired(streamers, set(live_info)))
        for sid in list(self.recorders):
            if sid not in want:
                await self.recorders.pop(sid).stop()
                for cb in self.listeners:
                    await cb(sid, None)
        for s in streamers:
            if s.id in want and s.id not in self.recorders:
                rec = Recorder(s)
                rec.start()
                self.recorders[s.id] = rec
                for cb in self.listeners:
                    await cb(s.id, rec)

    async def loop(self) -> None:
        try:
            while True:
                try:
                    await self.reconcile()
                except Exception as e:  # noqa: BLE001
                    L.exception("reconcile: %s", e)
                await asyncio.sleep(60)
        finally:
            await asyncio.gather(*(r.stop() for r in self.recorders.values()), return_exceptions=True)
