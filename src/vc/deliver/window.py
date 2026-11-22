"""Window delivery: every window_minutes (counted from pipeline start) the 2-3 clips with the highest virality V."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable

from vc import db, log
from vc.config import DeliveryCfg, get_config
from vc.detect import virality
from vc.models import Clip, ClipStatus, Source

L = log.get("window")


# ---------------------------------------------------------------- pure selection (tested)
def effective_v(c: Clip, window_start: datetime | None, cfg: DeliveryCfg) -> float | None:
    """Stored V; a clip from an earlier window loses age_decay_per_5min points per 5 minutes it waited."""
    v = virality.known_v(c)
    if v is None:
        return None
    if window_start and c.created_at and c.created_at < window_start:
        v -= cfg.age_decay_per_5min * (window_start - c.created_at).total_seconds() / 300
    return v


def select_batch(clips: list[Clip], cfg: DeliveryCfg, live_count: int | None = None,
                 window_start: datetime | None = None) -> list[Clip]:
    """1. drop V < min_v  2. sort by V, drop duplicates  3. max_per_streamer (unless only one streamer is live)
    4. the top per_window_min, more up to per_window_max only with V >= third_clip_min_v."""
    from vc.orchestrator import duplicate_loser
    scored = [(v, c) for c in clips if (v := effective_v(c, window_start, cfg)) is not None and v >= cfg.min_v]
    scored.sort(key=lambda x: x[0], reverse=True)
    kept: list[tuple[float, Clip]] = []
    per: dict[str, int] = {}
    cap = live_count != 1
    for v, c in scored:
        if duplicate_loser(c, [k for _, k in kept]) is not None:  # overlaps a better clip already kept
            continue
        if cap and per.get(c.streamer_id, 0) >= cfg.max_per_streamer:
            continue
        if len(kept) >= cfg.per_window_max or (len(kept) >= cfg.per_window_min and v < cfg.third_clip_min_v):
            break
        kept.append((v, c))
        per[c.streamer_id] = per.get(c.streamer_id, 0) + 1
    return [c for _, c in kept]


def batch_note(n_selected: int, n_candidates: int, cfg: DeliveryCfg, live_count: int) -> str | None:
    """Why a window brought fewer than per_window_min clips (no filler below min_v)."""
    if n_selected >= cfg.per_window_min:
        return None
    if n_selected == 0 and live_count == 0 and n_candidates == 0:
        return "keiner live"
    if n_selected == 0:
        return f"kein Clip mit V ≥ {cfg.min_v:g}" + (f" ({n_candidates} Kandidaten)" if n_candidates else "")
    return f"nur {n_selected} Clip mit V ≥ {cfg.min_v:g}"


def discard_targets(c: Clip, data: Path) -> list[Path]:
    """Media this pipeline made for a clip: raw cut, out/*.mp4, .ass, .txt, .words.json. Never outside data/."""
    root = data.resolve()
    cands = [c.raw_path, c.out_path, c.ass_path, c.text_path, data / "out" / f"{c.id}.mp4", data / "out" / f"{c.id}.txt",
             data / "candidates" / f"{c.id}.ass", data / "candidates" / f"{c.id}.words.json"]
    out = []
    for p in dict.fromkeys(str(x) for x in cands if x):
        rp = Path(p).resolve()
        if rp.is_relative_to(root) and rp.is_file():
            out.append(rp)
    return out


def keep_always(c: Clip, busy: set[str]) -> bool:
    """Never discarded: delivered/posted, hotkey clips, clips a worker is on."""
    return (c.status in (ClipStatus.DELIVERED, ClipStatus.POSTED) or c.delivered_at is not None
            or c.source == Source.MANUAL or c.id in busy)


# ---------------------------------------------------------------- scheduler
class Window:
    def __init__(self, deliver: Callable[[str], Awaitable[None]], notify: Callable[[str], Awaitable[None]],
                 live_count: Callable[[], int] = lambda: 0, hub: Callable[[], object] = lambda: None,
                 clock: Callable[[], float] = time.time, busy: Callable[[], set[str]] = set) -> None:
        self.deliver, self.notify, self.live_count, self.hub, self.clock = deliver, notify, live_count, hub, clock
        self.busy = busy  # ids a worker is on right now
        self.start = 0.0
        self.number = 0
        self.delivered = 0
        self._last_note: str | None = None

    @property
    def length_s(self) -> float:
        return get_config().delivery.window_minutes * 60

    def minute(self) -> float:
        return max(0.0, (self.clock() - self.start) / 60)

    def status(self) -> str:
        return f"Fenster {self.number} · Minute {int(self.minute()) + 1}/{get_config().delivery.window_minutes:g} · {self.delivered} geliefert"

    async def _save(self) -> None:
        await db.kv_set("window_start", datetime.fromtimestamp(self.start, timezone.utc).isoformat())
        await db.kv_set("window_delivered", str(self.delivered))

    async def open(self, start: float | None = None) -> None:
        """A (re)start always opens a fresh window; its start lives in kv."""
        self.start = self.clock() if start is None else start
        self.number += 1
        self.delivered = 0
        await self._save()

    async def close(self) -> list[Clip]:
        cfg = get_config().delivery
        start_dt = datetime.fromtimestamp(self.start, timezone.utc)
        cutoff = db.now() - timedelta(minutes=cfg.stale_after_minutes)
        clips = [c for c in await db.list_clips(ClipStatus.RENDERED, limit=1000)
                 if c.delivered_at is None and c.created_at and c.created_at >= cutoff]
        for c in clips:  # viewers' clips and late chat links are in by now
            await virality.compute(c, hub=self.hub())
        live = self.live_count()
        batch = select_batch(clips, cfg, live, start_dt)
        if not cfg.auto or cfg.paused:
            await db.add_event("window_close", n=0, candidates=len(clips), skipped="pausiert" if cfg.paused else "auto aus")
            return []
        for c in batch:
            try:
                await self.deliver(c.id)
                self.delivered += 1
                await self._save()
            except Exception as e:  # noqa: BLE001  one broken clip must not cost the others
                L.exception("deliver %s", c.id)
                await db.add_event("error", clip_id=c.id, stage="window", error=str(e)[:300])
        await db.add_event("window_close", window=self.number, n=self.delivered, candidates=len(clips),
                           v=[virality.known_v(c) for c in batch])
        if cfg.delete_losers:
            await self.discard(clips, batch)
        note = batch_note(self.delivered, len(clips), cfg, live)
        if note and note != self._last_note:  # the same reason again (e.g. all night "keiner live") is not repeated
            await self.notify(f"⏱ Fenster {self.number}: {note}")
        self._last_note = note
        return batch

    async def discard(self, rendered: list[Clip], batch: list[Clip]) -> int:
        """Everything that did not make it: rendered losers of this close plus NEW/JUDGED older than one window.
        Status REJECTED with the reason, media deleted, the DB row (signals) stays as training data for vc tune."""
        chosen = {c.id for c in batch}
        old = db.now() - timedelta(seconds=self.length_s)
        pending = [c for st in (ClipStatus.NEW, ClipStatus.JUDGED) for c in await db.list_clips(st, limit=1000)
                   if c.created_at and c.created_at < old]
        data = get_config().data
        n = 0
        for c in rendered + pending:
            fresh = await db.get_clip(c.id)
            if c.id in chosen or not fresh or keep_always(fresh, set(self.busy())):
                continue
            v = virality.known_v(fresh)
            for p in discard_targets(fresh, data):
                p.unlink(missing_ok=True)
            await db.update_clip(c.id, status=ClipStatus.REJECTED, out_path="", ass_path="", text_path="",
                                 reason=f"nicht Top 3 (Fenster {self.number}, V {'–' if v is None else f'{v:.0f}'})")
            n += 1
        if n:
            await db.add_event("discard", window=self.number, n=n)
            L.info("Fenster %d: %d Clips verworfen, Medien geloescht", self.number, n)
        return n

    async def tick(self) -> bool:
        """Closes the window when it is over and opens the next one right behind it. True if it closed."""
        if self.clock() < self.start + self.length_s:
            return False
        await self.close()
        nxt = self.start + self.length_s
        await self.open(nxt if self.clock() < nxt + self.length_s else None)  # after a long stall: fresh window
        return True

    async def run(self, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, step_s: float = 5) -> None:
        await self.open()
        while True:
            await sleep(step_s)
            await self.tick()
