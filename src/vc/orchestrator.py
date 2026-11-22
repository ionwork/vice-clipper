"""Orchestrator: one process runs every worker, supervised, with priorities, dedup, resume and cleanup."""
from __future__ import annotations

import asyncio
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable

from vc import db, log
from vc.config import get_config
from vc.detect.virality import known_v
from vc.models import Clip, ClipStatus, Source

L = log.get("orchestrator")
TIER = {"manual": 0, "clipcall": 1, "live": 2, "harvest": 3, "vod": 4}
MAX_AGE_H = 2
ACTIVE = (ClipStatus.NEW, ClipStatus.JUDGED, ClipStatus.CAPTIONED)


# ---------------------------------------------------------------- pure rules (tested)
def tier(c: Clip) -> int:
    if c.source == Source.LIVE and (c.signals or {}).get("clipcalls", 0) >= get_config().trigger.clip_call_count:
        return TIER["clipcall"]
    return TIER[c.source.value]


def priority_key(c: Clip, now: datetime) -> tuple[int, float]:
    """Hotkey < clip calls < live < harvest < vod; inside a tier the higher virality V wins (V/10, falls back to
    hype or pre_score when no V is known); live candidates lose one point per 10 minutes of waiting."""
    v = known_v(c)
    score = (v / 10 if v is not None else c.hype if c.hype is not None else c.pre_score) + c.bonus
    if c.source in (Source.LIVE, Source.MANUAL) and c.created_at:
        score -= (now - c.created_at).total_seconds() / 600
    return tier(c), -score


def window(c: Clip) -> tuple[float, float] | None:
    """Wall-clock window of the moment (epoch seconds); None if unknown (VOD)."""
    if not c.t_trigger:
        return None
    t = c.t_trigger.timestamp()
    if c.source in (Source.HARVEST, Source.MANUAL):
        return t - 30, t
    tc = get_config().trigger
    return t - tc.pre_roll_s, t + tc.post_roll_s


def overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return max(0.0, inter) / max(1e-6, min(a[1] - a[0], b[1] - b[0]))


def duplicate_loser(c: Clip, others: list[Clip]) -> str | None:
    """Same streamer and windows overlapping by more than half: the lower score loses. Returns the loser's id."""
    w = window(c)
    if not w:
        return None
    for o in others:
        if o.id == c.id or o.streamer_id != c.streamer_id or o.status in (ClipStatus.REJECTED, ClipStatus.FAILED):
            continue
        wo = window(o)
        if wo and overlap_ratio(w, wo) > 0.5:
            mine = c.hype if c.hype is not None else c.pre_score
            theirs = o.hype if o.hype is not None else o.pre_score
            return c.id if (mine, -tier(c)) <= (theirs, -tier(o)) else o.id
    return None


def stale(c: Clip, now: datetime) -> bool:
    return c.status in ACTIVE and c.created_at is not None and now - c.created_at > timedelta(hours=MAX_AGE_H)


def cleanup_targets(data: Path, now: float) -> list[Path]:
    """Raw cuts and proxies after 24 h, finished clips after 7 days."""
    out: list[Path] = []
    for sub, days in (("candidates", 1), ("tmp", 1), ("out", 7)):
        d = data / sub
        if d.is_dir():
            out += [p for p in d.iterdir() if p.is_file() and now - p.stat().st_mtime > days * 86400]
    return out


# ---------------------------------------------------------------- live stack
class LiveStack:
    """Recorders + chat + trigger engine, wired: a started recorder joins chat and is tracked by the engine."""

    def __init__(self) -> None:
        from vc.detect.trigger import TriggerEngine
        from vc.sources.chat_kick import KickChat
        from vc.sources.chat_twitch import ChatHub, TwitchChat
        from vc.sources.recorder import RecorderManager
        from vc.sources.twitch_api import TwitchAPI
        self.api = TwitchAPI()
        self.hub = ChatHub()
        self.twitch_chat = TwitchChat(self.hub)
        self.kick_chat = KickChat(self.hub)
        self.engine = TriggerEngine(self.hub)
        self.recorders = RecorderManager(self.api)
        self.recorders.listeners.append(self._on_recorder)
        self.recorders.live_listeners.append(self._on_live)
        self.last_category: dict[str, str] = {}  # streamer id -> last Twitch category seen
        self._bg: set[asyncio.Task] = set()
        self.engine.category_of = self.category_of
        self.notify: Callable[[str], Awaitable[None]] | None = None

    async def _on_live(self, sid: str, is_live: bool, info: dict) -> None:
        from vc.config import get_streamer
        s = get_streamer(sid)
        name = s.display if s else sid
        game = info.get("game_name") or ""
        if is_live and game:
            self.last_category[sid] = game
        text = f"🔴 {name} live" + (f" · {game}" if game else "") if is_live else f"⚫ {name} offline"
        await db.add_event("live" if is_live else "offline", streamer_id=sid, game=game)
        if self.notify:
            await self.notify(text)

    async def _load_emotes(self, s) -> None:
        from vc.sources.emotes import channel_emotes
        try:
            tid = (await self.api.user_ids([s.login])).get(s.login) if s.platform == "twitch" else None
        except Exception as e:  # noqa: BLE001
            L.warning("Twitch-ID fuer %s nicht abrufbar: %s", s.login, str(e)[:100])
            tid = None
        names = await channel_emotes(s.login, tid)
        self.engine.set_channel_emotes(s.id, names)
        L.info("%s: %d Third-Party-Emotes", s.login, len(names))

    def category_of(self, sid: str) -> str:
        """Current category (live_info refreshes every live check, so a switch to Just Chatting is seen)."""
        game = (self.recorders.live_info.get(sid) or {}).get("game_name") or ""
        if game:
            self.last_category[sid] = game
        return game or self.last_category.get(sid, "")

    async def _on_recorder(self, sid: str, rec) -> None:
        from vc.config import get_streamer
        from vc.detect.audio import AudioMonitor
        s = get_streamer(sid)
        if rec is None or s is None:
            self.engine.untrack(sid)
            if s and s.platform == "twitch":
                await self.twitch_chat.part(s.login)
            return
        audio = AudioMonitor()
        rec.on_segment = audio.feed_segment
        self.engine.track(s, audio=audio)
        t = asyncio.create_task(self._load_emotes(s))  # 7TV/BTTV/FFZ once per channel, never blocks
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)
        if s.platform == "kick":
            await self.kick_chat.join(s.login)
        else:
            self.twitch_chat.join(s.login)

    def tasks(self) -> dict[str, Callable[[], Awaitable[None]]]:
        return {"recorders": self.recorders.loop, "twitch_chat": self.twitch_chat.run, "kick_chat": self.kick_chat.run,
                "triggers": self.engine.run}

    def status_lines(self) -> list[str]:
        live = self.recorders.live_info
        lines = [f"Live: {', '.join(sorted(live)) or 'keiner'} · überwacht: {', '.join(sorted(self.recorders.recorders)) or '–'}"]
        for sid, rec in self.recorders.recorders.items():
            st = self.engine.ch.get(sid)
            z = st.snap.get("z", 0) if st else 0
            last = rec.last_segment_at
            lines.append(f"● {sid}: z {z:+.1f}, Segment {last:%H:%M:%S}" if last else f"○ {sid}: wartet auf Stream")
        return lines


# ---------------------------------------------------------------- orchestrator
class Orchestrator:
    def __init__(self, live: bool = True, harvest: bool = True, vod: bool = True, web: bool = True, bot: bool = True,
                 port: int = 8000) -> None:
        self.flags = dict(live=live, harvest=harvest, vod=vod, web=web, bot=bot)
        self.port = port
        self.live: LiveStack | None = None
        self.bot = None
        self.claimed: set[str] = set()
        self._notified: set[str] = set()
        self._hb: dict = {}
        self._budget_day = ""
        self.window = None
        self.tasks: dict[str, asyncio.Task] = {}
        self._bg: set[asyncio.Task] = set()  # pending Telegram notices

    # ------------------------------------------------------------ supervision
    async def notify(self, text: str) -> None:
        """Fire and forget: the pipeline never waits for Telegram (which can hang for minutes)."""
        if self.bot:
            t = asyncio.create_task(self.bot.notify(text))
            self._bg.add(t)
            t.add_done_callback(self._bg.discard)

    async def supervise(self, name: str, fn: Callable[[], Awaitable[None]]) -> None:
        backoff = 2.0
        while True:
            started = time.time()
            try:
                await fn()
                L.warning("%s beendet, starte neu", name)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                L.exception("%s abgestuerzt: %s", name, e)
                await db.add_event("error", stage=name, error=str(e)[:300])
                if name not in self._notified:  # report each component once
                    self._notified.add(name)
                    await self.notify(f"⚠️ {name} abgestürzt: {str(e)[:150]} (startet neu)")
            if time.time() - started > 300:
                backoff = 2.0
            await asyncio.sleep(backoff)
            backoff = min(120.0, backoff * 2)

    # ------------------------------------------------------------ queue workers
    async def _next(self, statuses: tuple[ClipStatus, ...]) -> Clip | None:
        now = db.now()
        clips = [c for st in statuses for c in await db.list_clips(st, limit=1000) if c.id not in self.claimed]
        if not clips:
            return None
        return min(clips, key=lambda c: priority_key(c, now))

    async def _dedup(self, c: Clip) -> bool:
        """True if c survives the duplicate rule."""
        others = await db.list_clips(since=db.now() - timedelta(hours=MAX_AGE_H), limit=2000)
        loser = duplicate_loser(c, others)
        if loser:
            await db.update_clip(loser, status=ClipStatus.REJECTED, reason="Duplikat")
            await db.add_event("duplicate", clip_id=loser, streamer_id=c.streamer_id)
        return loser != c.id

    def hub(self):
        return self.live.hub if self.live else None

    async def shed(self) -> int:
        """Judge is the bottleneck: every NEW gets a provisional V; with more than judge_queue_max waiting,
        candidates with V < judge_min are rejected without judging (hotkey clips are never shed)."""
        from vc.detect import virality
        vc = get_config().virality
        new = [c for c in await db.list_clips(ClipStatus.NEW, limit=1000) if c.id not in self.claimed]
        for c in new:
            if virality.stored(c) is None:  # harvest / vod candidates are created without V
                await virality.compute(c, hub=self.hub())
        if len(new) <= vc.judge_queue_max:
            return 0
        n = 0
        for c in new:
            v = known_v(c)
            if c.source != Source.MANUAL and v is not None and v < vc.judge_min:
                await db.update_clip(c.id, status=ClipStatus.REJECTED, reason="V zu niedrig")
                await db.add_event("shed", clip_id=c.id, streamer_id=c.streamer_id, v=v)
                n += 1
        return n

    async def judge_worker(self) -> None:
        from vc.detect import virality
        from vc.judge.gemini import judge_candidate
        while True:
            await self.shed()
            c = await self._next((ClipStatus.NEW,))
            if not c:
                await asyncio.sleep(2)
                continue
            self.claimed.add(c.id)
            try:
                if await self._dedup(c):
                    await judge_candidate(c.id)
                    if (j := await db.get_clip(c.id)) and j.status == ClipStatus.JUDGED:
                        await virality.compute(j, hub=self.hub())  # V now includes the judge
            except Exception as e:  # noqa: BLE001
                L.exception("judge %s", c.id)
                await db.update_clip(c.id, status=ClipStatus.FAILED, reason=f"judge: {str(e)[:250]}")
            finally:
                self.claimed.discard(c.id)

    async def _render_allowed(self) -> set[str] | None:
        """Window mode: only clips that can realistically make the top 3 get rendered: the best render_max of all
        undelivered JUDGED/CAPTIONED/RENDERED by V (hotkey clips always). None = no limit (gap mode)."""
        d = get_config().delivery
        if not self.window:
            return None
        pool = [c for st in (ClipStatus.JUDGED, ClipStatus.CAPTIONED, ClipStatus.RENDERED)
                for c in await db.list_clips(st, limit=1000) if c.delivered_at is None]
        best = sorted((c for c in pool if known_v(c) is not None), key=known_v, reverse=True)[: d.render_max]
        return {c.id for c in best} | {c.id for c in pool if c.source == Source.MANUAL}

    async def _next_render(self) -> Clip | None:
        """From render_boost_minute of a window on, the best JUDGED by V render first (ready for the window close).
        Clips outside the realistic top wait; the window close discards them if they stay out."""
        d = get_config().delivery
        pend = [c for st in (ClipStatus.JUDGED, ClipStatus.CAPTIONED) for c in await db.list_clips(st, limit=1000)
                if c.id not in self.claimed]
        if not pend:  # idle loop: one query, as before
            return None
        allowed = await self._render_allowed()
        if allowed is not None:
            pend = [c for c in pend if c.id in allowed]
        if self.window and self.window.minute() >= d.render_boost_minute:
            if best := [c for c in pend if known_v(c) is not None]:
                return max(best, key=known_v)
        now = db.now()
        return min(pend, key=lambda c: priority_key(c, now)) if pend else None

    async def render_worker(self) -> None:
        from vc.render.render import render_from_db
        while True:
            c = await self._next_render()
            if not c:
                await asyncio.sleep(2)
                continue
            self.claimed.add(c.id)
            task = asyncio.create_task(render_from_db(c.id))
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                L.info("Shutdown: laufenden Render %s fertigstellen", c.id)
                await asyncio.gather(task, return_exceptions=True)
                raise
            except Exception:  # noqa: BLE001  render_from_db already set FAILED
                L.exception("render %s", c.id)
            finally:
                self.claimed.discard(c.id)

    # ------------------------------------------------------------ housekeeping
    async def resume(self) -> None:
        now = db.now()
        n_old = n_back = 0
        for st in ACTIVE:
            for c in await db.list_clips(st, limit=5000):
                if stale(c, now):
                    await db.update_clip(c.id, status=ClipStatus.REJECTED, reason="zu alt (Neustart)")
                    n_old += 1
                    continue
                if st == ClipStatus.CAPTIONED:
                    await db.update_clip(c.id, status=ClipStatus.JUDGED)
                n_back += 1
        async with db.connect() as conn:
            await conn.execute("UPDATE vod_jobs SET status = 'NEW' WHERE status = 'RUNNING'")
            await conn.commit()
        L.info("Wiederanlauf: %d Clips wieder eingereiht, %d zu alt verworfen", n_back, n_old)

    async def cleanup(self) -> None:
        while True:
            now = db.now()
            for st in ACTIVE:
                for c in await db.list_clips(st, limit=5000):
                    if stale(c, now) and c.id not in self.claimed:
                        await db.update_clip(c.id, status=ClipStatus.REJECTED, reason="zu alt")
            files = cleanup_targets(get_config().data, time.time())
            for p in files:
                p.unlink(missing_ok=True)
            if files:
                L.info("Aufraeumen: %d Dateien geloescht", len(files))
            await asyncio.sleep(3600)

    def heartbeat(self) -> dict:
        return self._hb

    async def heartbeat_loop(self) -> None:
        from vc.judge.gemini import over_budget
        while True:
            hour = await db.list_clips(since=db.now() - timedelta(hours=1), limit=5000)
            queue = {st.value: len(await db.list_clips(st, limit=5000)) for st in (ClipStatus.NEW, ClipStatus.JUDGED, ClipStatus.RENDERED)}
            _, _, usd = await db.cost_today()
            recs = self.live.recorders.recorders if self.live else {}
            self._hb = {
                "recorders": f"{sum(1 for r in recs.values() if r.last_segment_at)} live / {len(recs)}",
                "queue": " · ".join(f"{k} {v}" for k, v in queue.items()),
                "last_hour": " · ".join(f"{k} {v}" for k, v in Counter(c.status.value for c in hour).most_common()) or "–",
                "usd": round(usd, 3),
            }
            L.info("heartbeat | recorder %s | queue %s | 1h %s | %.3f $",
                   self._hb["recorders"], self._hb["queue"], self._hb["last_hour"], usd)
            day = datetime.now().date().isoformat()
            if self._budget_day != day and await over_budget():
                self._budget_day = day
                await self.notify("💸 Gemini-Tagesbudget erreicht: Bewertung läuft bis Mitternacht ohne KI.")
            await asyncio.sleep(60)

    def status_lines(self) -> list[str]:
        lines = [f"Pipeline: {'läuft' if self.running else 'gestoppt (/start startet)'}"]
        lines += self.live.status_lines() if self.live else ["Live: aus"]
        if self.window:
            lines.append(self.window.status())
        if self._hb:
            lines.append(f"Recorder {self._hb['recorders']} · Queue {self._hb['queue']}")
        return lines

    # ------------------------------------------------------------ pieces for vc start / vc run
    async def manual(self, streamer_id: str, seconds: int = 30) -> str:
        """Hotkey /clip; resolved at call time because /stop and /go rebuild the live stack."""
        if not self.live:
            raise RuntimeError("Pipeline läuft nicht (/start)")
        return await self.live.engine.manual(streamer_id, seconds)

    def make_bot(self, control=None):
        from vc.deliver.telegram_bot import ClipBot
        self.bot = ClipBot(trigger_manual=self.manual, status_extra=self.status_lines, control=control)
        return self.bot

    def side_jobs(self) -> dict[str, Callable[[], Awaitable[None]]]:
        """Always on while the process lives (also in standby): Telegram bot and dashboard."""
        jobs: dict[str, Callable[[], Awaitable[None]]] = {}
        if self.bot:
            jobs["bot"] = self.bot.run
        if self.flags["web"]:
            from vc.web.app import create_app, serve
            app = create_app(self)
            jobs["web"] = lambda: serve(app, self.port)
            L.info("Dashboard: http://127.0.0.1:%d", self.port)
        return jobs

    def pipeline_jobs(self) -> dict[str, Callable[[], Awaitable[None]]]:
        """Everything /go starts and /stop stops."""
        from vc.deliver.window import Window
        f = self.flags
        jobs: dict[str, Callable[[], Awaitable[None]]] = {}
        if f["live"]:
            self.live = LiveStack()
            self.live.notify = self.notify
            jobs.update(self.live.tasks())
        if f["harvest"]:
            from vc.sources.harvester import Harvester
            jobs["harvester"] = Harvester(api=self.live.api if self.live else None).loop
        if f["vod"]:
            from vc.sources import vod_scout
            jobs["vod_scout"] = lambda: vod_scout.worker(self.notify if self.bot else None)
        for i in range(max(1, get_config().judge.max_parallel)):
            jobs[f"judge{i + 1}"] = self.judge_worker
        jobs["render"] = self.render_worker
        jobs["cleanup"] = self.cleanup
        jobs["heartbeat"] = self.heartbeat_loop
        if self.bot and get_config().delivery.mode == "window":
            self.window = Window(self.bot.deliver, self.notify, self.live_count, self.hub, busy=lambda: set(self.claimed))
            jobs["window"] = self.window.run
        return jobs

    def live_count(self) -> int:
        return len(self.live.recorders.live_info) if self.live else 0

    @property
    def running(self) -> bool:
        return bool(self.tasks)

    async def start_pipeline(self) -> bool:
        """False if it is already running (no second set of workers) or STREAMERS in .env is empty."""
        from vc.config import NO_STREAMERS_MSG, get_streamers
        if self.tasks:
            return False
        if not get_streamers():
            L.warning(NO_STREAMERS_MSG)
            await self.notify("⚠️ " + NO_STREAMERS_MSG)
            return False
        await self.resume()
        jobs = self.pipeline_jobs()
        self.tasks = {n: asyncio.create_task(self.supervise(n, fn), name=n) for n, fn in jobs.items()}
        L.info("Pipeline gestartet: %s", ", ".join(jobs))
        await self.notify("▶️ Pipeline gestartet: " + ", ".join(k for k, v in self.flags.items() if v))
        return True

    async def stop_pipeline(self) -> bool:
        """Cancels the workers: the running render finishes its clip, recorders stop in RecorderManager.loop."""
        if not self.tasks:
            return False
        tasks = list(self.tasks.values())
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks = {}
        self.live = self.window = None
        self.claimed.clear()
        L.info("Pipeline gestoppt")
        return True
