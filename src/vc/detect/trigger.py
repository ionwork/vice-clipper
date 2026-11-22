"""TriggerEngine: chat + audio signals -> live candidates. Also replay over chat logs for tuning."""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from vc import db, log
from vc.config import get_config, get_streamer
from vc.detect.audio import AudioMonitor
from vc.detect.chat_stats import ChannelStats, stats_from_config
from vc.detect.virality import CHAT_KEYS, n_reactions
from vc.models import Candidate, Source, Streamer

L = log.get("trigger")
WARMUP_S = 200
Z_MIN, Z_MAX = 3.0, 7.0


@dataclass
class ChannelState:
    stats: ChannelStats
    started: float
    lang: str = "en"
    last_trigger: float = 0.0
    triggers: deque = field(default_factory=deque)        # trigger times, last hour
    adapt: list = field(default_factory=list)              # [(delta, expires)]
    last_adapt: float = 0.0
    audio: AudioMonitor | None = None
    spark: deque = field(default_factory=lambda: deque(maxlen=36))  # burst every 5 s, 3 min
    snap: dict = field(default_factory=dict)


def pre_score(z: float, lex: float, audio: bool, clipcall: bool) -> float:
    return round(min(10.0, 2 + 0.9 * min(max(z, 0), 6) + 2 * lex + 1.5 * audio + 2 * clipcall), 2)


class TriggerEngine:
    def __init__(self, hub=None) -> None:
        self.hub = hub
        self.ch: dict[str, ChannelState] = {}   # key: streamer id
        self.login_to_id: dict[str, str] = {}
        self.on_trigger = self._make_candidate  # replaced in replay mode
        self.category_of: Callable[[str], str] = lambda sid: ""  # Twitch category right now (set by the orchestrator)
        self._tasks: set[asyncio.Task] = set()
        if hub is not None:
            hub.listeners.append(lambda m: self.feed(m.channel, m.ts, m.user, m.text, m.emotes))

    # ------------------------------------------------------------ tracking
    def track(self, s: Streamer, started: float | None = None, audio: AudioMonitor | None = None) -> None:
        self.ch[s.id] = ChannelState(stats_from_config(s.lang), started or time.time(), s.lang, audio=audio)
        self.login_to_id[s.login.lower()] = s.id

    def untrack(self, sid: str) -> None:
        self.ch.pop(sid, None)
        self.login_to_id = {k: v for k, v in self.login_to_id.items() if v != sid}

    def feed(self, login: str, ts: float, user: str, text: str, emotes: frozenset[str] = frozenset()) -> None:
        sid = self.login_to_id.get(login.lower())
        if sid and sid in self.ch:
            self.ch[sid].stats.add(ts, user, text, emotes)

    def set_channel_emotes(self, sid: str, names: frozenset[str]) -> None:
        if sid in self.ch:
            self.ch[sid].stats.channel_emotes = names

    def z_threshold(self, st: ChannelState, now: float) -> float:
        st.adapt = [(d, e) for d, e in st.adapt if e > now]
        return max(Z_MIN, min(Z_MAX, get_config().trigger.z_threshold + sum(d for d, _ in st.adapt)))

    # ------------------------------------------------------------ evaluation
    def evaluate(self, sid: str, now: float) -> dict | None:
        st = self.ch[sid]
        c = get_config().trigger
        snap = st.stats.snapshot(now)
        st.snap = {**snap, "zt": self.z_threshold(st, now)}
        if int(now) % 5 == 0:
            st.spark.append(snap["burst"])
        if now - st.started < WARMUP_S:
            return None
        zt = st.snap["zt"]
        strong = snap["z"] >= zt and snap["effective"] >= c.min_burst_msgs and snap["unique"] >= c.min_unique_ratio
        clipcall = snap["clipcalls"] >= c.clip_call_count
        self._adapt_down(st, now, snap)
        if not (strong or clipcall) or now - st.last_trigger < c.cooldown_s:
            return None
        audio_db = st.audio.spike(now) if st.audio else 0.0
        audio = audio_db > 0
        react = n_reactions(snap.get("reaction_lift", 1.0), get_config().virality)  # V2: reactions count like hype words
        pre = pre_score(snap["z"], max(snap["lex"], react), audio, clipcall)
        if pre < c.min_pre_score and not clipcall:  # "clip" calls always go through
            return None
        st.last_trigger = now
        st.triggers.append(now)
        while st.triggers and st.triggers[0] < now - 3600:
            st.triggers.popleft()
        if len(st.triggers) > 12 and now - st.last_adapt > 600:
            st.adapt.append((0.5, now + 3600))
            st.last_adapt = now
            L.info("%s: >12 Ausloeser/h, z_threshold -> %.1f", sid, self.z_threshold(st, now))
        return {**snap, "t": now, "pre_score": pre, "audio": audio, "audio_db": audio_db, "clipcall": clipcall, "zt": zt,
                "top": st.stats.top_tokens(now)}

    def _adapt_down(self, st: ChannelState, now: float, snap: dict) -> None:
        quiet_since = max(st.started + WARMUP_S, st.last_trigger)
        if (now - quiet_since > 3600 and snap["burst"] >= get_config().trigger.min_burst_msgs
                and now - st.last_adapt > 3600):
            st.adapt.append((-0.5, now + 3600))
            st.last_adapt = now
            L.info("Kein Ausloeser seit 60 min bei viel Chat, z_threshold -> %.1f", self.z_threshold(st, now))

    # ------------------------------------------------------------ live
    async def run(self) -> None:
        while True:
            now = time.time()
            for sid in list(self.ch):
                try:
                    if trig := self.evaluate(sid, now):
                        await db.add_event("trigger", streamer_id=sid, **{k: v for k, v in trig.items() if k != "t"})
                        t = asyncio.create_task(self.on_trigger(sid, trig))
                        self._tasks.add(t)
                        t.add_done_callback(self._tasks.discard)
                except Exception as e:  # noqa: BLE001
                    L.exception("%s: %s", sid, e)
            await asyncio.sleep(max(0.0, 1 - (time.time() - now)))

    def _chat_context(self, sid: str, t: float) -> list[str]:
        """Last 40 chat lines before t (from the hub's 10-minute buffer, else the stats' 10 s)."""
        login = next((lg for lg, i in self.login_to_id.items() if i == sid), sid)
        if self.hub is not None:
            msgs = [(m.user, m.text) for m in self.hub.recent(login, 180) if m.ts <= t]
        else:
            st = self.ch.get(sid)
            msgs = [(m.user, m.text) for m in (st.stats.recent if st else []) if m.ts <= t]
        return [f"{u}: {x}" for u, x in msgs][-40:]

    async def _make_candidate(self, sid: str, trig: dict, pre_s: float | None = None, post_s: float | None = None) -> str | None:
        from vc.render.ffmpeg import probe
        from vc.sources.recorder import cut
        c = get_config().trigger
        pre_s = c.pre_roll_s if pre_s is None else pre_s
        post_s = c.post_roll_s if post_s is None else post_s
        s = get_streamer(sid)
        if not s:
            return None
        t = trig["t"]
        await asyncio.sleep(max(0.0, t + post_s - time.time()))
        out = get_config().data / "candidates" / f"live_{sid}_{int(t)}.mp4"
        try:
            path, first = await cut(sid, datetime.fromtimestamp(t - pre_s), datetime.fromtimestamp(t + post_s), out)
            if (await probe(path)).duration < 12:
                raise RuntimeError("Kandidat unter 12 s (Luecke im Puffer), verworfen")
        except Exception as e:  # noqa: BLE001
            await db.add_event("error", streamer_id=sid, stage="cut", error=str(e)[:300])
            return None
        signals = {k: trig[k] for k in (*CHAT_KEYS, "audio", "top", "audio_db") if k in trig}
        if cat := self.category_of(sid):
            signals["stream_category"] = cat
        clip = await db.create_clip(Candidate(
            source=Source.MANUAL if trig.get("manual") else Source.LIVE, streamer=s, raw_path=str(path),
            t_trigger=datetime.fromtimestamp(t).astimezone(), trigger_offset_s=round(t - first.timestamp(), 2),
            signals=signals, pre_score=trig["pre_score"], chat_context=self._chat_context(sid, t),
            origin=f"live:{sid}:{int(t)}"))
        from vc.detect import virality  # provisional V (no judge yet, viewer clips mostly still missing)
        await virality.compute(clip, hub=self.hub)
        return clip.id

    async def manual(self, streamer_id: str, seconds: int = 30) -> str:
        """Hotkey: the last N seconds of a channel, pre_score 10."""
        now = time.time()
        cid = await self._make_candidate(streamer_id, {"t": now, "pre_score": 10.0, "manual": True}, pre_s=seconds, post_s=0)
        if not cid:
            raise RuntimeError(f"{streamer_id}: kein Puffer (Recorder laeuft nicht?)")
        await db.add_event("trigger", streamer_id=streamer_id, manual=True, pre_score=10)
        return cid


# ---------------------------------------------------------------- replay
def replay(path: Path, speed: float = 0, timing: dict | None = None) -> list[dict]:
    """Feed a chat log through the stats/trigger logic with a simulated clock. No recorder needed.
    timing (optional) receives the evaluate() cost per channel and tick: avg_ms, p99_ms, max_ms, ticks."""
    msgs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not msgs:
        return []
    msgs.sort(key=lambda m: m["ts"])
    eng = TriggerEngine()
    triggers: list[dict] = []
    for login in {m["channel"] for m in msgs}:
        s = get_streamer(login) or Streamer(id=login, login=login, lang="en")
        eng.track(s, started=msgs[0]["ts"])
    i, t = 0, int(msgs[0]["ts"]) + 1
    costs: list[float] = []
    end = msgs[-1]["ts"] + 1
    while t <= end:
        while i < len(msgs) and msgs[i]["ts"] < t:
            m = msgs[i]
            eng.feed(m["channel"], m["ts"], m["user"], m["text"], frozenset(m.get("emotes") or ()))
            i += 1
        for sid in list(eng.ch):
            t0 = time.perf_counter()
            trig = eng.evaluate(sid, t)
            costs.append(time.perf_counter() - t0)
            if trig:
                triggers.append({"channel": sid, **trig})
        if speed:
            time.sleep(1 / speed)
        t += 1
    by_channel: dict[str, list[tuple[float, str, str]]] = {}
    for m in msgs:
        by_channel.setdefault(m["channel"], []).append((m["ts"], m["user"], m["text"]))
    for trig in triggers:  # provisional V; the log holds the whole chat, so the clip-call window is complete
        trig["virality"] = replay_virality(trig, by_channel[trig["channel"]])
    if timing is not None and costs:
        costs.sort()
        timing.update(avg_ms=1000 * sum(costs) / len(costs), p99_ms=1000 * costs[int(0.99 * (len(costs) - 1))],
                      max_ms=1000 * costs[-1], ticks=len(costs))
    return triggers


def replay_virality(trig: dict, msgs: list[tuple[float, str, str]]):
    """V from what a chat log alone can tell: velocity, reaction, convergence, intensity, clip calls, clip links
    (no viewer clips, voice, judge)."""
    from vc.detect import virality as vr
    cfg = get_config()
    c = cfg.virality
    links = vr.chat_links(msgs, trig["t"], c)
    comps: dict = {}
    vr.chat_comps(trig, c, comps, {})
    comps["clip_calls"] = vr.n_clip_calls(vr.clipcall_users(msgs, trig["t"], c, cfg.chat.ignore_users), c)
    comps["library"] = vr.n_library(0, links, c.library_expected_min, False, c) if links else None
    return vr.score(comps, c)


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts > 10**8 else str(timedelta(seconds=int(ts)))
