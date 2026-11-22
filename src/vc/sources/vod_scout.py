"""Source C: Gemini watches a YouTube video by URL and names the best moments; only those get downloaded.
Twitch VODs: chat replay through the trigger logic instead of video analysis."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Awaitable, Callable

from pydantic import BaseModel

from vc import db, log
from vc.config import get_config, get_streamers
from vc.models import CATEGORIES, Candidate, Source, Streamer
from vc.proc import run_ok

L = log.get("vod")
PAD_S = 5
MERGE_GAP_S = 10


class MomentFlags(BaseModel):
    cutscene: bool = False
    story_spoiler: bool = False
    music_dominant: bool = False


class Moment(BaseModel):
    start_s: float
    end_s: float
    hype: int
    category: str = "other"
    summary: str = ""
    has_facecam: bool = False
    flags: MomentFlags = MomentFlags()


class ScoutResult(BaseModel):
    moments: list[Moment]


# ---------------------------------------------------------------- pure helpers (tested)
def plan_chunks(duration: float, chunk_minutes: int) -> list[tuple[float, float]]:
    size = chunk_minutes * 60
    if duration <= size:
        return [(0.0, duration)]
    n = -(-int(duration) // size)
    step = duration / n  # equal chunks, none shorter than ~half
    return [(round(i * step, 1), round(min(duration, (i + 1) * step), 1)) for i in range(n)]


def merge_moments(moments: list[Moment], gap: float = MERGE_GAP_S) -> list[Moment]:
    out: list[Moment] = []
    for m in sorted(moments, key=lambda m: m.start_s):
        if out and m.start_s - out[-1].end_s < gap:
            prev = out[-1]
            best = prev if prev.hype >= m.hype else m
            out[-1] = best.model_copy(update={"start_s": min(prev.start_s, m.start_s), "end_s": max(prev.end_s, m.end_s),
                                              "hype": max(prev.hype, m.hype)})
        else:
            out.append(m)
    return out


def hms(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}"


def section(start: float, end: float, pad: float = PAD_S, duration: float | None = None) -> str:
    hi = end + pad if duration is None else min(duration, end + pad)
    return f"*{hms(start - pad)}-{hms(hi)}"


def select_moments(moments: list[Moment], min_hype: int, max_n: int) -> list[Moment]:
    keep = [m for m in moments if m.hype >= min_hype and not m.flags.cutscene and not m.flags.story_spoiler
            and m.end_s > m.start_s]
    return sorted(keep, key=lambda m: m.hype, reverse=True)[:max_n]


def to_absolute(moments: list[Moment], chunk: tuple[float, float]) -> list[Moment]:
    """We ask for segment-relative times; if the model answered with absolute ones, keep them."""
    c0, c1 = chunk
    length = c1 - c0
    absolute = c0 > 0 and any(m.start_s > length + 5 for m in moments)
    return [m if absolute else m.model_copy(update={"start_s": m.start_s + c0, "end_s": m.end_s + c0}) for m in moments]


def is_youtube(url: str) -> bool:
    return bool(re.search(r"(youtube\.com|youtu\.be)/", url))


def is_twitch_vod(url: str) -> bool:
    return bool(re.search(r"twitch\.tv/videos/\d+", url))


# ---------------------------------------------------------------- scouting
async def video_info(url: str) -> dict:
    r = await run_ok("yt-dlp", "--dump-json", "--no-warnings", "--skip-download", url, timeout=120)
    return json.loads(r.out)


def loudness_track(path: Path, rate: int = 4000) -> list[float]:
    """dB per second of a (long) audio file, streamed so a 4 h VOD never sits in memory."""
    import subprocess

    import numpy as np

    from vc.proc import tool
    p = subprocess.Popen([tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
                          "-ar", str(rate), "-f", "f32le", "pipe:1"], stdout=subprocess.PIPE)
    out: list[float] = []
    assert p.stdout
    while chunk := p.stdout.read(rate * 4):
        a = np.frombuffer(chunk[: len(chunk) // 4 * 4], dtype=np.float32)
        out.append(float(20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-6)) if a.size else -120.0)
    p.wait()
    return out


def loudness_peaks(db_per_s: list[float], n: int, spacing_s: int = 60, base_window_s: int = 300) -> list[tuple[int, float]]:
    """(second, excess dB) of the n strongest 3-s bursts above the rolling 5-min median, at least spacing_s apart.
    ponytail: audio-only heuristic; the local vision judge filters the false positives afterwards."""
    import numpy as np
    a = np.array(db_per_s, dtype=float)
    if a.size < 10:
        return []
    smooth = np.convolve(np.pad(a, 1, mode="edge"), np.ones(3) / 3, mode="valid")  # edge pad: no fake peaks at the ends
    half = base_window_s // 2
    base = np.array([np.median(a[max(0, i - half):i + half]) for i in range(0, a.size, 10)]).repeat(10)[: a.size]
    excess = smooth - base
    picked: list[tuple[int, float]] = []
    for i in np.argsort(-excess):
        if excess[i] <= 3 or len(picked) >= n:
            break
        if all(abs(int(i) - j) >= spacing_s for j, _ in picked):
            picked.append((int(i), round(float(excess[i]), 1)))
    return picked


async def scout_youtube_local(url: str, duration: float) -> list[Moment]:
    """Free replacement for the Gemini URL scout: audio track -> loudness bursts -> moments (judged later, locally)."""
    cfg = get_config()
    tmp = cfg.data / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    audio = tmp / f"scout-{abs(hash(url))}.m4a"
    try:
        await run_ok("yt-dlp", "-q", "--no-warnings", "-f", "ba[ext=m4a]/ba", "-o", audio, url, timeout=3600)
        peaks = loudness_peaks(await asyncio.to_thread(loudness_track, audio), cfg.vod.max_moments * 2)
    finally:
        audio.unlink(missing_ok=True)
    c = cfg.trigger
    return [Moment(start_s=max(0.0, t - c.pre_roll_s), end_s=min(duration or t + c.post_roll_s, t + c.post_roll_s),
                   hype=max(cfg.vod.min_hype, 9 - rank // 4), category="other", summary=f"Lautstaerke +{ex} dB")
            for rank, (t, ex) in enumerate(peaks)]


async def scout_youtube(url: str, duration: float, tag: str) -> list[Moment]:
    if get_config().judge.provider == "ollama":
        return await scout_youtube_local(url, duration)
    from google.genai import types

    from vc.judge.gemini import call_gemini, over_budget
    from vc.judge.prompts import VOD_SCOUT_PROMPT
    cfg = get_config()
    model = cfg.vod.model or cfg.judge.model
    found: list[Moment] = []
    for c0, c1 in plan_chunks(duration, cfg.vod.chunk_minutes):
        if await over_budget():  # long VODs must not blow the daily Gemini budget
            L.warning("Tagesbudget erreicht, Scout stoppt bei %s", hms(c0))
            break
        part = types.Part(file_data=types.FileData(file_uri=url, mime_type="video/*"),
                          video_metadata=types.VideoMetadata(start_offset=f"{int(c0)}s", end_offset=f"{int(c1)}s"))
        text = (f"You are watching the segment from {hms(c0)} to {hms(c1)} of this video. "
                "Give start_s and end_s in seconds relative to the start of THIS segment.")
        res = await call_gemini([part, text], VOD_SCOUT_PROMPT, ScoutResult, model, clip_id=tag, media_low=True)
        found += to_absolute(res.moments, (c0, c1))
    for m in found:
        if m.category not in CATEGORIES:
            m.category = "other"
    return merge_moments(found)


def twitch_vod_triggers(url: str, login: str) -> list[float]:
    """Download the VOD chat, write a chat log, replay it through the trigger logic -> trigger offsets (s)."""
    from chat_downloader import ChatDownloader

    from vc.detect.trigger import replay
    d = get_config().data / "chatlogs"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"vod-{re.search(r'videos/(\d+)', url).group(1)}.jsonl"  # type: ignore[union-attr]
    with path.open("w", encoding="utf-8") as f:
        for m in ChatDownloader().get_chat(url):
            t = m.get("time_in_seconds")
            if t is None:
                continue
            f.write(json.dumps({"ts": float(t), "channel": login, "user": (m.get("author") or {}).get("name", "?").lower(),
                                "text": m.get("message", "")}) + "\n")
    return [tr["t"] for tr in replay(path)]


def streamer_for_video(info: dict) -> Streamer:
    names = {str(info.get(k, "")).lower().lstrip("@") for k in ("uploader_id", "uploader", "channel", "channel_id")}
    for s in get_streamers().values():
        if s.login.lower() in names or (s.name and s.name.lower() in names):
            return s
    handle = str(info.get("uploader_id") or info.get("channel") or "unknown").lstrip("@")
    slug = re.sub(r"[^a-z0-9_]", "", handle.lower()) or "unknown"
    lang = (info.get("language") or "en")[:2]  # known streamers returned above with their .env lang
    lang = lang if lang in ("de", "en") else "en"
    twitch = "twitch" in str(info.get("extractor", "")).lower()
    return Streamer(id=slug, platform="twitch" if twitch else "youtube", login=slug, lang=lang, name=info.get("uploader") or handle,
                    account=get_config().harvest.account_by_lang.get(lang, "account_other"),
                    credit=f"twitch.tv/{slug}" if twitch else f"youtube.com/@{handle}", cam=None)


async def download_section(url: str, start: float, end: float, duration: float, out_stem: str) -> Path:
    d = get_config().data / "candidates"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{out_stem}.mp4"
    await run_ok("yt-dlp", "-q", "--no-warnings", "--download-sections", section(start, end, PAD_S, duration),
                 "--force-keyframes-at-cuts", "-f", "bv*[height<=1080]+ba/b", "--merge-output-format", "mp4",
                 "-o", out, url, timeout=600)
    return out


async def scout(url: str, dry_run: bool = False, notify: Callable[[str], Awaitable[None]] | None = None,
                tag: str = "vod") -> list[Moment]:
    from vc.sources.harvester import suggest_cam
    cfg = get_config().vod
    say = notify or (lambda _m: asyncio.sleep(0))
    info = await video_info(url)
    duration = float(info.get("duration") or 0)
    s = streamer_for_video(info)
    await say(f"Scout gestartet: {info.get('title', url)[:80]} ({hms(duration)})")
    if is_twitch_vod(url):
        c = get_config().trigger
        offsets = await asyncio.to_thread(twitch_vod_triggers, url, s.login)
        moments = [Moment(start_s=max(0, t - c.pre_roll_s), end_s=min(duration, t + c.post_roll_s), hype=7, category="other",
                          summary="chat trigger") for t in offsets]
        moments = merge_moments(moments)[: cfg.max_moments]
    elif is_youtube(url):
        moments = select_moments(await scout_youtube(url, duration, tag), cfg.min_hype, cfg.max_moments)
    else:
        raise ValueError(f"Unbekannter Link: {url}")
    await say(f"{len(moments)} Momente gefunden")
    if dry_run:
        return moments
    vid = re.sub(r"[^A-Za-z0-9_-]", "", str(info.get("id", "vod")))
    for m in moments:
        try:
            path = await download_section(url, m.start_s, m.end_s, duration, f"vod_{vid}_{int(m.start_s)}")
            st = s
            if st.cam is None and (cam := await asyncio.to_thread(suggest_cam, path)):
                st = s.model_copy(update={"cam": cam})
            await db.create_clip(Candidate(
                source=Source.VOD, streamer=st, raw_path=str(path),
                signals={"vod_hype": m.hype, "category": m.category, "summary": m.summary, "has_facecam": m.has_facecam},
                pre_score=float(m.hype), origin=f"{url}#t={int(m.start_s)}"))
        except Exception as e:  # noqa: BLE001
            L.warning("Abschnitt %s: %s", hms(m.start_s), str(e)[:200])
            await say(f"Fehler bei {hms(m.start_s)}: {str(e)[:120]}")
    return moments


async def worker(notify: Callable[[str], Awaitable[None]] | None = None) -> None:
    while True:
        async with db.connect() as conn:
            async with conn.execute("SELECT id, url FROM vod_jobs WHERE status = 'NEW' ORDER BY id LIMIT 1") as cur:
                row = await cur.fetchone()
        if not row:
            await asyncio.sleep(10)
            continue
        job_id, url = row
        await _set_job(job_id, "RUNNING")
        try:
            moments = await scout(url, notify=notify, tag=f"vod:{job_id}")
            await db.add_event("vod_job", job=job_id, url=url, moments=len(moments), usd=await _job_cost(job_id))
            await _set_job(job_id, "DONE")
        except Exception as e:  # noqa: BLE001
            L.exception("vod job %s", job_id)
            await _set_job(job_id, "ERROR", str(e)[:300])
            if notify:
                await notify(f"Scout-Fehler: {str(e)[:150]}")


async def _job_cost(job_id: int) -> float:
    return round(sum(e["data"].get("usd", 0) for e in await db.list_events("gemini_usage", limit=10000)
                     if e["clip_id"] == f"vod:{job_id}"), 4)


async def _set_job(job_id: int, status: str, error: str = "") -> None:
    async with db.connect() as conn:
        await conn.execute("UPDATE vod_jobs SET status = ?, error = ? WHERE id = ?", (status, error, job_id))
        await conn.commit()
