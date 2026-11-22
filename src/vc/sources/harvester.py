"""Source B: Twitch clips of the category that are gaining views fast."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from vc import db, log
from vc.config import get_config, get_streamers
from vc.models import Candidate, Rect, Source, Streamer
from vc.proc import run_ok, tool
from vc.sources.twitch_api import TwitchAPI

L = log.get("harvest")
DUP_WINDOW_S = 40


# ---------------------------------------------------------------- pure logic (tested)
def velocity(view_count: int, created_at: datetime, now: datetime, prev: tuple[datetime, int] | None) -> float:
    """Views per minute since the previous poll; for new clips: views / age."""
    if prev:
        minutes = max((now - prev[0]).total_seconds() / 60, 0.5)
        return max(0.0, (view_count - prev[1]) / minutes)
    return view_count / max((now - created_at).total_seconds() / 60, 1.0)


def passes_filters(c: dict, whitelist: set[str]) -> bool:
    h = get_config().harvest
    if c["view_count"] < h.min_views or c.get("duration", 0) > h.max_duration_s:
        return False
    login = c.get("broadcaster_login", c["broadcaster_name"].lower())
    if h.languages and login not in whitelist and c.get("language") not in h.languages:  # own streamers: lang from .env
        return False
    if h.only_whitelisted and login not in whitelist:
        return False
    return True


def pre_score_for_rank(rank: int) -> float:
    return max(5.0, 9.0 - rank)


def streamer_for(c: dict) -> Streamer:
    login = c.get("broadcaster_login") or c["broadcaster_name"].lower()
    known = get_streamers().get(login) or next((s for s in get_streamers().values() if s.login == login), None)
    if known:  # lang from .env, never Twitch's clip language field
        return known
    lang = c.get("language") if c.get("language") in ("de", "en") else "en"
    return Streamer(id=login, platform="twitch", login=login, lang=lang, name=c["broadcaster_name"],
                    account=get_config().harvest.account_by_lang.get(lang, "account_other"), credit=f"twitch.tv/{login}", cam=None)


# ---------------------------------------------------------------- facecam suggestion
def suggest_cam(video: Path, at_s: float = 2.0) -> Rect | None:
    """Haar face detection on one frame, expanded to a 16:9 cam box snapped to the nearest edges.
    ponytail: Haar + one frame is a heuristic; upgrade to YuNet + several frames if suggestions miss."""
    import subprocess

    import cv2
    import numpy as np
    from vc.render.layouts import NORM, clamp_rect
    raw = subprocess.run([tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", str(at_s), "-i", str(video),
                          "-frames:v", "1", "-vf", NORM, "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
                         capture_output=True, timeout=60).stdout
    if len(raw) != 1920 * 1080:
        return None
    img = np.frombuffer(raw, np.uint8).reshape(1080, 1920)
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = det.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
    # facecams sit at the sides; faces in the middle band are usually game characters
    faces = [f for f in faces if not 0.3 * 1920 < f[0] + f[2] / 2 < 0.7 * 1920]
    if not len(faces):
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # box in the shape of the top area (50/50 split: 1080x960, nearly square), face ~1/3 of its height
    rc = get_config().render
    ch = min(1080, int(h * 3.2))
    cw = min(1920, int(ch * rc.width / rc.split_cam_height))
    cx, cy = int(x + w / 2 - cw / 2), int(y + h / 2 - ch * 0.42)
    snap = 80
    cx = 0 if cx < snap else (1920 - cw if cx + cw > 1920 - snap else cx)
    cy = 0 if cy < snap else (1080 - ch if cy + ch > 1080 - snap else cy)
    return clamp_rect(Rect(x=cx, y=cy, w=cw, h=ch))


# ---------------------------------------------------------------- harvester
class Harvester:
    def __init__(self, api: TwitchAPI | None = None) -> None:
        self.api = api or TwitchAPI()

    async def _fetch(self) -> list[dict]:
        h = get_config().harvest
        now = db.now()
        start = now - timedelta(minutes=h.lookback_minutes)
        clips: dict[str, dict] = {}
        for name in h.game_names:
            if gid := await self.api.get_game_id(name):
                for c in await self.api.get_clips(game_id=gid, started_at=start, ended_at=now):
                    clips[c["id"]] = c
        logins = [s.login for s in get_streamers().values() if s.platform == "twitch"]
        for login, bid in (await self.api.user_ids(logins)).items() if logins else []:
            for c in await self.api.get_clips(broadcaster_id=bid, started_at=start, ended_at=now, first=50):
                clips[c["id"]] = {**c, "broadcaster_login": login}
        # display names are not logins (credit link!): resolve the real login by id
        ids = list({c["broadcaster_id"] for c in clips.values() if "broadcaster_login" not in c and c.get("broadcaster_id")})
        if ids:
            logins = await self.api.logins_for_ids(ids)
            for c in clips.values():
                if "broadcaster_login" not in c and c.get("broadcaster_id") in logins:
                    c["broadcaster_login"] = logins[c["broadcaster_id"]]
        return list(clips.values())

    async def _record(self, clips: list[dict], now: datetime) -> dict[str, float]:
        vel: dict[str, float] = {}
        async with db.connect() as conn:
            for c in clips:
                async with conn.execute("SELECT ts, view_count FROM harvest_views WHERE clip_id = ? ORDER BY ts DESC LIMIT 1",
                                        (c["id"],)) as cur:
                    row = await cur.fetchone()
                prev = (datetime.fromisoformat(row[0]), row[1]) if row else None
                vel[c["id"]] = velocity(c["view_count"], datetime.fromisoformat(c["created_at"].replace("Z", "+00:00")), now, prev)
                await conn.execute(
                    "INSERT INTO harvest_clips (id, broadcaster, broadcaster_id, title, url, created_at, duration, language) "
                    "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
                    (c["id"], c["broadcaster_name"], c.get("broadcaster_id"), c.get("title"), c["url"], c["created_at"],
                     c.get("duration"), c.get("language")))
                await conn.execute("INSERT INTO harvest_views VALUES (?,?,?)", (c["id"], now.isoformat(), c["view_count"]))
            await conn.commit()
        return vel

    async def _processed(self, clip_id: str) -> bool:
        async with db.connect() as conn:
            async with conn.execute("SELECT processed FROM harvest_clips WHERE id = ?", (clip_id,)) as cur:
                row = await cur.fetchone()
        return bool(row and row[0])

    async def _mark(self, clip_id: str, v: int) -> None:
        async with db.connect() as conn:
            await conn.execute("UPDATE harvest_clips SET processed = ? WHERE id = ?", (v, clip_id))
            await conn.commit()

    async def _duplicate(self, streamer_id: str, t: datetime) -> bool:
        lo, hi = (t - timedelta(seconds=DUP_WINDOW_S)).isoformat(), (t + timedelta(seconds=DUP_WINDOW_S)).isoformat()
        async with db.connect() as conn:
            async with conn.execute("SELECT 1 FROM clips WHERE streamer_id = ? AND t_trigger BETWEEN ? AND ? LIMIT 1",
                                    (streamer_id, lo, hi)) as cur:
                return await cur.fetchone() is not None

    async def poll(self) -> list[tuple[dict, float, str]]:
        """One poll. Returns (clip, velocity, outcome) for the picked clips."""
        h = get_config().harvest
        now = db.now()
        clips = await self._fetch()
        vel = await self._record(clips, now)
        whitelist = {s.login for s in get_streamers().values()}
        eligible = [c for c in clips if passes_filters(c, whitelist) and not await self._processed(c["id"])]
        eligible.sort(key=lambda c: vel[c["id"]], reverse=True)
        picked: list[tuple[dict, float, str]] = []
        for rank, c in enumerate(eligible[: h.top_n]):
            s = streamer_for(c)
            t = datetime.fromisoformat(c["created_at"].replace("Z", "+00:00"))
            if await self._duplicate(s.id, t):
                await self._mark(c["id"], 3)
                picked.append((c, vel[c["id"]], "duplicate"))
                continue
            try:
                clip_id = await self._ingest(c, s, t, rank, vel[c["id"]])
                picked.append((c, vel[c["id"]], clip_id))
            except Exception as e:  # noqa: BLE001
                L.warning("Clip %s: %s", c["id"], str(e)[:200])
                await self._mark(c["id"], 2)
                picked.append((c, vel[c["id"]], f"error: {str(e)[:60]}"))
        L.info("harvest: %d clips gesehen, %d geeignet, %d uebernommen", len(clips), len(eligible), len(picked))
        return picked

    async def _ingest(self, c: dict, s: Streamer, t: datetime, rank: int, v: float) -> str:
        out = get_config().data / "candidates" / f"tw_{c['id']}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        await run_ok("yt-dlp", "-q", "--no-warnings", "-f", "best", "-o", out, c["url"], timeout=300)
        signals: dict = {"views": c["view_count"], "velocity": round(v, 1), "rank": rank, "twitch_clip": c["url"]}
        if cat := await self.api.game_name(c.get("game_id") or ""):
            signals["stream_category"] = cat
        if s.cam is None:
            if cam := await asyncio.to_thread(suggest_cam, out):
                signals["cam_suggestion"] = cam.model_dump()
                s = s.model_copy(update={"cam": cam})  # used for this clip only, never written to streamers.yaml
        clip = await db.create_clip(Candidate(source=Source.HARVEST, streamer=s, raw_path=str(out), t_trigger=t,
                                              signals=signals, pre_score=pre_score_for_rank(rank), origin=c["url"]))
        await self._mark(c["id"], 1)
        return clip.id

    async def loop(self) -> None:
        while True:
            try:
                await self.poll()
            except Exception as e:  # noqa: BLE001
                L.exception("harvest poll: %s", e)
                await db.add_event("error", stage="harvest", error=str(e)[:300])
            await asyncio.sleep(get_config().harvest.poll_seconds)
