"""Dataset for `vc tune`: moments with chat features and a label.

Positives: Twitch clips of the listed streamers (harvest_clips). Chat around the moment comes from our own chat logs
(data/chatlogs) or, if they do not cover it, from the VOD chat replay via chat-downloader (cached in data/tune/chat,
VODs expire after 7-60 days, so `vc tune --collect-only` fills the cache early). Negatives: 3 random moments per
covered stream stretch, at least 5 min away from every clip, label 0. Features are computed for every grid value
of the baseline floor, the rest of the grid is pure normalisation (detect/tune.py)."""
from __future__ import annotations

import asyncio
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime

from vc import db, log
from vc.config import get_config, get_streamers

L = log.get("tune")
FLOORS = (0.01, 0.02, 0.04)
BEFORE, AFTER = 200, 40          # chat window around a moment
SEARCH = (-60, 10)               # clip creation time is a few seconds after the moment (stream delay)
NEG_PER_STRETCH, NEG_GAP_S = 3, 300


def _ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def parse_duration(s: str) -> float:
    """Helix '3h2m1s' -> seconds."""
    return sum(int(v) * {"h": 3600, "m": 60, "s": 1}[u] for v, u in re.findall(r"(\d+)([hms])", s))


def chatlog(login: str) -> list[tuple[float, str, str]]:
    out = []
    for p in sorted((get_config().data / "chatlogs").glob(f"{login.lower()}-*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                m = json.loads(line)
                out.append((float(m["ts"]), m.get("user", ""), m.get("text", "")))
            except (ValueError, KeyError):
                continue
    return sorted(out)


def stretches(ts: list[float], gap: float = 120) -> list[tuple[float, float]]:
    """Covered intervals of a chat log (a pause > gap s ends one)."""
    out: list[list[float]] = []
    for t in ts:
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(a, b) for a, b in out if b - a >= 600]


def covered(t: float, spans: list[tuple[float, float]]) -> bool:
    return any(a <= t - (BEFORE - 40) and t + 5 <= b for a, b in spans)


def window(msgs: list[tuple[float, str, str]], t: float) -> list[tuple[float, str, str]]:
    import bisect
    i = bisect.bisect_left(msgs, (t - BEFORE,))
    j = bisect.bisect_right(msgs, (t + AFTER, "￿"))
    return msgs[i:j]


def features(msgs: list[tuple[float, str, str]], t: float, lang: str) -> dict | None:
    """Per floor: z, lifts, clip callers. None if the chat does not reach far enough back."""
    from vc.detect.chat_stats import features_at
    from vc.detect.virality import clipcall_users
    cfg = get_config()
    out = {}
    for floor in FLOORS:
        snap = features_at(msgs, t, lang, SEARCH, min_history_s=100, base_min=floor)
        if snap is None:
            return None
        out[str(floor)] = {"z": snap["z"], "reaction_lift": snap.get("reaction_lift", 1.0),
                           "react_lift": snap.get("react_lift", 1.0), "conv_lift": snap.get("conv_lift", 1.0),
                           "intensity_lift": snap.get("intensity_lift", 1.0), "label": snap.get("reaction_label", ""),
                           "calls": clipcall_users(msgs, snap["t"], cfg.virality, cfg.chat.ignore_users)}
    return out


async def harvest_rows(logins: list[str]) -> list[dict]:
    q = ("SELECT h.id, lower(replace(h.broadcaster, ' ', '')), h.created_at, h.duration, "
         "(SELECT view_count FROM harvest_views v WHERE v.clip_id = h.id ORDER BY ts DESC LIMIT 1) FROM harvest_clips h")
    async with db.connect() as conn:
        async with conn.execute(q) as cur:
            rows = await cur.fetchall()
    return [{"id": r[0], "login": r[1], "t": _ts(r[2]), "duration": r[3] or 30, "views": r[4] or 0}
            for r in rows if r[1] in logins]


def _vod_chat_sync(video_id: str, a: float, b: float) -> list[tuple[float, str, str]]:
    from chat_downloader import ChatDownloader
    chat = ChatDownloader().get_chat(f"https://www.twitch.tv/videos/{video_id}", start_time=a, end_time=b,
                                     max_attempts=2, retry_timeout=1, interruptible_retry=False)
    return [(float(m["time_in_seconds"]), (m.get("author") or {}).get("name", ""), m.get("message", "")) for m in chat]


async def vod_chat(video_id: str, a: float, b: float, cache: dict) -> list[tuple[float, str, str]] | None:
    """Chat replay of a VOD stretch (VOD seconds); cached; None if chat-downloader cannot get it (timeout 60 s)."""
    key = f"{video_id}:{int(a)}:{int(b)}"
    if key in cache:
        return [tuple(m) for m in cache[key]] if cache[key] is not None else None
    if cache.get(f"{video_id}:failed"):
        return None
    try:
        msgs = await asyncio.wait_for(asyncio.to_thread(_vod_chat_sync, video_id, a, b), 60)
    except Exception as e:  # noqa: BLE001
        L.warning("VOD %s: Chat nicht abrufbar (%s), alle Clips dieses VODs uebersprungen", video_id, str(e)[:120])
        cache[f"{video_id}:failed"] = True
        return None
    cache[key] = msgs
    return msgs


async def vods(api, login: str) -> list[tuple[str, float, float]]:
    """(video id, start epoch, end epoch) of the channel's archive VODs. Twitch's clips endpoint leaves video_id
    empty for these channels, so the offset is estimated from the timestamps (error = stream delay, ~5-15 s)."""
    uid = (await api.user_ids([login])).get(login)
    if not uid:
        return []
    data = (await api.get("videos", [("user_id", uid), ("type", "archive"), ("first", 20)])).get("data") or []
    return [(v["id"], _ts(v["created_at"]), _ts(v["created_at"]) + parse_duration(v["duration"])) for v in data]


async def collect(use_vods: bool = True) -> tuple[list[dict], Counter]:
    """All moments with features (cached in data/tune/moments.json) and a counter for the report."""
    cfg = get_config()
    d = cfg.data / "tune"
    d.mkdir(parents=True, exist_ok=True)
    cache_p, mom_p = d / "vod_chat.json", d / "moments.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    done = {m["id"]: m for m in json.loads(mom_p.read_text(encoding="utf-8"))} if mom_p.exists() else {}
    streamers = {s.login: s for s in get_streamers().values() if s.platform == "twitch"}
    clips = await harvest_rows(list(streamers))
    n: Counter = Counter(clips=len(clips), channels=len({c["login"] for c in clips}))
    by_login: dict[str, list[dict]] = defaultdict(list)
    for c in clips:
        by_login[c["login"]].append(c)
    api = None
    for login, cl in by_login.items():
        lang = streamers[login].lang
        log_msgs = chatlog(login)
        spans = stretches([m[0] for m in log_msgs])
        n["log_stretches"] += len(spans)
        clip_ts = [c["t"] for c in cl]
        for c in cl:
            if c["id"] in done:
                n["src_" + done[c["id"]]["source"]] += 1
                continue
            if covered(c["t"], spans):
                f, src = features(window(log_msgs, c["t"]), c["t"], lang), "log"
            elif use_vods:
                if api is None:
                    from vc.sources.twitch_api import TwitchAPI
                    api = TwitchAPI()
                    vod_cache: dict[str, list] = {}
                if login not in vod_cache:
                    try:
                        vod_cache[login] = await vods(api, login)
                    except Exception as e:  # noqa: BLE001
                        L.warning("%s: VOD-Liste nicht abrufbar: %s", login, str(e)[:100])
                        vod_cache[login] = []
                vod = next((v for v in vod_cache[login] if v[1] + BEFORE <= c["t"] <= v[2]), None)
                if vod is None:
                    n["skip_no_vod"] += 1
                    continue
                off = c["t"] - vod[1]
                msgs = await vod_chat(vod[0], off - BEFORE, off + AFTER, cache)
                cache_p.write_text(json.dumps(cache), encoding="utf-8")
                if msgs is None:
                    n["skip_vod_chat"] += 1
                    continue
                f, src = features(msgs, off, lang), "vod"
            else:
                n["skip_not_covered"] += 1
                continue
            if f is None:
                n["skip_thin_chat"] += 1
                continue
            done[c["id"]] = {"id": c["id"], "login": login, "abs_t": c["t"], "views": c["views"], "positive": True,
                             "source": src, "feats": f}
            n["src_" + src] += 1
        rng = random.Random(login)  # negatives from our own logs (VOD negatives need working VOD chat)
        for i, (a, b) in enumerate(spans):
            for k in range(NEG_PER_STRETCH):
                nid = f"neg:{login}:{int(a)}:{k}"
                if nid in done:
                    continue
                for _ in range(50):
                    t = rng.uniform(a + BEFORE, b - AFTER)
                    if all(abs(t - x) >= NEG_GAP_S for x in clip_ts):
                        if f := features(window(log_msgs, t), t, lang):
                            done[nid] = {"id": nid, "login": login, "abs_t": t, "views": 0, "positive": False,
                                         "source": "log", "feats": f}
                        break
    if api is not None:
        await api.close()
    mom_p.write_text(json.dumps(list(done.values())), encoding="utf-8")
    n["moments"] = len(done)
    n["positives"] = sum(m["positive"] for m in done.values())
    return list(done.values()), n


async def decisions() -> list[tuple[float, int]]:
    """My Telegram decisions as a second, weaker label: (stored V, POSTED=1 / discarded by hand=0)."""
    from vc.detect.virality import known_v
    from vc.models import ClipStatus
    out = []
    for c in await db.list_clips(ClipStatus.POSTED) + await db.list_clips(ClipStatus.REJECTED):
        v = known_v(c)
        if v is not None and (c.status == ClipStatus.POSTED or (c.reason or "").startswith("von Hand")):
            out.append((v, int(c.status == ClipStatus.POSTED)))
    return out

