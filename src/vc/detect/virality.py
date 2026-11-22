"""Virality score V (0-100): weighted mean of independent, normalised signals (V2: eight components).

velocity     chat spike, robust z against the channel's own baseline
reaction     max over laugh/shock/hype/cringe/confusion (one reaction, never summed), lift of distinct users
library      viewer-made Twitch clips around the moment + clip links in chat (+ view velocity)
clip_calls   distinct users writing "clip that" / !clip; the streamer saying it counts as 3
judge        AI hype 1-10, the weakest independent signal (guesses from stills): gate + tie-breaker
convergence  many distinct users writing the same token/emote, or emote-only messages
voice        loudness peak / speech-rate jump of the streamer + keyword and silence-after-peak bonus
intensity    CAPS / "!!!" / stretched words among distinct users, small weight (correlates with velocity)
Old clips keep their V: components named chat_spike/reactions/audio are scored with the legacy weights.

score() is pure (no I/O). components_for() gathers the raw signals of one clip from its stored signals,
the live chat buffer, the harvester's tables (harvest_clips / harvest_views) and the judge result."""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel

from vc.config import ViralityCfg, get_config
from vc.models import Clip, Source

COMPONENTS = ("velocity", "reaction", "library", "clip_calls", "judge", "convergence", "voice", "intensity")
LEGACY_COMPONENTS = ("chat_spike", "reactions", "clip_calls", "library", "audio", "judge")
ALIAS = {"velocity": "chat_spike", "reaction": "reactions", "voice": "audio"}  # new -> old name
CHAT_KEYS = ("z", "lex", "unique", "unique_n", "burst", "baseline", "clipcalls", "react_lift", "react_share",
             "reaction_lift", "reaction_label", "reaction_lifts", "conv_lift", "conv_token", "intensity_lift")
REACTION_EMOJI = {"laugh": "😂", "shock": "😱", "hype": "🔥", "cringe": "😬", "confusion": "❓"}
URL_RE = re.compile(r"https?://\S+|\S+\.(?:tv|com)/\S+")


class ViralityResult(BaseModel):
    v: float
    components: dict[str, float | None]
    weights_used: dict[str, float]
    bonus: float = 1.0
    penalties: list[str] = []
    excluded: bool = False
    reason: str = ""

    @property
    def known(self) -> bool:
        """False when no component was available (V carries no information)."""
        return bool(self.weights_used) or self.excluded


# ---------------------------------------------------------------- normalisers (pure)
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def n_chat_spike(z: float, c: ViralityCfg) -> float:
    return clamp((z - c.z_floor) / c.z_span)


def n_reactions(lift: float, c: ViralityCfg) -> float:
    return clamp(math.log2(max(lift, 1.0)) / c.reaction_log_span)


def n_clip_calls(n_users: int, c: ViralityCfg) -> float:
    return 1 - math.exp(-n_users / c.clipcall_scale)


def n_library(n_tw: int, n_links: int, expected: float, fast: bool, c: ViralityCfg) -> float:
    x = (n_tw + c.library_link_weight * n_links) / max(expected, c.library_expected_min)
    base = clamp(math.log2(1 + x) / c.library_log_span)
    return min(1.0, base + (c.velocity_bonus if fast else 0.0))


def n_lift(lift: float, span: float) -> float:
    """Log scale with saturation: lift 2^span (8x) is full, more brings nothing."""
    return clamp(math.log2(max(lift, 1.0)) / span)


def n_audio(db_above: float, c: ViralityCfg) -> float:
    return clamp(db_above / c.audio_full_db)


def n_judge(hype: float) -> float:
    return clamp((hype - 1) / 9)


# ---------------------------------------------------------------- the score (pure)
def score(components: dict[str, float | None], cfg: ViralityCfg, flags: dict | None = None,
          allow_gta5: bool = False, irl_ok: bool = False) -> ViralityResult:
    """V = 100 * sum(w*s) / sum(w) over available components, agreement bonus, judge-flag penalties."""
    flags = flags or {}
    legacy = any(k in components for k in ("chat_spike", "reactions", "audio"))
    weights = cfg.legacy_weights if legacy else cfg.weights
    comps = {k: components.get(k) for k in (LEGACY_COMPONENTS if legacy else COMPONENTS)}
    blocked = [f for f in ("cutscene", "story_spoiler") if flags.get(f)]
    if flags.get("not_gta6") and not allow_gta5 and not irl_ok:  # irl_ok: allow_irl and content irl
        blocked.append("not_gta6")
    if blocked:
        return ViralityResult(v=0.0, components=comps, weights_used={}, excluded=True, reason="ausgeschlossen: " + ",".join(blocked))
    avail = {k: clamp(s) for k, s in comps.items() if s is not None and weights.get(k, 0) > 0}
    if not avail:
        return ViralityResult(v=0.0, components=comps, weights_used={}, reason="keine Daten")
    w = {k: weights[k] for k in avail}
    v = 100 * sum(w[k] * avail[k] for k in avail) / sum(w.values())
    bonus = 1.0
    if sum(s >= cfg.agree_level for s in avail.values()) >= cfg.agree_min_components:
        bonus = cfg.agree_factor
        v = min(100.0, v * bonus)
    penalties = []
    if flags.get("needs_context"):
        v -= cfg.needs_context_penalty
        penalties.append(f"needs_context -{cfg.needs_context_penalty:g}")
    v = round(clamp(v, 0.0, 100.0), 1)
    top = max(avail, key=lambda k: w[k] * avail[k])
    return ViralityResult(v=v, components=comps, weights_used=w, bonus=bonus, penalties=penalties, reason=f"staerkstes Signal: {top}")


# ---------------------------------------------------------------- chat helpers (pure)
def _phrase_re(c: ViralityCfg) -> re.Pattern:
    alts = "|".join(re.escape(p) for p in sorted(c.clipcall_phrases, key=len, reverse=True))
    return re.compile(rf"(?i)(?<!\w)(?:{alts})(?!\w)")


def clipcall_users(msgs: Iterable[tuple[float, str, str]], t: float, c: ViralityCfg, ignore: Iterable[str] = ()) -> int:
    """Distinct users writing a clip phrase or !clip in [t + lo, t + hi]; links and bots do not count."""
    lo, hi = c.clipcall_window_s
    rx = _phrase_re(c)
    skip = {u.lower() for u in ignore}
    return len({u.lower() for ts, u, text in msgs if t + lo <= ts <= t + hi and u.lower() not in skip
                and (rx.search(URL_RE.sub(" ", text)) or text.lower().startswith("!clip"))})


def chat_links(msgs: Iterable[tuple[float, str, str]], t: float, c: ViralityCfg) -> int:
    """Distinct clip links posted in chat in the library window."""
    lo, hi = c.library_window_s
    rx = re.compile(c.link_pattern, re.I)
    links = {tok for ts, _, text in msgs if t + lo <= ts <= t + hi for tok in text.split() if rx.search(tok)}
    return len(links)


# ---------------------------------------------------------------- clip library (harvester tables)
def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


async def library_counts(login: str, t: datetime, c: ViralityCfg, now: datetime | None = None) -> dict | None:
    """Twitch clips of this channel around t, the channel's usual clip rate and the velocity bonus.
    None when the harvester has no data for the channel (component then counts as missing)."""
    from vc import db
    from vc.sources.harvester import velocity
    now = now or db.now()
    since = now - timedelta(days=c.library_rate_days)
    async with db.connect() as conn:
        async with conn.execute("SELECT id, created_at FROM harvest_clips WHERE lower(replace(broadcaster, ' ', '')) = ?",
                                (login.lower(),)) as cur:
            rows = [(r[0], _ts(r[1])) for r in await cur.fetchall()]
        rows = [(i, ct) for i, ct in rows if ct >= since]
        if not rows:
            return None
        ids = [i for i, _ in rows]
        vel: dict[str, float] = {}
        q = f"SELECT clip_id, ts, view_count FROM harvest_views WHERE clip_id IN ({','.join('?' * len(ids))})"
        async with conn.execute(q, ids) as cur:
            created = dict(rows)
            for cid, ts, views in await cur.fetchall():
                vel[cid] = max(vel.get(cid, 0.0), velocity(views, created[cid], _ts(ts), None))
    lo, hi = c.library_window_s
    hits = [i for i, ct in rows if t + timedelta(seconds=lo) <= ct <= t + timedelta(seconds=hi)]
    # usual rate = clips per ACTIVE hour (hours that had any clip), so offline time does not dilute it
    active_h = len({ct.astimezone(timezone.utc).strftime("%Y%m%d%H") for _, ct in rows})
    expected = max(c.library_expected_min, len(rows) / max(1, active_h) * (hi - lo) / 3600)
    fast = False
    vs = sorted(vel.values())
    if hits and len(vs) >= c.velocity_min_clips:
        threshold = vs[min(len(vs) - 1, int(c.velocity_quantile * len(vs)))]
        fast = max(vel.get(i, 0.0) for i in hits) >= threshold
    return {"n_tw": len(hits), "expected": round(expected, 3), "fast": fast}


# ---------------------------------------------------------------- one clip
async def components_for(clip: Clip, hub=None) -> tuple[dict[str, float | None], dict[str, Any]]:
    cfg = get_config()
    c = cfg.virality
    sig = clip.signals or {}
    prev = (sig.get("virality") or {}).get("raw") or {}
    raw: dict[str, Any] = {}
    comps: dict[str, float | None] = dict.fromkeys(COMPONENTS)
    t = clip.t_trigger.timestamp() if clip.t_trigger else None
    s = clip.streamer_obj
    msgs = [(m.ts, m.user, m.text) for m in hub.recent(s.login, 10**6)] if hub is not None and t else []
    if t and sig.get("z") is None and clip.source == Source.HARVEST:  # viewer clip: same chat components if chat exists
        from vc.detect.chat_stats import features_at
        snap = features_at(msgs or chatlog_msgs(s.login, t), t, s.lang)
        if snap:
            sig = {**sig, **{k: snap[k] for k in CHAT_KEYS if k in snap}, "chat_from": "log"}
            clip.signals = sig
    chat_comps(sig, c, comps, raw)
    n_calls = n_links = None
    ignore = cfg.chat.ignore_users
    if msgs:  # chat buffer can drop old lines: never go below what was counted before
        n_calls = max(clipcall_users(msgs, t, c, ignore), prev.get("clip_calls_n", 0))
        n_links = max(chat_links(msgs, t, c), prev.get("links_n", 0))
    elif "clip_calls_n" in prev:
        n_calls, n_links = prev["clip_calls_n"], prev.get("links_n", 0)
    if sig.get("streamer_clipcall"):  # the streamer said "clip that" (transcript)
        n_calls = (n_calls or 0) + c.streamer_clipcall_users
        raw["streamer_clipcall"] = True
    if n_calls is not None:
        raw["clip_calls_n"], raw["links_n"] = n_calls, n_links
        comps["clip_calls"] = n_clip_calls(n_calls, c)
    lib = await library_counts(s.login, clip.t_trigger, c) if clip.t_trigger and s.platform == "twitch" else None
    if lib is not None or n_links is not None:
        lib = lib or {"n_tw": 0, "expected": c.library_expected_min, "fast": False}
        raw.update(lib_n_tw=lib["n_tw"], lib_expected=lib["expected"], lib_fast=lib["fast"])
        comps["library"] = n_library(lib["n_tw"], n_links or 0, lib["expected"], lib["fast"], c)
    if sig.get("voice") is not None:  # from the judge step (Whisper words + loudness)
        raw["voice"] = sig["voice"]
        comps["voice"] = clamp(float(sig["voice"]))
    elif sig.get("audio_db") is not None:  # live trigger only knows the loudness spike
        raw["audio_db"] = sig["audio_db"]
        comps["voice"] = n_audio(float(sig["audio_db"]), c)
    if clip.judge and clip.hype is not None:
        raw["hype"] = clip.hype
        comps["judge"] = n_judge(clip.hype)
    return comps, raw


def chat_comps(sig: dict, c: ViralityCfg, comps: dict, raw: dict) -> None:
    """Pure: chat components from stored signals (old clips only have z and react_lift)."""
    if sig.get("z") is not None:
        raw["z"] = sig["z"]
        comps["velocity"] = n_chat_spike(float(sig["z"]), c)
    lift = sig.get("reaction_lift", sig.get("react_lift"))
    if lift is not None:
        raw["reaction_lift"] = lift
        comps["reaction"] = n_reactions(float(lift), c)
        if sig.get("reaction_label"):
            raw["reaction_label"] = sig["reaction_label"]
    for k, key in (("convergence", "conv_lift"), ("intensity", "intensity_lift")):
        if sig.get(key) is not None:
            raw[key] = sig[key]
            comps[k] = n_lift(float(sig[key]), c.log_span)


def chatlog_msgs(login: str, t: float, before: float = 200, after: float = 40) -> list[tuple[float, str, str]]:
    """Chat around t from data/chatlogs (the hub writes one jsonl per channel and day)."""
    import json
    d = get_config().data / "chatlogs"
    out = []
    for day in {datetime.fromtimestamp(x).date().isoformat() for x in (t - before, t + after)}:
        p = d / f"{login.lower()}-{day}.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if t - before <= m.get("ts", 0) <= t + after:
                    out.append((m["ts"], m.get("user", ""), m.get("text", "")))
    return sorted(out)


async def compute(clip: Clip, hub=None, store: bool = True) -> ViralityResult:
    """Score one clip and keep the result (with raw inputs) in clips.signals['virality']."""
    from vc import db
    cfg = get_config()
    comps, raw = await components_for(clip, hub)
    res = score(comps, cfg.virality, (clip.judge or {}).get("flags"), cfg.judge.allow_gta5,
                cfg.judge.allow_irl and (clip.judge or {}).get("content") == "irl")
    if store:
        fresh = await db.get_clip(clip.id)
        sig = dict((fresh.signals if fresh else clip.signals) or {})
        if (clip.signals or {}).get("chat_from"):  # chat features found for a viewer clip: keep them
            sig.update({k: clip.signals[k] for k in (*CHAT_KEYS, "chat_from") if k in clip.signals})
        sig["virality"] = {**res.model_dump(), "raw": raw}
        await db.update_clip(clip.id, signals=sig)
        clip.signals = sig
    return res


def stored(clip: Clip) -> ViralityResult | None:
    v = (clip.signals or {}).get("virality")
    return ViralityResult.model_validate({k: v[k] for k in v if k != "raw"}) if v else None


def known_v(clip: Clip) -> float | None:
    r = stored(clip)
    return r.v if r and r.known else None


def comp(res: ViralityResult, k: str) -> float | None:
    """Component by V2 name, falling back to the pre-V2 name for old clips."""
    s = res.components.get(k)
    return res.components.get(ALIAS[k]) if s is None and k in ALIAS else s


def summary_line(res: ViralityResult, label: str = "") -> str:
    """'V 78 · Chat 0.9 · 😂 0.8 · Clips 0.7 · KI 0.8' for the Telegram text block (label = reaction category)."""
    def f(k: str) -> str:
        s = comp(res, k)
        return "–" if s is None else f"{s:.1f}"
    return (f"V {res.v:.0f} · Chat {f('velocity')} · {REACTION_EMOJI.get(label, '💀😂')} {f('reaction')} · "
            f"Clips {f('library')} · KI {f('judge')}")


RAW_KEY = {"velocity": "z", "reaction": "reaction_lift", "clip_calls": "clip_calls_n", "library": "lib_n_tw",
           "judge": "hype", "convergence": "conv_lift", "voice": "voice", "intensity": "intensity_lift",
           "chat_spike": "z", "reactions": "react_lift", "audio": "audio_db"}


def breakdown(res: ViralityResult, raw: dict) -> list[tuple[str, str, str, str, str]]:
    """Rows (component, raw, norm, weight share, contribution in V points before bonus/penalties)."""
    total = sum(res.weights_used.values()) or 1.0
    rows = []
    for k in (LEGACY_COMPONENTS if "chat_spike" in res.components else COMPONENTS):
        s = res.components.get(k)
        w = res.weights_used.get(k)
        r = raw.get(RAW_KEY[k])
        if r is None and k == "voice":
            r = raw.get("audio_db") and f"{raw['audio_db']} dB"
        if r is None and k == "reaction":
            r = raw.get("react_lift")
        if r is not None and k == "reaction" and raw.get("reaction_label"):
            r = f"{r} ({raw['reaction_label']})"
        if k == "library" and r is not None:
            r = f"{r} Clips, {raw.get('links_n', 0)} Links, erw. {raw.get('lib_expected')}" + (" +schnell" if raw.get("lib_fast") else "")
        rows.append((k, "–" if r is None else str(r), "–" if s is None else f"{s:.2f}",
                     "–" if w is None else f"{w / total:.2f}", "–" if w is None else f"{100 * w * s / total:.1f}"))
    return rows
