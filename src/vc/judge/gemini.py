"""One Gemini call per candidate: keep or reject, exact cut, all texts."""
from __future__ import annotations

import asyncio
from functools import cache
from pathlib import Path

from vc import db, log
from vc.config import get_config, get_secrets
from vc.judge.fallback import fallback_judge
from vc.judge.prompts import judge_prompt
from vc.models import CATEGORIES, Clip, ClipStatus, JudgeResult
from vc.proc import run_ok
from vc.render.ffmpeg import probe

L = log.get("judge")


@cache
def client():
    from google import genai
    key = get_secrets().gemini_api_key
    if not key:
        raise RuntimeError("GEMINI_API_KEY fehlt in .env")
    return genai.Client(api_key=key)


async def make_proxy(src: Path, out: Path) -> Path:
    j = get_config().judge
    await run_ok("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                 "-vf", f"scale=-2:{j.proxy_height},fps={j.proxy_fps}", "-c:v", "libx264", "-crf", "30", "-preset", "veryfast",
                 "-c:a", "aac", "-b:a", "48k", "-ac", "1", out, timeout=300)
    return out


def estimate_usd(prompt_tokens: int, output_tokens: int) -> float:
    j = get_config().judge
    return prompt_tokens / 1e6 * j.price_in_per_m + output_tokens / 1e6 * j.price_out_per_m


def _retryable(e: Exception) -> bool:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    return isinstance(code, int) and (code == 429 or code >= 500)


async def call_gemini(contents: list, system: str, schema: type, model: str, clip_id: str | None = None,
                      media_low: bool = False, retries: int = 2):
    """Structured-JSON call with backoff on 429/5xx and on unparsable output. Logs token usage."""
    from google.genai import types
    cfg = types.GenerateContentConfig(
        system_instruction=system, response_mime_type="application/json", response_schema=schema,
        temperature=0.2, max_output_tokens=4096,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW if media_low else None,
    )
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = await client().aio.models.generate_content(model=model, contents=contents, config=cfg)
            u = resp.usage_metadata
            p, o = (u.prompt_token_count or 0, (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else (0, 0)
            await db.add_event("gemini_usage", clip_id=clip_id, model=model, prompt=p, output=o, usd=estimate_usd(p, o))
            parsed = resp.parsed if resp.parsed is not None else schema.model_validate_json(resp.text or "")
            return parsed
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries and (_retryable(e) or "validation error" in str(e).lower() or "json" in str(e).lower()):
                await asyncio.sleep(2 * 3 ** attempt)
                continue
            raise
    raise last  # pragma: no cover


def _words(s: str, n: int) -> str:
    return " ".join(s.replace('"', "").split()[:n])


def _tags(tags: list[str], n: int) -> list[str]:
    out: list[str] = []
    for t in ["gta6", *tags]:
        t = "".join(ch for ch in t.lower().lstrip("#") if ch.isalnum() or ch == "_")
        if t and t not in out:
            out.append(t)
    return out[:n]


def validate(r: JudgeResult, duration: float) -> JudgeResult:
    """Clamp the window into the candidate, enforce min/max length around peak_s, sanitize texts."""
    rc = get_config().render
    r = r.model_copy(deep=True)
    r.hype = max(1, min(10, int(r.hype)))
    r.subject_x = max(0.0, min(1.0, r.subject_x))
    if r.category not in CATEGORIES:
        r.category = "other"
    start, end = max(0.0, min(r.start_s, duration)), max(0.0, min(r.end_s, duration))
    if end < start:
        start, end = end, start
    peak = max(0.0, min(r.peak_s, duration))
    if not (start <= peak <= end):
        peak = (start + end) / 2
    if r.flags.violence_first_3s:  # start on the reaction, not on the violence
        start = min(start + 3, max(start, peak - 2))
    lo, hi = rc.min_len_s, rc.max_len_s
    if end - start > hi:  # keep ~3 s after the peak, trim the lead-in
        end = min(end, max(peak + 3, start + hi))
        start = max(start, end - hi)
    if end - start < lo:  # extend on both sides, inside the candidate
        need = min(lo, duration) - (end - start)
        post = min(duration - end, need / 2)
        pre = min(start, need - post)
        post = min(duration - end, need - pre)
        start, end = start - pre, end + post
    r.start_s, r.end_s, r.peak_s = round(start, 2), round(end, 2), round(max(start, min(peak, end)), 2)
    r.hook = _words(r.hook, 6)
    if len(r.title) > 60:
        r.title = r.title[:60].rsplit(" ", 1)[0]
    r.hashtags.youtube = _tags(r.hashtags.youtube, 3)
    r.hashtags.tiktok = _tags(r.hashtags.tiktok, 5)
    r.hashtags.instagram = _tags(r.hashtags.instagram, 5)
    return r


def render_rule(r: JudgeResult, min_hype: int, allow_irl: bool = True) -> tuple[bool, int, str]:
    """(render?, effective hype, reason). music_dominant costs one point. allow_irl: real life is not 'not_gta6'."""
    hype = r.hype - (1 if r.flags.music_dominant else 0)
    blocked = [f for f in ("cutscene", "story_spoiler", "not_gta6") if getattr(r.flags, f)]
    if allow_irl and r.content == "irl" and "not_gta6" in blocked:
        blocked.remove("not_gta6")
    if blocked:
        return False, hype, "flag: " + ",".join(blocked)
    if not r.keep:
        return False, hype, "keep=false"
    if hype < min_hype:
        return False, hype, f"hype {hype} < {min_hype}"
    return True, hype, ""


def _user_text(clip: Clip | None, duration: float, name: str = "", lang: str = "", platform: str = "") -> str:
    from vc.judge.lang import LANG_NAMES
    out_lang = lang if lang in LANG_NAMES else "en"
    lines = [f"Streamer: {name}", f"Output language: {LANG_NAMES[out_lang]} ({out_lang})", f"Platform: {platform}",
             f"Candidate duration: {duration:.1f} s"]
    if clip and clip.trigger_offset_s is not None:
        lines.append(f"Chat reacted at about {clip.trigger_offset_s:.1f} s into this cut.")
    if clip and clip.chat_context:
        lines.append("Chat lines right before the cut (oldest first):")
        lines += clip.chat_context[-40:]
    return "\n".join(lines)


async def _gemini_judge(src: Path, text: str, clip_id: str | None) -> JudgeResult:
    from google.genai import types
    cfg = get_config()
    tmp = cfg.data / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    proxy = await make_proxy(src, tmp / f"proxy-{clip_id or src.stem}.mp4")
    try:
        contents = [types.Part.from_bytes(data=proxy.read_bytes(), mime_type="video/mp4"), text]
        return await call_gemini(contents, judge_prompt(cfg.judge.allow_gta5, cfg.judge.allow_irl), JudgeResult, cfg.judge.model, clip_id)
    finally:
        proxy.unlink(missing_ok=True)


async def judge_media(src: Path, text: str, clip_id: str | None, lang: str) -> JudgeResult:
    """Provider switch: local Ollama (free) or Gemini (paid). Same prompt rules, same JudgeResult."""
    cfg = get_config()
    if cfg.judge.provider == "ollama":
        from vc.judge.local import ollama_judge
        return await ollama_judge(src, text, clip_id, lang, judge_prompt(cfg.judge.allow_gta5, cfg.judge.allow_irl))
    return await _gemini_judge(src, text, clip_id)


async def rewrite_texts(system: str, user: str, clip_id: str | None = None):
    """Text-only call (no frames, no video) that rewrites hook/title/caption into the target language."""
    from vc.judge.lang import RewrittenTexts
    cfg = get_config()
    if cfg.judge.provider == "ollama":
        from vc.judge.local import chat_json
        return await chat_json(system, user, RewrittenTexts, clip_id=clip_id)
    return await call_gemini([user], system, RewrittenTexts, cfg.judge.model, clip_id)


async def over_budget() -> bool:
    return (await db.cost_today())[2] >= get_config().judge.daily_budget_usd


async def judge_candidate(clip_id: str) -> JudgeResult:
    clip = await db.get_clip(clip_id)
    if not clip:
        raise KeyError(clip_id)
    src = Path(clip.raw_path)
    duration = (await probe(src)).duration
    s = clip.streamer_obj
    used_ai = True
    try:
        if await over_budget():
            raise RuntimeError("Tagesbudget erreicht")
        result = await judge_media(src, _user_text(clip, duration, s.display, s.lang, s.platform), clip_id, s.lang)
    except Exception as e:  # noqa: BLE001
        L.warning("%s: KI-Bewertung fehlgeschlagen, Rueckfall ohne KI: %s", clip_id, str(e)[:200])
        await db.add_event("judge_fallback", clip_id=clip_id, error=str(e)[:300])
        result, used_ai = fallback_judge(clip, duration), False
    result = validate(result, duration)
    sig = dict(clip.signals or {})
    if used_ai:  # texts must be in the streamer's language (.env), not the one the model drifted into
        from vc.judge.lang import ensure_lang
        result, mismatch = await ensure_lang(result, s.lang, lambda sy, u: rewrite_texts(sy, u, clip_id))
        result = validate(result, duration)
        if mismatch:
            sig["lang_mismatch"] = True
            L.warning("%s: Texte nicht in %s, markiert", clip_id, s.lang)
    cfg = get_config()
    from vc.judge.local import load_words
    if saved := load_words(clip_id):  # voice component from the judge's Whisper words + loudness
        from vc.detect.voice import voice_features
        sig.update(voice_features([tuple(w) for w in saved.get("words", [])], saved.get("loud", []), s.lang, cfg.virality))
    from vc.render.layouts import choose_layout, content_kind
    result.content = content_kind(sig.get("stream_category") or "", result.content, cfg.render.irl_categories)
    sig["content"] = result.content
    ok, hype, reason = render_rule(result, cfg.judge.min_hype, cfg.judge.allow_irl)
    # gameplay stays SPLIT while a cam may still be auto-detected at render time (effective_layout falls back)
    layout = choose_layout(result.content, s.cam is not None or cfg.render.auto_cam, cfg.render.force_layout).value
    await db.update_clip(clip_id, judge=result.model_dump(mode="json"), hype=hype, layout=layout, signals=sig,
                         status=ClipStatus.JUDGED if ok else ClipStatus.REJECTED, reason=reason)
    await db.add_event("judged", clip_id=clip_id, streamer_id=clip.streamer_id, hype=hype, keep=ok, reason=reason,
                       category=result.category, ai=used_ai)
    L.info("%s judged: hype %s %s %s", clip_id, hype, "KEEP" if ok else "REJECT", reason)
    return result


async def judge_file(path: Path, name: str = "", lang: str = "en") -> dict:
    duration = (await probe(path)).duration
    r = validate(await judge_media(path, _user_text(None, duration, name, lang, "unknown"), None, lang), duration)
    ok, hype, reason = render_rule(r, get_config().judge.min_hype, get_config().judge.allow_irl)
    return {**r.model_dump(mode="json"), "_render": ok, "_hype_effective": hype, "_reason": reason}
