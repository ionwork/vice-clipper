"""Text package per clip: YouTube / TikTok / Instagram blocks, ready to copy."""
from __future__ import annotations

import html
from datetime import datetime

from vc.config import get_config
from vc.models import Clip


def _sentence(s: str) -> str:
    s = s.strip()
    return s if not s or s[-1] in ".!?" else s + "."


# words that end up in the posted text, in the streamer's language (the header above them stays German, it is for me)
POST_WORDS = {"de": {"clip_by": "Clip von"}, "en": {"clip_by": "Clip by"}}


def _tags(tags: list[str], n: int) -> str:
    return " ".join(f"#{t.lstrip('#')}" for t in tags[:n])


def blocks(clip: Clip, slot: datetime | None = None) -> tuple[str, dict[str, str]]:
    j = clip.judge_result
    if j is None:
        raise ValueError(f"{clip.id}: kein Bewertungsergebnis")
    s = clip.streamer_obj
    header = f"#{clip.id} · Hype {clip.hype or j.hype} · {j.category} · {s.display}"
    header += f"\nAccount: {clip.account}"
    if slot:
        header += f" · frühester Slot: {slot.astimezone():%H:%M}" + ("" if slot.astimezone().date() == datetime.now().date() else f" ({slot.astimezone():%d.%m.})")
    if s.permission in ("", "none"):
        header += "\n⚠️ keine Clip-Erlaubnis hinterlegt"
    if s.campaign_rules:
        header += f"\nKampagne: {s.campaign_rules}"
    if (clip.signals or {}).get("lang_mismatch"):
        header += "\n⚠️ Sprache prüfen"
    words = POST_WORDS.get(s.lang, POST_WORDS["en"])
    hook = _sentence(j.hook)
    caption = _sentence(j.caption)
    ig_handle = f"@{s.login}" if s.login else s.credit
    out = {
        "YOUTUBE SHORTS": f"Titel: {j.title}\nBeschreibung: {caption}\n{words['clip_by']} {s.credit}\n{_tags(j.hashtags.youtube, 3)}",
        "TIKTOK": f"{hook} {caption} {_tags(j.hashtags.tiktok, 5)}".strip(),
        "INSTAGRAM": f"{hook} {caption}\nClip: {ig_handle}\n{_tags(j.hashtags.instagram, 5)}".strip(),
    }
    return header, out


def build(clip: Clip, slot: datetime | None = None) -> str:
    header, b = blocks(clip, slot)
    text = header + "\n\n" + "\n\n".join(f"{k}\n{v}" for k, v in b.items()) + "\n"
    p = get_config().data / "out" / f"{clip.id}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return text


def build_html(clip: Clip, slot: datetime | None = None) -> str:
    """Each platform block in <pre>, so one tap copies it in Telegram."""
    header, b = blocks(clip, slot)
    parts = [html.escape(header)]
    parts += [f"<b>{k}</b>\n<pre>{html.escape(v)}</pre>" for k, v in b.items()]
    return "\n\n".join(parts)
