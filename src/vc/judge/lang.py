"""Target language of a clip's texts = the streamer's lang from .env. detect_lang() checks the judge's texts
(stopword heuristic, no dependency); a mismatch gets one text-only rewrite call, then a flag."""
from __future__ import annotations

import re
from typing import Awaitable, Callable

from pydantic import BaseModel

from vc.models import JudgeResult

LANG_NAMES = {"de": "German", "en": "English"}
_WORD = re.compile(r"[a-zäöüß']+")
# ponytail: stopword lists, fine for 6-word hooks + one-line captions; a real classifier only if mismatches slip through
_STOP = {
    "de": set("der die das den dem des ein eine einen einem und oder aber ist sind war hat haben nicht kein keine ich du "
              "er sie es wir ihr mein dein sein was wie warum wer wo mit auf fuer für von zu zum zur im in ins am auch noch "
              "nur schon so dann wenn als bis nach vor über ueber unter jetzt hier da doch mal ja nein gott alter digga "
              "krass einfach komplett wart warte ab gleich los geht sieht sich man wird wurde gibt".split()),
    "en": set("the a an and or but is are was were has have not no i you he she it we they my your his her what how why "
              "who where with on for of to in at into also only just so then when as until after before over under now "
              "here there this that these those yes god bro wait for it watch see sees gets got goes went can't don't "
              "didn't did does do will would be been never ever out up was also in so".split()),
}
_SHARED = _STOP["de"] & _STOP["en"]  # 'was', 'in', 'so', 'also' exist in both languages: no evidence
_STOP = {k: v - _SHARED for k, v in _STOP.items()}


def detect_lang(text: str) -> str:
    """'de' | 'en' | '?' (too short or undecided)."""
    words = _WORD.findall(text.lower())
    score = {k: sum(w in stop for w in words) for k, stop in _STOP.items()}
    score["de"] += sum(ch in "äöüß" for ch in text.lower())
    if score["de"] == score["en"]:
        return "?"
    return "de" if score["de"] > score["en"] else "en"


def texts_lang(r: JudgeResult) -> str:
    return detect_lang(" ".join((r.hook, r.title, r.caption)))


class RewrittenTexts(BaseModel):
    hook: str
    title: str
    caption: str


REWRITE_SYSTEM = ("You rewrite short-video texts into a target language. Keep meaning, tone and length limits: hook max 6 "
                  "words, title max 60 characters and keep 'GTA 6' and names as they are, caption one sentence. "
                  "Return JSON that matches the schema and nothing else.")


def rewrite_prompt(r: JudgeResult, lang: str) -> str:
    return (f"Output language: {LANG_NAMES[lang]} ({lang})\nhook: {r.hook}\ntitle: {r.title}\ncaption: {r.caption}")


Rewriter = Callable[[str, str], Awaitable[RewrittenTexts]]  # (system, user) -> texts


async def ensure_lang(r: JudgeResult, lang: str, rewrite: Rewriter) -> tuple[JudgeResult, bool]:
    """(result, mismatch). Texts in another language -> exactly one rewrite call; still wrong -> mismatch True.
    'Undecided' counts as fine (a hook like 'GTA 6 KEKW' has no language)."""
    r = r.model_copy(update={"language": lang})
    if lang not in LANG_NAMES or texts_lang(r) in (lang, "?"):
        return r, False
    try:
        t = await rewrite(REWRITE_SYSTEM, rewrite_prompt(r, lang))
        r = r.model_copy(update={"hook": t.hook or r.hook, "title": t.title or r.title, "caption": t.caption or r.caption})
    except Exception:  # noqa: BLE001  a failed rewrite keeps the clip, only flagged
        pass
    return r, texts_lang(r) not in (lang, "?")
