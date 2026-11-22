"""V2 Phase 2: texts in the streamer's language (.env), not the detected one."""
from vc.config import get_streamers
from vc.deliver import textpack
from vc.judge import gemini
from vc.judge.lang import RewrittenTexts, detect_lang, ensure_lang
from vc.models import Clip, Hashtags, JudgeResult, Source


def jr(**kw) -> JudgeResult:
    base = dict(keep=True, hype=8, category="crash", summary="x", start_s=0, end_s=12, peak_s=6, hook="Wart's ab",
                title="GTA 6 Crash", caption="Das ist komplett eskaliert.", hashtags=Hashtags(youtube=["gta6"]))
    base.update(kw)
    return JudgeResult(**base)


def test_user_text_names_output_language():
    t = gemini._user_text(None, 20.0, "streamer_de", "de", "twitch")
    assert "Output language: German (de)" in t
    assert "Output language: English (en)" in gemini._user_text(None, 20.0, "x", "en", "twitch")


def test_detect_lang():
    for de in ("Er hat es nicht kommen sehen", "Das war komplett geplant", "Warte bis zum Ende"):
        assert detect_lang(de) == "de", de
    for en in ("He did not see that coming", "Wait for it", "This is why you never trust him"):
        assert detect_lang(en) == "en", en
    assert detect_lang("GTA 6 KEKW") == "?"  # no evidence either way
    assert detect_lang("Über die Brücke") == "de"  # umlauts count


def _clip(lang: str) -> Clip:
    s = get_streamers()["streamer_en"].model_copy(update={"lang": lang})
    return Clip(id="kc-1", source=Source.LIVE, streamer_id="zz_adhoc", platform="twitch", lang=lang, hype=8,
                judge=jr().model_dump(mode="json"), account="x", streamer=s.model_dump(mode="json"))


def test_textpack_post_words_localised():
    _, de = textpack.blocks(_clip("de"))
    _, en = textpack.blocks(_clip("en"))
    assert "Clip von twitch.tv/streamer_en" in de["YOUTUBE SHORTS"]
    assert "Clip by twitch.tv/streamer_en" in en["YOUTUBE SHORTS"]
    c = _clip("de")
    c.signals = {"lang_mismatch": True}
    assert "⚠️ Sprache prüfen" in textpack.blocks(c)[0]


async def test_wrong_language_one_rewrite():
    calls = []

    async def rewrite(system, user):
        calls.append(user)
        return RewrittenTexts(hook="Er hat es nicht gesehen", title="GTA 6 Crash mit dem Boot", caption="Das ging schief.")
    english = jr(hook="He did not see that", title="GTA 6 crash on the boat", caption="This went wrong fast.")
    r, mismatch = await ensure_lang(english, "de", rewrite)
    assert len(calls) == 1 and "Output language: German (de)" in calls[0]
    assert not mismatch and r.hook == "Er hat es nicht gesehen" and r.language == "de"

    async def stubborn(system, user):
        calls.append(user)
        return RewrittenTexts(hook="He did not see that", title="GTA 6 crash", caption="This went wrong fast.")
    calls.clear()
    r, mismatch = await ensure_lang(english, "de", stubborn)
    assert len(calls) == 1 and mismatch  # kept, only flagged

    calls.clear()
    r, mismatch = await ensure_lang(jr(), "de", rewrite)  # already German: no call
    assert calls == [] and not mismatch
