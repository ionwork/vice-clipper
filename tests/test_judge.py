from vc.config import get_streamers
from vc.judge import gemini
from vc.judge.fallback import fallback_judge
from vc.models import Candidate, ClipStatus, Flags, Hashtags, JudgeResult, Source


def jr(**kw) -> JudgeResult:
    base = dict(keep=True, hype=8, category="police_chase", summary="x", start_s=6.5, end_s=24.0, peak_s=17.2,
                subject_x=0.55, layout="split", hook="He did not see that", title="GTA 6 police chase ends on a boat",
                caption="c", hashtags=Hashtags(youtube=["#GTA6", "GTAVI", "streamer_en", "extra"], tiktok=["gaming"]),
                language="en", flags=Flags())
    base.update(kw)
    return JudgeResult(**base)


def test_validate_clamps_and_sanitizes():
    r = gemini.validate(jr(start_s=-3, end_s=80, peak_s=30, hook="one two three four five six seven",
                           title="x" * 70, subject_x=1.7, hype=12, category="weird"), duration=40)
    assert 0 <= r.start_s < r.end_s <= 40 and r.end_s - r.start_s <= 25
    assert r.start_s <= r.peak_s <= r.end_s and r.end_s >= r.peak_s + 3 - 1e-6
    assert len(r.hook.split()) == 6 and len(r.title) <= 60 and r.subject_x == 1 and r.hype == 10
    assert r.category == "other"
    assert r.hashtags.youtube == ["gta6", "gtavi", "streamer_en"] and r.hashtags.tiktok == ["gta6", "gaming"]


def test_validate_extends_short_window():
    r = gemini.validate(jr(start_s=10, end_s=14, peak_s=12), duration=40)
    assert abs((r.end_s - r.start_s) - 10) < 1e-6 and r.start_s <= 12 <= r.end_s
    r = gemini.validate(jr(start_s=0, end_s=3, peak_s=1), duration=40)  # at the edge
    assert r.start_s == 0 and abs(r.end_s - 10) < 1e-6


def test_violence_shifts_start():
    r = gemini.validate(jr(start_s=5, end_s=22, peak_s=15, flags=Flags(violence_first_3s=True)), duration=40)
    assert r.start_s == 8


def test_render_rule():
    assert gemini.render_rule(jr(), 7) == (True, 8, "")
    assert gemini.render_rule(jr(flags=Flags(music_dominant=True), hype=7), 7)[0] is False
    assert gemini.render_rule(jr(flags=Flags(cutscene=True)), 7)[0] is False
    assert gemini.render_rule(jr(flags=Flags(not_gta6=True)), 7)[2] == "flag: not_gta6"
    assert gemini.render_rule(jr(keep=False), 7)[0] is False


async def test_judge_candidate_and_fallback(db, synth_video, monkeypatch):
    s = get_streamers()["streamer_en"]
    c = await db.create_clip(Candidate(source=Source.LIVE, streamer=s, raw_path=str(synth_video), pre_score=8,
                                       trigger_offset_s=22))

    async def fake(src, text, clip_id, lang):
        assert "Chat reacted at about 22.0 s" in text
        return jr()
    monkeypatch.setattr(gemini, "judge_media", fake)
    await gemini.judge_candidate(c.id)
    got = await db.get_clip(c.id)
    assert got.status == ClipStatus.JUDGED and got.hype == 8 and got.judge["hook"] == "He did not see that"

    async def boom(*a):
        raise RuntimeError("503 unavailable")
    monkeypatch.setattr(gemini, "judge_media", boom)
    r = await gemini.judge_candidate(c.id)
    assert r.summary.startswith("unknown") and r.start_s == 2 and r.end_s == 27
    assert (await db.get_clip(c.id)).status == ClipStatus.JUDGED  # pre_score 8 -> keep

    await db.add_event("gemini_usage", prompt=1, output=1, usd=999)
    assert await gemini.over_budget()


def test_fallback_without_trigger():
    from vc.models import Clip
    s = get_streamers()["streamer_en"]
    clip = Clip(id="x", source=Source.HARVEST, streamer_id=s.id, platform="twitch", lang="en", pre_score=5,
                streamer=s.model_dump(mode="json"))
    r = fallback_judge(clip, 30)
    assert r.keep is False and 0 <= r.start_s < r.end_s <= 30


async def test_provider_switch_ollama(monkeypatch, tmp_path):
    """provider: ollama routes to the local judge with the same system prompt rules."""
    from vc.judge import gemini, local
    seen = {}

    async def fake(src, text, clip_id, lang, system):
        seen.update(lang=lang, system=system)
        return "R"
    monkeypatch.setattr(local, "ollama_judge", fake)
    assert await gemini.judge_media(tmp_path / "x.mp4", "t", None, "de") == "R"
    assert seen["lang"] == "de" and "senior YouTube Shorts editor" in seen["system"]


def test_local_text_layout():
    from vc.judge.local import build_user_text, frame_times
    assert frame_times(24, 4) == [3.0, 9.0, 15.0, 21.0]
    t = build_user_text("Streamer: x", [3.0, 9.0], ["[1.0s] hi"], [-20.0, -5.4])
    assert "image 2 at 9.0s" in t and "[1.0s] hi" in t and "1s:-5" in t
