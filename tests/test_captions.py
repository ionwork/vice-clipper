from vc.models import Layout
from vc.render.captions import Word, ass_color, ass_time, build_ass_text, caption_events, censor, clean_word, group_words


def W(t, s, e):
    return Word(t, s, e)


def test_ass_color_and_time():
    assert ass_color("FFE500") == "&H0000E5FF"
    assert ass_time(0) == "0:00:00.00" and ass_time(61.555) == "0:01:01.56" and ass_time(3600) == "1:00:00.00"


def test_grouping_rules():
    ws = [W("no", 0, .2), W("way", .2, .4), W("bro", .4, .6), W("look", .7, .9),  # max 3 words
          W("at", .9, 1.0), W("that!", 1.0, 1.2), W("wait", 1.25, 1.4),          # punctuation break
          W("what", 2.0, 2.2)]                                                   # gap > 0.35
    g = [[w.text for w in c] for c in group_words(ws, 3, 16)]
    assert g == [["no", "way", "bro"], ["look", "at", "that!"], ["wait"], ["what"]]
    long = [W("incredible", 0, .3), W("unbelievable", .3, .6)]  # 23 chars > 16
    assert len(group_words(long, 3, 16)) == 2


def test_word_events_timing_and_highlight():
    ws = [W("no", 1.20, 1.50), W("way", 1.55, 1.85), W("bro", 1.90, 2.1), W("next", 2.2, 2.4)]
    lines = caption_events(ws, 790, "FFE500", 3, 16)
    assert len(lines) == 4
    assert lines[0].startswith("Dialogue: 0,0:00:01.20,0:00:01.55,Cap")
    assert r"{\pos(540,790)}{\c&H0000E5FF&}NO{\c&HFFFFFF&} WAY BRO" in lines[0]
    assert r"NO {\c&H0000E5FF&}WAY{\c&HFFFFFF&} BRO" in lines[1]
    # last word of a chunk: until chunk end + 0.15, but never into the next chunk (starts 2.2)
    assert ",0:00:01.90,0:00:02.20," in lines[2]


def test_clean_and_censor():
    assert clean_word("that.") == "THAT" and clean_word("what?!") == "WHAT?!"
    assert censor("FUCK", "en") == "F***" and censor("HELLO", "en") == "HELLO"


def test_build_ass_text():
    txt = build_ass_text([W("a", 0, .1)], "He did not see that", "twitch.tv/x", Layout.GAMEPLAY, 18)
    assert "HE DID NOT SEE THAT" in txt and "twitch.tv/x" in txt and "Cap,," not in txt  # <2 words -> no captions
    txt = build_ass_text([W("a", 0, .1), W("b", .1, .2)], "", "c{x}", Layout.GAMEPLAY, 18)
    assert ",Hook," not in txt and "(540,1500)" in txt and "c(x)" in txt


async def test_rerender_keeps_delivered_status(db, monkeypatch):
    """A re-render of a delivered clip must not put it back into the render queue."""
    from vc.config import get_streamers
    from vc.models import Candidate, ClipStatus, Source
    from vc.render import captions
    from test_deliver import JUDGE
    monkeypatch.setattr(captions, "_transcribe_safe", lambda *a: [])
    c = await db.create_clip(Candidate(source=Source.LIVE, streamer=get_streamers()["streamer_en"], raw_path="x"))
    await db.update_clip(c.id, judge=JUDGE, status=ClipStatus.JUDGED)
    await captions.build_ass(c.id)
    assert (await db.get_clip(c.id)).status == ClipStatus.CAPTIONED
    await db.update_clip(c.id, status=ClipStatus.DELIVERED)
    await captions.build_ass(c.id)
    assert (await db.get_clip(c.id)).status == ClipStatus.DELIVERED
