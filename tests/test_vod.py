from vc.sources.vod_scout import (Moment, MomentFlags, hms, is_twitch_vod, is_youtube, merge_moments, plan_chunks,
                                  section, select_moments, streamer_for_video, to_absolute)


def M(s, e, h=8, **kw):
    return Moment(start_s=s, end_s=e, hype=h, **kw)


def test_plan_chunks():
    assert plan_chunks(1200, 40) == [(0.0, 1200)]
    ch = plan_chunks(3 * 3600, 40)  # 3 h -> 5 chunks of 36 min
    assert len(ch) == 5 and ch[0][0] == 0 and ch[-1][1] == 3 * 3600
    assert all(abs((b - a) - 2160) < 1 for a, b in ch)
    assert all(ch[i][1] == ch[i + 1][0] for i in range(len(ch) - 1))


def test_merge_moments():
    out = merge_moments([M(100, 120, 7, summary="a"), M(125, 140, 9, summary="b"), M(300, 320)])
    assert [(m.start_s, m.end_s, m.hype, m.summary) for m in out] == [(100, 140, 9, "b"), (300, 320, 8, "")]


def test_section_string():
    assert hms(2470) == "00:41:10"
    assert section(2475, 2500) == "*00:41:10-00:41:45"
    assert section(2, 20) == "*00:00:00-00:00:25"
    assert section(590, 598, duration=600) == "*00:09:45-00:10:00"


def test_select_and_absolute():
    ms = [M(1, 20, 6), M(1, 20, 9, flags=MomentFlags(cutscene=True)), M(30, 50, 8), M(60, 80, 10)]
    assert [m.hype for m in select_moments(ms, 7, 12)] == [10, 8]
    assert [m.hype for m in select_moments(ms, 7, 1)] == [10]
    rel = to_absolute([M(10, 30)], (2400, 4800))
    assert rel[0].start_s == 2410
    already = to_absolute([M(2500, 2520)], (2400, 4800))  # model answered absolute
    assert already[0].start_s == 2500


def test_url_kinds_and_streamer():
    assert is_youtube("https://youtu.be/abc") and is_twitch_vod("https://www.twitch.tv/videos/123")
    s = streamer_for_video({"uploader_id": "@SomeGamer", "uploader": "Some Gamer", "extractor": "youtube", "language": "de"})
    assert s.login == "somegamer" and s.credit == "youtube.com/@SomeGamer" and s.account == "account_de"
    assert streamer_for_video({"uploader_id": "streamer_en", "extractor": "twitch:vod"}).cam is not None  # known streamer


def test_loudness_peaks():
    from vc.sources.vod_scout import loudness_peaks
    track = [-30.0] * 1200
    for t, lvl in ((200, -5.0), (230, -8.0), (700, -10.0), (1000, -27.0)):  # 230 too close to 200; 1000 too weak
        track[t - 1:t + 2] = [lvl] * 3
    peaks = loudness_peaks(track, n=5)
    assert [t for t, _ in peaks] == [200, 700]


async def test_scout_local_moments(monkeypatch):
    from vc.sources import vod_scout
    async def fake_run_ok(*a, **k):
        return None
    monkeypatch.setattr(vod_scout, "run_ok", fake_run_ok)
    monkeypatch.setattr(vod_scout, "loudness_track", lambda p: [-30.0] * 300 + [-5.0] * 3 + [-30.0] * 300)
    ms = await vod_scout.scout_youtube_local("https://youtu.be/x", 603)
    assert len(ms) == 1 and ms[0].start_s == 301 - 28 and ms[0].hype >= 7
