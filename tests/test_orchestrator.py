import asyncio
import os
import time
from datetime import timedelta

from vc.config import get_streamers
from vc.models import Candidate, Clip, ClipStatus, Source
from vc.orchestrator import Orchestrator, cleanup_targets, duplicate_loser, overlap_ratio, priority_key


def C(cid, source, pre=6.0, age_min=0, hype=None, t_off=0, sid="streamer_en", clipcalls=0, status=ClipStatus.NEW):
    from vc.db import now
    n = now()
    return Clip(id=cid, source=source, streamer_id=sid, platform="twitch", lang="en", pre_score=pre, hype=hype,
                status=status, created_at=n - timedelta(minutes=age_min), t_trigger=n - timedelta(seconds=t_off),
                signals={"clipcalls": clipcalls})


def test_queue_order():
    from vc.db import now
    clips = [C("vod", Source.VOD, 9), C("harv", Source.HARVEST, 9), C("live", Source.LIVE, 6),
             C("call", Source.LIVE, 5, clipcalls=3), C("hot", Source.MANUAL, 10)]
    order = [c.id for c in sorted(clips, key=lambda c: priority_key(c, now()))]
    assert order == ["hot", "call", "live", "harv", "vod"]


def test_freshness_decay():
    from vc.db import now
    fresh, old = C("fresh", Source.LIVE, 6), C("old", Source.LIVE, 8, age_min=30)  # 8 - 3 = 5 < 6
    assert sorted([old, fresh], key=lambda c: priority_key(c, now()))[0].id == "fresh"


def test_duplicate_rule():
    assert overlap_ratio((0, 38), (10, 48)) > 0.5 and overlap_ratio((0, 38), (30, 68)) < 0.5
    a, b = C("a", Source.LIVE, 6, t_off=0), C("b", Source.LIVE, 8, t_off=10)
    assert duplicate_loser(a, [a, b]) == "a"
    assert duplicate_loser(b, [a, b]) == "a"
    far = C("far", Source.LIVE, 9, t_off=60)
    assert duplicate_loser(a, [a, far]) is None
    other = C("o", Source.LIVE, 9, t_off=5, sid="streamer_x")  # same moment, other streamer: allowed
    assert duplicate_loser(a, [a, other]) is None


async def test_resume(db):
    s = get_streamers()["streamer_en"]
    fresh = await db.create_clip(Candidate(source=Source.LIVE, streamer=s, raw_path="x"))
    capt = await db.create_clip(Candidate(source=Source.LIVE, streamer=s, raw_path="y"))
    old = await db.create_clip(Candidate(source=Source.LIVE, streamer=s, raw_path="z"))
    await db.update_clip(capt.id, status=ClipStatus.CAPTIONED)
    await db.update_clip(old.id, created_at=db.now() - timedelta(hours=3))
    await Orchestrator(live=False, bot=False, web=False).resume()
    assert (await db.get_clip(fresh.id)).status == ClipStatus.NEW
    assert (await db.get_clip(capt.id)).status == ClipStatus.JUDGED
    assert (await db.get_clip(old.id)).status == ClipStatus.REJECTED


async def test_judge_worker_with_fake_judge(db, monkeypatch):
    """Two NEW clips of the same moment: only the better one gets judged."""
    from vc.judge import gemini
    s = get_streamers()["streamer_en"]
    t = db.now()
    weak = await db.create_clip(Candidate(source=Source.LIVE, streamer=s, raw_path="a", pre_score=5.5, t_trigger=t))
    strong = await db.create_clip(Candidate(source=Source.LIVE, streamer=s, raw_path="b", pre_score=8, t_trigger=t))
    judged = []

    async def fake_judge(cid):
        judged.append(cid)
        await db.update_clip(cid, status=ClipStatus.JUDGED)
    monkeypatch.setattr(gemini, "judge_candidate", fake_judge)
    o = Orchestrator(live=False, bot=False, web=False)
    task = asyncio.create_task(o.judge_worker())
    for _ in range(50):
        await asyncio.sleep(0.1)
        if judged:
            break
    task.cancel()
    assert judged == [strong.id]
    assert (await db.get_clip(weak.id)).reason == "Duplikat"


def test_cleanup_targets(tmp_path):
    for sub in ("candidates", "out"):
        (tmp_path / sub).mkdir()
    old_raw, new_raw, old_out = tmp_path / "candidates/a.mp4", tmp_path / "candidates/b.mp4", tmp_path / "out/c.mp4"
    for p in (old_raw, new_raw, old_out):
        p.write_bytes(b"x")
    now = time.time()
    os.utime(old_raw, (now - 2 * 86400, now - 2 * 86400))
    os.utime(old_out, (now - 2 * 86400, now - 2 * 86400))  # out keeps 7 days
    assert cleanup_targets(tmp_path, now) == [old_raw]


async def test_shed_low_v_when_queue_full(db):
    """More than judge_queue_max NEW: candidates with V < judge_min are rejected unjudged; hotkey clips stay."""
    s = get_streamers()["streamer_en"]
    ids = {}
    for i, (z, src) in enumerate([(10, Source.LIVE)] * 4 + [(3.5, Source.LIVE)] * 3 + [(3.5, Source.MANUAL)]):
        c = await db.create_clip(Candidate(source=src, streamer=s, raw_path=str(i), signals={"z": z},
                                           t_trigger=db.now() - timedelta(minutes=i)))  # ids are unique per minute
        ids[c.id] = (z, src)
    o = Orchestrator(live=False, bot=False, web=False)
    assert await o.shed() == 3
    for cid, (z, src) in ids.items():
        got = await db.get_clip(cid)
        low = z < 5 and src == Source.LIVE
        assert (got.status == ClipStatus.REJECTED and got.reason == "V zu niedrig") == low
    # queue no longer over the limit: nothing more is shed
    assert await o.shed() == 0


async def test_render_only_realistic_top(db, monkeypatch):
    """Window mode: only the best render_max candidates by V are rendered; hotkey clips always."""
    from vc.config import get_config, get_streamers
    from vc.deliver.window import Window
    monkeypatch.setattr(get_config().delivery, "render_max", 2)
    s = get_streamers()["streamer_en"]
    ids = {}
    for name, v, src in (("a", 90, Source.LIVE), ("b", 80, Source.LIVE), ("c", 30, Source.LIVE), ("m", 10, Source.MANUAL)):
        c = await db.create_clip(Candidate(source=src, streamer=s, raw_path=name, t_trigger=db.now()))
        await db.update_clip(c.id, status=ClipStatus.JUDGED, signals={"virality": {"v": v, "components": {"judge": 1},
                                                                                  "weights_used": {"judge": 1}}})
        ids[name] = c.id
    o = Orchestrator(live=False, bot=False, web=False)

    async def noop(cid):
        pass
    o.window = Window(noop, noop)
    allowed = await o._render_allowed()
    assert allowed == {ids["a"], ids["b"], ids["m"]}
    seen = set()
    for _ in range(3):
        c = await o._next_render()
        seen.add(c.id)
        o.claimed.add(c.id)
    assert ids["c"] not in seen and await o._next_render() is None
