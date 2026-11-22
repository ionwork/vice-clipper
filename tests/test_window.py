from datetime import timedelta

from vc.config import get_config, get_streamers
from vc.deliver import window as wmod
from vc.deliver.window import Window, batch_note, effective_v, select_batch
from vc.models import Candidate, Clip, ClipStatus, Source

D = get_config().delivery


def vsig(v: float) -> dict:
    return {"virality": {"v": v, "components": {"judge": v / 100}, "weights_used": {"judge": 0.2}}}


def mk(cid: str, v: float, sid: str | None = None, t_off_min: float = 0, age_min: float = 0) -> Clip:
    from vc.db import now
    n = now()
    return Clip(id=cid, source=Source.LIVE, streamer_id=sid or cid, platform="twitch", lang="de", signals=vsig(v),
                status=ClipStatus.RENDERED, created_at=n - timedelta(minutes=age_min), t_trigger=n - timedelta(minutes=t_off_min))


def ids(batch):
    return [c.id for c in batch]


def test_select_batch_sizes():
    assert ids(select_batch([mk(f"c{v}", v) for v in (90, 70, 60, 40, 20)], D, 5)) == ["c90", "c70", "c60"]
    assert ids(select_batch([mk(f"c{v}", v) for v in (90, 70, 50)], D, 5)) == ["c90", "c70"]   # 50 < third_clip_min_v
    one = select_batch([mk("c90", 90)], D, 5)
    assert ids(one) == ["c90"] and batch_note(len(one), 1, D, 5) == f"nur 1 Clip mit V ≥ {D.min_v:g}"
    assert select_batch([mk("low", 34.9)], D, 5) == []                                          # no filler under min_v
    assert batch_note(0, 0, D, 0) == "keiner live" and batch_note(2, 5, D, 3) is None


def test_select_batch_per_streamer_cap():
    same = [mk(f"k{i}", v, sid="streamer_de", t_off_min=5 * i) for i, v in enumerate((90, 80, 70))]  # 5 min apart: no duplicates
    assert len(select_batch(same, D, live_count=3)) == 2
    assert len(select_batch(same, D, live_count=1)) == 3   # only one streamer live: no cap


def test_select_batch_duplicates_and_decay():
    a, b = mk("a", 90, sid="x"), mk("b", 80, sid="x", t_off_min=0.1)   # same moment, same streamer
    assert ids(select_batch([b, a], D, 3)) == ["a"]
    start = mk("ref", 0).created_at
    old = mk("old", 40, age_min=30)                                     # 30 min before the window: -6 points
    assert abs(effective_v(old, start, D) - (40 - 6 * D.age_decay_per_5min)) < 0.1
    assert select_batch([old], D, 3, start) == []


async def test_three_windows_in_45_fake_minutes(db, monkeypatch):
    """Fake clock, no real sleep: 45 minutes -> exactly 3 window closes with 2-3 deliveries each."""
    import time
    monkeypatch.setattr(get_config().delivery, "delete_losers", False)  # scheduling only; discarding: test below
    s = get_streamers()["streamer_en"]
    for i in range(12):
        c = await db.create_clip(Candidate(source=Source.LIVE, streamer=s, raw_path=str(i),
                                           t_trigger=db.now() - timedelta(minutes=3 * i)))
        await db.update_clip(c.id, status=ClipStatus.RENDERED, signals=vsig(60 + 2 * i))

    async def keep(c, hub=None, store=True):
        return None
    monkeypatch.setattr(wmod.virality, "compute", keep)
    delivered: list[str] = []

    async def deliver(cid):
        delivered.append(cid)
        await db.update_clip(cid, status=ClipStatus.DELIVERED, delivered_at=db.now())
    notes: list[str] = []

    async def notify(text):
        notes.append(text)
    clock = {"t": time.time()}
    w = Window(deliver, notify, live_count=lambda: 1, clock=lambda: clock["t"])
    await w.open()
    assert await db.kv_get("window_start") is not None
    closes, per_close = 0, []
    for _ in range(90):                          # 90 x 30 s = 45 min
        clock["t"] += 30
        before = len(delivered)
        if await w.tick():
            closes += 1
            per_close.append(len(delivered) - before)
    assert closes == 3 and all(2 <= n <= 3 for n in per_close), per_close
    assert len(set(delivered)) == len(delivered) and notes == []
    assert w.number == 4 and w.delivered == 0     # the 4th window just opened


async def test_window_note_when_nothing_qualifies(db):
    notes: list[str] = []

    async def notify(text):
        notes.append(text)

    async def deliver(cid):
        raise AssertionError("nothing to deliver")
    w = Window(deliver, notify, live_count=lambda: 0)
    await w.open()
    assert await w.close() == [] and notes == ["⏱ Fenster 1: keiner live"]
    await w.open()
    await w.close()
    assert len(notes) == 1  # same reason in the next window: not repeated


async def _eight(db, vc_root):
    """8 rendered clips with media files, plus a DELIVERED and a hotkey (MANUAL) clip."""
    s = get_streamers()["streamer_en"]
    out = vc_root / "data" / "out"
    cand = vc_root / "data" / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    cand.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, v in enumerate((90, 85, 80, 70, 60, 50, 45, 40)):
        c = await db.create_clip(Candidate(source=Source.LIVE, streamer=s, raw_path=str(cand / f"raw{i}.mp4"),
                                           t_trigger=db.now() - timedelta(minutes=3 * i)))
        files = [cand / f"raw{i}.mp4", out / f"{c.id}.mp4", cand / f"{c.id}.ass", cand / f"{c.id}.words.json"]
        for f in files:
            f.write_bytes(b"x")
        await db.update_clip(c.id, status=ClipStatus.RENDERED, signals=vsig(v), out_path=str(files[1]), ass_path=str(files[2]))
        clips.append((c.id, files))
    keep = []
    for src, st in ((Source.LIVE, ClipStatus.DELIVERED), (Source.MANUAL, ClipStatus.RENDERED)):
        c = await db.create_clip(Candidate(source=src, streamer=s, raw_path=str(cand / f"k{src.value}.mp4"),
                                           t_trigger=db.now() - timedelta(hours=1)))
        (cand / f"k{src.value}.mp4").write_bytes(b"x")
        await db.update_clip(c.id, status=st, signals=vsig(20))
        keep.append((c.id, st, cand / f"k{src.value}.mp4"))
    return clips, keep


async def test_window_delivers_top3_and_discards_the_rest(db, vc_root, monkeypatch):
    async def keepv(c, hub=None, store=True):
        return None
    monkeypatch.setattr(wmod.virality, "compute", keepv)
    clips, keep = await _eight(db, vc_root)
    delivered: list[str] = []

    async def deliver(cid):
        delivered.append(cid)
        await db.update_clip(cid, status=ClipStatus.DELIVERED, delivered_at=db.now())

    async def notify(text):
        pass
    w = Window(deliver, notify, live_count=lambda: 1)
    await w.open()
    await w.close()
    assert delivered == [cid for cid, _ in clips[:3]]
    for cid, files in clips[3:]:
        c = await db.get_clip(cid)
        assert c is not None and c.status == ClipStatus.REJECTED and c.reason.startswith("nicht Top 3 (Fenster 1, V")
        assert not any(f.exists() for f in files) and c.signals["virality"]["v"] > 0  # row + signals stay
    for cid, files in clips[:3]:
        assert files[0].exists()
    for cid, st, f in keep:
        assert (await db.get_clip(cid)).status == st and f.exists()


async def test_delete_losers_off_keeps_everything(db, vc_root, monkeypatch):
    async def keepv(c, hub=None, store=True):
        return None
    monkeypatch.setattr(wmod.virality, "compute", keepv)
    monkeypatch.setattr(get_config().delivery, "delete_losers", False)
    clips, _ = await _eight(db, vc_root)

    async def deliver(cid):
        await db.update_clip(cid, status=ClipStatus.DELIVERED, delivered_at=db.now())

    async def notify(text):
        pass
    w = Window(deliver, notify, live_count=lambda: 1)
    await w.open()
    await w.close()
    for cid, files in clips[3:]:
        assert (await db.get_clip(cid)).status == ClipStatus.RENDERED and all(f.exists() for f in files)


def test_discard_targets_stay_inside_data(vc_root, tmp_path):
    from vc.deliver.window import discard_targets
    outside = tmp_path / "mine.mp4"
    outside.write_bytes(b"x")
    c = mk("z", 50)
    c.raw_path = str(outside)
    assert discard_targets(c, vc_root / "data") == []


def test_head_lines():
    from vc.deliver.telegram_bot import head_lines
    from vc.detect.virality import ViralityResult
    c = mk("h", 78)
    c.signals = {**c.signals, "reaction_label": "laugh", "content": "irl"}
    c.judge = {"category": "laugh"}
    r = ViralityResult(v=78, components={"velocity": 0.9, "reaction": 0.8, "library": 0.7, "judge": 0.8},
                       weights_used={"velocity": 0.2})
    h = head_lines(c, r)
    assert "V 78 · Chat 0.9 · 😂 0.8 · Clips 0.7 · KI 0.8" in h and "Echt-Leben" in h and "Reaktion 😂 laugh" in h
