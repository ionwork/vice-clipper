from datetime import datetime, timedelta
from pathlib import Path

from vc.models import Streamer
from vc.sources.recorder import RecorderManager, expired, select_segments, seg_time

T = datetime(2026, 11, 19, 12, 0, 0)


def segs(n, start=T, step=4):
    return [(start + timedelta(seconds=i * step), Path(f"{(start + timedelta(seconds=i * step)):%Y%m%d-%H%M%S}.ts")) for i in range(n)]


def test_seg_time():
    assert seg_time(Path("20261119-120004.ts")) == T + timedelta(seconds=4)
    assert seg_time(Path("snapshot.jpg")) is None


def test_select_window():
    s = segs(20)  # 12:00:00 .. 12:01:16
    sizes = {p: 100_000 for _, p in s}
    t0, t1 = T + timedelta(seconds=30), T + timedelta(seconds=50)
    got = [t for t, _ in select_segments(s, t0, t1, 4, sizes)]
    # overlap with [26, 54]: segments starting 24 (ends 28) .. 52
    assert got[0] == T + timedelta(seconds=24) and got[-1] == T + timedelta(seconds=52)
    sizes[s[8][1]] = 1000  # broken segment at 32 s is skipped
    assert T + timedelta(seconds=32) not in [t for t, _ in select_segments(s, t0, t1, 4, sizes)]


def test_janitor():
    s = segs(100)  # 400 s
    old = expired(s, T + timedelta(seconds=400), 320)
    assert len(old) == 20 and old[-1] == s[19][1]


def test_desired_priority_and_cap(monkeypatch):
    from vc import config
    cfg = config.get_config().model_copy(deep=True)
    cfg.recorder.max_parallel = 2
    monkeypatch.setattr("vc.sources.recorder.get_config", lambda: cfg)
    ss = [Streamer(id="a", login="a", priority=3), Streamer(id="b", login="b", priority=1),
          Streamer(id="c", login="c", priority=2), Streamer(id="k", login="k", platform="kick", priority=2)]
    assert RecorderManager().desired(ss, {"a", "b", "c"}) == ["b", "c"]
    assert RecorderManager().desired(ss, {"a"}) == ["a"]            # Kick offline -> not recorded
    assert RecorderManager().desired(ss, {"a", "k"}) == ["k", "a"]  # Kick live (livestream != null)
    assert RecorderManager().desired(ss, set()) == []


async def test_live_transitions_notify(db, monkeypatch):
    """reconcile(): Twitch + Kick live check, one event per live/offline change, recorders follow."""
    from vc.sources import recorder as rec
    ss = {"tw": Streamer(id="tw", login="tw"), "kk": Streamer(id="kk", login="kk", platform="kick")}
    monkeypatch.setattr(rec, "get_streamers", lambda: ss)
    live = {"tw": True, "kk": True}

    class Api:
        async def get_streams(self, logins):
            return [{"user_login": "tw", "game_name": "Grand Theft Auto VI"}] if live["tw"] else []

    async def kick(slug):
        return {"categories": [{"name": "GTA 6"}]} if live["kk"] else None
    monkeypatch.setattr(rec, "kick_livestream", kick)

    class FakeRec:
        def __init__(self, s):
            self.s = s

        def start(self):
            pass

        async def stop(self):
            pass
    monkeypatch.setattr(rec, "Recorder", FakeRec)
    m = RecorderManager(Api())
    seen = []

    async def cb(sid, is_live, info):
        seen.append((sid, is_live, info.get("game_name")))
    m.live_listeners.append(cb)
    await m.reconcile()
    assert sorted(seen) == [("kk", True, "GTA 6"), ("tw", True, "Grand Theft Auto VI")] and set(m.recorders) == {"tw", "kk"}
    seen.clear()
    await m.reconcile()
    assert seen == []  # no change, no message
    live["tw"] = False
    await m.reconcile()
    assert seen == [("tw", False, "Grand Theft Auto VI")] and set(m.recorders) == {"kk"}


async def test_offline_untracks_and_parts_chat(monkeypatch):
    from vc.orchestrator import LiveStack
    stack = LiveStack()
    parted = []

    async def part(login):
        parted.append(login)
    monkeypatch.setattr(stack.twitch_chat, "part", part)
    s = Streamer(id="streamer_en", login="streamer_en")
    stack.engine.track(s)
    assert "streamer_en" in stack.engine.ch
    await stack._on_recorder("streamer_en", None)
    assert "streamer_en" not in stack.engine.ch and "streamer_en" not in stack.engine.login_to_id.values() and parted == ["streamer_en"]
