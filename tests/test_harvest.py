import pytest
from datetime import datetime, timedelta, timezone

from vc.config import get_streamers
from vc.models import Candidate, Source
from vc.sources import harvester as hv

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)


def clip(i, views, age_min=10, lang="en", dur=30, name="SomeOne"):
    return {"id": f"c{i}", "url": f"https://clips.twitch.tv/c{i}", "broadcaster_name": name, "broadcaster_id": "1",
            "title": "t", "view_count": views, "created_at": (NOW - timedelta(minutes=age_min)).isoformat().replace("+00:00", "Z"),
            "duration": dur, "language": lang}


@pytest.fixture(autouse=True)
def open_harvest(monkeypatch):
    """These tests cover the open mode (clips of any streamer); the whitelist mode has its own test."""
    from vc.config import get_config
    cfg = get_config().model_copy(deep=True)
    cfg.harvest.only_whitelisted = False
    monkeypatch.setattr(hv, "get_config", lambda: cfg)
    return cfg


def test_whitelist_mode(open_harvest):
    open_harvest.harvest.only_whitelisted = True
    assert hv.passes_filters({**clip(1, 200), "broadcaster_login": "streamer_de"}, {"streamer_de"})
    assert not hv.passes_filters(clip(1, 200, name="Stranger"), {"streamer_de"})


def test_velocity():
    created = NOW - timedelta(minutes=10)
    assert hv.velocity(500, created, NOW, None) == 50
    assert hv.velocity(500, created, NOW, (NOW - timedelta(minutes=3), 200)) == 100
    assert hv.velocity(100, NOW, NOW, None) == 100  # age < 1 min counts as 1


def test_filters():
    assert hv.passes_filters(clip(1, 200), set())
    assert not hv.passes_filters(clip(1, 100), set())             # min_views 150
    assert not hv.passes_filters(clip(1, 200, lang="fr"), set())  # language
    assert not hv.passes_filters(clip(1, 200, dur=90), set())     # max_duration


def test_adhoc_streamer():
    s = hv.streamer_for(clip(1, 200, lang="de", name="GermanGuy"))
    assert s.login == "germanguy" and s.account == "account_de" and s.cam is None and s.credit == "twitch.tv/germanguy"
    assert hv.streamer_for(clip(1, 200, name="Streamer_En")).cam is not None  # known streamer from streamers.yaml
    # display name != login: the resolved login drives id and credit
    s = hv.streamer_for({**clip(1, 200, name="Der Echte"), "broadcaster_login": "derechte_tv"})
    assert s.login == "derechte_tv" and s.credit == "twitch.tv/derechte_tv" and s.name == "Der Echte"


class FakeAPI:
    def __init__(self, clips):
        self.clips = clips

    async def get_game_id(self, name):
        return "1"

    async def game_name(self, gid):
        return "Grand Theft Auto V" if gid else ""

    async def get_clips(self, **kw):
        return self.clips if kw.get("game_id") else []

    async def user_ids(self, logins):
        return {}

    async def logins_for_ids(self, ids):
        return {}


async def test_poll_picks_fastest_and_skips_duplicates(db, monkeypatch):
    clips = [clip(1, 1000, age_min=5), clip(2, 300, age_min=60), clip(3, 2000, age_min=8, name="Streamer_En"), clip(4, 50)]
    h = hv.Harvester(api=FakeAPI(clips))
    ingested = []

    async def fake_ingest(c, s, t, rank, v):
        ingested.append((c["id"], rank))
        return f"id{c['id']}"
    monkeypatch.setattr(h, "_ingest", fake_ingest)
    # existing candidate of streamer_en 20 s before clip 3 -> duplicate
    s = get_streamers()["streamer_en"]
    t3 = datetime.fromisoformat(clips[2]["created_at"].replace("Z", "+00:00"))
    await db.create_clip(Candidate(source=Source.LIVE, streamer=s, raw_path="x", t_trigger=t3 - timedelta(seconds=20)))
    picked = await h.poll()
    outcomes = {c["id"]: o for c, _, o in picked}
    assert outcomes["c3"] == "duplicate" and "c4" not in outcomes
    assert [i for i, _ in ingested] == ["c1", "c2"]  # by velocity: c1 200/min > c2 5/min
    assert hv.pre_score_for_rank(0) == 9 and hv.pre_score_for_rank(10) == 5


def test_suggest_cam(tmp_path):
    """No face in a test pattern -> no suggestion."""
    import subprocess

    from vc.proc import tool
    v = tmp_path / "p.mp4"
    subprocess.run([tool("ffmpeg"), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:duration=3",
                    "-c:v", "libx264", "-preset", "ultrafast", str(v)], check=True)
    assert hv.suggest_cam(v) is None
