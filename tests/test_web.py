import httpx

from vc.config import get_streamers
from vc.models import Candidate, ClipStatus, Source
from vc.web.app import create_app


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://t")


async def test_pages_and_clip_actions(db, synth_video):
    s = get_streamers()["streamer_en"]
    c = await db.create_clip(Candidate(source=Source.LIVE, streamer=s, raw_path=str(synth_video), pre_score=7))
    async with client() as cl:
        for path in ("/", "/streamers", "/settings", "/partials/clips", "/partials/tiles", "/partials/header"):
            r = await cl.get(path)
            assert r.status_code == 200, path
        assert c.id in (await cl.get("/partials/clips")).text
        assert [x["id"] for x in (await cl.get("/api/clips")).json()] == [c.id]
        assert (await cl.get(f"/clips/{c.id}/preview")).status_code == 200
        assert (await cl.post(f"/api/clips/{c.id}/deliver")).status_code == 409  # not rendered
        assert (await cl.post("/api/clip/streamer_en")).status_code == 409          # no live runtime
        assert (await cl.post(f"/api/clips/{c.id}/reject")).status_code == 200
    assert (await db.get_clip(c.id)).status == ClipStatus.REJECTED


async def test_cam_and_settings_write_yaml(db):
    from vc.config import get_config, get_streamer
    async with client() as cl:
        r = await cl.post("/api/streamers/streamer_en/cam", json={"x": 10, "y": 700, "w": 500, "h": 380})
        assert r.status_code == 200
        assert get_streamer("streamer_en").cam.w == 500
        r = await cl.post("/settings", data={"judge.min_hype": "8", "trigger.z_threshold": "4,5", "delivery.auto": "on"})
        assert r.status_code == 303
    cfg = get_config()
    assert cfg.judge.min_hype == 8 and cfg.trigger.z_threshold == 4.5 and cfg.delivery.auto and not cfg.delivery.paused
    from vc.config import set_config_value
    set_config_value("judge.min_hype", 7)
