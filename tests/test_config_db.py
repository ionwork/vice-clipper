from vc.config import get_config, get_streamers, lexicon_for
from vc.models import Candidate, ClipStatus, Source


def test_config_loads():
    cfg = get_config()
    assert cfg.trigger.z_threshold == 4.0
    assert cfg.render.width == 1080
    assert "account_en" in cfg.accounts
    assert "KEKW" in lexicon_for("de") and "digga" in lexicon_for("de")


def test_streamers_load():
    s = get_streamers()["streamer_en"]
    assert s.cam and s.cam.w == 480 and s.account == "account_en"


async def test_db_roundtrip(db):
    s = get_streamers()["streamer_en"]
    c = await db.create_clip(Candidate(source=Source.LIVE, streamer=s, raw_path="x.mp4", pre_score=6.5,
                                       signals={"z": 5.1}, chat_context=["a", "b"]))
    assert c.id.startswith("st-")
    got = await db.get_clip(c.id)
    assert got.status == ClipStatus.NEW and got.signals == {"z": 5.1} and got.chat_context == ["a", "b"]
    await db.update_clip(c.id, status=ClipStatus.JUDGED, hype=8, judge={"keep": True})
    got = await db.get_clip(c.id)
    assert got.status == ClipStatus.JUDGED and got.hype == 8 and got.judge == {"keep": True}
    assert [x.id for x in await db.list_clips(ClipStatus.JUDGED)] == [c.id]
    assert await db.list_clips(ClipStatus.NEW) == []
    await db.add_event("gemini_usage", clip_id=c.id, prompt=1000, output=100, usd=0.01)
    p, o, usd = await db.cost_today()
    assert (p, o) == (1000, 100) and abs(usd - 0.01) < 1e-9
    await db.kv_set("a", "1")
    assert await db.kv_get("a") == "1"


def test_env_streamers(vc_root, monkeypatch):
    from vc.config import get_streamers, parse_env_streamers, upsert_streamer
    s = parse_env_streamers("streamer_x, twitch:Streamer_Y:de ;kick:streamer_z:en:campaign\n@foo", {"de": "account_de", "en": "account_en"})
    assert list(s) == ["streamer_x", "streamer_y", "streamer_z", "foo"]
    assert s["streamer_y"].lang == "de" and s["streamer_y"].account == "account_de" and s["streamer_y"].credit == "twitch.tv/streamer_y"
    assert s["streamer_z"].platform == "kick" and s["streamer_z"].permission == "campaign" and s["streamer_z"].priority == 1
    assert s["streamer_x"].platform == "twitch" and s["streamer_x"].lang == "en"
    monkeypatch.setenv("STREAMERS", "streamer_x, streamer_en")
    import vc.config as c
    c._env_cache = None
    all_ = get_streamers()
    assert "streamer_x" in all_ and all_["streamer_en"].cam is not None  # yaml entry wins
    upsert_streamer({"id": "streamer_x", "cam": {"x": 0, "y": 0, "w": 480, "h": 270}}, defaults=all_["streamer_x"].model_dump(mode="json"))
    assert get_streamers()["streamer_x"].cam.w == 480 and get_streamers()["streamer_x"].login == "streamer_x"
    c._env_cache = None


def test_env_list_is_the_whole_list(vc_root, monkeypatch):
    """STREAMERS is the only list: yaml-only streamers are ignored, empty STREAMERS -> nobody (no yaml fallback)."""
    import vc.config as c
    monkeypatch.setenv("STREAMERS", "streamer_de:de, streamer_de2:de")
    assert sorted(c.get_streamers()) == ["streamer_de", "streamer_de2"]
    monkeypatch.setenv("STREAMERS", "")
    assert c.get_streamers() == {}


def test_parse_env_streamers_cases():
    from vc.config import parse_env_streamers
    acc = {"de": "account_de", "en": "account_en"}
    err: list[str] = []
    s = parse_env_streamers("twitch:streamer_de:de:campaign", acc, err)
    assert s["streamer_de"].lang == "de" and s["streamer_de"].permission == "campaign" and not err      # valid
    assert parse_env_streamers("twitch:somebody", acc)["somebody"].lang == "en"              # no lang -> en
    assert parse_env_streamers("camp:en:campaign", acc)["camp"].permission == "campaign"     # platform optional
    err = []
    assert parse_env_streamers("twitch:x_y:fr, ok_login:de", acc, err).keys() == {"ok_login"}  # bad lang skipped
    assert len(err) == 1 and "x_y" in err[0] and "Sprache" in err[0]
    err = []
    assert parse_env_streamers("youtube:abc:de", acc, err) == {} and "Plattform" in err[0]
    err = []
    s = parse_env_streamers("dup:de, twitch:DUP:en", acc, err)                                # duplicate once
    assert list(s) == ["dup"] and s["dup"].lang == "de" and "doppelt" in err[0]
    err = []
    assert parse_env_streamers("", acc, err) == {} and err == []                              # empty string
    assert parse_env_streamers(" , ;", acc, err) == {} and err == []


def test_env_edit_is_picked_up_live(vc_root):
    """The recorder calls get_streamers() every live check: an edited .env applies without restart (mtime cache)."""
    import os
    import vc.config as c
    env = vc_root / ".env"
    old = env.read_text(encoding="utf-8")
    try:
        env.write_text("STREAMERS=twitch:streamer_en:en, twitch:neuer_login:de\n", encoding="utf-8")
        st = os.stat(env)
        os.utime(env, (st.st_atime, st.st_mtime + 5))
        assert set(c.get_streamers()) == {"streamer_en", "neuer_login"} and c.get_streamers()["neuer_login"].lang == "de"
    finally:
        env.write_text(old, encoding="utf-8")
        os.utime(env, (st.st_atime, st.st_mtime + 10))
    assert set(c.get_streamers()) == {"streamer_en"}


def test_yaml_does_not_override_env_lang(vc_root, monkeypatch):
    import vc.config as c
    monkeypatch.setenv("STREAMERS", "streamer_en:de")
    s = c.get_streamers()["streamer_en"]
    assert s.lang == "de" and s.cam is not None  # lang from .env, cam from streamers.yaml


async def test_start_refused_without_streamers(monkeypatch):
    from vc.config import NO_STREAMERS_MSG
    from vc.control import Controller
    from vc.orchestrator import Orchestrator
    monkeypatch.setenv("STREAMERS", "")
    o = Orchestrator(web=False, bot=False)
    assert await Controller(o).go() == NO_STREAMERS_MSG
    assert await o.start_pipeline() is False and not o.running
