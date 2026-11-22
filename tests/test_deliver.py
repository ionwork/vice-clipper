from datetime import datetime, timedelta, timezone

from vc.config import AccountCfg, get_streamers
from vc.deliver import planner, textpack
from vc.models import Clip, Source

JUDGE = {"keep": True, "hype": 8, "category": "police_chase", "summary": "s", "start_s": 6.5, "end_s": 24, "peak_s": 17,
         "subject_x": 0.5, "layout": "split", "hook": "He did not see that", "title": "GTA 6 police chase ends on a boat",
         "caption": "Day one and he already broke the game.",
         "hashtags": {"youtube": ["gta6", "gtavi", "streamer_en"], "tiktok": ["gta6", "gtavi", "streamer_en", "gaming"],
                      "instagram": ["gta6", "gtavi", "streamer_en", "gaming"]}, "language": "en", "flags": {}}


def clip() -> Clip:
    s = get_streamers()["streamer_en"]
    return Clip(id="kc-1119-0612-a3", source=Source.LIVE, streamer_id=s.id, platform="twitch", lang="en", hype=8,
                judge=JUDGE, account="account_en", streamer=s.model_dump(mode="json"))


def test_textpack_format():
    slot = datetime.now().astimezone().replace(hour=6, minute=45)
    t = textpack.build(clip(), slot)
    assert t.startswith("#kc-1119-0612-a3 · Hype 8 · police_chase · streamer_en\nAccount: account_en · frühester Slot: 06:45")
    assert ("YOUTUBE SHORTS\nTitel: GTA 6 police chase ends on a boat\nBeschreibung: Day one and he already broke the game.\n"
            "Clip by twitch.tv/streamer_en\n#gta6 #gtavi #streamer_en") in t  # en streamer -> English post words (V2)
    assert "TIKTOK\nHe did not see that. Day one and he already broke the game. #gta6 #gtavi #streamer_en #gaming" in t
    assert "INSTAGRAM\nHe did not see that. Day one and he already broke the game.\nClip: @streamer_en\n#gta6 #gtavi #streamer_en #gaming" in t
    h = textpack.build_html(clip())
    assert h.count("<pre>") == 3


def test_planner():
    cfg = AccountCfg(min_gap_minutes=50, daily_cap=2)
    now = datetime.now(timezone.utc).astimezone().replace(hour=12, minute=0).astimezone(timezone.utc)
    assert planner.next_slot_from([], cfg, now) == now
    last = now - timedelta(minutes=20)
    assert planner.next_slot_from([last], cfg, now) == last + timedelta(minutes=50)
    assert planner.next_slot_from([now - timedelta(hours=3)], cfg, now) == now
    two = [now - timedelta(hours=3), now - timedelta(hours=2)]  # cap reached -> tomorrow
    assert planner.next_slot_from(two, cfg, now).astimezone().date() > now.astimezone().date()


def test_url_regex():
    from vc.deliver.telegram_bot import URL_RE
    assert URL_RE.findall("look https://www.youtube.com/watch?v=abc and https://youtu.be/xyz")
    assert URL_RE.search("https://www.twitch.tv/videos/123456")
    assert not URL_RE.search("https://twitch.tv/streamer_en")


def test_config_writer(vc_root):
    from vc.config import get_config, set_config_value
    set_config_value("judge.min_hype", 8)
    assert get_config().judge.min_hype == 8
    set_config_value("judge.min_hype", 7)


def test_planner_pending_holds_gap_not_cap():
    cfg = AccountCfg(min_gap_minutes=50, daily_cap=1)
    now = datetime.now(timezone.utc).astimezone().replace(hour=12, minute=0).astimezone(timezone.utc)
    pend = now - timedelta(minutes=10)
    assert planner.next_slot_from([], cfg, now, [pend]) == pend + timedelta(minutes=50)
    assert planner.next_slot_from([], cfg, now, [now - timedelta(hours=2)]) == now  # unposted delivery: no cap


async def test_pick_next_slots_and_campaign(db, monkeypatch):
    from vc.deliver.telegram_bot import ClipBot
    from vc.models import Candidate, ClipStatus
    monkeypatch.setenv("STREAMERS", "streamer_en:en, camp:en:campaign, dede:de")  # .env is the only list (V2 Phase 1)
    s = get_streamers()

    async def rendered(sid, hype):
        c = await db.create_clip(Candidate(source=Source.LIVE, streamer=s[sid], raw_path="x"))
        await db.update_clip(c.id, status=ClipStatus.RENDERED, hype=hype)
        return c.id
    plain, camp, de = await rendered("streamer_en", 8), await rendered("camp", 7), await rendered("dede", 6)
    bot = object.__new__(ClipBot)  # pick_next needs no Telegram
    assert (await bot.pick_next()).id == camp  # 7 + campaign bonus 2 > 8
    other = await db.create_clip(Candidate(source=Source.LIVE, streamer=s["streamer_en"], raw_path="y"))
    await db.update_clip(other.id, status=ClipStatus.POSTED, posted_at=db.now())  # account_en just posted
    assert (await bot.pick_next(slots=True)).id == de
    assert plain


def test_setup_helpers(tmp_path):
    from vc.setup import best_flash_lite, write_env
    env = tmp_path / ".env"
    env.write_text("# Kommentar\nTWITCH_CLIENT_ID=\nOTHER=1\n", encoding="utf-8")
    write_env({"twitch_client_id": "abc", "gemini_api_key": "k"}, env)
    assert env.read_text(encoding="utf-8") == "# Kommentar\nTWITCH_CLIENT_ID=abc\nOTHER=1\nGEMINI_API_KEY=k\n"
    names = ["gemini-2.5-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash-lite-preview-09", "gemini-3.5-flash"]
    assert best_flash_lite(names) == "gemini-3.5-flash-lite" and best_flash_lite(["gemini-3.5-pro"]) is None
