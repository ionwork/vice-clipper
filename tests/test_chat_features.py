"""Phase 4: chat features (lift vs own baseline, distinct users, bots ignored) and the V2 golden ranking."""
from vc.config import get_config
from vc.detect import virality as vr
from vc.detect.chat_features import Classifier
from vc.detect.chat_stats import stats_from_config
from vc.detect.voice import voice_features

C = get_config().virality
T0 = 1_800_000_000.0
BURST = T0 + 200


def fx() -> Classifier:
    cfg = get_config()
    return Classifier(C.categories, cfg.chat.ignore_users, cfg.chat.ignore_patterns, C.clipcall_phrases)


def chat(base_text=lambda k: f"hello there {k}", burst=(), rate=2, users=300):
    """200 s of ordinary chat, then the burst messages [(dt, user, text)], snapshot 5 s after the burst start."""
    st = stats_from_config("en")
    k = 0
    for sec in range(200):
        for j in range(rate):
            k += 1
            st.add(T0 + sec + j / rate, f"u{k % users}", base_text(k))
    for dt, user, text in burst:
        st.add(BURST + dt, user, text)
    return st, st.snapshot(BURST + 5)


def spread(n, user, text):
    return [(j * 4.5 / n, user(j), text) for j in range(n)]


# ---------------------------------------------------------------- features
def test_distinct_laughers_lift_reaction_one_spammer_does_not():
    _, crowd = chat(burst=spread(20, lambda j: f"l{j}", "KEKW"))
    _, spam = chat(burst=spread(20, lambda j: "spammer", "KEKW"))
    assert crowd["reaction_label"] == "laugh" and vr.n_reactions(crowd["reaction_lift"], C) > 0.5
    assert spam["reaction_lifts"]["laugh"] == 1.0 and vr.n_reactions(spam["reaction_lift"], C) == 0


def test_constant_w_chat_has_no_lift_sudden_w_has():
    always_w = lambda k: "W" if k % 2 else f"hello {k}"  # noqa: E731
    burst = [(j * 0.1, f"w{j}", "W" if j % 2 else f"hi {j}") for j in range(40)]
    _, steady = chat(base_text=always_w, burst=burst)
    _, sudden = chat(burst=spread(20, lambda j: f"w{j}", "W"))
    assert steady["reaction_lifts"]["hype"] <= 1.2 and vr.n_reactions(steady["reaction_lifts"]["hype"], C) < 0.1
    assert sudden["reaction_label"] == "hype" and vr.n_reactions(sudden["reaction_lift"], C) > 0.5


def test_intensity():
    f = fx()
    assert not f.classify("a", "KEKW KEKW").intense
    assert f.classify("a", "NEEEEIN WAS").intense and f.classify("a", "nooooo").intense and f.classify("a", "what!!!").intense
    assert not f.classify("a", "Hallo zusammen").intense


def test_question_marks_are_confusion():
    _, s = chat(burst=spread(10, lambda j: f"q{j}", "???"))
    assert s["reaction_label"] == "confusion" and s["reaction_lift"] > 2
    assert "confusion" not in fx().classify("a", "was ist das?").cats


def test_giveaway_wave_and_bots_are_ignored():
    f = fx()
    assert f.classify("nightbot", "follow the stream!").ignored and f.classify("x", "!enter").ignored
    assert f.classify("x", "GIVEAWAY ist live").ignored and not f.classify("x", "!clip").ignored
    _, quiet = chat()
    _, wave = chat(burst=spread(40, lambda j: f"g{j}", "!enter") + spread(40, lambda j: "Nightbot", "Join the giveaway"))
    assert wave["burst"] == quiet["burst"] and wave["z"] == quiet["z"]


def test_same_emote_from_many_users_lifts_convergence():
    st, _ = chat()
    st2, _ = chat()
    st2.channel_emotes = frozenset({"catJAM"})
    for j in range(8):
        st2.add(BURST + j * 0.5, f"c{j}", "catJAM")
    s = st2.snapshot(BURST + 5)
    assert s["conv_lift"] > 2 and vr.n_lift(s["conv_lift"], C.log_span) > 0.3
    assert fx().classify("a", "catJAM catJAM", channel=frozenset({"catJAM"})).emote_only
    assert fx().classify("a", "NEEEIN").token == "nein"


def test_voice_features():
    words = [(i * 0.5, i * 0.5 + 0.4, "ja") for i in range(10)] + [(5 + i * 0.1, 5.1 + i * 0.1, "oh") for i in range(20)]
    words += [(7.5, 7.8, "oh"), (7.8, 8.0, "my"), (8.0, 8.3, "god"), (8.3, 8.6, "clip"), (8.6, 8.9, "that")]
    loud = [-30.0] * 7 + [-10.0] + [-40.0] * 4 + [-30.0] * 3
    v = voice_features(words, loud, "en", C)
    assert v["voice_keyword"] == "oh my god" and v["streamer_clipcall"] and v["voice_silence"]
    assert v["voice"] == 1.0 and voice_features([], [], "de", C)["voice"] == 0


# ---------------------------------------------------------------- golden ranking
def moment(burst, *, calls=0, viewer_clips=0, voice_db=None) -> float:
    _, snap = chat(burst=burst)
    comps: dict = {}
    vr.chat_comps(snap, C, comps, {})
    msgs = [(BURST + dt, u, x) for dt, u, x in burst]
    comps["clip_calls"] = vr.n_clip_calls(vr.clipcall_users(msgs, BURST, C, get_config().chat.ignore_users), C)
    if viewer_clips:
        comps["library"] = vr.n_library(viewer_clips, 0, 0.25, False, C)
    if voice_db is not None:
        comps["voice"] = vr.n_audio(voice_db, C)
    return vr.score(comps, C).v


def test_golden_ranking():
    spike = [(j * 0.05, f"s{j}", f"look at this {j}") for j in range(80)]
    laugh = [(j * 0.08, f"l{j}", "KEKW") for j in range(30)]
    shock = [(j * 0.08, f"k{j}", "WTF") for j in range(30)]
    calls = [(1 + j * 0.3, f"c{j}", "clip that") for j in range(6)]
    bots = spread(60, lambda j: "streamelements", "Follow!") + spread(60, lambda j: f"b{j}", "!enter")
    vs = [moment(spike + laugh + calls, viewer_clips=2),
          moment(spike + shock),
          moment(spike),
          moment([], voice_db=12),
          moment(bots)]
    assert vs == sorted(vs, reverse=True) and len(set(vs)) == 5, vs


def test_old_component_names_keep_old_v():
    old = {"chat_spike": 0.8, "reactions": 0.5, "clip_calls": 0.2, "library": None, "judge": 0.4, "audio": 0.1}
    w = {k: C.legacy_weights[k] for k, s in old.items() if s is not None}
    before = 100 * sum(w[k] * old[k] for k in w) / sum(w.values())
    r = vr.score(old, C)
    assert abs(r.v - before) <= 0.5 and "chat_spike" in r.weights_used
    assert "Chat 0.8" in vr.summary_line(r) and vr.breakdown(r, {"z": 8})[0][0] == "chat_spike"


async def test_emote_services_never_block(monkeypatch, vc_root):
    from vc.sources import emotes
    monkeypatch.setattr(emotes, "sources", lambda tid: [("7tv", "http://127.0.0.1:9/x", emotes._names_7tv)])
    assert await emotes.channel_emotes("nobody", "1") == frozenset()
    assert emotes._names_bttv([{"code": "catJAM", "imageType": "gif"}]) == {"catJAM": True}


async def test_captions_reuse_judge_words(db, monkeypatch):
    from test_deliver import JUDGE
    from vc.config import get_streamers
    from vc.judge.local import save_words
    from vc.models import Candidate, Source
    from vc.render import captions
    from vc.render.captions import Word

    def boom(*a):
        raise AssertionError("second Whisper pass")
    monkeypatch.setattr(captions, "_transcribe_safe", boom)
    c = await db.create_clip(Candidate(source=Source.LIVE, streamer=get_streamers()["streamer_en"], raw_path="x"))
    await db.update_clip(c.id, judge=JUDGE)
    save_words(c.id, [Word("hello", JUDGE["start_s"] + 1, JUDGE["start_s"] + 1.4),
                      Word("world", JUDGE["start_s"] + 1.5, JUDGE["start_s"] + 2)], [-20.0])
    ass = (await captions.build_ass(c.id)).read_text(encoding="utf-8")
    assert "HELLO" in ass.upper() and "WORLD" in ass.upper()


async def test_harvest_clip_gets_chat_components_from_log(db, vc_root):
    import json
    from datetime import datetime

    from vc.config import get_streamers
    from vc.models import Candidate, Source
    t = db.now().timestamp() - 3600
    d = vc_root / "data" / "chatlogs"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"ts": t - 200 + k * 0.5, "channel": "streamer_en", "user": f"u{k % 200}", "text": f"hi {k}"} for k in range(390)]
    rows += [{"ts": t - 20 + k * 0.1, "channel": "streamer_en", "user": f"l{k}", "text": "KEKW"} for k in range(40)]
    day = datetime.fromtimestamp(t).date().isoformat()
    (d / f"streamer_en-{day}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    c = await db.create_clip(Candidate(source=Source.HARVEST, streamer=get_streamers()["streamer_en"], raw_path="x",
                                       t_trigger=datetime.fromtimestamp(t).astimezone(), signals={"views": 10}))
    r = await vr.compute(c)
    got = (await db.get_clip(c.id)).signals
    assert r.components["velocity"] > 0 and r.components["reaction"] > 0 and got["chat_from"] == "log"
    assert got["reaction_label"] == "laugh" and got["views"] == 10
