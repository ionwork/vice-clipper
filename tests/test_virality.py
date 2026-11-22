import itertools
from datetime import timedelta

from vc.config import get_config
from vc.detect import virality as vr
from vc.detect.chat_stats import ChannelStats, Lexicon, Reactions

C = get_config().virality
ALL = list(vr.COMPONENTS)


def stats() -> ChannelStats:
    return ChannelStats(Lexicon([]), 5, 180, Reactions(C.reaction_emojis, C.reaction_emotes, C.reaction_patterns),
                        C.reaction_base_min)


# ---------------------------------------------------------------- score: range, single, redistribution
def test_range_and_single_component():
    for k in ALL:
        for s in (0.0, 0.37, 1.0):
            r = vr.score({k: s}, C)
            assert 0 <= r.v <= 100 and abs(r.v - round(100 * s, 1)) < 1e-9
    r = vr.score(dict.fromkeys(ALL, 1.0), C)
    assert r.v == 100 and r.bonus == C.agree_factor  # bonus capped at 100
    assert vr.score({}, C).v == 0 and not vr.score({}, C).known


def test_monotonic_over_grid():
    grid = (0.0, 0.3, 0.6, 1.0)
    for vals in itertools.product(grid, repeat=len(ALL)):
        comps = dict(zip(ALL, vals))
        v0 = vr.score(comps, C).v
        for k in ALL:
            for up in grid:
                if up > comps[k]:
                    assert vr.score({**comps, k: up}, C).v >= v0, (comps, k, up)


def test_missing_component_weights_are_redistributed():
    comps = {"velocity": 0.8, "reaction": 0.5, "clip_calls": 0.2, "library": None, "judge": 0.4, "voice": 0.1}
    w = {k: C.weights[k] for k, s in comps.items() if s is not None}
    manual = 100 * sum(w[k] * comps[k] for k in w) / sum(w.values())
    r = vr.score(comps, C)
    assert abs(r.v - round(manual, 1)) < 1e-9 and "library" not in r.weights_used


# ---------------------------------------------------------------- bonus and penalties at the thresholds
def test_agreement_bonus_at_threshold():
    at = {"velocity": 0.6, "reaction": 0.6, "clip_calls": 0.6}
    below = {**at, "clip_calls": 0.599}
    assert vr.score(at, C).bonus == C.agree_factor and abs(vr.score(at, C).v - 66.0) < 1e-9
    assert vr.score(below, C).bonus == 1.0


def test_flag_penalties_and_exclusions():
    comps = {"velocity": 0.5, "judge": 0.5}
    assert vr.score(comps, C, {"needs_context": True}).v == 50 - C.needs_context_penalty
    assert vr.score({"judge": 0.1}, C, {"needs_context": True}).v == 0  # never below 0
    for f in ("cutscene", "story_spoiler"):
        r = vr.score(comps, C, {f: True}, allow_gta5=True)
        assert r.excluded and r.v == 0
    assert vr.score(comps, C, {"not_gta6": True}, allow_gta5=False).excluded
    assert not vr.score(comps, C, {"not_gta6": True}, allow_gta5=True).excluded


# ---------------------------------------------------------------- normalisers
def test_normalisers():
    assert vr.n_chat_spike(3, C) == 0 and vr.n_chat_spike(10, C) == 1 and vr.n_chat_spike(6.5, C) == 0.5
    assert vr.n_reactions(8, C) == 1 and vr.n_reactions(0.5, C) == 0 and abs(vr.n_reactions(2, C) - 1 / 3) < 1e-9
    assert vr.n_clip_calls(0, C) == 0 and abs(vr.n_clip_calls(4, C) - (1 - 2.718281828 ** -1)) < 1e-6
    assert vr.n_audio(7.5, C) == 0.5 and vr.n_judge(1) == 0 and vr.n_judge(10) == 1
    assert vr.n_library(0, 0, 1, False, C) == 0 and vr.n_library(7, 0, 1, False, C) == 1  # log2(8)/3
    assert vr.n_library(1, 0, 1, True, C) == 1 / 3 + C.velocity_bonus


# ---------------------------------------------------------------- chat signals
def _feed_base(st: ChannelStats, t0: float, rate: int, react_every: int, users: int):
    k = 0
    for sec in range(200):
        for j in range(rate):
            k += 1
            st.add(t0 + sec + j / rate, f"u{k % users}", "LUL" if k % react_every == 0 else f"hello {k}")


def test_fairness_big_and_small_chat():
    """10x different chat size, same relative spike: same lift, so with the same z the same V (+-1)."""
    t0, lifts = 1_800_000_000.0, []
    for scale in (1, 10):
        st = stats()
        _feed_base(st, t0, 2 * scale, 20, 300 * scale)            # 5 % reactions in the baseline
        burst_t = t0 + 200
        for j in range(100 * scale):                               # burst: half the messages from distinct laughers
            st.add(burst_t + j * 4 / (100 * scale), f"b{j}", "KEKW" if j % 2 == 0 else f"wow {j}")
        lifts.append(st.snapshot(burst_t + 5)["react_lift"])
    assert abs(lifts[0] - lifts[1]) < 0.01
    vs = [vr.score({"velocity": vr.n_chat_spike(6, C), "reaction": vr.n_reactions(lf, C)}, C).v for lf in lifts]
    assert abs(vs[0] - vs[1]) <= 1


def test_emoji_spam_is_weaker_than_many_laughers():
    """50x '💀💀💀💀' from 2 users < 30x '💀' from 30 users: the burst share counts UNIQUE reacting users."""
    t0 = 1_800_000_000.0
    spam, crowd = stats(), stats()
    for st in (spam, crowd):
        _feed_base(st, t0, 2, 10**9, 300)                         # no reactions in the baseline -> floor 0.02
    t = t0 + 200
    for j in range(50):
        spam.add(t + j * 0.08, f"s{j % 2}", "💀💀💀💀")
    for j in range(30):
        crowd.add(t + j * 0.13, f"c{j}", "💀")
    a, b = spam.snapshot(t + 5), crowd.snapshot(t + 5)
    assert vr.n_reactions(a["react_lift"], C) < vr.n_reactions(b["react_lift"], C)
    assert a["react_share"] == 2 / 50 and b["react_share"] == 1.0


def test_reaction_signal_once_per_message():
    r = Reactions(C.reaction_emojis, C.reaction_emotes, C.reaction_patterns)
    assert all(r.hit(x) for x in ("😂", "☠️", "lul", "OMEGALUL", "hahaha", "HAHAH", "xddd"))
    assert not any(r.hit(x) for x in ("ha", "hallo", "lulu", "xbox"))
    st = stats()
    st.add(1.0, "a", "💀💀💀💀 KEKW hahaha")
    assert st.counts[-1] == [1, 1, 1]


def test_clipcall_users_window_and_links():
    msgs = [(100 - 16, "early", "clip"), (100 - 15, "a", "CLIP IT"), (101, "a", "clip that"), (110, "b", "clippt das"),
            (120, "c", "check clips.twitch.tv/Abc"), (125, "d", "eclipse"), (130, "e", "clip das"), (131, "late", "clip")]
    assert vr.clipcall_users(msgs, 100, C) == 3  # a, b, e; link, 'eclipse' and out-of-window do not count
    assert vr.chat_links(msgs + [(200, "f", "https://www.twitch.tv/x/clip/Def"), (201, "g", "https://www.twitch.tv/x/clip/Def")], 100, C) == 2


# ---------------------------------------------------------------- clip library from the harvester tables
async def _harvest(db, rows):
    async with db.connect() as conn:
        for cid, name, ct, views in rows:
            await conn.execute("INSERT INTO harvest_clips (id, broadcaster, broadcaster_id, title, url, created_at, duration, language) "
                               "VALUES (?,?,?,?,?,?,?,?)", (cid, name, "1", "t", "u", ct.isoformat(), 30, "de"))
            await conn.execute("INSERT INTO harvest_views VALUES (?,?,?)", (cid, (ct + timedelta(minutes=10)).isoformat(), views))
        await conn.commit()


async def test_library_window(db):
    t = db.now() - timedelta(hours=1)
    await _harvest(db, [("in", "Streamer_De", t + timedelta(seconds=90), 100),
                        ("out", "Streamer_De", t + timedelta(seconds=400), 100),
                        ("other", "Streamer_De2", t + timedelta(seconds=30), 100)])
    lib = await vr.library_counts("streamer_de", t, C)
    assert lib["n_tw"] == 1 and lib["expected"] == C.library_expected_min
    assert await vr.library_counts("streamer_en2", t, C) is None  # no harvester data -> component missing


async def test_library_velocity_bonus(db):
    t = db.now() - timedelta(hours=2)
    old = [(f"o{i}", "chan", t - timedelta(hours=1, minutes=i), 10) for i in range(9)]   # 1 view/min
    await _harvest(db, old + [("hot", "chan", t + timedelta(seconds=60), 500)])             # 50 views/min
    lib = await vr.library_counts("chan", t, C)
    assert lib["n_tw"] == 1 and lib["fast"]


async def test_compute_stores_and_keeps_signals(db):
    from vc.config import get_streamers
    from vc.models import Candidate, Source
    s = get_streamers()["streamer_en"]
    c = await db.create_clip(Candidate(source=Source.LIVE, streamer=s, raw_path="x", t_trigger=db.now(),
                                       signals={"z": 8.0, "react_lift": 4.0, "audio_db": 9.0, "clipcalls": 2, "top": ["KEKW"]}))
    r = await vr.compute(c)  # no chat buffer, no harvest rows: clip_calls and library missing
    got = await db.get_clip(c.id)
    assert got.signals["clipcalls"] == 2 and got.signals["top"] == ["KEKW"]
    assert got.signals["virality"]["v"] == r.v and r.components["library"] is None and r.components["judge"] is None
    assert vr.known_v(got) == r.v and set(r.weights_used) == {"velocity", "reaction", "voice"}
    await db.update_clip(c.id, judge={"flags": {}}, hype=9)
    r2 = await vr.compute(await db.get_clip(c.id))
    assert r2.components["judge"] == 8 / 9 and "judge" in r2.weights_used


def test_replay_fixture_gives_provisional_v():
    from pathlib import Path

    from vc.detect.trigger import replay
    trs = replay(Path(__file__).parent / "fixtures" / "chat_spike.jsonl")
    assert len(trs) == 1
    r = trs[0]["virality"]
    assert r.v >= 70 and r.components["clip_calls"] > 0.7 and r.components["library"] > 0  # 6 callers, 1 clip link
    assert r.components["voice"] is None and r.components["judge"] is None
