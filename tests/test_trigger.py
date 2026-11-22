import json
import random

from vc.detect.chat_stats import ChannelStats, Lexicon
from vc.detect.trigger import TriggerEngine, pre_score, replay
from vc.models import Streamer

T0 = 1_800_000_000.0


def steady(n_s, rate=2, start=T0):
    rng = random.Random(1)
    return [(start + s + k / rate, f"user{rng.randint(0, 400)}", f"just chatting {rng.randint(0, 9999)}")
            for s in range(n_s) for k in range(rate)]


def run(msgs, until):
    eng = TriggerEngine()
    eng.track(Streamer(id="ch", login="ch", lang="en"), started=T0)
    msgs = sorted(msgs)
    fired, i = [], 0
    for t in range(int(T0) + 1, int(until)):
        while i < len(msgs) and msgs[i][0] < t:
            eng.feed("ch", *msgs[i])
            i += 1
        if tr := eng.evaluate("ch", t):
            fired.append(tr)
    return fired


def burst(at, n, users, text):
    return [(at + k * 4 / n, users[k % len(users)], text(k)) for k in range(n)]


FANS = [f"fan{k}" for k in range(60)]


def test_lexicon():
    lx = Lexicon(["KEKW", "W", "NO WAY", "💀", "clip it"])
    assert lx.hit("kekw") and lx.hit("that is a W") and lx.hit("no way bro") and lx.hit("lol 💀")
    assert not lx.hit("week") and not lx.hit("noway")


def test_stats_formula():
    st = ChannelStats(Lexicon([]))
    for ts, u, x in steady(200):
        st.add(ts, u, x)
    snap = st.snapshot(T0 + 200)
    assert snap["baseline"] == 10 and snap["burst"] == 10 and abs(snap["z"]) < 1


def test_steady_no_trigger():
    assert run(steady(400), T0 + 400) == []


def test_single_spike_triggers():
    msgs = steady(400) + burst(T0 + 300, 50, FANS, lambda k: ["KEKW", "NO WAY", "what was that", f"lol {k}"][k % 4])
    fired = run(msgs, T0 + 400)
    assert len(fired) == 1 and T0 + 300 <= fired[0]["t"] <= T0 + 306
    assert fired[0]["pre_score"] >= 5 and fired[0]["lex"] > 0.3


def test_two_spammers_no_trigger():
    msgs = steady(400) + burst(T0 + 300, 50, ["spam1", "spam2"], lambda k: f"buy my stuff {k}")
    assert run(msgs, T0 + 400) == []


def test_clip_calls_trigger_immediately():
    msgs = steady(400) + [(T0 + 300, "a", "CLIP IT"), (T0 + 302, "b", "clip that"), (T0 + 304, "c", "someone clip")]
    fired = run(msgs, T0 + 400)
    assert len(fired) == 1 and fired[0]["clipcall"]


def test_sub_train_copypasta_no_trigger():
    msgs = steady(400) + burst(T0 + 300, 50, FANS, lambda k: "bleedPurple bleedPurple HYPE TRAIN")
    assert run(msgs, T0 + 400) == []


def test_warmup_and_cooldown():
    early = steady(400) + burst(T0 + 100, 50, FANS, lambda k: f"KEKW {k}")
    assert run(early, T0 + 200) == []  # baseline not ready
    twice = steady(500) + burst(T0 + 300, 50, FANS, lambda k: f"KEKW {k}") + burst(T0 + 320, 50, FANS, lambda k: f"W {k}")
    assert len(run(twice, T0 + 500)) == 1  # second spike inside the 45 s cooldown


def test_pre_score():
    assert pre_score(6, 0.5, True, False) == 9.9 and pre_score(20, 1, True, True) == 10 and pre_score(0, 0, False, False) == 2


def test_replay(tmp_path):
    msgs = steady(400) + burst(T0 + 300, 50, FANS, lambda k: f"KEKW {k}")
    p = tmp_path / "ch.jsonl"
    p.write_text("\n".join(json.dumps({"ts": t, "channel": "ch", "user": u, "text": x}) for t, u, x in msgs), encoding="utf-8")
    trs = replay(p)
    assert len(trs) == 1 and "kekw" in trs[0]["top"]


def test_audio_spike():
    import numpy as np

    from vc.detect.audio import AudioMonitor, rms_dbfs
    assert abs(rms_dbfs(np.full(100, 0.5, dtype=np.float32)) - (-6.02)) < 0.1
    a = AudioMonitor()
    for i in range(60):
        a.push(T0 + i * 2, -30.0)
    assert a.spike(T0 + 118) == 0
    a.push(T0 + 120, -18.0)  # 12 dB above median, threshold 8
    assert a.spike(T0 + 125) == 12.0 and a.spike(T0 + 140) == 0
