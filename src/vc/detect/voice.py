"""voice component (pure): what the streamer's own audio says about the moment.

Inputs come for free from the local judge: Whisper words (start, end, text) and loudness per second.
voice = clamp(max(loudness peak, speech-rate jump) + keyword bonus + silence-after-peak bonus).
Pitch, stutter and in-game sound effects are left out on purpose (mixed audio, Whisper smooths stutter,
would need a sound classifier)."""
from __future__ import annotations

import re
import statistics

from vc.config import ViralityCfg

PUNCT_RE = re.compile(r"[^\w\s']+")


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def speech_rate_jump(words: list[tuple[float, float, str]], win: float = 3.0) -> float:
    """Fastest 3 s window / median of the windows with speech (1.0 = steady)."""
    if len(words) < 6:
        return 1.0
    end = max(w[1] for w in words)
    bins = [sum(1 for w in words if k * win <= w[0] < (k + 1) * win) for k in range(int(end // win) + 1)]
    spoken = [b for b in bins if b]
    if len(spoken) < 3:
        return 1.0
    return max(spoken) / max(1.0, statistics.median(spoken))


def silence_after_peak(words: list[tuple[float, float, str]], loud: list[float], peak: int, need_s: float) -> bool:
    """No word for >= need_s within 3 s after the loudest second, and that gap is quieter than usual."""
    med = statistics.median(loud)
    times = sorted([w[0] for w in words] + [w[1] for w in words])
    edges = [peak + 1.0] + [x for x in times if x > peak + 1.0] + [float(len(loud))]
    for a, b in zip(edges, edges[1:]):
        if a > peak + 4.0:
            break
        if b - a >= need_s:
            quiet = loud[int(a):int(b)] or [med]
            if statistics.mean(quiet) < med:
                return True
    return False


def voice_features(words: list[tuple[float, float, str]], loud: list[float], lang: str, c: ViralityCfg) -> dict:
    text = " " + PUNCT_RE.sub(" ", " ".join(w[2] for w in words).lower()) + " "
    text = " ".join(text.split())
    keywords = c.voice_keywords.get(lang, []) + c.voice_keywords.get("en", []) * (lang != "en")
    hit = next((k for k in keywords if f" {k} " in f" {text} "), "")
    clip_said = any(f" {p} " in f" {text} " for p in c.clipcall_phrases if " " in p)  # "clip that", not "clip"
    loud_n, peak, silence = 0.0, 0, False
    if loud:
        peak = max(range(len(loud)), key=loud.__getitem__)
        loud_n = _clamp((loud[peak] - statistics.median(loud)) / c.audio_full_db)
        silence = silence_after_peak(words, loud, peak, c.voice_silence_s)
    jump = speech_rate_jump(words)
    pace_n = _clamp((jump - 1) / max(0.01, c.voice_pace_full - 1))
    v = _clamp(max(loud_n, pace_n) + (c.voice_keyword_bonus if hit else 0) + (c.voice_silence_bonus if silence else 0))
    return {"voice": round(v, 3), "voice_loud": round(loud_n, 2), "voice_pace": round(jump, 2), "voice_keyword": hit,
            "voice_silence": silence, "streamer_clipcall": clip_said}
