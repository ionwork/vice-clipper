"""Loudness monitor per recorder: RMS dBFS in 2-s steps vs the median of the last 120 s."""
from __future__ import annotations

import statistics
from collections import deque
from pathlib import Path

import numpy as np

from vc import log
from vc.config import get_config
from vc.proc import run
from vc.sources.recorder import seg_time

L = log.get("audio")
STEP_S = 2.0
HISTORY_S = 120


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -90.0
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return 20 * np.log10(max(rms, 1e-9))


class AudioMonitor:
    def __init__(self) -> None:
        self.hist: deque[tuple[float, float]] = deque()  # (epoch ts, dBFS)

    def push(self, ts: float, db: float) -> None:
        self.hist.append((ts, db))
        while self.hist and self.hist[0][0] < ts - HISTORY_S:
            self.hist.popleft()

    def spike(self, now: float, within_s: float = 12) -> float:
        """dB above the median for the loudest step in the last `within_s`, 0 if below the threshold."""
        if len(self.hist) < 10:
            return 0.0
        med = statistics.median(d for _, d in self.hist)
        recent = [d for t, d in self.hist if t >= now - within_s]
        if not recent:
            return 0.0
        excess = max(recent) - med
        return round(excess, 1) if excess >= get_config().trigger.audio_spike_db else 0.0

    async def feed_segment(self, path: Path) -> None:
        """Decode a finished segment to 16 kHz mono and push one value per 2 s."""
        t = seg_time(path)
        if not t:
            return
        r = await run("ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
                      "-f", "s16le", "pipe:1", timeout=20)
        if r.code != 0:
            return
        pcm = np.frombuffer(r.out, dtype=np.int16).astype(np.float32) / 32768
        step = int(16000 * STEP_S)
        base = t.astimezone().timestamp()
        for i in range(0, len(pcm) - step // 2, step):
            self.push(base + i / 16000, rms_dbfs(pcm[i:i + step]))
