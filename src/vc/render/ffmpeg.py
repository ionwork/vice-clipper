"""ffmpeg/ffprobe helpers: encoder detection, probing, encoder arguments."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vc import log
from vc.config import get_config
from vc.proc import run, run_ok

L = log.get("ffmpeg")

ENCODERS = ("h264_nvenc", "h264_videotoolbox", "libx264")
_detected: str | None = None


async def _test_encode(enc: str) -> bool:
    r = await run("ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=1",
                  "-c:v", enc, "-pix_fmt", "yuv420p", "-f", "null", "-", timeout=60)
    return r.code == 0


async def detect_encoder(force: bool = False) -> str:
    """Best working H.264 encoder (1-second test encode), cached per process."""
    global _detected
    want = get_config().render.encoder
    if want != "auto":
        return want
    if _detected and not force:
        return _detected
    for enc in ENCODERS:
        if await _test_encode(enc):
            _detected = enc
            L.info("encoder: %s", enc)
            return enc
    raise RuntimeError("kein H.264-Encoder funktioniert")


def encoder_args(enc: str, low_rate: bool = False) -> list[str]:
    if enc == "h264_nvenc":
        a = ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "21", "-b:v", "0"]
        a += ["-maxrate", "8M", "-bufsize", "16M"] if low_rate else ["-maxrate", "14M", "-bufsize", "28M"]
    elif enc == "h264_videotoolbox":
        a = ["-c:v", "h264_videotoolbox", "-b:v", "6M" if low_rate else "10M"]
    else:
        a = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19"]
        if low_rate:
            a += ["-maxrate", "8M", "-bufsize", "16M"]
    return a + ["-profile:v", "high", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart"]


@dataclass
class Probe:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool


async def probe(path: Path) -> Probe:
    r = await run_ok("ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path)
    d = json.loads(r.out)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    if not v:
        raise RuntimeError(f"kein Videostream in {path}")
    num, _, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    dur = float(d.get("format", {}).get("duration") or v.get("duration") or 0)
    return Probe(int(v["width"]), int(v["height"]), fps, dur, any(s["codec_type"] == "audio" for s in d["streams"]))
