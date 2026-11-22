"""End-to-end: producer | segmenter over an OS pipe (as with streamlink), then cut()."""
import asyncio
import os
from datetime import datetime, timedelta

from vc.proc import tool
from vc.render.ffmpeg import probe
from vc.sources.recorder import SEG_FMT, buffer_dir, cut, list_segments


async def test_pipe_segment_and_cut():
    d = buffer_dir("pipetest")
    d.mkdir(parents=True, exist_ok=True)
    for p in d.glob("*.ts"):
        p.unlink()
    r, w = os.pipe()
    prod = await asyncio.create_subprocess_exec(
        tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-re", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000", "-t", "14", "-c:v", "libx264", "-preset", "ultrafast",
        "-g", "30", "-c:a", "aac", "-f", "mpegts", "pipe:1", stdout=w, stdin=asyncio.subprocess.DEVNULL)
    seg = await asyncio.create_subprocess_exec(
        tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-fflags", "+genpts", "-i", "pipe:0", "-c", "copy",
        "-f", "segment", "-segment_time", "4", "-segment_format", "mpegts", "-reset_timestamps", "1", "-strftime", "1",
        str(d / f"{SEG_FMT}.ts"), stdin=r)
    os.close(r)
    os.close(w)
    await prod.wait()
    await seg.wait()
    segs = list_segments(d)
    assert len(segs) >= 3
    end = segs[-1][0] + timedelta(seconds=2)
    out, first = await cut("pipetest", end - timedelta(seconds=8), end, d.parent / "pipetest_cut.mp4")
    p = await probe(out)
    assert 6 <= p.duration <= 16 and p.has_audio
    assert first.tzinfo is not None and abs((first.replace(tzinfo=None) - segs[0][0]).total_seconds()) < 60
