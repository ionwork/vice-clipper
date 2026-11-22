"""render_clip(job): landscape source + window -> finished 1080x1920 clip."""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel

from vc import log
from vc.config import get_config
from vc.models import Layout, Rect
from vc.proc import run
from vc.render.ffmpeg import detect_encoder, encoder_args, probe
from vc.render.layouts import audio_graph, effective_layout, video_graph

L = log.get("render")


class RenderJob(BaseModel):
    src: Path
    start_s: float
    end_s: float
    layout: Layout = Layout.SPLIT
    cam: Rect | None = None
    subject_x: float = 0.5
    ass_path: Path | None = None
    peak_s: float | None = None
    out_path: Path
    sfx: bool = True


def _build_cmd(job: RenderJob, enc: str, fps: int, has_audio: bool, jobdir: Path, low_rate: bool) -> list[str]:
    cfg = get_config()
    dur = job.end_s - job.start_s
    args: list[str] = ["-y", "-hide_banner", "-loglevel", "error",
                       "-ss", f"{job.start_s:.3f}", "-t", f"{dur:.3f}", "-i", str(job.src.resolve())]
    idx = 1
    main = "0:a"
    if not has_audio:
        args += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        main, idx = "1:a", 2
    sfx_dir = cfg.path("sfx")
    whoosh_idx = impact_idx = None
    if job.sfx and (w := sfx_dir / "whoosh.wav").exists():
        args += ["-i", str(w.resolve())]
        whoosh_idx, idx = idx, idx + 1
    if job.sfx and job.peak_s is not None and (i := sfx_dir / "impact.wav").exists():
        args += ["-i", str(i.resolve())]
        impact_idx, idx = idx, idx + 1

    vg = video_graph(job.layout, job.cam, job.subject_x, cfg.render.split_cam_height, cfg.render.width, cfg.render.height)
    vg += ";[stack]subtitles=captions.ass:fontsdir=fonts[vout]" if job.ass_path else ";[stack]null[vout]"
    peak_ms = int(((job.peak_s or job.start_s) - job.start_s) * 1000)
    ag = audio_graph(main, cfg.render.loudness_lufs, whoosh_idx, impact_idx, peak_ms)
    args += ["-filter_complex", f"{vg};{ag}", "-map", "[vout]", "-map", "[aout]", "-r", str(fps),
             *encoder_args(enc, low_rate), "-t", f"{dur:.3f}", str(job.out_path.resolve())]
    return args


async def _encode(job: RenderJob, enc: str, fps: int, has_audio: bool, jobdir: Path, low_rate: bool = False) -> None:
    args = _build_cmd(job, enc, fps, has_audio, jobdir, low_rate)
    r = await run("ffmpeg", *args, cwd=jobdir, timeout=600)
    if r.code != 0:
        raise RuntimeError(f"ffmpeg ({enc}) exit {r.code}:\n{r.tail()}")


async def render_clip(job: RenderJob) -> Path:
    cfg = get_config()
    if job.end_s <= job.start_s:
        raise ValueError("end_s muss groesser als start_s sein")
    job = job.model_copy(update={"layout": effective_layout(job.layout, job.cam)})
    src = await probe(job.src)
    fps = 60 if src.fps >= 50 else 30
    enc = await detect_encoder()
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = cfg.data / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    jobdir = Path(tempfile.mkdtemp(prefix="render-", dir=tmp_root))
    try:
        # Windows: no drive letters inside the filter string -> relative paths from cwd=jobdir
        if job.ass_path:
            shutil.copy(job.ass_path, jobdir / "captions.ass")
            fonts = jobdir / "fonts"
            fonts.mkdir()
            for f in cfg.path("fonts").glob("*.[ot]tf"):
                shutil.copy(f, fonts / f.name)
        try:
            await _encode(job, enc, fps, src.has_audio, jobdir)
        except RuntimeError as e:
            if enc == "libx264":
                L.error("render failed: %s", e)
                raise
            L.warning("GPU-Encode fehlgeschlagen, Rueckfall libx264: %s", str(e).splitlines()[-1:])
            enc = "libx264"
            await _encode(job, enc, fps, src.has_audio, jobdir)

        size_mb = job.out_path.stat().st_size / 1e6
        if size_mb > cfg.render.max_file_mb:
            L.info("%.1f MB > %s MB, re-encode mit 8M maxrate", size_mb, cfg.render.max_file_mb)
            await _encode(job, enc, fps, src.has_audio, jobdir, low_rate=True)

        out = await probe(job.out_path)
        want = job.end_s - job.start_s
        problems = []
        if (out.width, out.height) != (cfg.render.width, cfg.render.height):
            problems.append(f"Aufloesung {out.width}x{out.height}")
        if abs(out.duration - want) > 0.3:
            problems.append(f"Dauer {out.duration:.2f}s statt {want:.2f}s")
        if not out.has_audio:
            problems.append("kein Ton")
        if problems:
            raise RuntimeError("Render-Pruefung fehlgeschlagen: " + ", ".join(problems))
        L.info("rendered %s (%s, %.1f MB, %s)", job.out_path.name, job.layout.value, job.out_path.stat().st_size / 1e6, enc)
        return job.out_path
    finally:
        shutil.rmtree(jobdir, ignore_errors=True)


async def render_from_db(clip_id: str, layout: Layout | None = None, shorter: bool = False) -> Path:
    """JUDGED -> CAPTIONED -> RENDERED for one clip. Also used for 'Anderes Layout' / 'Kuerzer'."""
    from vc import db
    from vc.models import ClipStatus
    from vc.render.captions import build_ass
    clip = await db.get_clip(clip_id)
    if not clip or not clip.judge:
        raise ValueError(f"{clip_id}: nicht bewertet")
    j = clip.judge_result
    start, end = j.start_s, j.end_s
    if shorter:  # 12 s around the peak, the payoff stays in
        dur = (await probe(Path(clip.raw_path))).duration
        end = min(dur, j.peak_s + 4)
        start = max(0.0, end - 12)
        end = min(dur, start + 12)
    if get_config().render.auto_cam and clip.streamer_obj.cam is None and Path(clip.raw_path).exists():
        from vc.sources.harvester import suggest_cam  # no frame drawn yet: look for the facecam in this clip
        if cam := await asyncio.to_thread(suggest_cam, Path(clip.raw_path), start + 1):
            await db.update_clip(clip_id, streamer={**clip.streamer, "cam": cam.model_dump()})
            clip = await db.get_clip(clip_id)
    lay = effective_layout(layout or Layout(clip.layout or j.layout), clip.streamer_obj.cam)
    try:
        ass = await build_ass(clip_id, lay, (start, end))
        out = get_config().data / "out" / f"{clip_id}.mp4"
        await render_clip(RenderJob(src=Path(clip.raw_path), start_s=start, end_s=end, layout=lay, cam=clip.streamer_obj.cam,
                                    subject_x=j.subject_x, ass_path=ass, peak_s=j.peak_s, out_path=out))
    except Exception as e:
        if clip.status not in (ClipStatus.DELIVERED, ClipStatus.POSTED):  # a failed re-render keeps the delivered clip
            await db.update_clip(clip_id, status=ClipStatus.FAILED, reason=f"render: {str(e)[:300]}")
        await db.add_event("error", clip_id=clip_id, stage="render", error=str(e)[:300])
        raise
    await db.update_clip(clip_id, status=ClipStatus.RENDERED, out_path=str(out), layout=lay.value)
    await db.add_event("rendered", clip_id=clip_id, streamer_id=clip.streamer_id, layout=lay.value)
    return out
