"""Filtergraphs for the four 9:16 layouts and the pure layout rule. Cam rectangles refer to the normalised 1920x1080 frame."""
from __future__ import annotations

from vc.models import Layout, Rect

NORM = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1"


def _even(v: float) -> int:
    return int(round(v / 2) * 2)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def clamp_rect(r: Rect) -> Rect:
    w, h = _clamp(_even(r.w), 16, 1920), _clamp(_even(r.h), 16, 1080)
    return Rect(x=_clamp(_even(r.x), 0, 1920 - w), y=_clamp(_even(r.y), 0, 1080 - h), w=w, h=h)


def split_gx(subject_x: float, crop_w: int = 1012) -> int:
    """GX = clamp(round(subject_x*1920 - crop_w/2), 0, 1920-crop_w), even."""
    return _clamp(_even(subject_x * 1920 - crop_w / 2), 0, 1920 - crop_w)


def gameplay_gx(subject_x: float) -> int:
    return _clamp(_even(subject_x * 1920 - 720), 0, 480)


def effective_layout(layout: Layout, cam: Rect | None) -> Layout:
    return Layout.GAMEPLAY if layout in (Layout.SPLIT, Layout.FULLCAM) and cam is None else layout


def content_kind(category: str, judge_content: str, irl_categories: list[str]) -> str:
    """Twitch category first (cheap, reliable); without one the judge decides. -> 'irl' | 'gameplay'."""
    if category.strip():
        return "irl" if category.strip().lower() in {c.lower() for c in irl_categories} else "gameplay"
    return "irl" if judge_content == "irl" else "gameplay"


def choose_layout(kind: str, has_cam: bool, force: str = "") -> Layout:
    """irl -> IRL; gameplay -> SPLIT (50/50) with a cam, else GAMEPLAY. force_layout overrides."""
    if force:
        return Layout(force)
    if kind == "irl":
        return Layout.IRL
    return Layout.SPLIT if has_cam else Layout.GAMEPLAY


def video_graph(layout: Layout, cam: Rect | None, subject_x: float, cam_h: int = 768, W: int = 1080, H: int = 1920) -> str:
    """Returns a graph ending in [stack]."""
    layout = effective_layout(layout, cam)
    if layout == Layout.SPLIT:
        c = clamp_rect(cam)  # type: ignore[arg-type]
        game_h = H - cam_h
        crop_w = min(1920, int(1080 * W / game_h) // 2 * 2)  # 1012 for 1152, matches the handbook
        return (
            f"[0:v]{NORM},split=2[v1][v2];"
            f"[v1]crop={c.w}:{c.h}:{c.x}:{c.y},scale={W}:{cam_h}:force_original_aspect_ratio=increase,crop={W}:{cam_h}[cam];"
            f"[v2]crop={crop_w}:1080:{split_gx(subject_x, crop_w)}:0,scale={W}:{game_h},setsar=1[game];"
            f"[cam][game]vstack=inputs=2[stack]"
        )
    if layout == Layout.GAMEPLAY:
        return (
            f"[0:v]{NORM},split=2[bg][fg];"
            f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},gblur=sigma=30,eq=brightness=-0.15[b];"
            f"[fg]crop=1440:1080:{gameplay_gx(subject_x)}:0,scale=1080:810[f];"
            f"[b][f]overlay=0:(H-h)/2[stack]"
        )
    if layout == Layout.IRL:  # whole 16:9 frame centred, blurred + darkened stream above and below
        fg_h = _even(W * 9 / 16)
        return (
            f"[0:v]{NORM},split=2[bg][fg];"
            f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},gblur=sigma=30,eq=brightness=-0.15[b];"
            f"[fg]scale={W}:{fg_h},setsar=1[f];"
            f"[b][f]overlay=0:(H-h)/2[stack]"
        )
    c = clamp_rect(cam)  # type: ignore[arg-type]
    return (
        f"[0:v]{NORM},split=2[v1][v2];"
        f"[v1]crop={c.w}:{c.h}:{c.x}:{c.y},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[bg];"
        f"[v2]scale=1000:562,pad=1008:570:4:4:color=white[g];"
        f"[bg][g]overlay=(W-w)/2:1236[stack]"
    )


def audio_graph(main: str, lufs: float, whoosh_idx: int | None, impact_idx: int | None, peak_ms: int) -> str:
    """Main chain plus optional SFX, ends in [aout]."""
    parts = [f"[{main}]highpass=f=70,acompressor=threshold=-20dB:ratio=3:attack=5:release=150,loudnorm=I={lufs}:TP=-1.5:LRA=11[a0]"]
    mix = ["[a0]"]
    if whoosh_idx is not None:
        parts.append(f"[{whoosh_idx}:a]volume=0.6[s1]")
        mix.append("[s1]")
    if impact_idx is not None:
        parts.append(f"[{impact_idx}:a]adelay={max(0, peak_ms)}:all=1,volume=0.7[s2]")
        mix.append("[s2]")
    if len(mix) == 1:
        parts[0] = parts[0].replace("[a0]", "[aout]")
    else:
        parts.append(f"{''.join(mix)}amix=inputs={len(mix)}:duration=first:normalize=0[aout]")
    return ";".join(parts)
