import pytest

from vc.models import Layout, Rect
from vc.render.ffmpeg import probe
from vc.render.layouts import audio_graph, gameplay_gx, split_gx
from vc.render.render import RenderJob, render_clip


def test_gx_math():
    assert split_gx(0.5) == 454 and split_gx(0.0) == 0 and split_gx(1.0) == 908
    assert gameplay_gx(0.5) == 240 and gameplay_gx(0.0) == 0 and gameplay_gx(1.0) == 480


def test_audio_graph_without_sfx():
    g = audio_graph("0:a", -14, None, None, 0)
    assert g.endswith("[aout]") and "amix" not in g


@pytest.mark.parametrize("layout", list(Layout))
async def test_render_layouts(layout, synth_video, vc_root):
    out = vc_root / "data" / "out" / f"t_{layout.value}.mp4"
    await render_clip(RenderJob(src=synth_video, start_s=5, end_s=17, layout=layout,
                                cam=Rect(x=0, y=720, w=480, h=360), subject_x=0.6, peak_s=10, out_path=out))
    p = await probe(out)
    assert (p.width, p.height) == (1080, 1920) and abs(p.duration - 12) <= 0.3 and p.has_audio
