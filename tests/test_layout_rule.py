"""Phase 3: layout by content (gameplay 50/50, real life centred with blurred stream)."""
from vc.judge import gemini
from vc.judge.prompts import judge_prompt
from vc.models import Flags, JudgeResult, Layout
from vc.render.layouts import choose_layout, content_kind, video_graph

IRL = ["Just Chatting", "IRL", "Travel & Outdoors"]


def test_irl_graph():
    g = video_graph(Layout.IRL, None, 0.5)
    assert "gblur" in g and "scale=1080:608" in g and g.endswith("[stack]")


def test_layout_rule():
    assert choose_layout(content_kind("Just Chatting", "gameplay", IRL), True) == Layout.IRL
    assert choose_layout(content_kind("Grand Theft Auto VI", "irl", IRL), True) == Layout.SPLIT  # category wins
    assert choose_layout(content_kind("Grand Theft Auto VI", "gameplay", IRL), False) == Layout.GAMEPLAY
    assert choose_layout(content_kind("", "irl", IRL), True) == Layout.IRL
    assert choose_layout("irl", True, force="split") == Layout.SPLIT


def test_layout_next_cycles_through_irl():
    seen, cur = set(), Layout.SPLIT
    for _ in range(4):
        seen.add(cur)
        cur = cur.next()
    assert Layout.IRL in seen and cur == Layout.SPLIT


def test_old_judge_json_without_content_loads():
    j = JudgeResult.model_validate({"keep": True, "hype": 8, "category": "crash", "summary": "x", "start_s": 0,
                                    "end_s": 12, "peak_s": 5, "layout": "split"})
    assert j.content == "gameplay"


def test_irl_not_blocked_when_allowed():
    r = JudgeResult(keep=True, hype=8, category="reaction", summary="x", start_s=0, end_s=12, peak_s=5,
                    flags=Flags(not_gta6=True), content="irl")
    assert gemini.render_rule(r, 7, allow_irl=True)[0] is True
    assert gemini.render_rule(r, 7, allow_irl=False)[2] == "flag: not_gta6"


def test_prompt_irl_switch():
    assert "content irl" in judge_prompt(False, True) and "just chatting, real life" not in judge_prompt(False, True)
    assert "just chatting, real life" in judge_prompt(False, False)
    assert "GTA 5 or GTA 6" in judge_prompt(True, True)
