"""Judge without AI: used when Gemini fails or the daily budget is spent."""
from __future__ import annotations

from vc.models import Clip, Flags, Hashtags, JudgeResult, Layout

HOOKS = {"en": "Wait for it", "de": "Wart's ab"}
CAPTIONS = {"en": "This escalated fast.", "de": "Das ist komplett eskaliert."}


def fallback_judge(clip: Clip, duration: float) -> JudgeResult:
    s = clip.streamer_obj
    t = clip.trigger_offset_s if clip.trigger_offset_s is not None else min(duration, 25.0) - 5
    start, end = max(0.0, t - 20), min(duration, t + 5)
    tag = "".join(ch for ch in s.login.lower() if ch.isalnum())
    tags = ["gta6", "gtavi", tag] if tag else ["gta6", "gtavi", "gaming"]
    return JudgeResult(
        keep=clip.pre_score >= 7, hype=max(1, min(10, round(clip.pre_score))), category="other",
        summary="unknown (fallback without AI)", start_s=start, end_s=end, peak_s=max(start, end - 3),
        subject_x=0.5, layout=Layout.SPLIT if s.cam else Layout.GAMEPLAY,
        hook=HOOKS.get(s.lang, HOOKS["en"]), title=f"GTA 6 {s.display} moment"[:60],
        caption=CAPTIONS.get(s.lang, CAPTIONS["en"]),
        hashtags=Hashtags(youtube=tags, tiktok=tags + ["gaming"], instagram=tags + ["gaming"]),
        language=s.lang, flags=Flags(),
    )
