"""faster-whisper word timestamps -> one ASS file with captions, hook and credit."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from vc import db, log
from vc.config import get_config
from vc.models import ClipStatus, Layout
from vc.proc import tool

L = log.get("captions")

ASS_HEADER = r"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Noto Sans Black,84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,7,0,5,60,60,0,1
Style: Hook,Noto Sans Black,70,&H00101010,&H00101010,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,3,14,0,8,70,70,110,1
Style: Credit,Noto Sans,34,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,1,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

CAPTION_Y = {Layout.SPLIT: 790, Layout.GAMEPLAY: 1500, Layout.FULLCAM: 1180, Layout.IRL: 1340}  # irl: video 656-1264
PROFANITY = {
    "en": {"fuck", "fucking", "shit", "bitch", "motherfucker", "cunt", "nigga", "nigger", "faggot"},
    "de": {"fick", "ficken", "scheiße", "scheisse", "hurensohn", "fotze", "wichser", "missgeburt"},
}
GAP_S = 0.35


@dataclass
class Word:
    text: str
    start: float
    end: float


# ---------------------------------------------------------------- pure helpers (tested)
def ass_color(hex_rgb: str) -> str:
    """'FFE500' -> '&H0000E5FF' (ASS is &HAABBGGRR)."""
    h = hex_rgb.lstrip("#")
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def ass_time(t: float) -> str:
    cs = max(0, round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\r", " ").replace("\n", " ")


def clean_word(w: str) -> str:
    w = w.strip().upper()
    while w and w[-1] in ".,;:-…\"'" and w[-1] not in "?!":
        w = w[:-1]
    return w


def censor(word: str, lang: str) -> str:
    bad = PROFANITY.get(lang, set()) | PROFANITY["en"]
    core = word.strip("?!").lower()
    return word[0] + "*" * (len(word) - 1) if core in bad and len(word) > 1 else word


def group_words(words: list[Word], max_words: int = 3, max_chars: int = 16) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur:
            text_len = len(" ".join(x.text for x in cur + [w]))
            prev = cur[-1]
            punct = prev.text.rstrip()[-1:] in ".,!?;:"
            if len(cur) >= max_words or text_len > max_chars or punct or w.start - prev.end > GAP_S:
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def caption_events(words: list[Word], y: int, highlight: str, max_words: int, max_chars: int,
                   lang: str = "en", do_censor: bool = False) -> list[str]:
    hl = ass_color(highlight)
    chunks = group_words(words, max_words, max_chars)
    lines: list[str] = []
    for ci, chunk in enumerate(chunks):
        texts = [clean_word(w.text) for w in chunk]
        if do_censor:
            texts = [censor(t, lang) for t in texts]
        texts = [escape(t) for t in texts]
        next_start = chunks[ci + 1][0].start if ci + 1 < len(chunks) else float("inf")
        chunk_end = min(chunk[-1].end + 0.15, next_start)
        for i, w in enumerate(chunk):
            start = w.start
            end = chunk[i + 1].start if i + 1 < len(chunk) else chunk_end
            if end <= start:
                end = start + 0.05
            body = " ".join(f"{{\\c{hl}&}}{t}{{\\c&HFFFFFF&}}" if j == i else t for j, t in enumerate(texts))
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{{\\pos(540,{y})}}{body}")
    return lines


def build_ass_text(words: list[Word], hook: str, credit: str, layout: Layout, duration: float, lang: str = "en") -> str:
    c = get_config().captions
    out = [ASS_HEADER.rstrip("\n")]
    if hook.strip():
        out.append(f"Dialogue: 1,{ass_time(0)},{ass_time(min(2.2, duration))},Hook,,0,0,0,,{escape(hook.strip().upper())}")
    if len([w for w in words if w.text.strip()]) >= 2:
        # split: captions sit on the seam between cam and gameplay, wherever split_cam_height puts it
        y = get_config().render.split_cam_height + 22 if layout == Layout.SPLIT else CAPTION_Y[layout]
        out += caption_events(words, y, c.highlight_color, c.max_words, c.max_chars, lang, c.censor)
    if credit.strip():
        out.append(f"Dialogue: 2,{ass_time(0)},{ass_time(duration)},Credit,,0,0,0,,{escape(credit.strip())}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- whisper
_model = None
_model_lock = threading.Lock()


def _add_cuda_dlls() -> None:
    """pip-installed cuBLAS/cuDNN on Windows: make their DLLs findable for ctranslate2."""
    if sys.platform != "win32":
        return
    base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for sub in ("cublas", "cudnn", "cuda_nvrtc"):
        d = base / sub / "bin"
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"


def get_model():
    global _model
    with _model_lock:
        if _model is not None:
            return _model
        from faster_whisper import WhisperModel
        c = get_config().captions
        if c.device in ("auto", "cuda"):
            try:
                _add_cuda_dlls()
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    _model = WhisperModel(c.whisper_model, device="cuda", compute_type="float16")
                    L.info("whisper %s on cuda", c.whisper_model)
                    return _model
            except Exception as e:  # noqa: BLE001
                L.warning("whisper cuda nicht verfuegbar (%s), CPU mit Modell small", e)
        _model = WhisperModel("small" if c.device == "auto" else c.whisper_model, device="cpu", compute_type="int8")
        L.info("whisper on cpu")
        return _model


def transcribe(path: Path, start_s: float, end_s: float, lang: str) -> list[Word]:
    """Words with times relative to start_s. Audio goes in as a float32 array (no PyAV decode)."""
    import numpy as np
    pcm = subprocess.run([tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", f"{start_s:.3f}",
                          "-t", f"{end_s - start_s:.3f}", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "pipe:1"], check=True, timeout=120, capture_output=True).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    if audio.size < 16000 * 0.5:
        return []
    segments, _ = get_model().transcribe(
        audio, language=lang, word_timestamps=True, vad_filter=True, condition_on_previous_text=False,
        initial_prompt=get_config().captions.initial_prompt or None,
    )
    return [Word(w.word.strip(), w.start, w.end) for seg in segments for w in (seg.words or []) if w.word.strip()]


def _transcribe_safe(path: Path, start_s: float, end_s: float, lang: str) -> list[Word]:
    try:
        return transcribe(path, start_s, end_s, lang)
    except Exception as e:  # noqa: BLE001  rather no captions than a dead pipeline
        L.warning("Transkription fehlgeschlagen (%s), nur Hook und Credit", e)
        return []


async def build_ass(clip_id: str, layout: Layout | None = None, window: tuple[float, float] | None = None) -> Path:
    clip = await db.get_clip(clip_id)
    if not clip or not clip.judge:
        raise ValueError(f"{clip_id}: nicht bewertet")
    j = clip.judge_result
    s = clip.streamer_obj
    start, end = window or (j.start_s, j.end_s)
    lay = layout or Layout(clip.layout or j.layout)
    from vc.render.layouts import effective_layout
    lay = effective_layout(lay, s.cam)
    from vc.judge.local import load_words
    saved = load_words(clip_id)
    if saved and saved.get("words"):  # the local judge already transcribed the whole candidate: no second pass
        words = [Word(x[2], x[0] - start, x[1] - start) for x in saved["words"] if x[0] >= start - 0.1 and x[1] <= end + 0.3]
    else:
        words = await asyncio.to_thread(_transcribe_safe, Path(clip.raw_path), start, end, s.lang)
    out = get_config().data / "candidates" / f"{clip_id}.ass"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_ass_text(words, j.hook, s.credit, lay, end - start, s.lang), encoding="utf-8")
    # only the first pass advances the status; a re-render of a delivered clip must not re-enter the render queue
    status = ClipStatus.CAPTIONED if clip.status in (ClipStatus.NEW, ClipStatus.JUDGED) else clip.status
    await db.update_clip(clip_id, ass_path=str(out), status=status)
    return out
