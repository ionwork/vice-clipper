"""Local judge via Ollama (no API costs). A vision LLM cannot take video or audio, so the clip is
translated into what it can read: N timestamped frames, a Whisper transcript with times, loudness per
second and the chat lines. Output is constrained to the same JudgeResult JSON schema as Gemini."""
from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import tempfile
from pathlib import Path

import httpx

from vc import db, log
from vc.config import get_config
from vc.models import JudgeResult
from vc.proc import tool

L = log.get("ollama")

LOCAL_PREAMBLE = """You cannot play the clip. Instead you receive, in time order:
- FRAMES: still images taken from the clip, image k was taken at the time listed for it,
- TRANSCRIPT: what is said, with start times in seconds,
- LOUDNESS: audio level per second in dB (screams, explosions and laughter are loud spikes),
- chat lines if any.
Reconstruct what happens from these and judge it exactly as if you had watched it. All times are seconds from the start of the clip."""


def frame_times(duration: float, n: int) -> list[float]:
    return [round((i + 0.5) * duration / n, 2) for i in range(n)]


def extract_frames(src: Path, duration: float, n: int, height: int) -> list[tuple[float, bytes]]:
    out = []
    with tempfile.TemporaryDirectory() as d:
        for i, t in enumerate(frame_times(duration, n)):
            p = Path(d) / f"{i:02d}.jpg"
            subprocess.run([tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(src),
                            "-frames:v", "1", "-vf", f"scale=-2:{height}", "-q:v", "4", str(p)], timeout=60, check=False)
            if p.exists():
                out.append((t, p.read_bytes()))
    return out


def loudness_per_second(src: Path, duration: float) -> list[float]:
    import numpy as np
    pcm = subprocess.run([tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1",
                          "-ar", "8000", "-f", "f32le", "pipe:1"], capture_output=True, timeout=120).stdout
    a = np.frombuffer(pcm, dtype=np.float32)
    secs = max(1, min(int(duration) + 1, len(a) // 8000))
    return [round(float(20 * np.log10(np.sqrt(np.mean(a[i * 8000:(i + 1) * 8000] ** 2)) + 1e-6)), 1) for i in range(secs)]


def words_path(clip_id: str) -> Path:
    """Whisper words + loudness of the whole candidate, written by the local judge, reused by captions and voice."""
    return get_config().data / "candidates" / f"{clip_id}.words.json"


def save_words(clip_id: str, words: list, loud: list[float]) -> None:
    p = words_path(clip_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"words": [[w.start, w.end, w.text] for w in words], "loud": loud}, ensure_ascii=False),
                 encoding="utf-8")


def load_words(clip_id: str) -> dict | None:
    try:
        return json.loads(words_path(clip_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def transcript_lines(words: list) -> list[str]:
    lines, cur, t0 = [], [], 0.0
    for w in words:
        if not cur:
            t0 = w.start
        cur.append(w.text)
        if len(cur) >= 8 or w.text.endswith((".", "!", "?")):
            lines.append(f"[{t0:.1f}s] {' '.join(cur)}")
            cur = []
    if cur:
        lines.append(f"[{t0:.1f}s] {' '.join(cur)}")
    return lines


def build_user_text(base: str, times: list[float], transcript: list[str], loud: list[float]) -> str:
    parts = [base, "", "FRAMES: " + ", ".join(f"image {i + 1} at {t:.1f}s" for i, t in enumerate(times)), "",
             "TRANSCRIPT:", *(transcript or ["(no speech)"]), "",
             "LOUDNESS dB per second: " + " ".join(f"{i}s:{v:.0f}" for i, v in enumerate(loud))]
    return "\n".join(parts)


async def chat_json(system: str, user: str, schema: type, images: list[bytes] = (), clip_id: str | None = None):
    """One structured call to Ollama's /api/chat; retries once on unparsable output."""
    j = get_config().judge
    body = {
        "model": j.ollama_model, "stream": False, "keep_alive": j.ollama_keep_alive, "format": schema.model_json_schema(),
        "options": {"temperature": 0.2, "num_ctx": j.ollama_num_ctx},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user, "images": [base64.b64encode(b).decode() for b in images]}],
    }
    last: Exception | None = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(j.ollama_timeout_s, connect=10)) as h:
        for _ in range(2):
            r = await h.post(f"{j.ollama_url.rstrip('/')}/api/chat", json=body)
            if r.status_code != 200:
                raise RuntimeError(f"Ollama {r.status_code}: {r.text[:200]}")
            data = r.json()
            await db.add_event("gemini_usage", clip_id=clip_id, model=j.ollama_model, prompt=data.get("prompt_eval_count", 0),
                               output=data.get("eval_count", 0), usd=0.0)
            try:
                return schema.model_validate_json(data["message"]["content"])
            except Exception as e:  # noqa: BLE001
                last = e
    raise RuntimeError(f"Ollama-Antwort nicht lesbar: {str(last)[:200]}")


async def ollama_judge(src: Path, base_text: str, clip_id: str | None, lang: str, system: str) -> JudgeResult:
    from vc.render.ffmpeg import probe
    j = get_config().judge
    duration = (await probe(src)).duration
    from vc.render.captions import _transcribe_safe
    frames, words, loud = await asyncio.gather(
        asyncio.to_thread(extract_frames, src, duration, j.local_frames, j.proxy_height),
        asyncio.to_thread(_transcribe_safe, src, 0.0, duration, lang),
        asyncio.to_thread(loudness_per_second, src, duration),
    )
    if clip_id:  # captions and the voice component reuse them instead of a second Whisper pass
        save_words(clip_id, words, loud)
    text = build_user_text(base_text, [t for t, _ in frames], transcript_lines(words), loud)
    return await chat_json(system + "\n\n" + LOCAL_PREAMBLE, text, JudgeResult, [b for _, b in frames], clip_id)


async def check_ollama() -> str:
    """Reachable and model pulled? Returns a status line, raises otherwise."""
    j = get_config().judge
    async with httpx.AsyncClient(timeout=10) as h:
        tags = (await h.get(f"{j.ollama_url.rstrip('/')}/api/tags")).json()
    names = [m["name"] for m in tags.get("models", [])]
    if j.ollama_model not in names:
        raise RuntimeError(f"Modell {j.ollama_model} fehlt (ollama pull {j.ollama_model}); vorhanden: {', '.join(names)}")
    return f"{j.ollama_model} bereit"
