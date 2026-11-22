"""Structured logging: one line per event to console and data/logs/vc.log."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

_done = False


def setup(level: int = logging.INFO) -> None:
    global _done
    if _done:
        return
    from vc.config import get_config
    logdir = get_config().data / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    console.setLevel(level)
    file = RotatingFileHandler(logdir / "vc.log", maxBytes=20_000_000, backupCount=3, encoding="utf-8")
    file.setFormatter(fmt)
    file.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file)
    for noisy in ("httpx", "httpx2", "httpcore", "aiogram", "websockets", "asyncio", "faster_whisper", "google_genai", "urllib3", "aiosqlite"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _done = True


def get(name: str) -> logging.Logger:
    return logging.getLogger(name)
