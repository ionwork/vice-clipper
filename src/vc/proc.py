"""Locate external tools and run them with asyncio (never shell=True)."""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from vc import log

L = log.get("proc")


@cache
def tool(name: str) -> str:
    """PATH, then the venv's Scripts dir (streamlink, yt-dlp), then winget's link dir (fresh ffmpeg installs)."""
    extra = [Path(sys.executable).parent]
    if la := os.environ.get("LOCALAPPDATA"):
        extra.append(Path(la) / "Microsoft" / "WinGet" / "Links")
    found = shutil.which(name) or next((shutil.which(name, path=str(d)) for d in extra if shutil.which(name, path=str(d))), None)
    if not found:
        raise FileNotFoundError(f"{name} nicht gefunden (PATH, venv, WinGet). Siehe README (Voraussetzungen).")
    return found


@dataclass
class Result:
    code: int
    out: bytes
    err: str

    def tail(self, n: int = 20) -> str:
        return "\n".join(self.err.strip().splitlines()[-n:])


async def run(name: str, *args: str | Path, cwd: Path | None = None, timeout: float | None = None, stdin: bytes | None = None) -> Result:
    argv = [tool(name), *map(str, args)]
    if name == "yt-dlp" and "--version" not in args:  # use the OS cert store (antivirus HTTPS scanning), not certifi
        argv[1:1] = ["--compat-options", "no-certifi"]
    L.debug("exec: %s", " ".join(argv))
    p = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(p.communicate(stdin), timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        p.kill()
        await p.wait()
        raise
    return Result(p.returncode or 0, out, err.decode("utf-8", "replace"))


async def run_ok(name: str, *args: str | Path, **kw) -> Result:
    r = await run(name, *args, **kw)
    if r.code != 0:
        raise RuntimeError(f"{name} exit {r.code}:\n{r.tail()}")
    return r
