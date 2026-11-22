"""One command for start and stop: single-instance lock (data/vc.lock with PID), kv control flag, /go and /stop."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Awaitable, Callable

from vc import db, log
from vc.config import get_config

L = log.get("control")


def lock_path() -> Path:
    return get_config().data / "vc.lock"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":  # os.kill(pid, 0) would send CTRL_C on Windows
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_holder(path: Path | None = None) -> int | None:
    """PID of a living instance holding the lock, else None (a stale lock counts as free)."""
    path = path or lock_path()
    try:
        pid = int(path.read_text(encoding="utf-8").strip() or 0)
    except (FileNotFoundError, ValueError):
        return None
    return pid if pid_alive(pid) else None


def acquire_lock(path: Path | None = None) -> bool:
    path = path or lock_path()
    if lock_holder(path) is not None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()), encoding="utf-8")
    return True


def release_lock(path: Path | None = None) -> None:
    path = path or lock_path()
    if lock_holder(path) in (os.getpid(), None):
        path.unlink(missing_ok=True)


async def request_stop(timeout_s: float = 60, poll_s: float = 0.5,
                       sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> bool | None:
    """vc stop: set kv control=stop and wait for the instance to release its lock. None = nothing was running."""
    if lock_holder() is None:
        return None
    await db.init_db()
    await db.kv_set("control", "stop")
    waited = 0.0
    while waited < timeout_s:
        if lock_holder() is None:
            return True
        await sleep(poll_s)
        waited += poll_s
    return False


class Controller:
    """What the Telegram commands /go and /stop talk to. The bot itself keeps running."""

    def __init__(self, orch) -> None:
        self.orch = orch
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        return "running" if self.orch.running else "stopped"

    async def go(self) -> str:
        from vc.config import NO_STREAMERS_MSG, get_streamers
        async with self._lock:
            if not self.orch.running and not get_streamers():
                return NO_STREAMERS_MSG
            return "▶️ Pipeline gestartet" if await self.orch.start_pipeline() else "läuft bereits"

    async def stop(self) -> str:
        async with self._lock:
            return "⏹ Pipeline gestoppt (Bot bleibt an, /start startet neu)" if await self.orch.stop_pipeline() else "ist schon gestoppt"


async def _telegram_check(bot) -> None:
    """Say it loud in the console when Telegram does not answer: then /start, /status and clips cannot arrive.
    The bot keeps retrying on its own; this only reports."""
    try:
        me = await asyncio.wait_for(bot.bot.get_me(), 15)
        L.info("Telegram verbunden: @%s - /start, /stop, /status im Chat", me.username)
    except Exception as e:  # noqa: BLE001
        L.warning("Telegram antwortet NICHT (%s). Befehle und Clips kommen erst an, wenn die Verbindung wieder steht; "
                  "der Bot versucht es selbst weiter. Pruefen: Internet, Virenscanner-HTTPS-Pruefung, vc status.",
                  type(e).__name__)


async def serve(orch, standby: bool = False, poll_s: float = 2.0) -> None:
    """vc start: one process with bot (+ dashboard) and, unless standby, the pipeline.
    Ends on kv control=stop (vc stop) or Ctrl+C; both shut down the same clean way."""
    path = lock_path()
    if not acquire_lock(path):
        raise RuntimeError(f"läuft bereits (PID {lock_holder(path)})")
    side: list[asyncio.Task] = []
    try:
        await db.init_db()
        await db.kv_set("control", "")
        if orch.flags["bot"]:
            orch.make_bot(control=Controller(orch))
        side = [asyncio.create_task(orch.supervise(n, fn), name=n) for n, fn in orch.side_jobs().items()]
        if orch.bot:
            side.append(asyncio.create_task(_telegram_check(orch.bot), name="telegram_check"))
        if get_config().judge.provider == "ollama":  # without it every clip is judged without AI
            from vc.judge.local import check_ollama
            try:
                await check_ollama()
            except Exception as e:  # noqa: BLE001
                await orch.notify(f"⚠️ Ollama nicht erreichbar ({str(e)[:100]}). Ollama starten, sonst bewertet die Pipeline ohne KI.")
        if standby:
            await orch.notify("⏸ Bereit (Standby). /start startet die Pipeline.")
        else:
            await orch.start_pipeline()
        while True:
            await asyncio.sleep(poll_s)
            if await db.kv_get("control") == "stop":
                L.info("vc stop empfangen")
                break
    finally:
        try:
            await orch.stop_pipeline()
            await db.kv_set("control", "")
            if orch.bot:  # awaited (the session closes next), but capped so a hanging Telegram cannot block vc stop
                await orch.bot.notify("⏹ Pipeline gestoppt", timeout=5)
        finally:
            for t in side:
                t.cancel()
            await asyncio.gather(*side, return_exceptions=True)
            if orch.bot:
                await orch.bot.bot.session.close()
            release_lock(path)
            L.info("beendet")
