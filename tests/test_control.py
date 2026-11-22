import asyncio
import os

from typer.testing import CliRunner

from vc import control
from vc.cli import app
from vc.orchestrator import Orchestrator


def quiet() -> Orchestrator:
    """Pipeline without network parts: judge, render, cleanup and heartbeat workers only."""
    return Orchestrator(live=False, harvest=False, vod=False, web=False, bot=False)


def test_second_start_exits_1_while_lock_is_held():
    path = control.lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()), encoding="utf-8")  # a living process holds it
    try:
        r = CliRunner().invoke(app, ["start"])
        assert r.exit_code == 1 and "läuft bereits" in r.output
    finally:
        path.unlink(missing_ok=True)


def test_stale_lock_is_free(tmp_path):
    p = tmp_path / "vc.lock"
    p.write_text("999999999", encoding="utf-8")  # no such process
    assert control.lock_holder(p) is None and control.acquire_lock(p) and control.lock_holder(p) == os.getpid()
    control.release_lock(p)
    assert not p.exists()


async def test_kv_stop_cancels_workers_and_removes_lock(db):
    orch = quiet()
    task = asyncio.create_task(control.serve(orch, poll_s=0.05))   # fake beat: 50 ms instead of 2 s
    for _ in range(100):
        await asyncio.sleep(0.02)
        if orch.running:
            break
    workers = list(orch.tasks.values())
    assert orch.running and workers and control.lock_path().exists()
    await db.kv_set("control", "stop")
    await asyncio.wait_for(task, timeout=3)
    assert not orch.running and all(t.done() for t in workers)
    assert not control.lock_path().exists() and await db.kv_get("control") == ""


async def test_go_stop_go_via_telegram_handlers(db, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:TESTtokenTESTtokenTESTtokenTESTtok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    from vc.deliver.telegram_bot import ClipBot
    orch = quiet()
    ctl = control.Controller(orch)
    bot = ClipBot(control=ctl)
    handlers = {h.callback.__name__: h.callback for r in bot.dp.sub_routers for h in r.message.handlers}

    class Msg:
        def __init__(self):
            self.answers: list[str] = []

        async def answer(self, text, **kw):
            self.answers.append(text)
    states, counts = [], []
    try:
        for cmd in ("go_", "stop_", "go_", "go_"):
            m = Msg()
            await handlers[cmd](m)
            states.append(ctl.state)
            counts.append(len(orch.tasks))
        assert states == ["running", "stopped", "running", "running"]
        assert counts[0] == counts[2] == counts[3] > 0 and counts[1] == 0   # no duplicate workers
        assert m.answers == ["läuft bereits"]
    finally:
        await orch.stop_pipeline()
        await bot.bot.session.close()
