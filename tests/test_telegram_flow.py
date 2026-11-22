"""Real Telegram updates through the real dispatcher (filters, middleware, handlers, error handler);
only the HTTP session to Telegram is replaced."""
import logging
from datetime import datetime, timezone

from aiogram.client.session.base import BaseSession
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from vc import control
from vc.orchestrator import Orchestrator


class FakeSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.sent: list[tuple[int, str]] = []

    async def make_request(self, bot, method, timeout=None):
        if isinstance(method, SendMessage):
            self.sent.append((method.chat_id, method.text))
            return Message(message_id=len(self.sent), date=datetime.now(timezone.utc),
                           chat=Chat(id=method.chat_id, type="private"), text=method.text)
        return True

    async def stream_content(self, *a, **kw):  # pragma: no cover
        yield b""

    async def close(self):
        pass


def update(text: str, chat_id: int = 42, n: int = 1) -> Update:
    return Update(update_id=n, message=Message(message_id=n, date=datetime.now(timezone.utc), text=text,
                                               chat=Chat(id=chat_id, type="private"),
                                               from_user=User(id=chat_id, is_bot=False, first_name="Rob")))


async def test_start_status_stop_through_dispatcher(db, monkeypatch, caplog):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:TESTtokenTESTtokenTESTtokenTESTtok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    orch = Orchestrator(live=False, harvest=False, vod=False, web=False, bot=False)
    bot = orch.make_bot(control=control.Controller(orch))
    fake = FakeSession()
    bot.bot.session = fake

    async def say(text, chat_id=42):
        before = len(fake.sent)
        await bot.dp.feed_update(bot.bot, update(text, chat_id))
        return [t for c, t in fake.sent[before:] if c == 42]
    try:
        assert any("Pipeline gestartet" in t for t in await say("/start")) and orch.running
        assert any("Pipeline: läuft" in t for t in await say("/status"))
        assert any("läuft bereits" in t for t in await say("/start"))
        assert any("gestoppt" in t for t in await say("/stop")) and not orch.running
        assert any("gestoppt (/start startet)" in t for t in await say("/status"))
        assert any("/start" in t and "/stop" in t for t in await say("/help"))
        # a stranger's chat is ignored (and logged, so a wrong TELEGRAM_CHAT_ID shows up in vc.log)
        with caplog.at_level(logging.WARNING, logger="bot"):
            assert await say("/start", chat_id=777) == [] and not orch.running
        assert "fremdem Chat 777" in caplog.text

        # a crashing command answers with the error instead of staying silent
        async def boom():
            raise RuntimeError("DB kaputt")
        monkeypatch.setattr(bot, "status_text", boom)
        assert any("Fehler: DB kaputt" in t for t in await say("/status"))
    finally:
        await orch.stop_pipeline()


async def test_notify_never_blocks_the_pipeline(db, monkeypatch):
    """Telegram hanging (seen on this PC): orchestrator notices are fire-and-forget."""
    import asyncio
    orch = Orchestrator(live=False, harvest=False, vod=False, web=False, bot=False)

    class SlowBot:
        async def notify(self, text, timeout=10):
            await asyncio.sleep(30)
    orch.bot = SlowBot()
    t0 = asyncio.get_running_loop().time()
    await orch.notify("x")
    assert asyncio.get_running_loop().time() - t0 < 0.5
    for t in list(orch._bg):
        t.cancel()
