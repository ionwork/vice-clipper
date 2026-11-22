"""Phase 6: one source for the operating guide."""
from typer.testing import CliRunner

from vc.cli import app
from vc.deliver.guide import plain, usage_text


def test_guide_contains_the_commands():
    t = usage_text()
    for s in ("vc start", "/start", "/pause", "/resume", "/stop", "vc stop", "--standby", "STREAMERS="):
        assert s in t, s


def test_cli_guide_prints_the_same_text():
    r = CliRunner().invoke(app, ["guide"])
    assert r.exit_code == 0 and r.output.strip() == plain(usage_text()).strip()


async def test_help_sends_the_same_text(monkeypatch):
    from aiogram.types import Chat, Message

    from vc.deliver.telegram_bot import ClipBot
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:TESTtokenTESTtokenTESTtokenTESTtok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    bot = ClipBot()
    sent: list[str] = []

    async def answer(self, text, **kw):
        sent.append(text)
    monkeypatch.setattr(Message, "answer", answer)
    from datetime import datetime
    from aiogram.types import Update, User
    msg = Message(message_id=1, date=datetime.now(), chat=Chat(id=42, type="private"),
                  from_user=User(id=1, is_bot=False, first_name="a"), text="/help")
    await bot.dp.feed_update(bot.bot, Update(update_id=1, message=msg))
    assert sent == [usage_text()]
