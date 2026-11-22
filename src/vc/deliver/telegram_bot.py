"""Telegram bot (aiogram 3, long polling): delivers clips and is the remote control."""
from __future__ import annotations

import asyncio
import re
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Awaitable, Callable

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandObject
from aiogram.types import (CallbackQuery, ErrorEvent, FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup,
                           LinkPreviewOptions, Message)

from vc import db, log
from vc.config import get_config, get_secrets, get_streamers, set_config_value, upsert_streamer
from vc.deliver import planner, textpack
from vc.models import Clip, ClipStatus, Layout
from vc.proc import run_ok

L = log.get("bot")
URL_RE = re.compile(r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch|live|shorts)\S*|youtu\.be/\S+|twitch\.tv/videos/\d+\S*)")


def keyboard(clip_id: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardButton
    return InlineKeyboardMarkup(inline_keyboard=[
        [b(text="✅ Gepostet", callback_data=f"p:{clip_id}"), b(text="🗑 Verwerfen", callback_data=f"r:{clip_id}")],
        [b(text="🔄 Anderes Layout", callback_data=f"l:{clip_id}"), b(text="✂️ Kürzer", callback_data=f"k:{clip_id}")],
    ])


def done_keyboard(label: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=label, callback_data="noop")]])


async def shrink(path: Path, max_mb: float) -> Path:
    """Last resort for Telegram's 50 MB limit: re-encode at a bitrate that fits."""
    from vc.render.ffmpeg import probe
    dur = (await probe(path)).duration or 25
    kbps = int(max_mb * 8000 / dur * 0.9) - 160
    out = path.with_name(path.stem + "_small.mp4")
    await run_ok("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-c:v", "libx264", "-preset", "veryfast",
                 "-b:v", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{2 * kbps}k", "-c:a", "copy", "-movflags", "+faststart", out)
    return out


def head_lines(clip: Clip, v) -> str:
    """V line plus category, content kind and reaction label above the text pack (HTML)."""
    import html

    from vc.detect import virality
    sig = clip.signals or {}
    label = sig.get("reaction_label", "")
    tags = [(clip.judge or {}).get("category") or "", "Echt-Leben" if sig.get("content") == "irl" else "Gameplay",
            f"Reaktion {virality.REACTION_EMOJI.get(label, '')} {label}" if label else ""]
    return f"<b>{html.escape(virality.summary_line(v, label))}</b>\n🏷 {html.escape(' · '.join(t for t in tags if t))}\n"


class ClipBot:
    def __init__(self, trigger_manual: Callable[[str, int], Awaitable[str]] | None = None,
                 status_extra: Callable[[], list[str]] | None = None, control=None):
        s = get_secrets()
        s.require("telegram_bot_token", "telegram_chat_id")
        self.chat_id = int(s.telegram_chat_id)
        self.bot = Bot(s.telegram_bot_token, default=DefaultBotProperties(parse_mode="HTML", link_preview=LinkPreviewOptions(is_disabled=True)))
        self.trigger_manual = trigger_manual
        self.status_extra = status_extra
        self.control = control  # vc.control.Controller: /go and /stop (only inside vc start)
        self._next_now = asyncio.Event()
        self._last_delivery = 0.0
        self._lock = asyncio.Lock()
        self.dp = Dispatcher()
        self.dp.message.outer_middleware(self._log_incoming)
        self.dp.include_router(self._router())

    async def _log_incoming(self, handler, event: Message, data: dict):
        """Every incoming message lands in the log, also the ones the chat filter drops (wrong TELEGRAM_CHAT_ID)."""
        if event.chat.id == self.chat_id:
            L.info("Telegram empfangen: %s", (event.text or "")[:60])
        else:
            L.warning("Telegram: Nachricht aus fremdem Chat %s ignoriert (TELEGRAM_CHAT_ID ist %s)", event.chat.id, self.chat_id)
        return await handler(event, data)

    # ------------------------------------------------------------ delivery
    async def notify(self, text: str, timeout: float = 10) -> bool:
        """Short status message. Never raises and never blocks longer than timeout (Telegram may hang)."""
        try:
            await self.bot.send_message(self.chat_id, text, parse_mode=None, request_timeout=int(timeout))
            return True
        except Exception as e:  # noqa: BLE001
            L.warning("Telegram-Meldung fehlgeschlagen (%s): %s", type(e).__name__, str(e)[:120])
            return False

    async def deliver(self, clip_id: str) -> None:
        async with self._lock:
            clip = await db.get_clip(clip_id)
            if not clip or not clip.out_path:
                raise ValueError(f"{clip_id}: nicht gerendert")
            path = Path(clip.out_path)
            limit = min(49.0, get_config().render.max_file_mb)
            if path.stat().st_size / 1e6 > limit:
                path = await shrink(path, limit)
            slot = await planner.next_slot(clip.account)
            textpack.build(clip, slot)
            file = FSInputFile(path, filename=f"{clip.id}.mp4")
            if get_config().delivery.send_as_video:  # plays inline; "Speichern" on the phone keeps the original file
                from vc.render.ffmpeg import probe
                info = await probe(path)
                await self.bot.send_video(self.chat_id, file, width=info.width, height=info.height,
                                          duration=round(info.duration), supports_streaming=True)
            else:
                await self.bot.send_document(self.chat_id, file, disable_content_type_detection=True)
            text = textpack.build_html(clip, slot)
            from vc.detect import virality
            if (v := virality.stored(clip)) is not None and v.known:  # 'V 78 · Chat 0.9 · 😂 0.8 · Clips 0.7 · KI 0.8'
                text = head_lines(clip, v) + text
            msg = await self.bot.send_message(self.chat_id, text, reply_markup=keyboard(clip.id))
            await db.update_clip(clip.id, status=ClipStatus.DELIVERED, delivered_at=db.now(), tg_message_id=msg.message_id,
                                 text_path=str(get_config().data / "out" / f"{clip.id}.txt"))
            await db.add_event("delivered", clip_id=clip.id, streamer_id=clip.streamer_id, hype=clip.hype)
            self._last_delivery = asyncio.get_running_loop().time()
            L.info("delivered %s", clip.id)

    async def pick_next(self, fresh_only: bool = True, slots: bool = False) -> Clip | None:
        """Best fresh clip; with slots=True only clips whose account may post within the next 5 minutes."""
        cfg = get_config().delivery
        # delivered_at set = re-render of an already delivered clip; _rerender sends it itself
        clips = [c for c in await db.list_clips(ClipStatus.RENDERED) if c.delivered_at is None]
        if fresh_only:
            cutoff = db.now() - timedelta(minutes=cfg.stale_after_minutes)
            clips = [c for c in clips if c.created_at and c.created_at >= cutoff]
        if slots and clips:
            soon = db.now() + timedelta(minutes=5)
            free = {a for a in {c.account for c in clips} if await planner.next_slot(a) <= soon}
            clips = [c for c in clips if c.account in free]
        clips.sort(key=lambda c: ((c.hype or 0) + c.bonus, c.created_at or db.now()), reverse=True)
        return clips[0] if clips else None

    async def pacing_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._next_now.wait(), timeout=15)
            except asyncio.TimeoutError:
                pass
            forced = self._next_now.is_set()
            self._next_now.clear()
            cfg = get_config().delivery
            try:
                if forced:
                    clip = await self.pick_next() or await self.pick_next(fresh_only=False)
                    if clip:
                        await self.deliver(clip.id)
                    else:
                        await self.notify("Keine fertigen Clips in der Warteschlange.")
                    continue
                gap_ok = asyncio.get_running_loop().time() - self._last_delivery >= cfg.min_gap_minutes * 60 or not self._last_delivery
                # mode window: the pipeline's Window scheduler delivers; here only /next (above)
                if cfg.mode == "gap" and cfg.auto and not cfg.paused and gap_ok and (clip := await self.pick_next(slots=cfg.follow_slots)):
                    await self.deliver(clip.id)
            except Exception as e:  # noqa: BLE001
                L.exception("delivery: %s", e)
                await db.add_event("error", stage="delivery", error=str(e)[:300])

    # ------------------------------------------------------------ handlers
    def _router(self) -> Router:
        r = Router()
        r.message.filter(F.chat.id == self.chat_id)
        r.callback_query.filter(F.message.chat.id == self.chat_id)

        @r.message(Command("help"))
        async def help_(m: Message) -> None:
            from vc.deliver.guide import usage_text
            await m.answer(usage_text())

        @r.message(Command("start", "go"))
        async def go_(m: Message) -> None:
            if not self.control:
                await m.answer("/start geht nur, wenn der Bot mit <code>vc start</code> läuft.")
                return
            await m.answer(await self.control.go(), parse_mode=None)

        @r.message(Command("stop"))
        async def stop_(m: Message) -> None:
            if not self.control:
                await m.answer("/stop geht nur, wenn der Bot mit <code>vc start</code> läuft.")
                return
            await m.answer(await self.control.stop(), parse_mode=None)

        @r.message(Command("status"))
        async def status(m: Message) -> None:
            await m.answer(f"<pre>{await self.status_text()}</pre>")

        @r.message(Command("next"))
        async def next_(m: Message) -> None:
            self._next_now.set()
            await m.answer("kommt …")

        @r.message(Command("pause"))
        async def pause(m: Message) -> None:
            set_config_value("delivery.paused", True)
            await m.answer("Lieferung pausiert.")

        @r.message(Command("resume"))
        async def resume(m: Message) -> None:
            set_config_value("delivery.paused", False)
            await m.answer("Lieferung läuft wieder.")

        @r.message(Command("launch"))
        async def launch(m: Message, command: CommandObject) -> None:
            from vc.setup import launch as do_launch
            await m.answer(await do_launch((command.args or "").strip() != "probe"), parse_mode=None)

        @r.message(Command("hype"))
        async def hype(m: Message, command: CommandObject) -> None:
            try:
                v = int((command.args or "").strip())
                assert 1 <= v <= 10
            except (ValueError, AssertionError):
                await m.answer("Nutzung: /hype 8")
                return
            set_config_value("judge.min_hype", v)
            await m.answer(f"min_hype = {v}")

        @r.message(Command("clip"))
        async def clip(m: Message, command: CommandObject) -> None:
            sid = (command.args or "").strip().lower()
            if sid not in get_streamers():
                await m.answer(f"Unbekannter Streamer: {sid or '?'}")
                return
            if not self.trigger_manual:
                await m.answer("/clip geht nur, wenn die Pipeline mit <code>vc run</code> läuft.")
                return
            try:
                cid = await self.trigger_manual(sid, 30)
                await m.answer(f"Kandidat {cid} angelegt.")
            except Exception as e:  # noqa: BLE001
                await m.answer(f"Fehler: {str(e)[:200]}", parse_mode=None)

        @r.message(Command("add"))
        async def add(m: Message, command: CommandObject) -> None:
            parts = (command.args or "").split()
            if len(parts) < 2 or parts[0] not in ("twitch", "kick"):
                await m.answer("Nutzung: /add twitch login en [campaign|public_ok|asked|none]")
                return
            platform, login = parts[0], parts[1].lower()
            lang = parts[2] if len(parts) > 2 else "en"
            permission = parts[3] if len(parts) > 3 else "none"  # campaign | public_ok | asked | none
            account = get_config().harvest.account_by_lang.get(lang, "account_other")
            upsert_streamer({"id": login, "platform": platform, "login": login, "lang": lang, "quality": "1080p60,1080p,best",
                             "priority": 1 if permission == "campaign" else 2, "account": account, "permission": permission,
                             "campaign_rules": "",
                             "credit": f"{'twitch.tv' if platform == 'twitch' else 'kick.com'}/{login}", "cam": None})
            await m.answer(f"{login}: Details gespeichert ({platform}, {lang}, {account}). Überwacht wird nur, wer in der "
                           f".env unter STREAMERS steht, z. B. {platform}:{login}:{lang}")

        @r.message(F.text.regexp(URL_RE))
        async def link(m: Message) -> None:
            for url in URL_RE.findall(m.text or ""):
                await db.add_vod_job(url)
            await m.answer("eingereiht")

        @r.callback_query(F.data == "noop")
        async def noop(q: CallbackQuery) -> None:
            await q.answer()

        @r.callback_query(F.data.regexp(r"^[prlk]:"))
        async def button(q: CallbackQuery) -> None:
            action, clip_id = q.data.split(":", 1)
            clip = await db.get_clip(clip_id)
            if not clip:
                await q.answer("Clip unbekannt")
                return
            if action == "p":
                await db.update_clip(clip_id, status=ClipStatus.POSTED, posted_at=db.now())
                await db.add_event("posted", clip_id=clip_id, account=clip.account)
                slot = await planner.next_slot(clip.account)
                await q.message.edit_reply_markup(reply_markup=done_keyboard(f"✅ gepostet · nächster Slot {slot.astimezone():%H:%M}"))
                await q.answer("gepostet")
            elif action == "r":
                await db.update_clip(clip_id, status=ClipStatus.REJECTED, reason="von Hand verworfen")
                await db.add_event("rejected_manual", clip_id=clip_id)
                await q.message.edit_reply_markup(reply_markup=done_keyboard("🗑 verworfen"))
                await q.answer("verworfen")
            else:
                await q.answer("rendert neu …")
                asyncio.create_task(self._rerender(clip, shorter=action == "k"))

        @r.error()
        async def on_error(ev: ErrorEvent) -> None:  # a failing command answers instead of staying silent
            L.error("Telegram-Befehl fehlgeschlagen: %r", ev.exception, exc_info=ev.exception)
            msg = ev.update.message or (ev.update.callback_query.message if ev.update.callback_query else None)
            if msg:
                try:
                    await msg.answer(f"Fehler: {str(ev.exception)[:300]}", parse_mode=None)
                except Exception:  # noqa: BLE001
                    pass

        return r

    async def _rerender(self, clip: Clip, shorter: bool) -> None:
        from vc.render.render import render_from_db
        try:
            layout = None
            if not shorter:
                cur = Layout(clip.layout or "split")
                # without a cam only the layouts that need none: gameplay <-> irl
                layout = cur.next() if clip.streamer_obj.cam else (Layout.GAMEPLAY if cur == Layout.IRL else Layout.IRL)
            await render_from_db(clip.id, layout=layout, shorter=shorter)
            await self.deliver(clip.id)
        except Exception as e:  # noqa: BLE001
            L.exception("rerender")
            await self.notify(f"{clip.id}: Neu-Render fehlgeschlagen: {str(e)[:200]}")

    async def status_text(self) -> str:
        midnight = db.local_midnight_utc()
        counts = Counter(c.status.value for c in await db.list_clips(since=midnight - timedelta(days=1), limit=100000))
        delivered = len(await db.list_events("delivered", since=midnight, limit=100000))
        posted = len(await db.list_events("posted", since=midnight, limit=100000))
        p, o, usd = await db.cost_today()
        cfg = get_config()
        lines = list(self.status_extra() if self.status_extra else ["Recorder: (nur mit vc run)"])
        lines += [
            "Warteschlange: " + (" ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "leer"),
            f"Heute: {delivered} geliefert, {posted} gepostet",
            (f"KI lokal ({cfg.judge.ollama_model}): {p:,} in / {o:,} out Tokens, 0 $" if cfg.judge.provider == "ollama" else f"Gemini heute: {p:,} in / {o:,} out Tokens ≈ {usd:.3f} $ (Budget {cfg.judge.daily_budget_usd} $)"),
            f"Modus: {'Probe (GTA V)' if cfg.judge.allow_gta5 else 'GTA 6'}",
            f"Auto: {'an' if cfg.delivery.auto else 'aus'} · Slots: {'an' if cfg.delivery.follow_slots else 'aus'} · Pause: {'ja' if cfg.delivery.paused else 'nein'} · min_hype {cfg.judge.min_hype}",
        ]
        return "\n".join(lines)

    async def run(self) -> None:
        pacing = asyncio.create_task(self.pacing_loop())
        polling = asyncio.create_task(self.dp.start_polling(self.bot, handle_signals=False))
        try:
            await asyncio.shield(polling)
        except asyncio.CancelledError:  # end the long poll the regular way (no "Server disconnected" error)
            try:
                await self.dp.stop_polling()
            except RuntimeError:
                pass
            await asyncio.gather(polling, return_exceptions=True)
            raise
        finally:
            pacing.cancel()
            await self.bot.session.close()


async def deliver_once(clip_id: str) -> None:
    bot = ClipBot()
    try:
        await bot.deliver(clip_id)
    finally:
        await bot.bot.session.close()
