"""Typer CLI: uv run vc --help"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

from vc import log, tls

tls.setup()  # antivirus HTTPS scanning (Avast here): trust the Windows certificate store

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)


def _run(coro):
    log.setup()
    return asyncio.run(coro)


def _ok(label: str, ok: bool, detail: str = "") -> bool:
    mark = typer.style(" OK ", fg="green") if ok else typer.style("FAIL", fg="red")
    typer.echo(f"[{mark}] {label:<28} {detail}")
    return ok


# ---------------------------------------------------------------- chapter 1
@app.command()
def doctor() -> None:
    """Check tools, encoder, fonts, sfx and secrets."""
    from vc.config import REQUIRED_SECRETS, get_config, get_secrets
    from vc.proc import tool
    from vc.render.ffmpeg import detect_encoder

    async def main() -> bool:
        good = True
        for name, arg in (("ffmpeg", "-version"), ("ffprobe", "-version"), ("streamlink", "--version"), ("yt-dlp", "--version")):
            try:
                from vc.proc import run
                r = await run(name, arg)
                good &= _ok(name, r.code == 0, (r.out.decode(errors="replace").splitlines() or [""])[0][:70])
            except FileNotFoundError as e:
                good &= _ok(name, False, str(e))
        try:
            tool("ffmpeg")
            enc = await detect_encoder(force=True)
            good &= _ok("H.264 encoder", True, enc)
        except Exception as e:  # noqa: BLE001
            good &= _ok("H.264 encoder", False, str(e)[:80])
        cfg = get_config()
        for f in ("NotoSans-Black.ttf", "NotoSans-Bold.ttf"):
            p = cfg.path("fonts") / f
            good &= _ok(f"font {f}", p.exists(), str(p))
        for f in ("whoosh.wav", "impact.wav"):
            p = cfg.path("sfx") / f
            good &= _ok(f"sfx {f}", p.exists(), str(p))
        s = get_secrets()
        for k in REQUIRED_SECRETS:
            if k == "gemini_api_key" and cfg.judge.provider == "ollama":
                continue
            good &= _ok(f".env {k.upper()}", bool(getattr(s, k)))
        from vc.config import get_streamers
        st = get_streamers()
        from vc.config import NO_STREAMERS_MSG, env_streamer_errors
        _ok("Streamer", bool(st), f"{len(st)}: " + ", ".join(f"{x.id}{'' if x.cam else ' (Cam auto)'}" for x in st.values())[:120]
            if st else NO_STREAMERS_MSG)
        for e in env_streamer_errors():
            _ok("  STREAMERS übersprungen", False, e)
        for label, ok, detail in await _lookup_streamers(st):
            good &= _ok(label, ok, detail)
        from vc.setup import online_checks
        for label, ok, detail in await online_checks():
            good &= _ok(label, ok, detail)
        typer.echo(f"Modus: {'Probe (GTA V erlaubt)' if cfg.judge.allow_gta5 else 'GTA 6'} · Harvest {cfg.harvest.game_names}"
                   f" · Auto-Lieferung {'an' if cfg.delivery.auto else 'aus'}")
        return good

    raise typer.Exit(0 if _run(main()) else 1)


async def _lookup_streamers(streamers: dict) -> list[tuple[str, bool, str]]:
    """Does each login exist on its platform? Twitch: Helix users + channels (language), Kick: channel API."""
    from vc.config import get_secrets
    out: list[tuple[str, bool, str]] = []
    tw = [s for s in streamers.values() if s.platform == "twitch"]
    if tw and get_secrets().twitch_client_id:
        from vc.sources.twitch_api import TwitchAPI
        api = TwitchAPI()
        try:
            users = {u["login"]: u for u in await api.get_users([s.login for s in tw])}
            for s in tw:
                if u := users.get(s.login):
                    ch = ((await api.get("channels", {"broadcaster_id": u["id"]})).get("data") or [{}])[0]
                    out.append((f"  {s.login}", True, f"gefunden · twitch · {ch.get('broadcaster_language', '?')}"))
                else:
                    out.append((f"  {s.login}", False, "auf Twitch nicht gefunden"))
        except Exception as e:  # noqa: BLE001
            out.append(("  Twitch-Streamer", False, str(e)[:80]))
        finally:
            await api.close()
    for s in (s for s in streamers.values() if s.platform == "kick"):
        try:
            from vc.sources.chat_kick import chatroom_id
            await chatroom_id(s.login)
            out.append((f"  {s.login}", True, f"gefunden · kick · {s.lang}"))
        except Exception as e:  # noqa: BLE001
            out.append((f"  {s.login}", False, f"auf Kick nicht gefunden ({str(e)[:50]})"))
    return out


@app.command()
def streamers(online: bool = typer.Option(True, help="Twitch-IDs per Helix pruefen")) -> None:
    """Die ueberwachten Streamer aus STREAMERS in .env (einzige Liste) als Tabelle."""
    from vc.config import NO_STREAMERS_MSG, env_streamer_errors, get_streamers
    st = get_streamers()
    for e in env_streamer_errors():
        typer.echo(f"übersprungen: {e}")
    if not st:
        typer.echo(NO_STREAMERS_MSG)
        raise typer.Exit(1)
    found = {label.strip(): ok for label, ok, _ in _run(_lookup_streamers(st))} if online else {}
    typer.echo(f"{'Login':<20} {'Plattform':<9} {'Sprache':<7} {'Erlaubnis':<10} {'Cam':<5} Twitch-ID")
    for s in st.values():
        tid = "–" if s.platform != "twitch" or not found else ("ja" if found.get(s.login) else "nein")
        typer.echo(f"{s.login:<20} {s.platform:<9} {s.lang:<7} {s.permission:<10} {'ja' if s.cam else 'nein':<5} {tid}")


@app.command()
def virality(clip_id: str, recompute: bool = typer.Option(True, help="neu berechnen (sonst gespeicherten Wert zeigen)")) -> None:
    """Viralitaetsscore V eines Clips mit Aufschluesselung (Komponente, Rohwert, Normwert, Gewicht, Beitrag)."""
    from vc import db
    from vc.detect import virality as vr

    async def main() -> None:
        await db.init_db()
        clip = await db.get_clip(clip_id)
        if not clip:
            raise typer.BadParameter(f"Clip {clip_id} unbekannt")
        res = await vr.compute(clip) if recompute else vr.stored(clip)
        if res is None:
            raise typer.BadParameter("kein gespeicherter Wert")
        raw = ((clip.signals or {}).get("virality") or {}).get("raw", {})
        typer.echo(f"{clip.id} · {clip.streamer_id} · {clip.status.value} · {vr.summary_line(res)}")
        typer.echo(f"{'Komponente':<11} {'Rohwert':<34} {'Norm':>5} {'Gewicht':>8} {'Beitrag':>8}")
        for k, r, n, w, c in vr.breakdown(res, raw):
            typer.echo(f"{k:<11} {r[:34]:<34} {n:>5} {w:>8} {c:>8}")
        extra = [f"Bonus x{res.bonus:.2f}"] if res.bonus != 1 else []
        typer.echo(f"V = {res.v:.1f}  " + " · ".join(extra + res.penalties + [res.reason]))

    _run(main())


@app.command()
def setup() -> None:
    """Gefuehrte Einrichtung: Schluessel abfragen, live pruefen, Telegram-Chat finden, .env schreiben."""
    from vc import setup as su
    from vc.config import get_config, get_secrets, set_config_value

    def ask(label: str, current: str, secret: bool = True) -> str:
        shown = f" [{current[:4]}…]" if current and secret else (f" [{current}]" if current else "")
        v = typer.prompt(f"{label}{shown} (Enter = behalten)", default="", show_default=False, hide_input=secret)
        return v.strip() or current

    async def main() -> None:
        s = get_secrets()
        env: dict[str, str] = {}
        typer.echo("\n1) Twitch  -> dev.twitch.tv/console/apps, App registrieren, Client-ID + neues Secret kopieren")
        while True:
            cid = ask("TWITCH_CLIENT_ID", s.twitch_client_id, secret=False)
            sec = ask("TWITCH_CLIENT_SECRET", s.twitch_client_secret)
            try:
                _ok("Twitch", True, await su.check_twitch(cid, sec))
                env.update(twitch_client_id=cid, twitch_client_secret=sec)
                break
            except Exception as e:  # noqa: BLE001
                _ok("Twitch", False, str(e))
                if not typer.confirm("Nochmal eingeben?", default=True):
                    break
        typer.echo("\n2) Telegram  -> in Telegram @BotFather oeffnen, /newbot, Token kopieren")
        while True:
            tok = ask("TELEGRAM_BOT_TOKEN", s.telegram_bot_token)
            try:
                name = await su.check_telegram(tok)
                _ok("Telegram", True, f"@{name}")
                env["telegram_bot_token"] = tok
                break
            except Exception as e:  # noqa: BLE001
                _ok("Telegram", False, str(e))
                if not typer.confirm("Nochmal eingeben?", default=True):
                    name = None
                    break
        if name and (not s.telegram_chat_id or typer.confirm("Chat-ID neu suchen?", default=False)):
            typer.echo(f"   Oeffne jetzt t.me/{name} und schick dem Bot /start (ich warte bis zu 3 Minuten) …")
            if chat := await su.find_chat_id(tok):
                _ok("Telegram Chat-ID", True, chat)
                env["telegram_chat_id"] = chat
            else:
                _ok("Telegram Chat-ID", False, "keine Nachricht gesehen - vc setup spaeter erneut starten")
        if get_config().judge.provider == "ollama":
            from vc.judge.local import check_ollama
            typer.echo("\n3) Bewertung lokal mit Ollama (kein Gemini-Key noetig)")
            try:
                _ok("Ollama", True, await check_ollama())
            except Exception as e:  # noqa: BLE001
                _ok("Ollama", False, f"{str(e)[:120]} - Ollama starten bzw. Modell laden")
        else:
            typer.echo("\n3) Gemini  -> aistudio.google.com/apikey, Key erstellen (Projekt mit Abrechnung + Budget-Limit)")
        while get_config().judge.provider == "gemini":
            key = ask("GEMINI_API_KEY", s.gemini_api_key)
            try:
                model = get_config().judge.model
                ok, alt = await su.check_gemini(key, model)
                if not ok and alt:
                    set_config_value("judge.model", alt)
                    model = alt
                _ok("Gemini", ok or bool(alt), f"Modell {model}" if ok or alt else "kein Flash-Lite-Modell gefunden")
                env["gemini_api_key"] = key
                break
            except Exception as e:  # noqa: BLE001
                _ok("Gemini", False, str(e))
                if not typer.confirm("Nochmal eingeben?", default=True):
                    break
        typer.echo("\n4) Streamer  -> wen soll ich schauen? Komma-Liste, z. B.: twitch:login_eins:de, twitch:login_zwei:en")
        env["streamers"] = ask("STREAMERS", s.streamers, secret=False)
        su.write_env(env)
        typer.echo(f"\nGespeichert in {su.ENV_PATH}. Weiter mit: uv run vc doctor")

    _run(main())


@app.command()
def launch(undo: bool = typer.Option(False, help="zurueck in den Probe-Modus (GTA V)")) -> None:
    """Auf GTA 6 umschalten: nur GTA-VI-Material, Harvester auf die GTA-VI-Kategorie."""
    from vc.setup import launch as do_launch
    typer.echo(_run(do_launch(not undo)))


@app.command("db-init")
def db_init() -> None:
    """Create the SQLite tables."""
    from vc.db import db_path, init_db
    _run(init_db())
    typer.echo(f"DB bereit: {db_path()}")


# ---------------------------------------------------------------- chapter 2
@app.command()
def render(
    input: Path = typer.Option(..., exists=True), start: float = typer.Option(...), end: float = typer.Option(...),
    layout: str = "split", cam: Optional[str] = typer.Option(None, help="x,y,w,h im 1920x1080-Bild"),
    subject_x: float = 0.5, ass: Optional[Path] = None, peak: Optional[float] = None, out: Path = typer.Option(...),
    no_sfx: bool = False,
) -> None:
    """Render a 9:16 clip from a landscape video."""
    import time
    from vc.models import Layout, Rect
    from vc.render.render import RenderJob, render_clip
    t = time.time()
    job = RenderJob(src=input, start_s=start, end_s=end, layout=Layout(layout), cam=Rect.parse(cam) if cam else None,
                    subject_x=subject_x, ass_path=ass, peak_s=peak, out_path=out, sfx=not no_sfx)
    p = _run(render_clip(job))
    typer.echo(f"{p}  ({time.time() - t:.1f}s)")


# ---------------------------------------------------------------- chapter 3
@app.command()
def judge(clip_id: Optional[str] = None, file: Optional[Path] = typer.Option(None, exists=True),
          name: str = "", lang: str = "en") -> None:
    """Judge a candidate from the DB (--clip-id) or an ad-hoc file (--file, prints JSON)."""
    from vc.judge.gemini import judge_candidate, judge_file
    if file:
        typer.echo(json.dumps(_run(judge_file(file, name, lang)), indent=2, ensure_ascii=False))
    elif clip_id:
        typer.echo(_run(judge_candidate(clip_id)).model_dump_json(indent=2))
    else:
        raise typer.BadParameter("--clip-id oder --file angeben")


# ---------------------------------------------------------------- chapter 4
@app.command()
def captions(clip_id: Optional[str] = None, file: Optional[Path] = typer.Option(None, exists=True),
             start: float = 0, end: float = 20, lang: str = "en", hook: str = "", credit: str = "",
             layout: str = "split", out: Optional[Path] = None) -> None:
    """Build the ASS file for a clip (--clip-id) or an ad-hoc file."""
    from vc.models import Layout
    from vc.render.captions import build_ass, build_ass_text, transcribe
    if clip_id:
        typer.echo(_run(build_ass(clip_id)))
        return
    if not file or not out:
        raise typer.BadParameter("--clip-id oder --file und --out angeben")
    log.setup()
    words = transcribe(file, start, end, lang)
    typer.echo(" ".join(f"{w.text}[{w.start:.2f}]" for w in words))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_ass_text(words, hook, credit, Layout(layout), end - start, lang), encoding="utf-8")
    typer.echo(out)


@app.command("render-clip")
def render_clip_cmd(clip_id: str, layout: Optional[str] = None, shorter: bool = False) -> None:
    """Captions + render for a JUDGED clip from the DB."""
    from vc.models import Layout
    from vc.render.render import render_from_db
    typer.echo(_run(render_from_db(clip_id, Layout(layout) if layout else None, shorter)))


# ---------------------------------------------------------------- chapter 5
@app.command()
def bot() -> None:
    """Run the Telegram bot (delivery + remote control)."""
    from vc.db import init_db
    from vc.deliver.telegram_bot import ClipBot

    async def main() -> None:
        await init_db()
        await ClipBot().run()
    _run(main())


@app.command()
def deliver(clip_id: str) -> None:
    """Send one rendered clip to Telegram now."""
    from vc.deliver.telegram_bot import deliver_once
    _run(deliver_once(clip_id))


# ---------------------------------------------------------------- chapter 6
@app.command()
def harvest(once: bool = False) -> None:
    """Poll Twitch clips of the category (--once prints the picked candidates)."""
    from vc.db import init_db
    from vc.sources.harvester import Harvester

    async def main() -> None:
        await init_db()
        h = Harvester()
        try:
            if not once:
                await h.loop()
            for c, v, outcome in await h.poll():
                typer.echo(f"{v:8.1f} v/min {c['view_count']:7d} views  {c['broadcaster_name'][:18]:<18} "
                           f"{c.get('language', ''):<3} {c['title'][:40]:<40} -> {outcome}")
        finally:
            await h.api.close()
    _run(main())


@app.command("run")
def run_all(no_live: bool = False, no_harvest: bool = False, no_vod: bool = False, no_web: bool = False,
            no_bot: bool = False, port: int = 8000) -> None:
    """Alias von 'vc start' (mit Schaltern zum Abschalten einzelner Teile)."""
    _start(False, no_live, no_harvest, no_vod, no_web, no_bot, port)


def _start(standby: bool, no_live: bool = False, no_harvest: bool = False, no_vod: bool = False, no_web: bool = False,
           no_bot: bool = False, port: int = 8000) -> None:
    from vc.control import lock_holder, serve
    from vc.orchestrator import Orchestrator
    from vc.config import NO_STREAMERS_MSG, env_streamer_errors, get_streamers
    if (pid := lock_holder()) is not None:
        typer.echo(f"vc läuft bereits (PID {pid}). Beenden mit: uv run vc stop")
        raise typer.Exit(1)
    for e in env_streamer_errors():
        typer.echo(f"STREAMERS: {e} (übersprungen)")
    if not standby and not get_streamers():
        typer.echo(NO_STREAMERS_MSG)
        raise typer.Exit(1)
    orch = Orchestrator(live=not no_live, harvest=not no_harvest, vod=not no_vod, web=not no_web, bot=not no_bot, port=port)
    try:
        _run(serve(orch, standby=standby))
    except KeyboardInterrupt:  # Ctrl+C: serve() has already shut down the same way as vc stop
        pass
    except RuntimeError as e:
        typer.echo(str(e))
        raise typer.Exit(1)
    typer.echo("gestoppt")


@app.command()
def start(standby: bool = typer.Option(False, help="nur der Telegram-Bot laeuft, die Pipeline wartet auf /start")) -> None:
    """Startet alles in einem Prozess: Bot, Dashboard und Pipeline (nur eine Instanz). Stoppen: vc stop, /stop oder Strg+C.

    \b
    Autostart unter Windows (Aufgabenplanung, einmalig von Hand):
      1. Aufgabenplanung oeffnen -> Einfache Aufgabe erstellen, Name "vice-clipper"
      2. Trigger: Bei Anmeldung
      3. Aktion: Programm starten
         Programm:      uv
         Argumente:     run vc start --standby
         Starten in:    C:\\pfad\\zu\\vice-clipper
      4. Fertig. Nach dem Anmelden laeuft der Bot, die Pipeline startet mit /start in Telegram.
    """
    _start(standby)


@app.command()
def status() -> None:
    """Zeigt, ob die Pipeline laeuft, wer live ist, das 15-Minuten-Fenster, die Warteschlange und ob Telegram/Ollama antworten."""
    from datetime import datetime, timezone

    from vc import db
    from vc.config import get_config, get_secrets, get_streamers
    from vc.control import lock_holder
    from vc.models import ClipStatus

    async def main() -> None:
        await db.init_db()
        cfg = get_config()
        pid = lock_holder()
        typer.echo(f"Pipeline:  {'läuft (PID ' + str(pid) + ')' if pid else 'nicht gestartet -> vc start'}")
        streamers = get_streamers()
        live: dict[str, str] = {}
        if get_secrets().twitch_client_id and any(s.platform == "twitch" for s in streamers.values()):
            from vc.sources.twitch_api import TwitchAPI
            api = TwitchAPI()
            try:
                tw = [s for s in streamers.values() if s.platform == "twitch"]
                for x in await asyncio.wait_for(api.get_streams([s.login for s in tw]), 15):
                    live[x["user_login"]] = x.get("game_name", "")
            except Exception as e:  # noqa: BLE001
                typer.echo(f"Twitch:    Abfrage fehlgeschlagen ({str(e)[:80]})")
            finally:
                await api.close()
        for s in streamers.values():
            state = f"live · {live[s.login]}" if s.login in live else "offline"
            typer.echo(f"  {s.login:<16} {s.platform:<7} {state}")
        if pid and (ws := await db.kv_get("window_start")):
            start = datetime.fromisoformat(ws)
            minute = int((datetime.now(timezone.utc) - start).total_seconds() // 60) + 1
            typer.echo(f"Fenster:   Minute {minute}/{cfg.delivery.window_minutes:g} · {await db.kv_get('window_delivered', '0')} geliefert")
        counts = {st.value: len(await db.list_clips(st, limit=5000)) for st in (ClipStatus.NEW, ClipStatus.JUDGED, ClipStatus.RENDERED)}
        delivered = len(await db.list_events("delivered", since=db.local_midnight_utc(), limit=100000))
        typer.echo(f"Queue:     NEW {counts['NEW']} · JUDGED {counts['JUDGED']} · RENDERED {counts['RENDERED']} · heute geliefert {delivered}")
        from vc.setup import check_telegram
        try:
            name = await asyncio.wait_for(check_telegram(get_secrets().telegram_bot_token), 12)
            typer.echo(f"Telegram:  antwortet (@{name})")
        except Exception:  # noqa: BLE001
            typer.echo("Telegram:  antwortet NICHT - Befehle und Clips kommen erst an, wenn die Verbindung wieder geht"
                       " (Internet / Virenscanner-HTTPS-Prüfung prüfen)")
        if cfg.judge.provider == "ollama":
            from vc.judge.local import check_ollama
            try:
                typer.echo(f"Ollama:    {await asyncio.wait_for(check_ollama(), 10)}")
            except Exception as e:  # noqa: BLE001
                typer.echo(f"Ollama:    nicht erreichbar ({str(e)[:80]}) - Ollama starten")

    _run(main())


@app.command()
def stop() -> None:
    """Haelt die laufende Instanz sauber an (laufender Render wird fertig, Recorder stoppen). Wartet bis zu 60 s."""
    from vc.control import request_stop
    log.setup()
    res = asyncio.run(request_stop())
    if res is None:
        typer.echo("vc läuft nicht.")
    elif res:
        typer.echo("gestoppt.")
    else:
        typer.echo("Keine Bestätigung nach 60 s - Prozess prüfen (Lock: data/vc.lock).")
        raise typer.Exit(1)


# ---------------------------------------------------------------- chapter 7
def _streamer_or_adhoc(sid: str):
    from vc.config import get_streamer
    from vc.models import Streamer
    return get_streamer(sid) or Streamer(id=sid.lower(), login=sid.lower(), credit=f"twitch.tv/{sid.lower()}")


@app.command()
def record(streamer: str) -> None:
    """Record one channel into the ring buffer (foreground)."""
    from vc.db import init_db
    from vc.sources.recorder import Recorder, list_segments

    async def main() -> None:
        await init_db()
        rec = Recorder(_streamer_or_adhoc(streamer))
        rec.start()
        try:
            while True:
                await asyncio.sleep(10)
                segs = list_segments(rec.dir)
                size = sum(p.stat().st_size for _, p in segs if p.exists()) / 1e6
                typer.echo(f"{len(segs)} Segmente, {size:.0f} MB, neuestes {segs[-1][0]:%H:%M:%S}" if segs else "warte auf Segmente …")
        finally:
            await rec.stop()
    try:
        _run(main())
    except KeyboardInterrupt:
        pass


@app.command()
def cut(streamer: str, last: float = 30, out: Path = typer.Option(...)) -> None:
    """Cut the last N seconds from a channel's buffer."""
    from datetime import datetime, timedelta
    from vc.sources.recorder import cut as do_cut
    now = datetime.now()
    p, first = _run(do_cut(streamer, now - timedelta(seconds=last), now, out))
    typer.echo(f"{p} (erstes Bild {first:%H:%M:%S})")


# ---------------------------------------------------------------- chapter 8
@app.command()
def chat(channel: list[str] = typer.Option(..., help="mehrfach angebbar"), platform: str = "twitch", seconds: int = 0) -> None:
    """Read chat anonymously; prints messages per second every 5 s."""
    import time
    from vc.db import init_db
    from vc.sources.chat_kick import KickChat
    from vc.sources.chat_twitch import ChatHub, TwitchChat

    async def main() -> None:
        await init_db()
        hub = ChatHub()
        if platform == "kick":
            client = KickChat(hub)
            for c in channel:
                await client.join(c)
        else:
            client = TwitchChat(hub)
            for c in channel:
                client.join(c)
        task = asyncio.create_task(client.run())
        t0 = time.time()
        while not seconds or time.time() - t0 < seconds:
            await asyncio.sleep(5)
            typer.echo("  ".join(f"{c}: {len(hub.recent(c, 5)) / 5:.1f} msg/s" for c in channel))
        task.cancel()
    _run(main())


# ---------------------------------------------------------------- chapter 9
@app.command()
def replay(chatlog: Path = typer.Option(..., exists=True), speed: float = 0) -> None:
    """Run a chat log through the trigger logic with a simulated clock and print the triggers with V2 breakdown."""
    log.setup()
    from vc.detect.trigger import fmt_ts, replay as rp
    from vc.detect.virality import breakdown, summary_line
    timing: dict = {}
    trs = rp(chatlog, speed, timing)
    for t in trs:
        v = t["virality"]
        flag = "  [clip]" if t["clipcall"] else ""
        typer.echo(f"\n{fmt_ts(t['t'])} {t['channel']} · pre {t['pre_score']:.1f} · {summary_line(v, t.get('reaction_label', ''))}"
                   f" · Top: {' '.join(t['top'])}{flag}")
        raw = {"z": t["z"], "reaction_lift": t.get("reaction_lift"), "reaction_label": t.get("reaction_label"),
               "conv_lift": t.get("conv_lift"), "intensity_lift": t.get("intensity_lift")}
        for k, r, n, w, c in breakdown(v, raw):
            if n != "–":
                typer.echo(f"  {k:<12} roh {r[:30]:<30} norm {n:>5} gew {w:>5} beitrag {c:>5}")
    if trs:
        hours = max((trs[-1]["t"] - trs[0]["t"]) / 3600, 1 / 60)
        typer.echo(f"\n{len(trs)} Ausloeser, ~{len(trs) / hours:.1f} pro Stunde (Ziel 5-15)")
    else:
        typer.echo("keine Ausloeser")
    if timing:
        typer.echo(f"evaluate(): {timing['avg_ms']:.3f} ms Mittel, {timing['p99_ms']:.3f} ms p99, {timing['max_ms']:.3f} ms max "
                   f"je Kanal und Takt ({timing['ticks']} Takte)")


@app.command()
def guide(send: bool = typer.Option(False, "--send", help="per Telegram an den eigenen Chat schicken")) -> None:
    """Bedien-Anleitung (dieselbe wie /help in Telegram)."""
    from vc.deliver.guide import plain, usage_text
    text = usage_text()
    if not send:
        typer.echo(plain(text))
        return

    async def main() -> None:
        from aiogram import Bot
        from aiogram.client.default import DefaultBotProperties
        from vc.config import get_secrets
        s = get_secrets()
        s.require("telegram_bot_token", "telegram_chat_id")
        bot = Bot(s.telegram_bot_token, default=DefaultBotProperties(parse_mode="HTML"))
        try:
            await asyncio.wait_for(bot.send_message(int(s.telegram_chat_id), text), 30)
        finally:
            await bot.session.close()
        typer.echo("Anleitung per Telegram geschickt.")
    _run(main())


@app.command()
def tune(collect_only: bool = typer.Option(False, "--collect-only", help="nur Chat-Cache fuellen (VODs verfallen)"),
         apply: bool = typer.Option(False, "--apply", help="Gewichte in config.yaml schreiben, wenn die Daten reichen"),
         no_vods: bool = typer.Option(False, "--no-vods", help="nur eigene Chatlogs, kein VOD-Chat")) -> None:
    """Viralitaets-Gewichte gegen echte Twitch-Clips pruefen (NDCG@3 je 15-min-Fenster, zeitliche Trennung)."""
    import time as _time
    from vc import db
    from vc.config import get_config, set_config_value
    from vc.detect import tune as tn
    from vc.sources import tune_data

    async def main() -> None:
        await db.init_db()
        moments, counts = await tune_data.collect(use_vods=not no_vods)
        if collect_only:
            typer.echo(f"Cache gefuellt: {counts['moments']} Momente ({counts['positives']} Clips) in {get_config().data / 'tune'}")
            return
        text, block = tn.report(moments, counts, get_config().virality, _time.time(), await tune_data.decisions())
        typer.echo(text)
        (get_config().data / "tune" / "report.txt").write_text(text, encoding="utf-8")
        if apply and block:
            for k, v in block.items():
                set_config_value(k, v)
            typer.echo("config.yaml aktualisiert.")
        elif apply:
            typer.echo("--apply: nichts geschrieben, die Daten reichen nicht (siehe oben).")

    log.setup()
    _run(main())


@app.command()
def triggers() -> None:
    """Live: recorders + chat + trigger engine for all streamers; prints snapshots and creates candidates."""
    from vc.db import init_db
    from vc.orchestrator import LiveStack

    async def main() -> None:
        await init_db()
        stack = LiveStack()
        tasks = [asyncio.create_task(f(), name=n) for n, f in stack.tasks().items()]
        try:
            while True:
                await asyncio.sleep(5)
                for sid, st in stack.engine.ch.items():
                    s = st.snap
                    typer.echo(f"{sid:<14} burst {s.get('burst', 0):>3} base {s.get('baseline', 0):>5.1f} "
                               f"z {s.get('z', 0):>5.1f}/{s.get('zt', 0):.1f} uniq {s.get('unique', 0):.2f} "
                               f"lex {s.get('lex', 0):.2f} clip {s.get('clipcalls', 0)}")
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    try:
        _run(main())
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------- chapter 10
@app.command()
def scout(url: str, dry_run: bool = False) -> None:
    """Scout a YouTube video or Twitch VOD for moments (--dry-run: table only, no download)."""
    from vc.db import init_db
    from vc.sources.vod_scout import hms, scout as do_scout

    async def main():
        await init_db()

        async def say(m: str) -> None:
            typer.echo(m)
        return await do_scout(url, dry_run=dry_run, notify=say)
    moments = _run(main())
    typer.echo(f"\n{'Start':<9} {'Ende':<9} {'Hype':>4}  {'Kategorie':<13} {'Cam':<4} Zusammenfassung")
    for m in moments:
        typer.echo(f"{hms(m.start_s):<9} {hms(m.end_s):<9} {m.hype:>4}  {m.category:<13} {'ja' if m.has_facecam else '-':<4} {m.summary[:70]}")


# ---------------------------------------------------------------- chapter 11
@app.command()
def web(port: int = 8000) -> None:
    """Dashboard on http://127.0.0.1:PORT (DB only; with 'vc run' it also shows live channels)."""
    from vc.db import init_db
    from vc.web.app import create_app, serve

    async def main() -> None:
        await init_db()
        typer.echo(f"http://127.0.0.1:{port}")
        await serve(create_app(), port)
    _run(main())
