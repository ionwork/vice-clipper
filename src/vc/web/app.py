"""Local dashboard: channel tiles, live events, candidates, cam editor, settings. Bind 127.0.0.1 only."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from vc import bus, db
from vc.config import get_config, get_streamer, get_streamers, set_config_value, upsert_streamer
from vc.models import ClipStatus, Layout, Rect

HERE = Path(__file__).parent
SETTINGS = [  # (dotted key, label, type)
    ("trigger.z_threshold", "z-Schwelle", float), ("trigger.min_burst_msgs", "Min. Burst-Nachrichten", int),
    ("trigger.min_unique_ratio", "Min. Anteil Schreiber", float), ("trigger.min_pre_score", "Min. Vorab-Score", float),
    ("trigger.pre_roll_s", "Vorlauf s", int), ("trigger.post_roll_s", "Nachlauf s", int),
    ("judge.min_hype", "Min. Hype", int), ("judge.daily_budget_usd", "Tagesbudget $", float),
    ("delivery.min_gap_minutes", "Lieferabstand min", float), ("delivery.auto", "Auto-Modus", bool),
    ("delivery.paused", "Pause", bool), ("delivery.follow_slots", "Nur liefern, wenn Account posten darf", bool),
    ("delivery.campaign_bonus", "Kampagnen-Bonus", float), ("judge.allow_gta5", "GTA V zulassen (Probe)", bool),
    ("harvest.min_views", "Harvest: min. Views", int), ("harvest.only_whitelisted", "Harvest: nur eigene Streamer", bool),
]


def _get(dotted: str) -> Any:
    node: Any = get_config()
    for part in dotted.split("."):
        node = getattr(node, part)
    return node


def create_app(runtime: Any = None) -> FastAPI:
    """runtime: the Orchestrator (live stack, bot, heartbeat) or None for a DB-only dashboard."""
    app = FastAPI(title="vice-clipper", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    tpl = Jinja2Templates(directory=HERE / "templates")
    live = getattr(runtime, "live", None)

    def tiles() -> list[dict]:
        out = []
        recs = live.recorders.recorders if live else {}
        for s in sorted(get_streamers().values(), key=lambda s: (s.priority, s.id)):
            st = live.engine.ch.get(s.id) if live else None
            info = live.recorders.live_info.get(s.id, {}) if live else {}
            out.append({"s": s, "live": s.id in recs, "since": (info.get("started_at") or "")[11:16],
                        "z": (st.snap.get("z", 0) if st else 0), "spark": list(st.spark) if st else [],
                        "viewers": info.get("viewer_count")})
        return out

    async def header() -> dict:
        p, o, usd = await db.cost_today()
        hb = runtime.heartbeat() if runtime and hasattr(runtime, "heartbeat") else {}
        return {"usd": usd, "budget": get_config().judge.daily_budget_usd, "auto": get_config().delivery.auto,
                "paused": get_config().delivery.paused, "hb": hb, "runtime": runtime is not None}

    # ------------------------------------------------------------ pages
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return tpl.TemplateResponse(request, "index.html", {"h": await header(), "tiles": tiles()})

    @app.get("/partials/tiles", response_class=HTMLResponse)
    async def p_tiles(request: Request):
        return tpl.TemplateResponse(request, "_tiles.html", {"tiles": tiles()})

    @app.get("/partials/clips", response_class=HTMLResponse)
    async def p_clips(request: Request, status: str = ""):
        clips = await db.list_clips(status or None, limit=60)
        return tpl.TemplateResponse(request, "_clips.html", {"clips": clips, "layouts": [x.value for x in Layout]})

    @app.get("/partials/header", response_class=HTMLResponse)
    async def p_header(request: Request):
        return tpl.TemplateResponse(request, "_header.html", {"h": await header()})

    @app.get("/streamers", response_class=HTMLResponse)
    async def streamers(request: Request):
        return tpl.TemplateResponse(request, "streamers.html", {"streamers": list(get_streamers().values())})

    @app.get("/settings", response_class=HTMLResponse)
    async def settings(request: Request, saved: int = 0):
        rows = [(k, label, t, _get(k)) for k, label, t in SETTINGS]
        return tpl.TemplateResponse(request, "settings.html", {"rows": rows, "saved": saved})

    @app.post("/settings")
    async def save_settings(request: Request):
        form = await request.form()
        for k, _, t in SETTINGS:
            if t is bool:
                set_config_value(k, k in form)
            elif k in form and str(form[k]).strip():
                set_config_value(k, t(str(form[k]).replace(",", ".")))
        return RedirectResponse("/settings?saved=1", status_code=303)

    # ------------------------------------------------------------ media
    @app.get("/snap/{sid}.jpg")
    async def snap(sid: str):
        p = get_config().data / "buffer" / sid / "snapshot.jpg"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, headers={"Cache-Control": "no-store"})

    @app.get("/clips/{clip_id}/preview")
    async def preview(clip_id: str):
        c = await db.get_clip(clip_id)
        if not c:
            raise HTTPException(404)
        p = Path(c.out_path) if c.out_path and Path(c.out_path).exists() else Path(c.raw_path)
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="video/mp4")

    # ------------------------------------------------------------ api
    @app.post("/api/clip/{sid}")
    async def clip_now(sid: str):
        if not live:
            raise HTTPException(409, "Clip jetzt geht nur mit 'vc run'")
        if not get_streamer(sid):
            raise HTTPException(404)
        return {"clip_id": await live.engine.manual(sid, 30)}

    @app.get("/api/clips")
    async def api_clips(status: str | None = None):
        return [c.model_dump(mode="json") for c in await db.list_clips(status, limit=200)]

    @app.post("/api/clips/{clip_id}/{action}")
    async def clip_action(clip_id: str, action: str, layout: str | None = None):
        c = await db.get_clip(clip_id)
        if not c:
            raise HTTPException(404)
        if action == "reject":
            await db.update_clip(clip_id, status=ClipStatus.REJECTED, reason="Dashboard")
        elif action == "deliver":
            if c.status not in (ClipStatus.RENDERED, ClipStatus.DELIVERED, ClipStatus.POSTED):
                raise HTTPException(409, "noch nicht gerendert")
            bot = getattr(runtime, "bot", None)
            if bot:
                await bot.deliver(clip_id)
            else:
                from vc.deliver.telegram_bot import deliver_once
                await deliver_once(clip_id)
        elif action == "rerender":
            from vc.render.render import render_from_db
            if not c.judge:
                raise HTTPException(409, "noch nicht bewertet")
            asyncio.create_task(render_from_db(clip_id, Layout(layout) if layout else None))
        else:
            raise HTTPException(400)
        return {"ok": True}

    @app.post("/api/streamers/{sid}/cam")
    async def save_cam(sid: str, rect: Rect | None = None):
        if not (s := get_streamer(sid)):
            raise HTTPException(404)
        # .env streamers get a full yaml entry on their first cam frame
        upsert_streamer({"id": sid, "cam": rect.model_dump() if rect else None}, defaults=s.model_dump(mode="json"))
        return {"ok": True}

    @app.post("/api/streamers/{sid}/suggest")
    async def suggest(sid: str):
        from vc.sources.harvester import suggest_cam
        snapshot = get_config().data / "buffer" / sid / "snapshot.jpg"
        src, at = (snapshot, 0.0) if snapshot.exists() else (None, 2.0)
        if src is None:
            clips = [c for c in await db.list_clips(limit=200) if c.streamer_id == sid and Path(c.raw_path).exists()]
            src = Path(clips[0].raw_path) if clips else None
        if src is None:
            raise HTTPException(404, "kein Standbild und kein Clip vorhanden")
        rect = await asyncio.to_thread(suggest_cam, src, at)
        return {"rect": rect.model_dump() if rect else None}

    @app.get("/api/streamers/{sid}/frame.jpg")
    async def frame(sid: str):
        """Snapshot if live, else a frame from the newest clip of this streamer (for the cam editor)."""
        snapshot = get_config().data / "buffer" / sid / "snapshot.jpg"
        if snapshot.exists():
            return FileResponse(snapshot, headers={"Cache-Control": "no-store"})
        clips = [c for c in await db.list_clips(limit=200) if c.streamer_id == sid and Path(c.raw_path).exists()]
        if not clips:
            raise HTTPException(404)
        from vc.proc import run_ok
        from vc.render.layouts import NORM
        out = get_config().data / "tmp" / f"frame-{sid}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        await run_ok("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", "2", "-i", clips[0].raw_path,
                     "-frames:v", "1", "-vf", f"{NORM},scale=960:540", out)
        return FileResponse(out, headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------ live events
    @app.get("/events")
    async def events(request: Request):
        q = bus.subscribe()

        async def gen():
            try:
                for e in reversed(await db.list_events(limit=30)):
                    yield f"data: {json.dumps({'kind': e['kind'], 'clip_id': e['clip_id'], 'streamer_id': e['streamer_id'], 'ts': e['ts'], **e['data']}, default=str)}\n\n"
                while not await request.is_disconnected():
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=1)
                        yield f"data: {json.dumps(msg, default=str)}\n\n"
                    except asyncio.TimeoutError:
                        if live:
                            tick = {sid: {"z": st.snap.get("z", 0), "burst": st.snap.get("burst", 0)} for sid, st in live.engine.ch.items()}
                            yield f"event: tick\ndata: {json.dumps(tick)}\n\n"
            finally:
                bus.unsubscribe(q)
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    async def health():
        return JSONResponse({"ok": True})

    return app


async def serve(app: FastAPI, port: int = 8000) -> None:
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:  # stop the regular way instead of killing the lifespan (no traceback)
        server.should_exit = True
        await asyncio.gather(task, return_exceptions=True)
        raise
