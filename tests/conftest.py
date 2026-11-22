"""Isolated VC_ROOT per test session: copies of config/streamers, real assets, temp data dir."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
_tmp = Path(tempfile.mkdtemp(prefix="vc-test-"))
cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
cfg["paths"] = {"data": str(_tmp / "data"), "fonts": str(REPO / "assets/fonts"), "sfx": str(REPO / "assets/sfx")}
(_tmp / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
shutil.copy(REPO / "tests/fixtures/streamers.yaml", _tmp / "streamers.yaml")
(_tmp / ".env").write_text("STREAMERS=twitch:streamer_en:en\n", encoding="utf-8")  # .env is the only streamer list
os.environ.pop("STREAMERS", None)
os.environ["VC_ROOT"] = str(_tmp)
for k in ("TWITCH_CLIENT_ID", "TWITCH_CLIENT_SECRET", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "GEMINI_API_KEY"):
    os.environ.setdefault(k, "test")


@pytest.fixture(scope="session")
def vc_root() -> Path:
    return _tmp


@pytest.fixture
async def db():
    import asyncio
    from vc import db as dbm
    p = dbm.db_path()
    for _ in range(20):  # Windows: antivirus or a just-cancelled connection can hold the file for a moment
        try:
            p.unlink(missing_ok=True)
            for extra in (p.with_name(p.name + '-wal'), p.with_name(p.name + '-shm')):
                extra.unlink(missing_ok=True)
            break
        except PermissionError:
            await asyncio.sleep(0.05)
    await dbm.init_db()
    async with dbm.connect() as conn:  # empty tables even when the file could not be deleted (never a stale DB)
        async with conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'") as cur:
            tables = [r[0] for r in await cur.fetchall()]
        for t in tables:
            await conn.execute(f"DELETE FROM {t}")
        await conn.commit()
    return dbm


@pytest.fixture(scope="session")
def synth_video() -> Path:
    """1920x1080 30 s testsrc2 + sine audio."""
    from vc.proc import tool
    out = _tmp / "synth.mp4"
    if not out.exists():
        subprocess.run([tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=30",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=30:sample_rate=48000",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(out)], check=True)
    return out
