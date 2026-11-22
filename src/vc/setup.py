"""Setup helpers: live key checks, Telegram chat-id discovery, .env writer, launch switch."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import httpx

from vc.config import ENV_PATH, get_config, set_config_value

GEMINI = "https://generativelanguage.googleapis.com/v1beta/models"
GTA5, GTA6 = "Grand Theft Auto V", "Grand Theft Auto VI"


def write_env(updates: dict[str, str], path: Path = ENV_PATH) -> None:
    """Set KEY=value lines, keep every other line (comments, unknown keys) as is."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    todo = {k.upper(): v for k, v in updates.items()}
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if key in todo:
            lines[i] = f"{key}={todo.pop(key)}"
    lines += [f"{k}={v}" for k, v in todo.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def check_twitch(cid: str, secret: str) -> str:
    async with httpx.AsyncClient(timeout=15) as h:
        r = await h.post("https://id.twitch.tv/oauth2/token",
                         data={"client_id": cid, "client_secret": secret, "grant_type": "client_credentials"})
    if r.status_code != 200:
        raise RuntimeError(f"Twitch lehnt ab ({r.status_code}): Client-ID/Secret pruefen")
    return "App-Token erhalten"


async def check_telegram(token: str) -> str:
    async with httpx.AsyncClient(timeout=15) as h:
        r = await h.get(f"https://api.telegram.org/bot{token}/getMe")
    if r.status_code != 200 or not r.json().get("ok"):
        raise RuntimeError("Telegram lehnt den Bot-Token ab")
    return r.json()["result"]["username"]


async def find_chat_id(token: str, wait_s: int = 180) -> str | None:
    """Waits for any private message to the bot and returns its chat id."""
    async with httpx.AsyncClient(timeout=40) as h:
        loop = asyncio.get_running_loop()
        end = loop.time() + wait_s
        while loop.time() < end:
            r = await h.get(f"https://api.telegram.org/bot{token}/getUpdates", params={"timeout": 25})
            for u in r.json().get("result", []):
                chat = (u.get("message") or {}).get("chat") or {}
                if chat.get("type") == "private":
                    return str(chat["id"])
    return None


def best_flash_lite(names: list[str]) -> str | None:
    """Newest stable gemini-X.Y-flash-lite from the model list."""
    def ver(n: str) -> float:
        m = re.search(r"gemini-(\d+(?:\.\d+)?)-flash-lite$", n)
        return float(m.group(1)) if m else -1
    good = [n for n in names if ver(n) >= 0]
    return max(good, key=ver) if good else None


async def check_gemini(key: str, model: str) -> tuple[bool, str | None]:
    """(configured model available?, best flash-lite alternative)."""
    async with httpx.AsyncClient(timeout=15) as h:
        r = await h.get(GEMINI, params={"pageSize": 1000}, headers={"x-goog-api-key": key})
    if r.status_code != 200:
        raise RuntimeError(f"Gemini lehnt den Schluessel ab ({r.status_code})")
    names = [m["name"].removeprefix("models/") for m in r.json().get("models", [])]
    return model in names, best_flash_lite(names)


async def online_checks() -> list[tuple[str, bool, str]]:
    """Live checks for doctor; skips keys that are missing."""
    from vc.config import get_secrets
    s, out = get_secrets(), []
    if s.twitch_client_id and s.twitch_client_secret:
        try:
            out.append(("Twitch Login", True, await check_twitch(s.twitch_client_id, s.twitch_client_secret)))
        except Exception as e:  # noqa: BLE001
            out.append(("Twitch Login", False, str(e)[:80]))
    if s.telegram_bot_token:
        try:
            out.append(("Telegram Bot", True, "@" + await check_telegram(s.telegram_bot_token)))
        except Exception as e:  # noqa: BLE001
            out.append(("Telegram Bot", False, str(e)[:80]))
    if get_config().judge.provider == "ollama":
        from vc.judge.local import check_ollama
        try:
            out.append(("Ollama (lokal)", True, await check_ollama()))
        except Exception as e:  # noqa: BLE001
            out.append(("Ollama (lokal)", False, str(e)[:100]))
    elif s.gemini_api_key:
        model = get_config().judge.model
        try:
            ok, alt = await check_gemini(s.gemini_api_key, model)
            out.append((f"Gemini {model}", ok, "verfuegbar" if ok else f"nicht gefunden, vorhanden: {alt} (vc setup stellt um)"))
        except Exception as e:  # noqa: BLE001
            out.append(("Gemini", False, str(e)[:80]))
    return out


async def launch(on: bool = True) -> str:
    """GTA 6 mode: only GTA VI material passes the judge, harvester watches the GTA VI category."""
    set_config_value("judge.allow_gta5", not on)
    set_config_value("harvest.game_names", [GTA6] if on else [GTA5])
    mode = "GTA-6-Modus an" if on else "Probe-Modus (GTA V) an"
    from vc.sources.twitch_api import TwitchAPI
    try:
        api = TwitchAPI()
    except SystemExit:
        return f"{mode}. Twitch-Kategorie nicht geprueft (Twitch-Schluessel fehlen)."
    try:
        gid = await api.get_game_id(GTA6 if on else GTA5)
    except Exception as e:  # noqa: BLE001
        return f"{mode}. Twitch-Kategorie nicht geprueft: {str(e)[:100]}"
    finally:
        await api.close()
    found = f"gefunden (id {gid})" if gid else "NICHT gefunden - Namen in config.yaml harvest.game_names pruefen"
    return f"{mode}. Twitch-Kategorie {found}"
