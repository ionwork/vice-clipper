"""Twitch Helix client with app access token (client credentials)."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from vc import log
from vc.config import get_secrets

L = log.get("twitch")
HELIX = "https://api.twitch.tv/helix"


def rfc3339(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TwitchAPI:
    def __init__(self) -> None:
        s = get_secrets()
        s.require("twitch_client_id", "twitch_client_secret")
        self.cid, self.secret = s.twitch_client_id, s.twitch_client_secret
        self.http = httpx.AsyncClient(timeout=20)
        self.token: str | None = None
        self._game_ids: dict[str, str] = {}
        self._game_names: dict[str, str] = {}
        self._user_ids: dict[str, str] = {}

    async def close(self) -> None:
        await self.http.aclose()

    async def _auth(self) -> None:
        r = await self.http.post("https://id.twitch.tv/oauth2/token",
                                 data={"client_id": self.cid, "client_secret": self.secret, "grant_type": "client_credentials"})
        r.raise_for_status()
        self.token = r.json()["access_token"]

    async def get(self, path: str, params: Any) -> dict:
        for attempt in range(3):
            if not self.token:
                await self._auth()
            r = await self.http.get(f"{HELIX}/{path}", params=params,
                                    headers={"Client-Id": self.cid, "Authorization": f"Bearer {self.token}"})
            if r.status_code == 401:
                self.token = None
                continue
            if r.status_code == 429 or r.headers.get("Ratelimit-Remaining") == "0":
                reset = float(r.headers.get("Ratelimit-Reset", time.time() + 5))
                await asyncio.sleep(max(1.0, reset - time.time()))
                if r.status_code == 429:
                    continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Twitch {path}: Authentifizierung fehlgeschlagen")

    async def get_game_id(self, name: str) -> str | None:
        """Category names drift; try a few spellings and log which one hit."""
        if name in self._game_ids:
            return self._game_ids[name]
        variants = [name]
        if "VI" in name or "6" in name:
            variants += ["Grand Theft Auto VI", "Grand Theft Auto 6", "GTA VI", "GTA 6"]
        for v in dict.fromkeys(variants):
            data = (await self.get("games", {"name": v})).get("data") or []
            if data:
                L.info("Kategorie '%s' -> '%s' (id %s)", name, data[0]["name"], data[0]["id"])
                self._game_ids[name] = data[0]["id"]
                return data[0]["id"]
        L.warning("Kategorie '%s' nicht gefunden (versucht: %s)", name, variants)
        return None

    async def game_name(self, gid: str) -> str:
        """Category name for a game_id ('' if unknown or Twitch is unreachable)."""
        if not gid:
            return ""
        if gid not in self._game_names:
            try:
                data = (await self.get("games", {"id": gid})).get("data") or []
            except Exception as e:  # noqa: BLE001
                L.warning("Kategorie %s nicht abrufbar: %s", gid, str(e)[:100])
                return ""
            self._game_names[gid] = data[0]["name"] if data else ""
        return self._game_names[gid]

    async def get_users(self, logins: list[str]) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(logins), 100):
            out += (await self.get("users", [("login", lg) for lg in logins[i:i + 100]])).get("data", [])
        for u in out:
            self._user_ids[u["login"]] = u["id"]
        return out

    async def logins_for_ids(self, ids: list[str]) -> dict[str, str]:
        """broadcaster_id -> login (clip payloads only carry the display name)."""
        out: dict[str, str] = {}
        for i in range(0, len(ids), 100):
            for u in (await self.get("users", [("id", x) for x in ids[i:i + 100]])).get("data", []):
                out[u["id"]] = u["login"]
                self._user_ids[u["login"]] = u["id"]
        return out

    async def user_ids(self, logins: list[str]) -> dict[str, str]:
        missing = [lg for lg in logins if lg not in self._user_ids]
        if missing:
            await self.get_users(missing)
        return {lg: self._user_ids[lg] for lg in logins if lg in self._user_ids}

    async def get_streams(self, logins: list[str]) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(logins), 100):
            out += (await self.get("streams", [("user_login", lg) for lg in logins[i:i + 100]] + [("first", 100)])).get("data", [])
        return out

    async def get_clips(self, *, game_id: str | None = None, broadcaster_id: str | None = None,
                        started_at: datetime, ended_at: datetime, first: int = 100) -> list[dict]:
        p: dict[str, Any] = {"started_at": rfc3339(started_at), "ended_at": rfc3339(ended_at), "first": first}
        p["game_id" if game_id else "broadcaster_id"] = game_id or broadcaster_id
        return (await self.get("clips", p)).get("data", [])
