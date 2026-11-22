"""Earliest next posting slot per account (min gap + daily cap)."""
from __future__ import annotations

from datetime import datetime, timedelta

from vc import db
from vc.config import AccountCfg, get_config


def next_slot_from(posted: list[datetime], cfg: AccountCfg, now: datetime, pending: list[datetime] = ()) -> datetime:
    """posted: posted_at times of this account (aware); pending: delivered but not yet marked posted
    (they hold the gap, not the daily cap). Pure function."""
    local_now = now.astimezone()
    today = [p for p in posted if p.astimezone().date() == local_now.date()]
    slot = now
    if posted or pending:
        slot = max(slot, max([*posted, *pending]) + timedelta(minutes=cfg.min_gap_minutes))
    if len(today) >= cfg.daily_cap:
        tomorrow = (local_now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        slot = max(slot, tomorrow)
    return slot


async def next_slot(account: str, now: datetime | None = None) -> datetime:
    cfg = get_config().accounts.get(account, AccountCfg())
    now = now or db.now()
    since = (now - timedelta(days=2)).isoformat()
    async with db.connect() as conn:
        async with conn.execute("SELECT posted_at FROM clips WHERE account = ? AND posted_at >= ?", (account, since)) as cur:
            posted = [datetime.fromisoformat(r[0]) for r in await cur.fetchall()]
        async with conn.execute("SELECT delivered_at FROM clips WHERE account = ? AND status = 'DELIVERED' AND delivered_at >= ?",
                                (account, since)) as cur:
            pending = [datetime.fromisoformat(r[0]) for r in await cur.fetchall()]
    return next_slot_from(posted, cfg, now, pending)
