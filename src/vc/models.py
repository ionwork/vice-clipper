"""Shared enums and Pydantic models."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ClipStatus(StrEnum):
    NEW = "NEW"
    JUDGED = "JUDGED"
    CAPTIONED = "CAPTIONED"
    RENDERED = "RENDERED"
    DELIVERED = "DELIVERED"
    POSTED = "POSTED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class Source(StrEnum):
    LIVE = "live"
    HARVEST = "harvest"
    VOD = "vod"
    MANUAL = "manual"


class Layout(StrEnum):
    SPLIT = "split"
    GAMEPLAY = "gameplay"
    FULLCAM = "fullcam"
    IRL = "irl"  # real life: whole 16:9 frame centred, same stream blurred above and below

    def next(self) -> "Layout":
        order = list(Layout)
        return order[(order.index(self) + 1) % len(order)]


CATEGORIES = ["explosion", "police_chase", "crash", "laugh", "gunfight", "stunt", "fail", "glitch", "reaction", "other"]


class Rect(BaseModel):
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def parse(cls, s: str) -> "Rect":
        x, y, w, h = (int(v) for v in s.split(","))
        return cls(x=x, y=y, w=w, h=h)


class Streamer(BaseModel):
    id: str
    platform: str = "twitch"
    login: str
    lang: str = "en"
    quality: str = "1080p60,1080p,best"
    priority: int = 2
    account: str = "account_en"
    permission: str = "none"
    campaign_rules: str = ""
    credit: str = ""
    cam: Rect | None = None
    name: str = ""

    @property
    def display(self) -> str:
        return self.name or self.login


class Flags(BaseModel):
    cutscene: bool = False
    story_spoiler: bool = False
    violence_first_3s: bool = False
    music_dominant: bool = False
    needs_context: bool = False
    not_gta6: bool = False


class Hashtags(BaseModel):
    youtube: list[str] = []
    tiktok: list[str] = []
    instagram: list[str] = []


class JudgeResult(BaseModel):
    keep: bool
    hype: int  # 1-10; no ge/le here, Gemini's schema subset rejects them -> clamped in judge.validate
    category: str
    summary: str
    start_s: float
    end_s: float
    peak_s: float
    subject_x: float = 0.5
    layout: Layout = Layout.SPLIT
    hook: str = ""
    title: str = ""
    caption: str = ""
    hashtags: Hashtags = Hashtags()
    language: str = "en"
    flags: Flags = Flags()
    content: Literal["gameplay", "irl"] = "gameplay"  # old judge JSON has no content -> gameplay


class Candidate(BaseModel):
    """What a source hands to the pipeline."""
    source: Source
    streamer: Streamer
    raw_path: str
    t_trigger: datetime | None = None
    trigger_offset_s: float | None = None  # trigger position inside raw_path
    signals: dict[str, Any] = {}
    pre_score: float = 0
    chat_context: list[str] = []
    origin: str = ""  # clip url, vod url+offset ... for dedup and credit


class Clip(BaseModel):
    id: str
    source: Source
    streamer_id: str
    platform: str
    lang: str
    t_trigger: datetime | None = None
    trigger_offset_s: float | None = None
    signals: dict[str, Any] = {}
    pre_score: float = 0
    judge: dict[str, Any] | None = None
    hype: int | None = None
    status: ClipStatus = ClipStatus.NEW
    reason: str = ""
    raw_path: str = ""
    ass_path: str = ""
    out_path: str = ""
    text_path: str = ""
    account: str = ""
    layout: str = ""
    streamer: dict[str, Any] = {}
    chat_context: list[str] = []
    origin: str = ""
    tg_message_id: int | None = None
    created_at: datetime | None = None
    delivered_at: datetime | None = None
    posted_at: datetime | None = None

    @property
    def streamer_obj(self) -> Streamer:
        """Live config wins (cam edits apply), snapshot is the fallback for ad-hoc streamers."""
        from vc.config import get_streamer
        live = get_streamer(self.streamer_id)
        if live is None:
            return Streamer.model_validate(self.streamer)
        if live.cam is None and (cam := self.streamer.get("cam")):  # harvester's per-clip cam suggestion
            return live.model_copy(update={"cam": Rect.model_validate(cam)})
        return live

    @property
    def bonus(self) -> float:
        """Campaign streamers pay per view: their clips jump the queue."""
        from vc.config import get_config
        return get_config().delivery.campaign_bonus if self.streamer_obj.permission == "campaign" else 0.0

    @property
    def judge_result(self) -> JudgeResult | None:
        return JudgeResult.model_validate(self.judge) if self.judge else None
