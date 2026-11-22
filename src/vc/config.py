"""Load config.yaml, streamers.yaml and .env into Pydantic models. Reloads on file change."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from vc.models import Streamer

ROOT = Path(os.environ.get("VC_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_PATH = ROOT / "config.yaml"
STREAMERS_PATH = ROOT / "streamers.yaml"
ENV_PATH = ROOT / ".env"


class Paths(BaseModel):
    data: Path = Path("data")
    fonts: Path = Path("assets/fonts")
    sfx: Path = Path("assets/sfx")


class RecorderCfg(BaseModel):
    segment_seconds: int = 4
    buffer_seconds: int = 300
    max_parallel: int = 8


class TriggerCfg(BaseModel):
    burst_window_s: int = 5
    baseline_window_s: int = 180
    z_threshold: float = 4.0
    min_burst_msgs: int = 15
    min_unique_ratio: float = 0.6
    lexicon_ratio_bonus: float = 0.35
    clip_call_count: int = 3
    cooldown_s: int = 45
    pre_roll_s: int = 28
    post_roll_s: int = 10
    audio_spike_db: float = 8
    min_pre_score: float = 5.0


class ChatCfg(BaseModel):
    log: bool = True
    kick_enabled: bool = True
    kick_pusher_key: str = "34bf7a0ff419a2a775b9"
    kick_pusher_cluster: str = "us2"
    # bots and copy waves never count as viewers (lower-case logins / regex on the text)
    ignore_users: list[str] = ["nightbot", "streamelements", "moobot", "fossabot", "streamlabs", "soundalerts",
                               "wizebot", "sery_bot"]
    ignore_patterns: list[str] = [r"(?i)\b(giveaway|gewinnspiel|raid)\b"]


class JudgeCfg(BaseModel):
    provider: str = "ollama"            # ollama (local, free) | gemini (API, paid)
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma3:12b"    # must support vision
    ollama_num_ctx: int = 8192
    ollama_keep_alive: str = "15m"
    ollama_timeout_s: float = 600
    local_frames: int = 12
    model: str = "gemini-3.5-flash-lite"
    min_hype: int = 7
    proxy_height: int = 480
    proxy_fps: int = 10
    max_parallel: int = 2
    daily_budget_usd: float = 10
    price_in_per_m: float = 0.30
    price_out_per_m: float = 2.50
    allow_gta5: bool = False
    allow_irl: bool = True      # real-life scenes of the listed streamers are valid clips


class RenderCfg(BaseModel):
    width: int = 1080
    height: int = 1920
    split_cam_height: int = 768
    force_layout: str = ""      # "" = rule by content (irl -> irl, gameplay -> split), else split | gameplay | fullcam | irl
    irl_categories: list[str] = ["Just Chatting", "IRL", "Travel & Outdoors", "Food & Drink", "Music",
                                 "Talk Shows & Podcasts"]
    auto_cam: bool = True       # no cam frame -> detect the facecam per clip
    min_len_s: float = 10
    max_len_s: float = 25
    encoder: str = "auto"
    loudness_lufs: float = -14
    max_file_mb: float = 49


class CaptionsCfg(BaseModel):
    whisper_model: str = "large-v3-turbo"
    device: str = "auto"
    max_words: int = 3
    max_chars: int = 16
    highlight_color: str = "FFE500"
    censor: bool = False
    initial_prompt: str = ""


class HarvestCfg(BaseModel):
    game_names: list[str] = ["Grand Theft Auto VI"]
    poll_seconds: int = 180
    lookback_minutes: int = 90
    min_views: int = 150
    top_n: int = 5
    languages: list[str] = ["en", "de"]
    only_whitelisted: bool = False
    max_duration_s: float = 60
    account_by_lang: dict[str, str] = {}


class VodCfg(BaseModel):
    model: str = ""
    chunk_minutes: int = 40
    min_hype: int = 7
    max_moments: int = 12


class DeliveryCfg(BaseModel):
    min_gap_minutes: float = 4
    stale_after_minutes: float = 45
    auto: bool = False
    paused: bool = False
    send_as_video: bool = True      # False = as file (document)
    mode: str = "window"            # window = best 2-3 per window | gap = one clip every min_gap_minutes (old)
    window_minutes: float = 15
    per_window_min: int = 2
    per_window_max: int = 3
    third_clip_min_v: float = 55
    min_v: float = 35
    max_per_streamer: int = 2
    age_decay_per_5min: float = 1     # V points lost per 5 min a clip waited before this window
    render_boost_minute: float = 10   # from this minute on the best JUDGED (by V) render first
    render_boost_top: int = 3
    render_max: int = 5               # only the best N undelivered candidates by V get rendered (hotkey clips always)
    delete_losers: bool = True        # window close: not chosen -> REJECTED, media deleted, DB row kept for vc tune
    follow_slots: bool = True       # auto delivery only when the clip's account may post now
    campaign_bonus: float = 2       # extra score for streamers with permission "campaign" (paid per view)


class CategorySpec(BaseModel):
    words: list[str] = []    # case-insensitive whole words or phrases, de + en (stretched letters are squeezed)
    emotes: list[str] = []
    emojis: list[str] = []
    regex: list[str] = []


DEFAULT_CATEGORIES = {
    "laugh": CategorySpec(words=["lmao", "lol", "lul", "xd", "haha", "hahaha", "rofl"],
                          emotes=["LUL", "LULW", "KEKW", "OMEGALUL", "ICANT", "KEKL", "pepeLaugh"],
                          emojis=["😂", "🤣", "💀", "😭", "☠️"], regex=[r"(?i)\b(ha){2,}h?\b"]),
    "shock": CategorySpec(words=["wtf", "wth", "what", "no way", "huh", "bro", "was zum", "alter", "omg", "junge"],
                          emotes=["monkaS", "monkaW", "D:", "WAYTOODANK"], emojis=["😱", "😳", "🤯"]),
    "hype": CategorySpec(words=["pog", "poggers", "let's go", "lets go", "gg", "w", "sheesh", "insane"],
                         emotes=["PogChamp", "Pog", "POGGERS", "PogU", "EZ"], emojis=["🔥"]),
    "cringe": CategorySpec(words=["cringe", "yikes", "fremdschämen"], emotes=["Pepepains", "WeirdChamp", "monkaHmm"],
                           emojis=["😬"]),
    "confusion": CategorySpec(regex=[r"^[\s?]*\?[\s?]*$"]),   # the whole message is question marks
}
LEGACY_WEIGHTS = {"chat_spike": 0.22, "reactions": 0.18, "clip_calls": 0.15, "library": 0.20, "judge": 0.20, "audio": 0.05}
LEGACY_NAMES = {"chat_spike": "velocity", "reactions": "reaction", "audio": "voice"}


class ViralityCfg(BaseModel):
    weights: dict[str, float] = {"velocity": 0.20, "reaction": 0.18, "library": 0.18, "clip_calls": 0.13, "judge": 0.13,
                                 "convergence": 0.07, "voice": 0.07, "intensity": 0.04}
    legacy_weights: dict[str, float] = LEGACY_WEIGHTS  # V of clips scored before V2 (old component names)
    categories: dict[str, CategorySpec] = DEFAULT_CATEGORIES
    reaction_min_users: int = 4          # a category / intensity counts only with >= 4 distinct users in the burst
    convergence_min_users: int = 5
    log_span: float = 3.0                # convergence / intensity = clamp(log2(lift) / span)
    streamer_clipcall_users: int = 3     # the streamer saying "clip that" counts like 3 viewers
    voice_keywords: dict[str, list[str]] = {
        "de": ["oh mein gott", "hast du das gesehen", "clip das", "clippt das", "alter", "was zum"],
        "en": ["oh my god", "did you see that", "clip that", "clip it", "no way", "what the"]}
    voice_pace_full: float = 2.5         # speech rate jump (fastest 3 s / median) that counts as full
    voice_keyword_bonus: float = 0.2
    voice_silence_bonus: float = 0.1
    voice_silence_s: float = 2.5

    @field_validator("weights")
    @classmethod
    def _rename_legacy(cls, w: dict[str, float]) -> dict[str, float]:
        """An old config.yaml with chat_spike/reactions/audio weights keeps working (mapped to V2 names)."""
        if not any(k in w for k in LEGACY_NAMES):
            return w
        out = {LEGACY_NAMES.get(k, k): v for k, v in w.items()}
        return {**{"convergence": 0.07, "intensity": 0.04}, **out}
    z_floor: float = 3.0                 # chat_spike = clamp((z - z_floor) / z_span)
    z_span: float = 7.0
    reaction_emojis: list[str] = ["😂", "🤣", "💀", "☠️", "😭"]
    reaction_emotes: list[str] = ["KEKW", "LUL", "LULW", "OMEGALUL", "ICANT"]
    reaction_patterns: list[str] = [r"(?i)\b(ha){2,}h?\b", r"(?i)\bxd+\b"]
    reaction_base_min: float = 0.02      # floor of the baseline reaction share
    reaction_log_span: float = 3.0       # reactions = clamp(log2(lift) / span): lift 8x = 1.0
    clipcall_phrases: list[str] = ["clip", "clip it", "clip that", "clippt das", "clip das"]
    clipcall_window_s: tuple[float, float] = (-15, 30)
    clipcall_scale: float = 4.0          # clip_calls = 1 - exp(-n / scale)
    link_pattern: str = r"clips\.twitch\.tv/|twitch\.tv/\w+/clip/|kick\.com/\w+/clips?/"
    library_window_s: tuple[float, float] = (-20, 240)
    library_rate_days: int = 7
    library_expected_min: float = 0.25
    library_link_weight: float = 0.5
    library_log_span: float = 3.0
    velocity_quantile: float = 0.9
    velocity_min_clips: int = 5
    velocity_bonus: float = 0.15
    audio_full_db: float = 15.0          # audio = clamp(dB above baseline / full)
    agree_min_components: int = 3        # agreement bonus: >= n components >= level -> V * factor
    agree_level: float = 0.6
    agree_factor: float = 1.10
    needs_context_penalty: float = 15.0
    judge_queue_max: int = 6             # more NEW than this -> shed candidates with V < judge_min
    judge_min: float = 25.0


class AccountCfg(BaseModel):
    min_gap_minutes: float = 50
    daily_cap: int = 12


class Config(BaseModel):
    paths: Paths = Paths()
    recorder: RecorderCfg = RecorderCfg()
    trigger: TriggerCfg = TriggerCfg()
    lexicon: dict[str, list[str]] = {}
    chat: ChatCfg = ChatCfg()
    judge: JudgeCfg = JudgeCfg()
    render: RenderCfg = RenderCfg()
    captions: CaptionsCfg = CaptionsCfg()
    harvest: HarvestCfg = HarvestCfg()
    vod: VodCfg = VodCfg()
    delivery: DeliveryCfg = DeliveryCfg()
    virality: ViralityCfg = ViralityCfg()
    accounts: dict[str, AccountCfg] = Field(default_factory=dict)

    def path(self, name: str) -> Path:
        p = getattr(self.paths, name)
        return p if p.is_absolute() else ROOT / p

    @property
    def data(self) -> Path:
        return self.path("data")


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_PATH), env_file_encoding="utf-8", extra="ignore")
    twitch_client_id: str = ""
    twitch_client_secret: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    gemini_api_key: str = ""
    twitch_user_oauth: str = ""
    streamers: str = ""  # STREAMERS=login_eins, twitch:login_zwei:de, kick:login_drei:en:campaign

    def require(self, *names: str) -> None:
        missing = [n.upper() for n in names if not getattr(self, n)]
        if missing:
            raise SystemExit(f"Fehlende Schluessel in {ENV_PATH}: {', '.join(missing)}")


REQUIRED_SECRETS = ("twitch_client_id", "twitch_client_secret", "telegram_bot_token", "telegram_chat_id", "gemini_api_key")

_cache: dict[Path, tuple[float, Any]] = {}


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Datei fehlt: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cached(path: Path, build):
    """Rebuild when the file's mtime changes, so running components pick up edits."""
    mtime = path.stat().st_mtime if path.exists() else 0.0
    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        value = build(_load_yaml(path))
    except ValidationError as e:
        raise SystemExit(f"Ungueltige Konfiguration in {path.name}:\n{e}") from None
    _cache[path] = (mtime, value)
    return value


def get_config() -> Config:
    return _cached(CONFIG_PATH, lambda d: Config.model_validate(d))


PLATFORMS = ("twitch", "kick")
LANGS = ("de", "en")
PERMISSIONS = ("campaign", "public_ok", "asked", "none")
NO_STREAMERS_MSG = "Keine Streamer in .env (STREAMERS=…). Format: twitch:login:de"
_LOGIN_RE = re.compile(r"[a-z0-9_-]{2,25}")  # kick slugs may hold "-"


def parse_env_streamers(raw: str, account_by_lang: dict[str, str], errors: list[str] | None = None) -> dict[str, Streamer]:
    """'login_eins, twitch:login_zwei:de, kick:login_drei:en:campaign' -> streamers.
    Entry = [platform:]login[:lang[:permission]]; platform defaults to twitch, lang to en.
    Invalid entries are skipped and described in `errors` (name + reason); duplicates are taken once."""
    errors = [] if errors is None else errors
    out: dict[str, Streamer] = {}
    for item in raw.replace(";", ",").replace("\n", ",").split(","):
        parts = [p.strip() for p in item.strip().split(":")]
        if not any(parts):
            continue
        name = item.strip()
        if parts[0].lower() in PLATFORMS and len(parts) > 1:
            platform = parts.pop(0).lower()
        elif len(parts) >= 3 and parts[1].lower() not in LANGS:  # x:login:lang -> x must be a platform
            errors.append(f"{name}: Plattform '{parts[0]}' unbekannt (erlaubt: {', '.join(PLATFORMS)})")
            continue
        else:
            platform = "twitch"
        login = parts[0].lower().lstrip("@")
        lang = (parts[1].lower() if len(parts) > 1 else "") or "en"
        permission = (parts[2].lower() if len(parts) > 2 else "") or "none"
        if not _LOGIN_RE.fullmatch(login):
            errors.append(f"{name}: Login '{login}' ungueltig")
        elif lang not in LANGS:
            errors.append(f"{name}: Sprache '{lang}' ungueltig (erlaubt: {', '.join(LANGS)})")
        elif permission not in PERMISSIONS:
            errors.append(f"{name}: Erlaubnis '{permission}' ungueltig (erlaubt: {', '.join(PERMISSIONS)})")
        elif login in out:
            errors.append(f"{name}: doppelt, nur der erste Eintrag zaehlt")
        else:
            out[login] = Streamer(id=login, platform=platform, login=login, lang=lang, permission=permission,
                                  priority=1 if permission == "campaign" else 2, account=account_by_lang.get(lang, "account_other"),
                                  credit=f"{'twitch.tv' if platform == 'twitch' else 'kick.com'}/{login}")
    return out


_env_cache: tuple[float, str, dict[str, Streamer], list[str]] | None = None


def _env_parsed() -> tuple[dict[str, Streamer], list[str]]:
    """Re-parsed when .env changes (mtime): the recorder's next live check picks up edits without a restart."""
    global _env_cache
    mtime = ENV_PATH.stat().st_mtime if ENV_PATH.exists() else 0.0
    raw = get_secrets().streamers
    if _env_cache is None or _env_cache[0] != mtime or _env_cache[1] != raw:
        errors: list[str] = []
        streamers = parse_env_streamers(raw, get_config().harvest.account_by_lang, errors)
        if errors:
            from vc import log
            for e in errors:
                log.get("config").warning("STREAMERS: %s", e)
        _env_cache = (mtime, raw, streamers, errors)
    return _env_cache[2], _env_cache[3]


def env_streamer_errors() -> list[str]:
    return _env_parsed()[1]


# streamers.yaml may add these details for a .env streamer; login, platform, lang and permission come from .env
YAML_DETAILS = ("quality", "priority", "account", "campaign_rules", "credit", "cam", "name")


def get_streamers() -> dict[str, Streamer]:
    """STREAMERS in .env is the only list (empty -> nothing is tracked). streamers.yaml only adds details for the
    same ids (cam, credit, campaign rules, display name)."""
    yaml_s = _cached(STREAMERS_PATH, lambda d: {s["id"]: Streamer.model_validate(s) for s in d.get("streamers") or []})
    out: dict[str, Streamer] = {}
    for sid, s in _env_parsed()[0].items():
        y = yaml_s.get(sid)
        out[sid] = s.model_copy(update={k: getattr(y, k) for k in YAML_DETAILS if k in y.model_fields_set}) if y else s
    return out


def get_streamer(streamer_id: str) -> Streamer | None:
    return get_streamers().get(streamer_id)


def get_secrets() -> Secrets:
    return Secrets()


def lexicon_for(lang: str) -> list[str]:
    lx = get_config().lexicon
    return list(lx.get("common", [])) + list(lx.get(lang, []))


# ---------------------------------------------------------------- writers (keep comments and order)
def _rt():
    from ruamel.yaml import YAML
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    return y


def set_config_value(dotted: str, value: Any) -> None:
    """set_config_value('judge.min_hype', 8) -> rewrites config.yaml in place."""
    y = _rt()
    with CONFIG_PATH.open(encoding="utf-8") as f:
        doc = y.load(f)
    node = doc
    *parents, leaf = dotted.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value
    Config.model_validate(_plain(doc))  # refuse to write an invalid config
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        y.dump(doc, f)


def upsert_streamer(data: dict, defaults: dict | None = None) -> None:
    """Add a streamer or update fields of an existing one (matched by id).
    defaults fill a new entry, e.g. a .env streamer that gets its first cam frame."""
    y = _rt()
    with STREAMERS_PATH.open(encoding="utf-8") as f:
        doc = y.load(f) or {}
    lst = doc.setdefault("streamers", None) or []
    doc["streamers"] = lst
    for s in lst:
        if s["id"] == data["id"]:
            s.update(data)
            break
    else:
        lst.append({**(defaults or {}), **data})
    Streamer.model_validate(_plain(next(s for s in lst if s["id"] == data["id"])))
    with STREAMERS_PATH.open("w", encoding="utf-8") as f:
        y.dump(doc, f)


def _plain(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _plain(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_plain(v) for v in node]
    return node
