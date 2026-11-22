"""Per-channel chat statistics: burst vs robust baseline (median/MAD), unique ratio, lexicon share, clip calls."""
from __future__ import annotations

import re
import statistics
from collections import Counter, deque
from dataclasses import dataclass

from vc.detect.chat_features import Classifier, Feat, FeatureWindow

TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
BASELINE_GAP_S = 10  # baseline excludes the last 10 s


class Lexicon:
    """Single words: whole-token match (case-insensitive). Multi-word / emoji entries: substring match."""

    def __init__(self, entries: list[str]) -> None:
        self.words = {e.lower() for e in entries if TOKEN_RE.fullmatch(e)}
        self.phrases = [e.lower() for e in entries if not TOKEN_RE.fullmatch(e)]

    def hit(self, text: str) -> bool:
        low = text.lower()
        return any(t in self.words for t in TOKEN_RE.findall(low)) or any(p in low for p in self.phrases)


class Reactions:
    """Laugh / skull signal. A message counts once, however many emojis it holds (spam guard)."""

    def __init__(self, emojis: list[str], emotes: list[str], patterns: list[str]) -> None:
        self.emojis = list(emojis)
        self.emotes = {e.lower() for e in emotes}
        self.patterns = [re.compile(p) for p in patterns]

    def hit(self, text: str) -> bool:
        return (any(e in text for e in self.emojis) or any(t in self.emotes for t in TOKEN_RE.findall(text.lower()))
                or any(p.search(text) for p in self.patterns))


@dataclass(slots=True)
class _Msg:
    ts: float
    user: str
    text: str
    lex: bool
    clip: bool
    react: bool = False
    feat: Feat | None = None


class ChannelStats:
    def __init__(self, lexicon: Lexicon, burst_s: int = 5, baseline_s: int = 180, reactions: Reactions | None = None,
                 reaction_base_min: float = 0.02, features: Classifier | None = None, min_users: int = 4,
                 conv_min_users: int = 5) -> None:
        self.lex = lexicon
        self.features = features          # V2: per-message features; None = legacy behaviour (old tests, tools)
        self.min_users, self.conv_min_users = min_users, conv_min_users
        self.channel_emotes: frozenset[str] = frozenset()  # 7TV / BTTV / FFZ names, loaded once per channel
        self.fw = FeatureWindow(baseline_s + BASELINE_GAP_S + burst_s)
        self.reactions = reactions
        self.reaction_base_min = reaction_base_min
        self.burst_s, self.baseline_s = burst_s, baseline_s
        self.recent: deque[_Msg] = deque()            # last 10 s of messages
        self.counts: deque[list[int]] = deque()       # [second, n, n_reaction] for the baseline window
        self.first_ts: float | None = None

    def add(self, ts: float, user: str, text: str, emotes: frozenset[str] = frozenset()) -> None:
        """O(1) amortised per message. With features, bots / commands / giveaway waves are dropped here."""
        f = self.features.classify(user, text, emotes, self.channel_emotes) if self.features else None
        if f is not None and f.ignored:
            return
        if self.first_ts is None:
            self.first_ts = ts
        low = text.lower()
        react = bool(self.reactions and self.reactions.hit(text))
        self.recent.append(_Msg(ts, user, low, self.lex.hit(text), f.clipcall if f else "clip" in low, react, f))
        if f is not None:
            self.fw.add(ts, f)
        sec = int(ts)
        if self.counts and self.counts[-1][0] == sec:
            self.counts[-1][1] += 1
            self.counts[-1][2] += react
        else:
            self.counts.append([sec, 1, int(react)])
        self._trim(ts)

    def _trim(self, now: float) -> None:
        while self.recent and self.recent[0].ts < now - 10:
            self.recent.popleft()
        keep = int(now) - self.baseline_s - BASELINE_GAP_S - self.burst_s
        while self.counts and self.counts[0][0] < keep:
            self.counts.popleft()
        self.fw.trim(now)

    def snapshot(self, now: float) -> dict:
        self._trim(now)
        burst_msgs = [m for m in self.recent if now - self.burst_s <= m.ts < now]
        burst = len(burst_msgs)
        # copypasta / sub trains: identical texts without a hype word count once
        effective = sum(m.lex for m in burst_msgs) + len({m.text for m in burst_msgs if not m.lex})
        per_sec = {c[0]: c[1] for c in self.counts}
        end_hi = int(now) - BASELINE_GAP_S
        # reactions: burst share = UNIQUE reacting users / burst messages (2 spammers stay small),
        # baseline share = reaction messages / messages over the baseline seconds, floored
        base_rows = [c for c in self.counts if end_hi - self.baseline_s < c[0] <= end_hi]
        base_n = sum(c[1] for c in base_rows)
        react_base = max(self.reaction_base_min, sum(c[2] for c in base_rows) / base_n if base_n else 0.0)
        react_share = len({m.user for m in burst_msgs if m.react}) / burst if burst else 0.0
        sums = [sum(per_sec.get(e - k, 0) for k in range(self.burst_s))
                for e in range(end_hi - self.baseline_s + self.burst_s, end_hi + 1)]
        baseline = statistics.median(sums) if sums else 0.0
        mad = statistics.median(abs(x - baseline) for x in sums) if sums else 0.0
        z = (effective - baseline) / (1.4826 * mad + 1)
        v2 = {}
        if self.features is not None:
            v2 = self.fw.snapshot([(m.user, m.feat) for m in burst_msgs], [m.feat for m in self.recent],
                                  end_hi - self.baseline_s, end_hi, self.reaction_base_min, self.min_users,
                                  self.conv_min_users)
        return {**v2,
            "burst": burst,
            "effective": effective,
            "baseline": baseline,
            "mad": mad,
            "z": round(z, 2),
            "unique": round(len({m.user for m in burst_msgs}) / burst, 2) if burst else 0.0,
            "lex": round(sum(m.lex for m in burst_msgs) / burst, 2) if burst else 0.0,
            # V2: distinct users; legacy: messages
            "clipcalls": len({m.user for m in self.recent if m.clip}) if self.features else sum(m.clip for m in self.recent),
            "react_share": round(react_share, 3),
            "react_base": round(react_base, 3),
            "react_lift": round(react_share / react_base, 2),
        }

    def top_tokens(self, now: float, n: int = 5) -> list[str]:
        c = Counter(t for m in self.recent if now - self.burst_s <= m.ts < now for t in TOKEN_RE.findall(m.text))
        return [t for t, _ in c.most_common(n)]


def stats_from_config(lang: str = "en") -> ChannelStats:
    """ChannelStats with the V2 classifier, as the live trigger engine builds it."""
    from vc.config import get_config, lexicon_for
    cfg = get_config()
    c, v = cfg.trigger, cfg.virality
    fx = Classifier(v.categories, cfg.chat.ignore_users, cfg.chat.ignore_patterns, v.clipcall_phrases)
    return ChannelStats(Lexicon(lexicon_for(lang)), c.burst_window_s, c.baseline_window_s,
                        Reactions(v.reaction_emojis, v.reaction_emotes, v.reaction_patterns), v.reaction_base_min,
                        fx, v.reaction_min_users, v.convergence_min_users)


def features_at(msgs: list[tuple[float, str, str]], t: float, lang: str = "en", search: tuple[int, int] = (-45, 5),
                min_history_s: float = 120, base_min: float | None = None) -> dict | None:
    """Chat snapshot of a moment known only by its time (viewer clip, tuning): replay the log and keep the second
    with the highest z in [t+search]. None if the log does not reach min_history_s before the moment."""
    msgs = sorted(m for m in msgs if m[0] <= t + search[1])
    if not msgs or msgs[0][0] > t + search[0] - min_history_s:
        return None
    st = stats_from_config(lang)
    if base_min is not None:  # tuning grid over the baseline floor
        st.reaction_base_min = base_min
    best: dict | None = None
    i = 0
    for sec in range(int(t + search[0]), int(t + search[1]) + 1):
        while i < len(msgs) and msgs[i][0] < sec:
            st.add(*msgs[i][:3])
            i += 1
        snap = st.snapshot(sec)
        if best is None or snap["z"] > best["z"]:
            best = {**snap, "t": sec}
    return best
