"""Chat features for virality V2 (pure, no I/O).

One pass per message classifies it (reaction categories, intensity, emote-only, normalised token, clip call,
ignored). FeatureWindow keeps per-second counters for the channel baseline and turns the last burst into lifts:
share of DISTINCT users with a feature in the burst / share of messages with it in the channel's own baseline.
So a chat that always writes "W" gets no bonus, only a sudden rise does; one spammer moves nothing."""
from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable

from vc.config import CategorySpec

TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
STRETCH_RE = re.compile(r"(\w)\1{2,}")          # NEEEIN -> nein, KEKWWW -> kekw
REPEAT4_RE = re.compile(r"([^\W\d_])\1{3,}")    # a letter 4+ times in a row
URL_RE = re.compile(r"https?://\S+|\S+\.(?:tv|com)/\S+")
CATS = ("laugh", "shock", "hype", "cringe", "confusion")


def squeeze(s: str) -> str:
    return STRETCH_RE.sub(r"\1", s.lower())


@dataclass(frozen=True, slots=True)
class Feat:
    cats: frozenset[str] = frozenset()
    intense: bool = False
    emote_only: bool = False
    token: str = ""
    clipcall: bool = False
    ignored: bool = False


IGNORED = Feat(ignored=True)


class Classifier:
    def __init__(self, categories: dict[str, CategorySpec], ignore_users: Iterable[str] = (),
                 ignore_patterns: Iterable[str] = (), clip_phrases: Iterable[str] = ("clip",)) -> None:
        self.ignore_users = {u.lower() for u in ignore_users}
        self.ignore = [re.compile(p) for p in ignore_patterns]
        self.cats: list[tuple[str, set[str], re.Pattern | None, list[str], list[re.Pattern]]] = []
        self.emote_words: set[str] = set()
        for name, spec in categories.items():
            single = {squeeze(w) for w in spec.words + spec.emotes if TOKEN_RE.fullmatch(w)}
            multi = [squeeze(w) for w in spec.words + spec.emotes if not TOKEN_RE.fullmatch(w)]
            rx = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(m) for m in multi) + r")(?!\w)") if multi else None
            self.cats.append((name, single, rx, list(spec.emojis), [re.compile(p) for p in spec.regex]))
            self.emote_words |= {squeeze(e) for e in spec.emotes}
        alts = "|".join(re.escape(p) for p in sorted(clip_phrases, key=len, reverse=True))
        self.clip_rx = re.compile(rf"(?i)(?<!\w)(?:{alts})(?!\w)")

    def is_emote(self, tok: str, emotes: frozenset[str], channel: frozenset[str] = frozenset()) -> bool:
        return tok in emotes or tok in channel or squeeze(tok) in self.emote_words

    def classify(self, user: str, text: str, emotes: frozenset[str] = frozenset(),
                 channel: frozenset[str] = frozenset()) -> Feat:
        """emotes: Twitch emote names of this message; channel: the channel's 7TV/BTTV/FFZ names."""
        t = text.strip()
        if user.lower() in self.ignore_users or not t:
            return IGNORED
        if t.startswith("!") and not t.lower().startswith("!clip"):
            return IGNORED
        if any(p.search(t) for p in self.ignore):
            return IGNORED
        words = t.split()
        toks = [squeeze(x) for x in TOKEN_RE.findall(t)]
        tokset = set(toks)
        sq = squeeze(t)
        cats = frozenset(name for name, single, rx, emojis, regex in self.cats
                         if tokset & single or (rx and rx.search(sq)) or any(e in t for e in emojis)
                         or any(r.search(t) for r in regex))
        plain = " ".join(w for w in words if not self.is_emote(w, emotes, channel))
        letters = [c for c in plain if c.isalpha()]
        intense = ((len(letters) >= 3 and sum(c.isupper() for c in letters) >= 0.7 * len(letters))
                   or bool(REPEAT4_RE.search(plain)) or t.count("!") >= 3)
        emoji = [w for w in words if not any(c.isalnum() for c in w) and any(ord(c) > 0x2100 for c in w)]
        emote_only = all(self.is_emote(w, emotes, channel) or not any(c.isalnum() for c in w) for w in words) and (
            any(self.is_emote(w, emotes, channel) for w in words) or bool(emoji))
        token = " ".join(dict.fromkeys(squeeze(w) for w in words))[:40]
        clip = t.lower().startswith("!clip") or bool(self.clip_rx.search(URL_RE.sub(" ", t)))
        return Feat(cats, intense, emote_only, token, clip)


def lift(n_users: int, n_burst_users: int, base_hits: float, base_n: float, floor: float) -> float:
    """Burst share of distinct users / baseline message share (floored). 1.0 = as usual."""
    if not n_burst_users:
        return 1.0
    base = max(floor, base_hits / base_n if base_n else 0.0)
    return (n_users / n_burst_users) / base


class FeatureWindow:
    """Per-second counters [sec, n, laugh, shock, hype, cringe, confusion, intense, emote_only] plus a rolling
    token counter. O(1) amortised per message; snapshot is O(burst + baseline seconds)."""

    def __init__(self, keep_s: int) -> None:
        self.keep_s = keep_s
        self.rows: deque[list[int]] = deque()
        self.tokens: deque[tuple[float, str]] = deque()
        self.tok_n: Counter[str] = Counter()

    def add(self, ts: float, f: Feat) -> None:
        sec = int(ts)
        if not self.rows or self.rows[-1][0] != sec:
            self.rows.append([sec] + [0] * 8)
        r = self.rows[-1]
        r[1] += 1
        for i, c in enumerate(CATS):
            r[2 + i] += c in f.cats
        r[7] += f.intense
        r[8] += f.emote_only
        if f.token:
            self.tokens.append((ts, f.token))
            self.tok_n[f.token] += 1
        self.trim(ts)

    def trim(self, now: float) -> None:
        keep = int(now) - self.keep_s
        while self.rows and self.rows[0][0] < keep:
            self.rows.popleft()
        while self.tokens and self.tokens[0][0] < keep:
            tok = self.tokens.popleft()[1]
            self.tok_n[tok] -= 1
            if not self.tok_n[tok]:
                del self.tok_n[tok]

    def snapshot(self, burst: list[tuple[str, Feat]], recent: list[Feat], lo: int, hi: int, floor: float,
                 min_users: int, conv_min_users: int) -> dict:
        """burst: (user, feat) of the burst; recent: feats of the last 10 s (excluded from the token baseline);
        baseline = seconds lo < s <= hi."""
        base = [0] * 9
        for r in self.rows:
            if lo < r[0] <= hi:
                for i in range(1, 9):
                    base[i] += r[i]
        users = {u for u, _ in burst}
        nu = len(users)
        lifts: dict[str, float] = {}
        for i, c in enumerate(CATS):
            hit = {u for u, f in burst if c in f.cats}
            lifts[c] = round(lift(len(hit), nu, base[2 + i], base[1], floor), 2) if len(hit) >= min_users else 1.0
        label = max(lifts, key=lambda k: lifts[k])
        intense = {u for u, f in burst if f.intense}
        emo = {u for u, f in burst if f.emote_only}
        by_tok: Counter[str] = Counter(f.token for f in {(u, f.token): f for u, f in burst if f.token}.values())
        tok, n_tok = by_tok.most_common(1)[0] if by_tok else ("", 0)
        rec_tok = sum(1 for f in recent if f.token == tok)
        tok_lift = (lift(n_tok, nu, self.tok_n[tok] - rec_tok, len(self.tokens) - len(recent), floor)
                    if n_tok >= conv_min_users else 1.0)
        emo_lift = lift(len(emo), nu, base[8], base[1], floor) if len(emo) >= conv_min_users else 1.0
        return {
            "unique_n": nu,
            "reaction_lifts": lifts,
            "reaction_lift": lifts[label],
            "reaction_label": label if lifts[label] > 1 else "",
            "intensity_lift": round(lift(len(intense), nu, base[7], base[1], floor), 2) if len(intense) >= min_users else 1.0,
            "conv_lift": round(max(tok_lift, emo_lift), 2),
            "conv_token": tok if tok_lift >= emo_lift else "(emotes)",
        }
