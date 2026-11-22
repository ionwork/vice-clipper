"""Calibration of the V2 chat weights against real outcomes (pure: no I/O, deterministic seed).

Target = the real decision: NDCG@3 per 15-minute window and channel (plus Spearman and Precision@3).
Label = percentile of log(views) inside the channel for clips >= 24 h old, 0 for random non-clip moments.
library is derived from the clips' existence (= the label) and is therefore never tuned (leak); judge and voice
are too expensive for hundreds of moments. Only the five chat components are searched:
5000 Dirichlet samples x grid (baseline floor, z_floor, log span), then coordinate refinement of the best 20.
Time split: older 70 % search, newest 30 % test, never shuffled."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CHAT = ("velocity", "reaction", "clip_calls", "convergence", "intensity")
GRID_Z = (2.0, 3.0, 4.0)
GRID_SPAN = (2.0, 3.0, 4.0)
WINDOW_S = 900


@dataclass
class Params:
    floor: float
    z_floor: float
    span: float
    w: np.ndarray  # weights over CHAT, sum 1

    def label(self) -> str:
        return f"floor {self.floor:g} · z_floor {self.z_floor:g} · log_span {self.span:g}"


@dataclass
class Data:
    moments: list[dict]
    labels: np.ndarray
    train: list[np.ndarray] = field(default_factory=list)  # index groups (windows)
    test: list[np.ndarray] = field(default_factory=list)
    test_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def labels(moments: list[dict], now: float, min_age_h: float = 24) -> np.ndarray:
    """Percentile of log(views) per channel among positives old enough; NaN = too young (left out)."""
    out = np.full(len(moments), np.nan)
    by: dict[str, list[int]] = {}
    for i, m in enumerate(moments):
        if not m["positive"]:
            out[i] = 0.0
        elif now - m["abs_t"] >= min_age_h * 3600:
            by.setdefault(m["login"], []).append(i)
    for idx in by.values():
        lv = np.log1p([moments[i]["views"] for i in idx])
        ranks = lv.argsort().argsort() + 1
        out[idx] = ranks / len(idx)
    return out


def split(moments: list[dict], lab: np.ndarray, train_share: float = 0.7) -> Data:
    keep = np.flatnonzero(~np.isnan(lab))
    d = Data(moments, lab)
    if not len(keep):
        return d
    ts = np.array([moments[i]["abs_t"] for i in keep])
    cut = np.quantile(ts, train_share)
    groups: dict[tuple, list[int]] = {}
    for i in keep:
        groups.setdefault((moments[i]["login"], int(moments[i]["abs_t"] // WINDOW_S)), []).append(i)
    for g in groups.values():
        if len(g) < 2 or lab[g].max() <= 0:  # a window needs a choice and something worth choosing
            continue
        (d.train if min(moments[i]["abs_t"] for i in g) < cut else d.test).append(np.array(g))
    d.test_idx = np.array([i for i in keep if moments[i]["abs_t"] >= cut], dtype=int)
    return d


def matrix(moments: list[dict], floor: float, z_floor: float, span: float, c) -> np.ndarray:
    """n x 5 normalised chat components (same formulas as virality.py)."""
    rows = []
    for m in moments:
        f = m["feats"][str(floor)]
        rows.append([min(1.0, max(0.0, (f["z"] - z_floor) / c.z_span)),
                     min(1.0, math.log2(max(f["reaction_lift"], 1.0)) / span),
                     1 - math.exp(-f["calls"] / c.clipcall_scale),
                     min(1.0, math.log2(max(f["conv_lift"], 1.0)) / span),
                     min(1.0, math.log2(max(f["intensity_lift"], 1.0)) / span)])
    return np.array(rows)


def scores(X: np.ndarray, W: np.ndarray, c) -> np.ndarray:
    """n x m V for m weight vectors, with the agreement bonus (it does not depend on the weights)."""
    bonus = np.where((X >= c.agree_level).sum(1) >= c.agree_min_components, c.agree_factor, 1.0)
    return np.minimum(100.0, 100 * (X @ (W / W.sum(1, keepdims=True)).T) * bonus[:, None])


def legacy_scores(moments: list[dict], c) -> np.ndarray:
    """Today's V on the same moments: chat_spike, reactions (laugh only), clip_calls with the old weights."""
    X = np.array([[min(1.0, max(0.0, (f["z"] - c.z_floor) / c.z_span)),
                   min(1.0, math.log2(max(f["react_lift"], 1.0)) / c.reaction_log_span),
                   1 - math.exp(-f["calls"] / c.clipcall_scale)] for f in (m["feats"]["0.02"] for m in moments)])
    w = np.array([[c.legacy_weights[k] for k in ("chat_spike", "reactions", "clip_calls")]])
    return scores(X, w, c)[:, 0]


def ndcg3(S: np.ndarray, lab: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Mean NDCG@3 over the windows, one value per score column."""
    if S.ndim == 1:
        S = S[:, None]
    tot = np.zeros(S.shape[1])
    for g in groups:
        k = min(3, len(g))
        disc = 1 / np.log2(np.arange(2, k + 2))
        order = np.argsort(-S[g], axis=0, kind="stable")[:k]
        dcg = (lab[g][order] * disc[:, None]).sum(0)
        ideal = (np.sort(lab[g])[::-1][:k] * disc).sum()
        tot += dcg / ideal
    return tot / max(1, len(groups))


def precision3(s: np.ndarray, lab: np.ndarray, groups: list[np.ndarray], rel: float = 0.5) -> float:
    hits = [np.mean(lab[g][np.argsort(-s[g], kind="stable")[:3]] >= rel) for g in groups]
    return float(np.mean(hits)) if hits else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def search(d: Data, c, start: Params, floors: tuple[float, ...], n_samples: int = 5000, seed: int = 0) -> Params:
    rng = np.random.default_rng(seed)
    W = np.vstack([start.w, rng.dirichlet(np.ones(len(CHAT)), n_samples)])
    cands: list[tuple[float, Params]] = []
    for floor in floors:
        for z in GRID_Z:
            for span in GRID_SPAN:
                X = matrix(d.moments, floor, z, span, c)
                nd = ndcg3(scores(X, W, c), d.labels, d.train)
                for j in np.argsort(-nd, kind="stable")[:20]:
                    cands.append((float(nd[j]), Params(floor, z, span, W[j])))
    cands.sort(key=lambda x: -x[0])
    best_v, best = cands[0]
    for v, p in cands[:20]:  # coordinate refinement around the best 20
        X = matrix(d.moments, p.floor, p.z_floor, p.span, c)
        w = p.w.copy()
        for step in (0.05, 0.02, 0.01):
            improved = True
            while improved:
                improved = False
                for i in range(len(CHAT)):
                    for sgn in (1, -1):
                        t = w.copy()
                        t[i] = max(0.0, t[i] + sgn * step)
                        t /= t.sum()
                        tv = float(ndcg3(scores(X, t[None], c), d.labels, d.train)[0])
                        if tv > v + 1e-9:
                            v, w, improved = tv, t, True
        if v > best_v:
            best_v, best = v, Params(p.floor, p.z_floor, p.span, w)
    return best


def evaluate(d: Data, c, p: Params) -> dict:
    X = matrix(d.moments, p.floor, p.z_floor, p.span, c)
    s = scores(X, p.w[None], c)[:, 0]
    return {"ndcg": float(ndcg3(s, d.labels, d.test)[0]) if d.test else float("nan"),
            "spearman": spearman(s[d.test_idx], d.labels[d.test_idx]), "p3": precision3(s, d.labels, d.test)}


def evaluate_legacy(d: Data, c) -> dict:
    s = legacy_scores(d.moments, c)
    return {"ndcg": float(ndcg3(s, d.labels, d.test)[0]) if d.test else float("nan"),
            "spearman": spearman(s[d.test_idx], d.labels[d.test_idx]), "p3": precision3(s, d.labels, d.test)}


def ablation(d: Data, c, p: Params) -> dict[str, float]:
    """NDCG@3 loss on the test part when one component is left out (weights renormalised)."""
    base = evaluate(d, c, p)["ndcg"]
    out = {}
    for i, k in enumerate(CHAT):
        w = p.w.copy()
        w[i] = 0
        out[k] = base - evaluate(d, c, Params(p.floor, p.z_floor, p.span, w / w.sum() if w.sum() else w))["ndcg"]
    return out


def chat_weights(p: Params, start_weights: dict[str, float]) -> dict[str, float]:
    """Tuned chat weights scaled to the start sum of the chat components; library/judge/voice keep their start."""
    total = sum(start_weights[k] for k in CHAT)
    out = dict(start_weights)
    out.update({k: round(float(v) * total, 3) for k, v in zip(CHAT, p.w)})
    return out


def report(moments: list[dict], counts: dict, c, now: float, decisions: list[tuple[float, int]],
           min_pos: int = 150, min_test_windows: int = 30, min_gain: float = 0.05) -> tuple[str, dict | None]:
    """Report text and the config block to apply (None when the data does not justify it)."""
    from vc.sources.tune_data import FLOORS
    lab = labels(moments, now)
    d = split(moments, lab)
    pos = int(np.sum((lab > 0) & ~np.isnan(lab)))
    young = int(np.sum(np.isnan(lab)))
    lines = [f"Daten: {counts.get('clips', 0)} Katalog-Clips, {counts.get('channels', 0)} Kanaele, "
             f"{counts.get('moments', 0)} Momente ({counts.get('positives', 0)} Clips mit Chat, Rest Zufall)",
             f"  Chat-Quelle: eigene Logs {counts.get('src_log', 0)}, VOD-Chat {counts.get('src_vod', 0)} · "
             f"uebersprungen: ohne VOD {counts.get('skip_no_vod', 0)}, VOD-Chat nicht abrufbar {counts.get('skip_vod_chat', 0)}, "
             f"zu wenig Chat {counts.get('skip_thin_chat', 0)}, nicht abgedeckt {counts.get('skip_not_covered', 0)}",
             f"  mit Label: {pos} Clips (>= 24 h alt), {young} noch zu jung · Fenster: {len(d.train)} Suche / {len(d.test)} Test"]
    missing = []
    if pos < min_pos:
        missing.append(f"{pos} < {min_pos} positive Momente")
    if len(d.test) < min_test_windows:
        missing.append(f"{len(d.test)} < {min_test_windows} Testfenster")
    dec = f"Telegram-Entscheidungen: {len(decisions)}"
    if len(decisions) >= 50:
        v, y = np.array([x[0] for x in decisions]), np.array([x[1] for x in decisions], dtype=float)
        dec += f", Spearman(V, gepostet) {spearman(v, y):.2f}"
    else:
        dec += " (< 50, noch nicht genutzt)"
    if not d.train or not d.test:
        lines += ["ZU WENIG DATEN fuer eine Suche: " + "; ".join(missing or ["keine Fenster"]),
                  "Startgewichte bleiben. judge/voice: nicht ausgewertet.", dec]
        return "\n".join(lines), None
    start = Params(c.reaction_base_min, c.z_floor, c.log_span, np.array([c.weights[k] for k in CHAT]))
    start.w = start.w / start.w.sum()
    tuned = search(d, c, start, FLOORS)
    old, v2s, v2t = evaluate_legacy(d, c), evaluate(d, c, start), evaluate(d, c, tuned)
    lines += ["", f"{'Testteil':<16} {'NDCG@3':>7} {'Spearman':>9} {'P@3':>6}"]
    for name, r in (("V heute", old), ("V2 Start", v2s), ("V2 getunt", v2t)):
        lines.append(f"{name:<16} {r['ndcg']:>7.3f} {r['spearman']:>9.2f} {r['p3']:>6.2f}")
    lines.append("Ablation (NDCG-Verlust ohne Komponente): " +
                 ", ".join(f"{k} {v:+.3f}" for k, v in ablation(d, c, tuned).items()))
    gain = (v2t["ndcg"] - v2s["ndcg"]) / v2s["ndcg"] if v2s["ndcg"] else 0.0
    if gain < min_gain:
        missing.append(f"Verbesserung {gain:+.1%} < {min_gain:.0%}")
    if v2s["ndcg"] < old["ndcg"]:
        lines.append("ACHTUNG: V2 (Start) ist auf dem Testteil schlechter als V heute.")
    block = {"virality.weights": chat_weights(tuned, c.weights), "virality.z_floor": tuned.z_floor,
             "virality.reaction_base_min": tuned.floor, "virality.reaction_log_span": tuned.span,
             "virality.log_span": tuned.span}
    lines += ["Vorschlag (" + tuned.label() + "):", *(f"  {k}: {v}" for k, v in block.items()),
              "judge/voice: nicht ausgewertet (Stichprobe braucht Video + Ollama je Moment).", dec]
    if missing:
        lines.append("Nicht uebernommen: " + "; ".join(missing))
        return "\n".join(lines), None
    return "\n".join(lines), block
