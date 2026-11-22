"""Phase 4f: calibration search (pure) and the honest 'too little data' path of vc tune."""
import random

import numpy as np

from vc.config import get_config
from vc.detect import tune as tn

C = get_config().virality
NOW = 1_900_000_000.0


def feats(z, react=1.0, calls=0, conv=1.0, inten=1.0):
    f = {"z": z, "reaction_lift": react, "react_lift": react, "conv_lift": conv, "intensity_lift": inten,
         "calls": calls, "label": ""}
    return {str(fl): f for fl in (0.01, 0.02, 0.04)}


def synthetic(n_windows=60, seed=1):
    """Views follow the reaction lift only; velocity is noise. Old enough, 4 moments per 15-min window."""
    rng = random.Random(seed)
    out = []
    for w in range(n_windows):
        for k in range(4):
            react = rng.choice([1.0, 2.0, 4.0, 8.0, 16.0])
            out.append({"id": f"{w}-{k}", "login": f"ch{w % 3}", "abs_t": NOW - 5 * 86400 + w * 900 + k * 60,
                        "views": int(react * 100 + rng.random() * 10), "positive": True, "source": "log",
                        "feats": feats(rng.uniform(0, 12), react)})
    return out


def test_labels_percentile_and_age():
    m = [{"login": "a", "abs_t": NOW - 2 * 86400, "views": v, "positive": True} for v in (10, 100, 1000)]
    m += [{"login": "a", "abs_t": NOW - 3600, "views": 5, "positive": True},
          {"login": "a", "abs_t": NOW - 3600, "views": 0, "positive": False}]
    lab = tn.labels(m, NOW)
    assert list(lab[:3]) == [1 / 3, 2 / 3, 1.0] and np.isnan(lab[3]) and lab[4] == 0


def test_time_split_never_mixes():
    m = synthetic()
    d = tn.split(m, tn.labels(m, NOW))
    last_train = max(m[i]["abs_t"] for g in d.train for i in g)
    first_test = min(m[i]["abs_t"] for g in d.test for i in g)
    assert last_train < first_test and len(d.test) >= 15


def test_search_finds_the_informative_component():
    m = synthetic()
    d = tn.split(m, tn.labels(m, NOW))
    start = tn.Params(0.02, 3.0, 3.0, np.array([C.weights[k] for k in tn.CHAT]))
    start.w = start.w / start.w.sum()
    best = tn.search(d, C, start, (0.02,), n_samples=500)
    assert tn.CHAT[int(np.argmax(best.w))] == "reaction"
    assert tn.evaluate(d, C, best)["ndcg"] >= tn.evaluate(d, C, start)["ndcg"]
    assert tn.ablation(d, C, best)["reaction"] > 0


def test_ndcg_perfect_and_worst():
    lab = np.array([1.0, 0.5, 0.0, 0.0])
    g = [np.arange(4)]
    assert tn.ndcg3(np.array([4.0, 3, 2, 1]), lab, g)[0] == 1.0
    assert tn.ndcg3(np.array([1.0, 2, 3, 4]), lab, g)[0] < 0.5


def test_report_says_too_little_data():
    m = synthetic(5)
    text, block = tn.report(m, {"clips": 20, "channels": 3, "moments": 20, "positives": 20}, C, NOW, [])
    assert block is None and ("ZU WENIG DATEN" in text or "Nicht uebernommen" in text)


def test_vc_tune_without_data_leaves_config(vc_root):
    from typer.testing import CliRunner

    from vc import db
    from vc.cli import app
    db.db_path().unlink(missing_ok=True)
    before = (vc_root / "config.yaml").read_text(encoding="utf-8")
    r = CliRunner().invoke(app, ["tune", "--apply", "--no-vods"])
    assert r.exit_code == 0, r.output
    assert "ZU WENIG DATEN" in r.output and "nichts geschrieben" in r.output
    assert (vc_root / "config.yaml").read_text(encoding="utf-8") == before
