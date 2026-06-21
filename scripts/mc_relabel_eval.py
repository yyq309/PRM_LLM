"""Does relabeling with MC-return beat the DQN labels on decision-relevant forks?

mc_return_labels.py showed: only ~20% of same-state candidate groups are outcome-relevant (the env
recovers from the rest), and on those the DQN value has ~0 correlation with realized return. This
tests the fix: train the strong-PRM score head on TRAIN oracle samples with (a) the DQN label and
(b) the MC goal-return label, then on HELD-OUT DECISION-RELEVANT groups measure which model's top
pick actually has the (near-)max realized MC return. Honest, non-circular: train and eval tasks are
disjoint (structural split), and the target metric is the REALIZED return, not the training label.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.feature_extraction import DictVectorizer  # noqa: E402

from train_prm_baseline import read_jsonl  # noqa: E402
from train_prm_strong import extract_features  # noqa: E402

EPS = 1e-9
MEANINGFUL = 0.02


def main() -> None:
    parser = argparse.ArgumentParser(description="MC-relabel vs DQN-label PRM on decision-relevant forks.")
    parser.add_argument("--mc-labels", type=Path, default=ROOT / "outputs" / "mc_return_labels.jsonl")
    parser.add_argument("--train-input", type=Path, default=ROOT / "outputs" / "prm_samples_train.jsonl")
    parser.add_argument("--heldout-input", type=Path, default=ROOT / "outputs" / "prm_samples_heldout.jsonl")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "mc_relabel_eval.json")
    args = parser.parse_args()

    mc = {r["sample_id"]: r for r in read_jsonl(args.mc_labels)}
    samples = {}
    for inp in (args.train_input, args.heldout_input):
        for r in read_jsonl(inp):
            if r.get("label_source") == "oracle" and r.get("sample_id") in mc:
                samples[r["sample_id"]] = r

    train_rows, heldout_rows = [], []
    for sid, r in samples.items():
        m = mc[sid]
        rec = {"feats": extract_features(r), "dqn": float(r.get("score", 0.0)),
               "mc": float(m["mc_g_goal"]), "group": m["candidate_group"], "split": m.get("dataset_split")}
        (train_rows if r.get("dataset_split") == "train" else heldout_rows).append(rec)

    vec = DictVectorizer(sparse=True)
    x_train = vec.fit_transform([r["feats"] for r in train_rows]).toarray()
    x_held = vec.transform([r["feats"] for r in heldout_rows]).toarray()

    def fit(target_key: str):
        y = np.asarray([r[target_key] for r in train_rows], dtype=np.float32)
        m = HistGradientBoostingRegressor(random_state=0, max_iter=300, learning_rate=0.08)
        m.fit(x_train, y)
        return np.clip(m.predict(x_held), 0.0, 1.0)

    pred_dqn = fit("dqn")
    pred_mc = fit("mc")
    raw_dqn = np.asarray([r["dqn"] for r in heldout_rows])

    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(heldout_rows):
        groups[r["group"]].append(i)

    def top1_realized_optimal(pred) -> dict[str, Any]:
        hit = relevant = 0
        for idxs in groups.values():
            mc_vals = [heldout_rows[i]["mc"] for i in idxs]
            if len(idxs) < 2 or (max(mc_vals) - min(mc_vals)) <= MEANINGFUL:
                continue  # flat group: no meaningful decision
            relevant += 1
            best = max(idxs, key=lambda i: pred[i])
            if heldout_rows[best]["mc"] >= max(mc_vals) - 1e-6:
                hit += 1
        return {"decision_relevant_groups": relevant, "top1_realized_optimal_rate": round(hit / max(relevant, 1), 4)}

    report = {
        "n_train_oracle": len(train_rows),
        "n_heldout_oracle": len(heldout_rows),
        "metric": "on HELD-OUT decision-relevant groups, does the model's top pick have the (near-)max REALIZED MC goal-return",
        "raw_dqn_label": top1_realized_optimal(raw_dqn),
        "prm_trained_on_dqn_labels": top1_realized_optimal(pred_dqn),
        "prm_trained_on_mc_labels": top1_realized_optimal(pred_mc),
        "note": (
            "All three are scored against the SAME realized-MC ground truth on disjoint held-out tasks. "
            "If prm_trained_on_mc_labels > the DQN-based ones, MC relabeling genuinely improves decisions "
            "where they matter. If all ~equal/low, the bottleneck is the PRM INPUT FEATURES (the observable "
            "state does not distinguish the better fork), not the label source — reported honestly either way."
        ),
    }
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("note", "metric")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
