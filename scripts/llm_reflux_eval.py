"""Evaluate the DAgger-style LLM rollout reflux (method §11.3).

Tests whether adding real DeepSeek-proposed, oracle-labeled candidate actions to PRM
training improves the PRM's ranking of REAL LLM actions on held-out states — i.e. whether
the PRM aligns to the LLM's action distribution rather than the scripted candidate
generator's.

Setup (all on the ORACLE-labeled subset, the honest metric):
- LLM rollout samples are split by task into reflux-train (TRAIN-family tasks) and
  llm-test (HELD-OUT tasks: unseen instances + unseen chains).
- Model A (no reflux): trained on the scripted PRM train set only.
- Model B (reflux):    trained on scripted PRM train + LLM reflux-train samples.
- Both are evaluated on llm-test. If reflux helps, B beats A on the held-out LLM actions.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from honest_eval import pairwise_within_group  # noqa: E402
from task_split import get_split  # noqa: E402
from train_prm_baseline import read_jsonl  # noqa: E402
from train_prm_strong import eval_group, extract_features, train_one  # noqa: E402


def matrix(rows: list[dict[str, Any]], vec: DictVectorizer, fit: bool) -> np.ndarray:
    feats = [extract_features(r) for r in rows]
    return (vec.fit_transform(feats) if fit else vec.transform(feats)).toarray()


def targets(rows: list[dict[str, Any]]):
    return (
        np.asarray([float(r["score"]) for r in rows], dtype=np.float32),
        [str(r["rank_label"]) for r in rows],
        [str(r["diagnosis"]) for r in rows],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM rollout reflux on real held-out LLM actions.")
    parser.add_argument("--scripted-train", type=Path, default=ROOT / "outputs" / "prm_samples_train.jsonl")
    parser.add_argument("--llm-samples", type=Path, default=ROOT / "outputs" / "llm_rollout_samples.jsonl")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "llm_reflux_eval.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairwise-eps", type=float, default=1e-6)
    args = parser.parse_args()

    scripted_train = read_jsonl(args.scripted_train)
    llm = read_jsonl(args.llm_samples)

    split = get_split()
    train_ids = {Path(p).stem for p in split["train"]}
    heldout_ids = {Path(p).stem for p in split["heldout_all"]}

    def task_file(r) -> str:
        return Path(r.get("task_path", "")).stem

    reflux_train = [r for r in llm if task_file(r) in train_ids]
    llm_test = [r for r in llm if task_file(r) in heldout_ids]
    llm_test_oracle = [i for i, r in enumerate(llm_test) if r.get("label_source") == "oracle"]

    # Fit a shared vectorizer on the union of all training text so both models share features.
    vec = DictVectorizer(sparse=True)
    vec.fit([extract_features(r) for r in scripted_train + reflux_train + llm_test])

    x_test = matrix(llm_test, vec, fit=False)

    def fit_eval(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
        x_tr = matrix(train_rows, vec, fit=False)
        ys, yr, yd = targets(train_rows)
        models = train_one(train_rows, x_tr, ys, yr, yd, args.seed)
        return eval_group(models, llm_test, x_test, llm_test_oracle, args.pairwise_eps)

    model_a = fit_eval(scripted_train)
    model_b = fit_eval(scripted_train + reflux_train)

    report = {
        "n_scripted_train": len(scripted_train),
        "n_llm_reflux_train": len(reflux_train),
        "n_llm_test_total": len(llm_test),
        "n_llm_test_oracle_subset": len(llm_test_oracle),
        "evaluated_on": "held-out tasks' real-DeepSeek candidate actions, oracle-labeled subset",
        "model_A_no_reflux": model_a,
        "model_B_with_reflux": model_b,
        "reflux_pairwise_gain": (
            (model_b.get("pairwise_accuracy") or 0) - (model_a.get("pairwise_accuracy") or 0)
            if model_a.get("pairwise_accuracy") is not None and model_b.get("pairwise_accuracy") is not None else None
        ),
        "reflux_rank_gain": (
            (model_b.get("rank_accuracy") or 0) - (model_a.get("rank_accuracy") or 0)
            if model_a.get("n") and model_b.get("n") else None
        ),
        "note": (
            "Positive gains mean refluxing real DeepSeek-proposed, oracle-labeled actions into PRM "
            "training improves ranking of the LLM's actions on held-out states (DAgger-style alignment)."
        ),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
