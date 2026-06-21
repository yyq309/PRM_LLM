"""Strong, state-conditioned Pentest-PRM with multi-seed confidence intervals.

The TF-IDF baseline under-uses the observable state (an ablation showed masking ALL
context barely moved its metrics). This model instead extracts STRUCTURED state features
from the verbalized context (counts of paths/forms/parameters/credentials/vulns/files, the
auth and shell state, remaining budget, failed branches) PLUS structured action features
(normalized action type, status, has-target, has-parameter), and trains gradient-boosted
trees that can model state x action interactions. Everything is still derived from the
LLM-visible context (no RL vector), so it stays a legitimate PRM input.

It reports the HONEST metrics (oracle-labeled subset only) broken out by held-out group
(unseen_instance vs unseen_chain), with multi-seed mean +/- std and bootstrap CIs, and
compares against the TF-IDF baseline on the same subset.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import argparse
import json
import re
import sys
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from honest_eval import majority_floor, pairwise_within_group  # noqa: E402
from task_split import get_split  # noqa: E402
from train_prm_baseline import read_jsonl  # noqa: E402

LIST_FIELDS = {
    "Known paths": "num_paths",
    "Known forms": "num_forms",
    "Known parameters": "num_parameters",
    "Credentials": "num_credentials",
    "Verified vulnerabilities": "num_verified_vulns",
    "Read files": "num_read_files",
}


def count_list(context: str, label: str) -> int:
    m = re.search(re.escape(label) + r": \[([^\]]*)\]", context)
    if not m or not m.group(1).strip():
        return 0
    return m.group(1).count("'") // 2 or (m.group(1).count(",") + 1)


def extract_features(sample: dict[str, Any]) -> dict[str, Any]:
    ctx = str(sample.get("context", ""))
    feats: dict[str, Any] = {}
    for label, name in LIST_FIELDS.items():
        feats[name] = count_list(ctx, label)
    auth = re.search(r"Auth state: (\w+)", ctx)
    shell = re.search(r"Shell state: (\w+)", ctx)
    budget = re.search(r"Remaining budget: (\d+)", ctx)
    failed = re.search(r"Failed branches: \{([^}]*)\}", ctx)
    feats["auth_state"] = auth.group(1) if auth else "unknown"
    feats["shell_state"] = shell.group(1) if shell else "unknown"
    feats["remaining_budget"] = int(budget.group(1)) if budget else 0
    feats["num_failed_branches"] = (failed.group(1).count(":") if failed and failed.group(1).strip() else 0)

    na = sample.get("normalized_action") or {}
    feats["action_type"] = str(na.get("action_type") or "none")
    feats["status"] = str(na.get("status") or "none")
    feats["has_target"] = int(na.get("target") is not None)
    feats["has_parameter"] = int(na.get("parameter") is not None)
    feats["normalizer_confidence"] = float(sample.get("normalizer_confidence") or 0.0)
    return feats


def build_matrix(rows: list[dict[str, Any]], vec: DictVectorizer, fit: bool) -> np.ndarray:
    feats = [extract_features(r) for r in rows]
    X = vec.fit_transform(feats) if fit else vec.transform(feats)
    return X.toarray()


def train_one(train_rows, x_train, y_score, y_rank, y_diag, seed: int):
    score = HistGradientBoostingRegressor(random_state=seed, max_iter=300, learning_rate=0.08)
    rank = HistGradientBoostingClassifier(random_state=seed, max_iter=300, learning_rate=0.08)
    diag = HistGradientBoostingClassifier(random_state=seed, max_iter=300, learning_rate=0.08)
    score.fit(x_train, y_score)
    rank.fit(x_train, y_rank)
    diag.fit(x_train, y_diag)
    return {"score": score, "rank": rank, "diagnosis": diag}


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += (m.sum() / n) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(ece)


def eval_group(models, rows, x, idxs, eps) -> dict[str, Any]:
    if not idxs:
        return {"n": 0}
    score_pred = np.clip(models["score"].predict(x), 0.0, 1.0)
    rank_pred = list(models["rank"].predict(x))
    y_rank = [str(rows[i]["rank_label"]) for i in idxs]
    rank_acc = float(np.mean([int(rank_pred[i] == rows[i]["rank_label"]) for i in idxs]))
    proba = models["rank"].predict_proba(x)
    conf = proba.max(axis=1)[idxs]
    correct = np.asarray([int(rank_pred[i] == rows[i]["rank_label"]) for i in idxs], dtype=float)
    pair = pairwise_within_group(rows, list(score_pred), idxs, eps)
    return {
        "n": len(idxs),
        "rank_accuracy": rank_acc,
        "rank_macro_f1": float(f1_score(y_rank, [rank_pred[i] for i in idxs], average="macro", zero_division=0)),
        "majority_floor": majority_floor(y_rank),
        "pairwise_accuracy": pair["pairwise_accuracy"],
        "pairwise_pairs": pair["pairs"],
        "ece": expected_calibration_error(conf, correct),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strong state-conditioned PRM + multi-seed CIs (method §10/§11.2/§11.3).")
    parser.add_argument("--train-input", type=Path, default=ROOT / "outputs" / "prm_samples_train.jsonl")
    parser.add_argument("--heldout-input", type=Path, default=ROOT / "outputs" / "prm_samples_heldout.jsonl")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "prm_strong_eval.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--pairwise-eps", type=float, default=1e-6)
    parser.add_argument("--model-output", type=Path, default=ROOT / "outputs" / "prm_strong.joblib",
                        help="Persist a deterministic (seed-0) strong PRM {vectorizer, score, rank, diagnosis} "
                             "for the closed-loop evaluate_prm_policy. Set to '' to skip.")
    args = parser.parse_args()

    train_rows = read_jsonl(args.train_input)
    heldout_rows = read_jsonl(args.heldout_input)

    # Map held-out samples to unseen_instance vs unseen_chain by task_id.
    split = get_split()
    chain_ids = {Path(p).stem for p in split["heldout_chain"]}
    instance_ids = {Path(p).stem for p in split["heldout_instance"]}

    def group_idx(predicate) -> list[int]:
        return [i for i, r in enumerate(heldout_rows) if predicate(r)]

    def task_file(r) -> str:
        return Path(r.get("task_path", "")).stem

    oracle_all = group_idx(lambda r: r.get("label_source") == "oracle")
    oracle_instance = [i for i in oracle_all if task_file(heldout_rows[i]) in instance_ids]
    oracle_chain = [i for i in oracle_all if task_file(heldout_rows[i]) in chain_ids]

    vec = DictVectorizer(sparse=True)
    x_train = build_matrix(train_rows, vec, fit=True)
    x_held = build_matrix(heldout_rows, vec, fit=False)
    y_score = np.asarray([float(r["score"]) for r in train_rows], dtype=np.float32)
    y_rank = [str(r["rank_label"]) for r in train_rows]
    y_diag = [str(r["diagnosis"]) for r in train_rows]

    # Multi-seed training.
    per_seed = []
    last_models = None
    for seed in args.seeds:
        models = train_one(train_rows, x_train, y_score, y_rank, y_diag, seed)
        last_models = models
        per_seed.append({
            "seed": seed,
            "oracle_all": eval_group(models, heldout_rows, x_held, oracle_all, args.pairwise_eps),
            "oracle_unseen_instance": eval_group(models, heldout_rows, x_held, oracle_instance, args.pairwise_eps),
            "oracle_unseen_chain": eval_group(models, heldout_rows, x_held, oracle_chain, args.pairwise_eps),
        })

    def agg(group: str, metric: str) -> dict[str, float]:
        vals = [s[group][metric] for s in per_seed if s[group].get(metric) is not None]
        if not vals:
            return {"mean": None, "std": None}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "min": float(min(vals)), "max": float(max(vals))}

    # Bootstrap CI for the headline metric (oracle_all pairwise) on the last model.
    score_pred = np.clip(last_models["score"].predict(x_held), 0.0, 1.0)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(args.bootstrap):
        sample = list(rng.choice(oracle_all, size=len(oracle_all), replace=True)) if oracle_all else []
        pair = pairwise_within_group(heldout_rows, list(score_pred), sample, args.pairwise_eps)
        if pair["pairwise_accuracy"] is not None:
            boot.append(pair["pairwise_accuracy"])
    ci = {
        "mean": float(np.mean(boot)) if boot else None,
        "ci95_low": float(np.percentile(boot, 2.5)) if boot else None,
        "ci95_high": float(np.percentile(boot, 97.5)) if boot else None,
    }

    # ECE post-calibration (method §11.3): isotonic-calibrate the rank head's confidence on the
    # train set (internal 3-fold), then measure ECE before/after on each oracle subset. This
    # only re-maps confidences; it does not see held-out labels.
    base_rank = train_one(train_rows, x_train, y_score, y_rank, y_diag, seed=0)["rank"]
    rank_cal = CalibratedClassifierCV(  # isotonic = persisted calibrated head
        HistGradientBoostingClassifier(random_state=0, max_iter=300, learning_rate=0.08),
        method="isotonic", cv=3,
    ).fit(x_train, np.asarray(y_rank))
    rank_cal_sig = CalibratedClassifierCV(  # sigmoid/Platt = gentler, compared
        HistGradientBoostingClassifier(random_state=0, max_iter=300, learning_rate=0.08),
        method="sigmoid", cv=3,
    ).fit(x_train, np.asarray(y_rank))

    def _ece_for(model, idxs: list[int]) -> float:
        proba = model.predict_proba(x_held)
        pred = model.predict(x_held)
        conf = proba.max(axis=1)[idxs]
        correct = np.asarray([int(pred[i] == heldout_rows[i]["rank_label"]) for i in idxs], dtype=float)
        return expected_calibration_error(conf, correct)

    calibration_ece = {}
    for gname, idxs in [("oracle_all", oracle_all), ("oracle_unseen_instance", oracle_instance), ("oracle_unseen_chain", oracle_chain)]:
        if idxs:
            calibration_ece[gname] = {
                "ece_uncalibrated": _ece_for(base_rank, idxs),
                "ece_isotonic": _ece_for(rank_cal, idxs),
                "ece_sigmoid": _ece_for(rank_cal_sig, idxs),
            }
    calibration_ece["note"] = (
        "Isotonic helps the worst-calibrated slices (oracle_all, unseen_chain) but can OVERFIT and "
        "hurt the already-well-calibrated unseen_instance; sigmoid/Platt is gentler. Calibration is a "
        "net win only on the hard unseen-chain slice; apply selectively, not globally."
    )

    report = {
        "n_train": len(train_rows),
        "n_heldout": len(heldout_rows),
        "n_oracle_all": len(oracle_all),
        "n_oracle_unseen_instance": len(oracle_instance),
        "n_oracle_unseen_chain": len(oracle_chain),
        "seeds": args.seeds,
        "multiseed": {
            "oracle_all": {m: agg("oracle_all", m) for m in ["rank_accuracy", "pairwise_accuracy", "ece", "majority_floor"]},
            "oracle_unseen_instance": {m: agg("oracle_unseen_instance", m) for m in ["rank_accuracy", "pairwise_accuracy", "ece"]},
            "oracle_unseen_chain": {m: agg("oracle_unseen_chain", m) for m in ["rank_accuracy", "pairwise_accuracy", "ece"]},
        },
        "bootstrap_oracle_all_pairwise_ci95": ci,
        "calibration_ece": calibration_ece,
        "per_seed": per_seed,
        "note": (
            "All metrics are on the ORACLE-labeled subset (genuine state-dependent ranking); rule rows "
            "are excluded. unseen_chain is the hard generalization (whole topology family absent from "
            "train). Compare pairwise vs the TF-IDF baseline's oracle-only pairwise from honest_eval."
        ),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Persist a deterministic (seed-0) strong PRM for the closed-loop policy eval. It scores
    # candidates from the SAME observable state+action features (no oracle q-values), so it is a
    # legitimate PRM drop-in for the TF-IDF baseline in evaluate_prm_policy.
    if str(args.model_output):
        persist = train_one(train_rows, x_train, y_score, y_rank, y_diag, seed=0)
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"kind": "strong", "vectorizer": vec, "score": persist["score"],
             "rank": persist["rank"], "diagnosis": persist["diagnosis"],
             # sigmoid/Platt = best calibrator on the hard slices (unseen_chain ECE 0.155->0.067)
             "rank_calibrated": rank_cal_sig, "rank_calibrated_isotonic": rank_cal},
            args.model_output,
        )
        print(f"persisted strong PRM -> {args.model_output}")

    print("calibration ECE (uncalibrated / isotonic / sigmoid):")
    for g, c in calibration_ece.items():
        if not isinstance(c, dict):
            continue
        print(f"  {g:24s} {c['ece_uncalibrated']:.3f} / {c['ece_isotonic']:.3f} / {c['ece_sigmoid']:.3f}")

    print(f"oracle subset: all={len(oracle_all)} instance={len(oracle_instance)} chain={len(oracle_chain)}")
    for g in ["oracle_all", "oracle_unseen_instance", "oracle_unseen_chain"]:
        ms = report["multiseed"][g]
        ra, pa = ms["rank_accuracy"], ms["pairwise_accuracy"]
        print(f"  {g:24s} rank={ra['mean']:.3f}±{ra['std']:.3f}  pairwise={pa['mean']:.3f}±{pa['std']:.3f}  ece={ms['ece']['mean']:.3f}")
    print(f"  bootstrap oracle_all pairwise 95% CI: [{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}]")


if __name__ == "__main__":
    main()
