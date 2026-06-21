"""Robust Pentest-PRM training + evaluation (method §6.1, §11.2, §11.3).

Upgrades the TF-IDF PRM from a clean-only baseline into a robustness-hardened
evaluator, then reports the robustness, calibration and ranking metrics the method
makes training-stage gates:

1. sim->real robustness injection (§6.1/§14): augment training with *degraded*
   observations — randomly masked context fields, an out-of-abstraction event token,
   and "path found but inputs not yet characterized" states — so the PRM does not rely
   on clean simulation contexts and stays stable on adapter-style dirty states.
2. label-confidence weighting (§11.2): weight each sample by its `label_confidence`
   (oracle margin x normalizer x schema confidence), down-weighting low-confidence
   rule/precondition labels instead of treating every label as equally trustworthy.
3. calibration + abstention (§11.2): expected calibration error, high-confidence
   precision, and the accuracy gain from abstaining on low-confidence predictions.
4. same-state pairwise ranking (§11.3): within each candidate group, does the PRM
   score order candidates the way the oracle does?

It trains a baseline (clean, unweighted) and a robust (augmented, weighted) model with
the *same* pipeline, and compares them on clean and dirty held-out splits. Robustness =
the robust model degrades less from clean to dirty.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import argparse
import json
import random
import sys
from collections import defaultdict
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from leakage_audit import CONTEXT_LABELS, mask_field  # noqa: E402
from train_prm_baseline import read_jsonl, regression_metrics, sample_to_text  # noqa: E402


# ---------------------------------------------------------------------------
# Dirty-observation degradation (adapter-style imperfect state)
# ---------------------------------------------------------------------------

def degrade_context(context: str, rng: random.Random, num_fields: int, ooa_prob: float) -> str:
    labels = list(CONTEXT_LABELS)
    rng.shuffle(labels)
    out = context
    for label in labels[:num_fields]:
        out = mask_field(out, label)
    if rng.random() < ooa_prob:
        out = out + " [out_of_abstraction_event: unmapped_tool_output]."
    return out


def degrade_action(normalized: dict[str, Any], rng: random.Random, drop_arg_prob: float, drop_type_prob: float) -> dict[str, Any]:
    """Simulate normalizer imperfection: drop target/parameter, occasionally blank the type.

    The PRM leans on the normalized-action representation, so adapter/normalizer
    degradation is the robustness axis that actually matters (method §10.1 — the
    normalizer is a single point of failure). We only DROP fields (set to None); we do
    NOT rewrite `status` to a label-bearing value, because (per the adversarial review)
    forcing status='schema_gap' on a row whose rank_label is not schema_gap corrupts the
    label the model is scored against rather than degrading the observation.
    """
    out = dict(normalized)
    if rng.random() < drop_arg_prob:
        out["target"] = None
    if rng.random() < drop_arg_prob:
        out["parameter"] = None
    if rng.random() < drop_type_prob:
        out["action_type"] = None  # blank only; keep status so the label is not corrupted
    return out


def degrade_row(
    row: dict[str, Any],
    rng: random.Random,
    num_fields: int,
    ooa_prob: float,
    *,
    degrade_action_repr: bool = True,
    drop_arg_prob: float = 0.4,
    drop_type_prob: float = 0.15,
) -> dict[str, Any]:
    dirty = dict(row)
    dirty["context"] = degrade_context(str(row.get("context", "")), rng, num_fields, ooa_prob)
    if degrade_action_repr and isinstance(row.get("normalized_action"), dict):
        dirty["normalized_action"] = degrade_action(row["normalized_action"], rng, drop_arg_prob, drop_type_prob)
    return dirty


# ---------------------------------------------------------------------------
# OUT-OF-DISTRIBUTION corruption family B (held out of training).
#
# The robust model trains on family A above (mask context fields + OOA token + drop action
# fields). To test GENUINE robustness, family B uses a structurally DIFFERENT corruption that
# never appears in training: rename every field label to a synonym, reorder the fields, inject
# plausible spurious lines, and jitter the budget number. Nothing here is "masking" or an OOA
# token, so "robust degrades less on B" is a real OOD-generalization claim, not in-distribution.
# ---------------------------------------------------------------------------

OOD_RELABEL = {
    "Known paths:": "Routes seen:",
    "Known forms:": "Forms found:",
    "Known parameters:": "Params:",
    "Credentials:": "Creds:",
    "Auth state:": "Session:",
    "Shell state:": "Foothold:",
    "Verified vulnerabilities:": "Confirmed vulns:",
    "Read files:": "Files read:",
    "Failed branches:": "Past failures:",
    "Remaining budget:": "Steps left:",
    "Recent feedback:": "Last result:",
}
OOD_DISTRACTORS = [
    " Scanner note: passive enumeration ongoing.",
    " Telemetry: response latency nominal.",
    " Proxy log: 200 OK on last probe.",
    " Tool hint: re-check the discovered surface.",
]


def parse_context_fields(context: str) -> tuple[str, list[tuple[str, str]]]:
    positions = sorted((context.find(label), label) for label in CONTEXT_LABELS if context.find(label) >= 0)
    if not positions:
        return context, []
    prefix = context[: positions[0][0]]
    fields: list[tuple[str, str]] = []
    for i, (start, label) in enumerate(positions):
        value_start = start + len(label)
        value_end = positions[i + 1][0] if i + 1 < len(positions) else len(context)
        fields.append((label, context[value_start:value_end]))
    return prefix, fields


def degrade_context_ood(context: str, rng: random.Random) -> str:
    import re

    prefix, fields = parse_context_fields(context)
    if not fields:
        return context + rng.choice(OOD_DISTRACTORS)
    new_fields: list[tuple[str, str]] = []
    for label, value in fields:
        if label == "Remaining budget:":
            m = re.search(r"-?\d+", value)
            if m:
                jittered = int(m.group(0)) + rng.choice([-2, -1, 1, 2])
                value = value[: m.start()] + str(jittered) + value[m.end():]
        new_fields.append((OOD_RELABEL.get(label, label), value))
    rng.shuffle(new_fields)
    out = prefix + "".join(f"{lab}{val}" for lab, val in new_fields)
    for _ in range(rng.choice([1, 2])):
        out = out + rng.choice(OOD_DISTRACTORS)
    return out


def degrade_row_ood(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    dirty = dict(row)
    dirty["context"] = degrade_context_ood(str(row.get("context", "")), rng)
    return dirty


# ---------------------------------------------------------------------------
# Model construction / training
# ---------------------------------------------------------------------------

def build_pipelines(args: argparse.Namespace) -> dict[str, Pipeline]:
    def tfidf() -> TfidfVectorizer:
        return TfidfVectorizer(ngram_range=(1, 2), min_df=args.min_df, max_features=args.max_features, lowercase=True)

    return {
        "score": Pipeline([("tfidf", tfidf()), ("ridge", Ridge(alpha=args.ridge_alpha, random_state=args.seed))]),
        "rank": Pipeline([("tfidf", tfidf()), ("logreg", LogisticRegression(max_iter=args.max_iter, class_weight="balanced", random_state=args.seed))]),
        "diagnosis": Pipeline([("tfidf", tfidf()), ("logreg", LogisticRegression(max_iter=args.max_iter, class_weight="balanced", random_state=args.seed))]),
    }


def build_training_set(
    rows: list[dict[str, Any]],
    *,
    augment_copies: int,
    use_confidence_weight: bool,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, list[Any]], np.ndarray]:
    rng = random.Random(args.seed)
    texts: list[str] = []
    targets: dict[str, list[Any]] = {"score": [], "rank": [], "diagnosis": []}
    weights: list[float] = []

    for row in rows:
        # Confidence weighting applies ONLY to oracle labels (method §11.2: down-weight
        # noisy oracle labels). Rule-based labels (precondition / unsafe / outside-scope /
        # ambiguous / schema_gap) are CERTAIN negatives and must keep full weight — their
        # low label_confidence reflects schema_confidence=0 ("not mappable"), not noise.
        if use_confidence_weight and row.get("label_source") == "oracle":
            oc = row.get("oracle_label_confidence")
            base_weight = max(float(oc), args.weight_floor) if oc is not None else 1.0
        else:
            base_weight = 1.0
        variants = [row]
        for _ in range(augment_copies):
            variants.append(degrade_row(row, rng, args.degrade_fields, args.ooa_prob))
        for variant in variants:
            texts.append(sample_to_text(variant))
            targets["score"].append(float(variant["score"]))
            targets["rank"].append(str(variant["rank_label"]))
            targets["diagnosis"].append(str(variant["diagnosis"]))
            weights.append(base_weight)

    return texts, targets, np.asarray(weights, dtype=np.float64)


def fit_models(models: dict[str, Pipeline], texts: list[str], targets: dict[str, list[Any]], weights: np.ndarray) -> None:
    models["score"].fit(texts, np.asarray(targets["score"], dtype=np.float32), ridge__sample_weight=weights)
    models["rank"].fit(texts, targets["rank"], logreg__sample_weight=weights)
    models["diagnosis"].fit(texts, targets["diagnosis"], logreg__sample_weight=weights)


# ---------------------------------------------------------------------------
# Evaluation: clean/dirty, calibration, abstention, pairwise
# ---------------------------------------------------------------------------

def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
    return float(ece)


def abstention_curve(confidences: np.ndarray, correct: np.ndarray, threshold: float) -> dict[str, float]:
    keep = confidences >= threshold
    coverage = float(keep.mean())
    high_conf_acc = float(correct[keep].mean()) if keep.any() else 0.0
    overall_acc = float(correct.mean())
    return {
        "threshold": threshold,
        "coverage": coverage,
        "high_confidence_accuracy": high_conf_acc,
        "overall_accuracy": overall_acc,
        "abstention_benefit": high_conf_acc - overall_acc,
    }


def pairwise_ranking_accuracy(rows: list[dict[str, Any]], pred_scores: list[float], eps: float) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[str(row.get("candidate_group"))].append(idx)
    concordant = 0
    total = 0
    for idxs in groups.values():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                true_diff = float(rows[a]["score"]) - float(rows[b]["score"])
                if abs(true_diff) <= eps:
                    continue
                pred_diff = pred_scores[a] - pred_scores[b]
                total += 1
                if (true_diff > 0 and pred_diff > 0) or (true_diff < 0 and pred_diff < 0):
                    concordant += 1
    return {"pairs": total, "pairwise_ranking_accuracy": (concordant / total) if total else None}


def evaluate(models: dict[str, Pipeline], rows: list[dict[str, Any]], texts: list[str], args: argparse.Namespace) -> dict[str, Any]:
    y_score = np.asarray([float(r["score"]) for r in rows], dtype=np.float32)
    y_rank = [str(r["rank_label"]) for r in rows]
    y_diag = [str(r["diagnosis"]) for r in rows]

    score_pred = np.clip(models["score"].predict(texts), 0.0, 1.0)
    rank_pred = list(models["rank"].predict(texts))
    diag_pred = list(models["diagnosis"].predict(texts))

    rank_proba = models["rank"].predict_proba(texts)
    rank_conf = rank_proba.max(axis=1)
    rank_correct = np.asarray([int(p == t) for p, t in zip(rank_pred, y_rank)], dtype=np.float64)

    return {
        "n": len(rows),
        "score_regression": regression_metrics(y_score, score_pred),
        "rank_accuracy": float(accuracy_score(y_rank, rank_pred)),
        "rank_macro_f1": float(f1_score(y_rank, rank_pred, average="macro", zero_division=0)),
        "diagnosis_accuracy": float(accuracy_score(y_diag, diag_pred)),
        "rank_calibration_error": expected_calibration_error(rank_conf, rank_correct),
        "rank_abstention": abstention_curve(rank_conf, rank_correct, args.abstain_threshold),
        "pairwise": pairwise_ranking_accuracy(rows, list(score_pred), args.pairwise_eps),
    }


def _drop(clean: dict[str, Any], dirty: dict[str, Any]) -> dict[str, Any]:
    cp = clean["pairwise"]["pairwise_ranking_accuracy"]
    dp = dirty["pairwise"]["pairwise_ranking_accuracy"]
    return {
        "rank_accuracy_drop": clean["rank_accuracy"] - dirty["rank_accuracy"],
        "score_mae_increase": dirty["score_regression"]["mae"] - clean["score_regression"]["mae"],
        "pairwise_drop": (cp - dp) if cp is not None and dp is not None else None,
    }


def eval_clean_and_dirty(models: dict[str, Pipeline], rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    clean_texts = [sample_to_text(r) for r in rows]
    rng = random.Random(args.seed + 1000)
    dirty_rows = [degrade_row(r, rng, args.degrade_fields, args.ooa_prob) for r in rows]
    dirty_texts = [sample_to_text(r) for r in dirty_rows]
    # Family B (OOD): relabel + reorder + distractors + budget jitter, never seen in training.
    ood_rng = random.Random(args.seed + 2000)
    ood_rows = [degrade_row_ood(r, ood_rng) for r in rows]
    ood_texts = [sample_to_text(r) for r in ood_rows]

    clean = evaluate(models, rows, clean_texts, args)
    dirty = evaluate(models, dirty_rows, dirty_texts, args)
    dirty_ood = evaluate(models, ood_rows, ood_texts, args)
    return {
        "clean": clean,
        "dirty": dirty,
        "dirty_ood": dirty_ood,
        "robustness_drop": _drop(clean, dirty),
        "robustness_drop_ood": _drop(clean, dirty_ood),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a robustness-hardened Pentest-PRM (method §6.1/§11.2/§11.3).")
    parser.add_argument("--train-input", type=Path, default=ROOT / "outputs" / "prm_samples_train.jsonl")
    parser.add_argument("--heldout-input", type=Path, default=ROOT / "outputs" / "prm_samples_heldout.jsonl")
    parser.add_argument("--model-output", type=Path, default=ROOT / "outputs" / "prm_robust.joblib")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "prm_robust_eval.json")
    parser.add_argument("--augment-copies", type=int, default=2, help="Dirty augmented copies per training sample (robust model).")
    parser.add_argument("--degrade-fields", type=int, default=4, help="Number of context fields masked per degraded copy.")
    parser.add_argument("--ooa-prob", type=float, default=0.5, help="Probability of injecting an out-of-abstraction token.")
    parser.add_argument("--weight-floor", type=float, default=0.3, help="Min sample weight for low-confidence oracle labels.")
    parser.add_argument("--abstain-threshold", type=float, default=0.6)
    parser.add_argument("--pairwise-eps", type=float, default=1e-6)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train_input)
    heldout_rows = read_jsonl(args.heldout_input)

    # Baseline: clean, unweighted.
    baseline_models = build_pipelines(args)
    b_texts, b_targets, b_weights = build_training_set(train_rows, augment_copies=0, use_confidence_weight=False, args=args)
    fit_models(baseline_models, b_texts, b_targets, b_weights)

    # Robust: dirty-augmented, confidence-weighted.
    robust_models = build_pipelines(args)
    r_texts, r_targets, r_weights = build_training_set(train_rows, augment_copies=args.augment_copies, use_confidence_weight=True, args=args)
    fit_models(robust_models, r_texts, r_targets, r_weights)

    baseline_eval = eval_clean_and_dirty(baseline_models, heldout_rows, args)
    robust_eval = eval_clean_and_dirty(robust_models, heldout_rows, args)

    base_drop = baseline_eval["robustness_drop"]["rank_accuracy_drop"]
    robust_drop = robust_eval["robustness_drop"]["rank_accuracy_drop"]
    base_ece = baseline_eval["clean"]["rank_calibration_error"]
    robust_ece = robust_eval["clean"]["rank_calibration_error"]
    base_dirty_pair = baseline_eval["dirty"]["pairwise"]["pairwise_ranking_accuracy"]
    robust_dirty_pair = robust_eval["dirty"]["pairwise"]["pairwise_ranking_accuracy"]
    # OOD (held-out corruption family B) comparison.
    base_drop_ood = baseline_eval["robustness_drop_ood"]["rank_accuracy_drop"]
    robust_drop_ood = robust_eval["robustness_drop_ood"]["rank_accuracy_drop"]
    base_ood_pair = baseline_eval["dirty_ood"]["pairwise"]["pairwise_ranking_accuracy"]
    robust_ood_pair = robust_eval["dirty_ood"]["pairwise"]["pairwise_ranking_accuracy"]
    report = {
        "train_input": str(args.train_input),
        "heldout_input": str(args.heldout_input),
        "config": {
            "augment_copies": args.augment_copies,
            "degrade_fields": args.degrade_fields,
            "ooa_prob": args.ooa_prob,
            "weight_floor": args.weight_floor,
            "confidence_weighting": "oracle_labels_only",
            "degrade_axes": "context_fields + normalized_action_representation",
        },
        "baseline": baseline_eval,
        "robust": robust_eval,
        "verdict": {
            "robust_no_clean_accuracy_cost": bool(robust_eval["clean"]["rank_accuracy"] >= baseline_eval["clean"]["rank_accuracy"] - 0.01),
            "robust_better_calibrated": bool(robust_ece <= base_ece),
            "robust_more_stable_dirty_ranking": bool(
                robust_dirty_pair is not None and base_dirty_pair is not None and robust_dirty_pair >= base_dirty_pair
            ),
            "calibration_error_improvement": round(base_ece - robust_ece, 4),
            "dirty_pairwise_improvement": round((robust_dirty_pair - base_dirty_pair), 4) if (robust_dirty_pair and base_dirty_pair) else None,
            "baseline_rank_drop_clean_to_dirty": round(base_drop, 4),
            "robust_rank_drop_clean_to_dirty": round(robust_drop, 4),
            "robust_clean_rank_accuracy": round(robust_eval["clean"]["rank_accuracy"], 4),
            "robust_dirty_rank_accuracy": round(robust_eval["dirty"]["rank_accuracy"], 4),
            "robust_calibration_error_clean": round(robust_ece, 4),
            "robust_pairwise_clean": round(robust_eval["clean"]["pairwise"]["pairwise_ranking_accuracy"], 4),
            "robust_abstention_benefit_clean": round(robust_eval["clean"]["rank_abstention"]["abstention_benefit"], 4),
            "OOD_family_B": "relabel + reorder + spurious-distractor + budget-jitter (held out of training; not masking/OOA)",
            "OOD_baseline_rank_drop_clean_to_B": round(base_drop_ood, 4),
            "OOD_robust_rank_drop_clean_to_B": round(robust_drop_ood, 4),
            "OOD_robust_smaller_rank_drop": bool(robust_drop_ood <= base_drop_ood),
            "OOD_baseline_dirty_pairwise": round(base_ood_pair, 4) if base_ood_pair is not None else None,
            "OOD_robust_dirty_pairwise": round(robust_ood_pair, 4) if robust_ood_pair is not None else None,
            "OOD_robust_pairwise_advantage": round(robust_ood_pair - base_ood_pair, 4) if (robust_ood_pair is not None and base_ood_pair is not None) else None,
            "caveats": (
                "HONEST CAVEATS: (1) ADDRESSED — robustness is now tested on a HELD-OUT corruption family B "
                "(OOD_* fields: relabel/reorder/distractor/jitter, never trained on), not only the training "
                "corruption; read OOD_robust_smaller_rank_drop / OOD_robust_pairwise_advantage for the genuine "
                "OOD result. The same-family ('dirty') numbers remain partly in-distribution by construction. "
                "(2) Full-set rank/pairwise are "
                "inflated by self-predictable rule rows (status copied into input, score=0); see "
                "honest_eval.py for oracle-subset metrics (rank ~0.60, pairwise ~0.60), which are the real "
                "evaluator quality. (3) Single seed, no CIs. The robustness deltas (ECE, dirty pairwise) are "
                "modest and should be read with these caveats."
            ),
            "note": (
                "Confidence weighting is applied to ORACLE labels only (rule-based negatives are certain). "
                "degrade_action drops fields without rewriting the label-bearing status."
            ),
        },
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "score_regressor": robust_models["score"],
            "rank_classifier": robust_models["rank"],
            "diagnosis_classifier": robust_models["diagnosis"],
            "metadata": {"robust": True, "features": "context + raw_llm_action + normalized schema", "config": report["config"]},
        },
        args.model_output,
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    def line(name: str, ev: dict[str, Any]) -> str:
        c, d, o = ev["clean"], ev["dirty"], ev["dirty_ood"]
        return (
            f"{name:9s} clean[rank={c['rank_accuracy']:.3f} pair={c['pairwise']['pairwise_ranking_accuracy']:.3f}]  "
            f"dirtyA[rank={d['rank_accuracy']:.3f} pair={d['pairwise']['pairwise_ranking_accuracy']:.3f}]  "
            f"OOD-B[rank={o['rank_accuracy']:.3f} pair={o['pairwise']['pairwise_ranking_accuracy']:.3f}]"
        )

    print(line("baseline", baseline_eval))
    print(line("robust", robust_eval))
    print("\nverdict:", json.dumps(report["verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
