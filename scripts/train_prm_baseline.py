from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows found: {path}")
    return rows


def sample_to_text(sample: dict[str, Any]) -> str:
    normalized = sample.get("normalized_action") or {}
    fields = [
        f"context: {sample.get('context', '')}",
        f"candidate action: {sample.get('raw_llm_action', '')}",
        f"normalized status: {normalized.get('status')}",
        f"normalized action type: {normalized.get('action_type')}",
        f"normalized target: {normalized.get('target')}",
        f"normalized parameter: {normalized.get('parameter')}",
        f"normalizer reason: {normalized.get('reason')}",
    ]
    return "\n".join(fields)


def build_texts(rows: list[dict[str, Any]]) -> list[str]:
    return [sample_to_text(row) for row in rows]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    if len(y_true) > 1 and float(np.std(y_true)) > 0 and float(np.std(y_pred)) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = 0.0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": pearson,
    }


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = read_jsonl(args.train_input)
    heldout_rows = read_jsonl(args.heldout_input)

    x_train = build_texts(train_rows)
    x_heldout = build_texts(heldout_rows)
    y_score_train = np.asarray([float(row["score"]) for row in train_rows], dtype=np.float32)
    y_score_heldout = np.asarray([float(row["score"]) for row in heldout_rows], dtype=np.float32)
    y_rank_train = [str(row["rank_label"]) for row in train_rows]
    y_rank_heldout = [str(row["rank_label"]) for row in heldout_rows]
    y_diagnosis_train = [str(row["diagnosis"]) for row in train_rows]
    y_diagnosis_heldout = [str(row["diagnosis"]) for row in heldout_rows]

    score_regressor = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=args.min_df,
                    max_features=args.max_features,
                    lowercase=True,
                ),
            ),
            ("ridge", Ridge(alpha=args.ridge_alpha, random_state=args.seed)),
        ]
    )
    rank_classifier = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=args.min_df,
                    max_features=args.max_features,
                    lowercase=True,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=args.max_iter,
                    class_weight="balanced",
                    random_state=args.seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    diagnosis_classifier = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=args.min_df,
                    max_features=args.max_features,
                    lowercase=True,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=args.max_iter,
                    class_weight="balanced",
                    random_state=args.seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    score_regressor.fit(x_train, y_score_train)
    rank_classifier.fit(x_train, y_rank_train)
    diagnosis_classifier.fit(x_train, y_diagnosis_train)

    score_pred = np.clip(score_regressor.predict(x_heldout), 0.0, 1.0)
    rank_pred = list(rank_classifier.predict(x_heldout))
    diagnosis_pred = list(diagnosis_classifier.predict(x_heldout))

    predictions = [
        {
            "sample_id": row["sample_id"],
            "task_id": row["task_id"],
            "step": row["step"],
            "true_score": float(row["score"]),
            "pred_score": float(score_pred[idx]),
            "true_rank_label": row["rank_label"],
            "pred_rank_label": rank_pred[idx],
            "true_diagnosis": row["diagnosis"],
            "pred_diagnosis": diagnosis_pred[idx],
            "raw_llm_action": row["raw_llm_action"],
        }
        for idx, row in enumerate(heldout_rows)
    ]

    metrics = {
        "train_input": str(args.train_input),
        "heldout_input": str(args.heldout_input),
        "model_output": str(args.model_output),
        "num_train": len(train_rows),
        "num_heldout": len(heldout_rows),
        "score_regression": regression_metrics(y_score_heldout, score_pred),
        "rank_label_classification": classification_metrics(y_rank_heldout, rank_pred),
        "diagnosis_classification": classification_metrics(y_diagnosis_heldout, diagnosis_pred),
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "score_regressor": score_regressor,
            "rank_classifier": rank_classifier,
            "diagnosis_classifier": diagnosis_classifier,
            "metadata": {
                "train_input": str(args.train_input),
                "heldout_input": str(args.heldout_input),
                "features": "context + raw_llm_action + normalized action schema",
            },
        },
        args.model_output,
    )

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_output.open("w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight PRM baseline on generated WebAttackSim labels.")
    parser.add_argument("--train-input", type=Path, default=ROOT / "outputs" / "prm_samples_train.jsonl")
    parser.add_argument("--heldout-input", type=Path, default=ROOT / "outputs" / "prm_samples_heldout.jsonl")
    parser.add_argument("--model-output", type=Path, default=ROOT / "outputs" / "prm_baseline.joblib")
    parser.add_argument("--metrics-output", type=Path, default=ROOT / "outputs" / "prm_baseline_eval.json")
    parser.add_argument("--predictions-output", type=Path, default=ROOT / "outputs" / "prm_baseline_predictions.jsonl")
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    metrics = train(args)
    compact = {
        "num_train": metrics["num_train"],
        "num_heldout": metrics["num_heldout"],
        "score_regression": metrics["score_regression"],
        "rank_label": {
            "accuracy": metrics["rank_label_classification"]["accuracy"],
            "macro_f1": metrics["rank_label_classification"]["macro_f1"],
            "weighted_f1": metrics["rank_label_classification"]["weighted_f1"],
        },
        "diagnosis": {
            "accuracy": metrics["diagnosis_classification"]["accuracy"],
            "macro_f1": metrics["diagnosis_classification"]["macro_f1"],
            "weighted_f1": metrics["diagnosis_classification"]["weighted_f1"],
        },
        "metrics_output": str(args.metrics_output),
        "model_output": str(args.model_output),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
