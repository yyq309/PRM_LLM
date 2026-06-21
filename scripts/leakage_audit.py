"""Leakage audit for the Pentest-PRM input (method §10.1 / plan §6.1).

The value_gap labels are produced by an oracle that knows the hidden ground
truth, while the PRM input must be a strict function of *observable* state and
history. If a hidden field (an undiscovered path, an unleaked credential, the
flag value, a target file path) leaks into the context, the PRM learns to
predict value from the leak: it looks great on the test split and fails on
deployment.

This audit has two independent parts:

1. Structural hidden-truth leak test. For every PRM sample, reconstruct the set
   of hidden-truth tokens for that task (credential passwords, flag values,
   internal file system paths, and not-yet-discovered hidden paths) and assert
   none of them appear in the verbalized context. This is a hard unit test:
   the leak count must be 0.

2. Masking ablation. Re-train the PRM with each observable context field masked
   in turn and measure held-out degradation. A graceful drop means the field is
   a normal, non-essential observable feature; a cliff would flag a single field
   the model over-relies on and warrant scrutiny. Combined with part 1 (which
   proves no hidden token is present), graceful degradation is the operational
   definition of "the PRM does not peek at hidden truth".
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

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_prm_baseline import (  # noqa: E402
    classification_metrics,
    read_jsonl,
    regression_metrics,
    sample_to_text,
)
from web_attack_sim.tasks import bundled_task_paths, load_task_config  # noqa: E402


# Context field labels emitted by demo_pipeline.context_from_observation, in order.
CONTEXT_LABELS = [
    "Known paths:",
    "Known forms:",
    "Known parameters:",
    "Credentials:",
    "Auth state:",
    "Shell state:",
    "Verified vulnerabilities:",
    "Read files:",
    "Failed branches:",
    "Remaining budget:",
    "Recent feedback:",
]


def mask_field(context: str, label: str) -> str:
    start = context.find(label)
    if start < 0:
        return context
    value_start = start + len(label)
    next_positions = [context.find(other, value_start) for other in CONTEXT_LABELS]
    next_positions = [pos for pos in next_positions if pos >= 0]
    value_end = min(next_positions) if next_positions else len(context)
    return context[:value_start] + " [MASKED]. " + context[value_end:]


# Passwords equal to these dictionary words coincide with the observable action
# vocabulary and credential identifiers, so string matching cannot distinguish a
# leak from legitimate observable text. They are reported, not failed.
COMMON_CREDENTIAL_WORDS = {"admin", "password", "user", "guest", "root", "test", "login", "username", "pass", "default"}


def visible_paths(context: str) -> set[str]:
    match = re.search(r"Known paths: \[([^\]]*)\]", context)
    if not match:
        return set()
    return set(re.findall(r"'(/[^']*)'", match.group(1)))


def strip_task_label(context: str, task_id: str) -> str:
    """Remove the scenario/task label so its name does not masquerade as a secret."""
    body = re.sub(r"^(Task|Scenario) [^.]*\.\s*", "", context)
    return body.replace(task_id, "")


def contains_token(text: str, token: str) -> bool:
    """Boundary-aware containment; falls back to substring for tokens with non-word chars."""
    if re.search(r"\W", token):
        return token in text
    return re.search(r"\b" + re.escape(token) + r"\b", text) is not None


def hidden_tokens_for_task(task: dict[str, Any]) -> dict[str, list[str]]:
    unique_secrets: list[str] = []
    dictionary_secrets: list[str] = []
    for credential in task.get("credentials", {}).values():
        password = credential.get("password")
        if not password:
            continue
        (dictionary_secrets if str(password).lower() in COMMON_CREDENTIAL_WORDS else unique_secrets).append(str(password))
    file_paths: list[str] = []
    for spec in task.get("files", {}).values():
        flag = spec.get("flag")
        if flag:
            unique_secrets.append(str(flag))  # flag values are always unique hidden truth
        path = spec.get("path")
        if path:
            file_paths.append(str(path))
    return {
        "unique_secrets": sorted(set(unique_secrets)),  # flags + non-dictionary passwords: must never appear
        "dictionary_secrets": sorted(set(dictionary_secrets)),  # reported only; not string-distinguishable
        "file_paths": sorted(set(file_paths)),
        "hidden_paths": list(task.get("hidden_paths", [])),
    }


def structural_leak_test(rows: list[dict[str, Any]], task_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    leaks: list[dict[str, Any]] = []
    secret_hits = 0
    file_path_hits = 0
    future_path_hits = 0
    descriptive_task_id_in_context = 0
    dictionary_secret_total = 0

    for row in rows:
        context = str(row.get("context", ""))
        task_id = str(row.get("task_id"))
        task = task_index.get(task_id)
        if task is None:
            continue
        tokens = hidden_tokens_for_task(task)
        dictionary_secret_total += len(tokens["dictionary_secrets"])
        # The descriptive task_id (e.g. ...sqli..., ...upload...) names the scenario family.
        if context.startswith("Task ") and any(part in task_id for part in ["sqli", "lfi", "rce", "upload", "backup", "leak", "password", "privesc"]):
            descriptive_task_id_in_context += 1

        scan_text = strip_task_label(context, task_id)
        visible = visible_paths(context)

        for secret in tokens["unique_secrets"]:
            if contains_token(scan_text, secret):
                secret_hits += 1
                leaks.append({"sample_id": row.get("sample_id"), "type": "unique_secret", "token": secret})

        for path in tokens["file_paths"]:
            if path not in visible and contains_token(scan_text, path):
                file_path_hits += 1
                leaks.append({"sample_id": row.get("sample_id"), "type": "file_path", "token": path})

        for path in tokens["hidden_paths"]:
            if path not in visible and contains_token(scan_text, path):
                future_path_hits += 1
                leaks.append({"sample_id": row.get("sample_id"), "type": "future_path", "token": path})

    total = secret_hits + file_path_hits + future_path_hits
    return {
        "num_samples": len(rows),
        "unique_secret_leaks": secret_hits,
        "file_path_leaks": file_path_hits,
        "future_path_leaks": future_path_hits,
        "total_leaks": total,
        "passed": total == 0,
        "dictionary_secret_occurrences_skipped": dictionary_secret_total,
        "samples_with_descriptive_task_id": descriptive_task_id_in_context,
        "examples": leaks[:20],
    }


def build_pipelines(args: argparse.Namespace) -> tuple[Pipeline, Pipeline, Pipeline]:
    def tfidf() -> TfidfVectorizer:
        return TfidfVectorizer(ngram_range=(1, 2), min_df=args.min_df, max_features=args.max_features, lowercase=True)

    score = Pipeline([("tfidf", tfidf()), ("ridge", Ridge(alpha=args.ridge_alpha, random_state=args.seed))])
    rank = Pipeline(
        [("tfidf", tfidf()), ("logreg", LogisticRegression(max_iter=args.max_iter, class_weight="balanced", random_state=args.seed))]
    )
    diagnosis = Pipeline(
        [("tfidf", tfidf()), ("logreg", LogisticRegression(max_iter=args.max_iter, class_weight="balanced", random_state=args.seed))]
    )
    return score, rank, diagnosis


def text_with_mask(row: dict[str, Any], mask_label: str | None) -> str:
    if mask_label is None:
        return sample_to_text(row)
    masked = dict(row)
    if mask_label == "__ALL_CONTEXT__":
        masked["context"] = "[MASKED]"
    else:
        masked["context"] = mask_field(str(row.get("context", "")), mask_label)
    return sample_to_text(masked)


def fit_eval(
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    mask_label: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    x_train = [text_with_mask(row, mask_label) for row in train_rows]
    x_heldout = [text_with_mask(row, mask_label) for row in heldout_rows]
    y_score_train = np.asarray([float(row["score"]) for row in train_rows], dtype=np.float32)
    y_score_heldout = np.asarray([float(row["score"]) for row in heldout_rows], dtype=np.float32)
    y_rank_train = [str(row["rank_label"]) for row in train_rows]
    y_rank_heldout = [str(row["rank_label"]) for row in heldout_rows]
    y_diag_train = [str(row["diagnosis"]) for row in train_rows]
    y_diag_heldout = [str(row["diagnosis"]) for row in heldout_rows]

    score_reg, rank_clf, diag_clf = build_pipelines(args)
    score_reg.fit(x_train, y_score_train)
    rank_clf.fit(x_train, y_rank_train)
    diag_clf.fit(x_train, y_diag_train)

    score_pred = np.clip(score_reg.predict(x_heldout), 0.0, 1.0)
    rank_pred = list(rank_clf.predict(x_heldout))
    diag_pred = list(diag_clf.predict(x_heldout))

    return {
        "score_regression": regression_metrics(y_score_heldout, score_pred),
        "rank_accuracy": classification_metrics(y_rank_heldout, rank_pred)["accuracy"],
        "rank_macro_f1": classification_metrics(y_rank_heldout, rank_pred)["macro_f1"],
        "diagnosis_accuracy": classification_metrics(y_diag_heldout, diag_pred)["accuracy"],
        "diagnosis_macro_f1": classification_metrics(y_diag_heldout, diag_pred)["macro_f1"],
    }


def classify_degradation(baseline: dict[str, Any], masked: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mae_increase = masked["score_regression"]["mae"] - baseline["score_regression"]["mae"]
    rank_drop = baseline["rank_accuracy"] - masked["rank_accuracy"]
    diag_drop = baseline["diagnosis_accuracy"] - masked["diagnosis_accuracy"]
    pearson_drop = baseline["score_regression"]["pearson"] - masked["score_regression"]["pearson"]
    is_cliff = rank_drop > args.cliff_rank_drop or mae_increase > args.cliff_mae_increase or pearson_drop > args.cliff_pearson_drop
    return {
        "mae_increase": round(float(mae_increase), 4),
        "pearson_drop": round(float(pearson_drop), 4),
        "rank_accuracy_drop": round(float(rank_drop), 4),
        "diagnosis_accuracy_drop": round(float(diag_drop), 4),
        "degradation": "cliff" if is_cliff else "graceful",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = read_jsonl(args.train_input)
    heldout_rows = read_jsonl(args.heldout_input)
    task_index = {str(load_task_config(path)["task_id"]): load_task_config(path) for path in bundled_task_paths()}

    structural = {
        "train": structural_leak_test(train_rows, task_index),
        "heldout": structural_leak_test(heldout_rows, task_index),
    }

    baseline = fit_eval(train_rows, heldout_rows, None, args)
    fields = [(label.rstrip(":"), label) for label in CONTEXT_LABELS]
    fields.append(("ALL_CONTEXT", "__ALL_CONTEXT__"))

    ablation: list[dict[str, Any]] = []
    for field_name, label in fields:
        masked = fit_eval(train_rows, heldout_rows, label, args)
        ablation.append(
            {
                "masked_field": field_name,
                "metrics": masked,
                "delta_vs_baseline": classify_degradation(baseline, masked, args),
            }
        )

    num_cliffs = sum(1 for item in ablation if item["delta_vs_baseline"]["degradation"] == "cliff")
    report = {
        "train_input": str(args.train_input),
        "heldout_input": str(args.heldout_input),
        "structural_leak_test": structural,
        "masking_ablation": {
            "baseline": baseline,
            "fields": ablation,
            "thresholds": {
                "cliff_rank_drop": args.cliff_rank_drop,
                "cliff_mae_increase": args.cliff_mae_increase,
                "cliff_pearson_drop": args.cliff_pearson_drop,
            },
        },
        "verdict": {
            "no_hidden_truth_leak": structural["train"]["passed"] and structural["heldout"]["passed"],
            "degrades_gracefully": num_cliffs == 0,
            "num_cliff_fields": num_cliffs,
            "metadata_hygiene_warning": (
                structural["train"]["samples_with_descriptive_task_id"] > 0
                or structural["heldout"]["samples_with_descriptive_task_id"] > 0
            ),
            "metadata_hygiene_recommendation": (
                "context_from_observation prefixes the verbalized state with the descriptive task_id "
                "(e.g. 'web_002_sqli_login'), which names the vulnerability family. This is not hidden "
                "instance truth (no path/credential/flag), but for clean leakage hygiene the context "
                "builder should use an opaque scenario id so the PRM cannot read the vuln type off the label."
            ),
            "note": (
                "A cliff on an *observable* field is not leakage (part 1 proves no hidden token is "
                "present); it only means the PRM leans on that legitimate field. Leakage would be a "
                "non-zero unique_secret/file_path/future_path count. Dictionary-word passwords (e.g. "
                "'admin', 'password') are skipped because they collide with the observable action "
                "vocabulary and cannot be string-distinguished from legitimate text."
            ),
        },
    }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Pentest-PRM input for hidden-truth leakage.")
    parser.add_argument("--train-input", type=Path, default=ROOT / "outputs" / "prm_samples_train.jsonl")
    parser.add_argument("--heldout-input", type=Path, default=ROOT / "outputs" / "prm_samples_heldout.jsonl")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "leakage_audit.json")
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cliff-rank-drop", type=float, default=0.15)
    parser.add_argument("--cliff-mae-increase", type=float, default=0.06)
    parser.add_argument("--cliff-pearson-drop", type=float, default=0.20)
    args = parser.parse_args()

    report = run(args)
    st = report["structural_leak_test"]
    print(
        json.dumps(
            {
                "structural_leak_test": {
                    "train": {
                        "num_samples": st["train"]["num_samples"],
                        "total_leaks": st["train"]["total_leaks"],
                        "unique_secret_leaks": st["train"]["unique_secret_leaks"],
                        "file_path_leaks": st["train"]["file_path_leaks"],
                        "future_path_leaks": st["train"]["future_path_leaks"],
                        "passed": st["train"]["passed"],
                    },
                    "heldout": {
                        "num_samples": st["heldout"]["num_samples"],
                        "total_leaks": st["heldout"]["total_leaks"],
                        "passed": st["heldout"]["passed"],
                    },
                },
                "verdict": report["verdict"],
                "report_output": str(args.report_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    base = report["masking_ablation"]["baseline"]
    print(
        f"\nbaseline: score_mae={base['score_regression']['mae']:.4f} "
        f"score_pearson={base['score_regression']['pearson']:.4f} "
        f"rank_acc={base['rank_accuracy']:.4f} diag_acc={base['diagnosis_accuracy']:.4f}"
    )
    print(f"{'masked_field':24s} {'rank_drop':>10s} {'mae_incr':>9s} {'pearson_drop':>12s} {'degradation':>12s}")
    for item in report["masking_ablation"]["fields"]:
        d = item["delta_vs_baseline"]
        print(
            f"{item['masked_field']:24s} {d['rank_accuracy_drop']:>10.4f} {d['mae_increase']:>9.4f} "
            f"{d['pearson_drop']:>12.4f} {d['degradation']:>12s}"
        )


if __name__ == "__main__":
    main()
