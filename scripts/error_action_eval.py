"""Error-action identification metrics for the Pentest-PRM (error-trajectory scoring).

The PRM's headline metric (pairwise ranking) measures ordering of GOOD candidates. This
script measures the complementary, previously-unreported question the simulator's reward
design is built around: **does the PRM correctly flag actions that should NOT be taken**
(precondition-missing, unsafe, out-of-scope, duplicate, premature, or simply low-value)?

Binary error label per held-out candidate:
    is_error = rank_label NOT in {high, medium}
    => low / precondition_missing / unsafe / outside_single_host_web_scope / ambiguous / schema_gap

Two detectors from the persisted strong PRM (prm_strong.joblib):
  - score detector:  error_score = 1 - prm_score   (threshold-free ROC-AUC / PR-AUC)
  - rank-head:       is_error_pred = predicted rank_label not in {high, medium}  (precision/recall/F1)

Reported on three honest cuts:
  - full_set      : inflated (rule rows copy the status into the input -> trivially separable)
  - oracle_subset : the GENUINE test -> among VALID, executed candidates, error == oracle "low"
  - rule_subset   : sanity (should be near-perfect)
Plus per-error-category recall of the rank head (does it catch each error type?).
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import argparse
import json
import sys
from collections import Counter
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_prm_baseline import read_jsonl  # noqa: E402
from train_prm_strong import extract_features  # noqa: E402

GOOD = {"high", "medium"}


def is_error(rank_label: str) -> int:
    return int(rank_label not in GOOD)


def detector_metrics(y_true: list[int], err_score: list[float], err_pred: list[int]) -> dict[str, Any]:
    """ROC/PR AUC for the threshold-free score detector + P/R/F1 for the rank-head detector."""
    n = len(y_true)
    pos = sum(y_true)
    out: dict[str, Any] = {"n": n, "n_error": pos, "error_base_rate": round(pos / max(n, 1), 4)}
    if 0 < pos < n:
        out["score_roc_auc"] = round(float(roc_auc_score(y_true, err_score)), 4)
        out["score_pr_auc"] = round(float(average_precision_score(y_true, err_score)), 4)
    else:
        out["score_roc_auc"] = None
        out["score_pr_auc"] = None
    p, r, f, _ = precision_recall_fscore_support(y_true, err_pred, average="binary", pos_label=1, zero_division=0)
    out["rankhead_precision"] = round(float(p), 4)
    out["rankhead_recall"] = round(float(r), 4)
    out["rankhead_f1"] = round(float(f), 4)
    # score-detector operating point at prm_score < 0.5 (err_score > 0.5)
    op_pred = [int(s > 0.5) for s in err_score]
    p2, r2, f2, _ = precision_recall_fscore_support(y_true, op_pred, average="binary", pos_label=1, zero_division=0)
    out["score_at_0.5_precision"] = round(float(p2), 4)
    out["score_at_0.5_recall"] = round(float(r2), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Error-action identification metrics for the PRM.")
    parser.add_argument("--heldout-input", type=Path, default=ROOT / "outputs" / "prm_samples_heldout.jsonl")
    parser.add_argument("--model", type=Path, default=ROOT / "outputs" / "prm_strong.joblib")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "error_action_eval.json")
    args = parser.parse_args()

    rows = read_jsonl(args.heldout_input)
    model = joblib.load(args.model)
    if model.get("kind") != "strong":
        raise SystemExit("expected a strong PRM joblib (run train_prm_strong.py first)")

    # Batched predictions over the whole held-out set.
    X = model["vectorizer"].transform([extract_features(r) for r in rows]).toarray()
    prm_score = np.clip(model["score"].predict(X), 0.0, 1.0)
    prm_rank = list(model["rank"].predict(X))

    y_true = [is_error(str(r["rank_label"])) for r in rows]
    err_score = [float(1.0 - prm_score[i]) for i in range(len(rows))]
    err_pred = [int(str(prm_rank[i]) not in GOOD) for i in range(len(rows))]

    oracle_idx = [i for i, r in enumerate(rows) if r.get("label_source") == "oracle"]
    rule_idx = [i for i, r in enumerate(rows) if r.get("label_source") != "oracle"]

    def cut(idxs: list[int]) -> dict[str, Any]:
        return detector_metrics([y_true[i] for i in idxs], [err_score[i] for i in idxs], [err_pred[i] for i in idxs])

    # On the oracle subset, "error" is specifically the valid-but-low-value action (oracle rank "low").
    oracle_cut = cut(oracle_idx)
    oracle_cut["error_definition"] = "oracle rank_label == 'low' (valid action, but a poor choice at this state)"

    # Per-error-category recall of the rank head: of true rows in each error category, what fraction
    # does the PRM flag as SOME error (not high/medium)?
    per_cat: dict[str, dict[str, Any]] = {}
    cats = sorted({str(r["rank_label"]) for r in rows if is_error(str(r["rank_label"]))})
    for cat in cats:
        idxs = [i for i, r in enumerate(rows) if str(r["rank_label"]) == cat]
        flagged = sum(err_pred[i] for i in idxs)
        exact = sum(int(str(prm_rank[i]) == cat) for i in idxs)
        per_cat[cat] = {"n": len(idxs), "recall_as_any_error": round(flagged / max(len(idxs), 1), 4),
                        "recall_exact_category": round(exact / max(len(idxs), 1), 4)}

    report = {
        "model": str(args.model),
        "n_heldout": len(rows),
        "error_label_rule": "is_error = rank_label not in {high, medium}",
        "full_set_INFLATED": {**cut(list(range(len(rows)))),
                              "CAVEAT": "rule rows copy the normalizer status into the PRM input, so errors are trivially separable here; not the genuine signal."},
        "oracle_subset_HEADLINE": oracle_cut,
        "rule_subset_sanity": cut(rule_idx),
        "per_error_category_recall": per_cat,
        "predicted_rank_distribution": dict(sorted(Counter(str(x) for x in prm_rank).items())),
        "note": (
            "HEADLINE = oracle_subset: among valid, executed candidates the PRM must flag the low-value "
            "('wrong choice') actions. Score ROC/PR-AUC is the threshold-free error-discrimination; the "
            "rank-head P/R is the interpretable operating point. Full-set is reported but inflated."
        ),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def show(name: str, c: dict[str, Any]) -> None:
        print(f"{name:22s} n={c['n']:4d} err_rate={c['error_base_rate']:.2f}  "
              f"score_ROC_AUC={c['score_roc_auc']}  score_PR_AUC={c['score_pr_auc']}  "
              f"rankhead P/R/F1={c['rankhead_precision']:.3f}/{c['rankhead_recall']:.3f}/{c['rankhead_f1']:.3f}")

    show("full_set (inflated)", report["full_set_INFLATED"])
    show("oracle_subset (REAL)", report["oracle_subset_HEADLINE"])
    show("rule_subset (sanity)", report["rule_subset_sanity"])
    print("per-error-category recall (flagged as any error / exact category):")
    for cat, c in per_cat.items():
        print(f"  {cat:32s} n={c['n']:4d}  any={c['recall_as_any_error']:.3f}  exact={c['recall_exact_category']:.3f}")


if __name__ == "__main__":
    main()
