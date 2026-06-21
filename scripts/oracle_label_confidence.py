"""Multi-seed oracle label confidence for PRM supervision (method §11.2).

Single-checkpoint labels carry no uncertainty estimate. The method asks for a per-label
Conf_oracle from seed agreement, value-gap variance, and the top-1/top-2 margin, and to
down-weight (or robustify) low-confidence labels rather than trusting every label equally.

This regenerates labels from all 3 de-templated oracle seeds (aligned by sample_id) and
emits a merged dataset with, per oracle-labeled sample:
  - robust labels: modal rank_label + mean score across seeds (more stable than one seed),
  - seed_rank_agreement (fraction of seeds at the modal rank),
  - score_std / value_gap_std (the label's cross-seed uncertainty),
  - margin_confidence (mean top1-top2 Q gap), and
  - multiseed_confidence = Conf_oracle (the product, clamped) used as a PRM sample weight.
Rule-based labels (precondition/normalizer) are oracle-independent and certain -> conf 1.0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_prm_baseline import read_jsonl  # noqa: E402


def index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["sample_id"]: r for r in rows}


def margin_of(sample: dict[str, Any]) -> float | None:
    rep = sample.get("oracle_q_report")
    if not rep:
        return None
    top = rep.get("top_actions") or []
    if len(top) < 2:
        return 1.0
    return abs(float(top[0]["q"]) - float(top[1]["q"]))


def merge(seed_rows: list[dict[str, dict[str, Any]]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = seed_rows[0]
    merged: list[dict[str, Any]] = []
    agreements: list[float] = []
    confidences: list[float] = []
    gap_stds: list[float] = []
    n_oracle = 0

    for sid, row in base.items():
        variants = [s[sid] for s in seed_rows if sid in s]
        out = dict(row)
        if row.get("label_source") == "oracle" and len(variants) == len(seed_rows):
            n_oracle += 1
            ranks = [str(v["rank_label"]) for v in variants]
            scores = [float(v["score"]) for v in variants]
            gaps = [float(v["value_gap"]) for v in variants if v.get("value_gap") is not None]
            margins = [m for m in (margin_of(v) for v in variants) if m is not None]

            modal, modal_count = Counter(ranks).most_common(1)[0]
            agreement = modal_count / len(ranks)
            score_mean = sum(scores) / len(scores)
            score_std = math.sqrt(sum((s - score_mean) ** 2 for s in scores) / len(scores))
            gap_mean = sum(gaps) / len(gaps) if gaps else None
            gap_std = math.sqrt(sum((g - gap_mean) ** 2 for g in gaps) / len(gaps)) if gaps and len(gaps) > 1 else 0.0
            margin_conf = min((sum(margins) / len(margins)) / args.margin_scale, 1.0) if margins else 0.5

            # Conf_oracle = agreement x (variance penalty) x margin confidence.
            var_penalty = math.exp(-args.var_weight * score_std)
            conf = max(0.0, min(1.0, agreement * var_penalty * (0.5 + 0.5 * margin_conf)))

            # Robust labels: modal rank + mean score.
            out["rank_label"] = modal
            out["score"] = round(score_mean, 6)
            out["multiseed_confidence"] = round(conf, 4)
            out["seed_rank_agreement"] = round(agreement, 4)
            out["score_std"] = round(score_std, 4)
            out["value_gap_mean"] = round(gap_mean, 4) if gap_mean is not None else None
            out["value_gap_std"] = round(gap_std, 4)
            out["margin_confidence"] = round(margin_conf, 4)
            agreements.append(agreement)
            confidences.append(conf)
            gap_stds.append(gap_std)
        else:
            # rule-based labels are certain and oracle-independent
            out["multiseed_confidence"] = 1.0
            out["seed_rank_agreement"] = 1.0
            out["score_std"] = 0.0
        merged.append(out)

    report = {
        "num_samples": len(merged),
        "num_oracle_labeled": n_oracle,
        "oracle_seed_rank_agreement_mean": round(sum(agreements) / max(len(agreements), 1), 4) if agreements else None,
        "oracle_frac_full_agreement": round(sum(1 for a in agreements if a >= 0.999) / max(len(agreements), 1), 4) if agreements else None,
        "oracle_multiseed_confidence_mean": round(sum(confidences) / max(len(confidences), 1), 4) if confidences else None,
        "oracle_value_gap_std_mean": round(sum(gap_stds) / max(len(gap_stds), 1), 4) if gap_stds else None,
    }
    return merged, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed oracle label confidence (method §11.2).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--margin-scale", type=float, default=3.0)
    parser.add_argument("--var-weight", type=float, default=2.0)
    parser.add_argument("--train-output", type=Path, default=ROOT / "outputs" / "prm_train_multiseed.jsonl")
    parser.add_argument("--heldout-output", type=Path, default=ROOT / "outputs" / "prm_heldout_multiseed.jsonl")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "label_confidence_report.json")
    args = parser.parse_args()

    report: dict[str, Any] = {}
    for split, out_path in [("train", args.train_output), ("heldout", args.heldout_output)]:
        seed_rows = [index_by_id(read_jsonl(ROOT / "outputs" / f"prm_{split}_seed_{s}.jsonl")) for s in args.seeds]
        merged, split_report = merge(seed_rows, args)
        write_jsonl(out_path, merged)
        report[split] = split_report

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
