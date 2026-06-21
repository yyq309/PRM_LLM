"""Data-scaling diagnostic: is the PRM starved for simulation tasks?

Answers, with data, whether the 9-family / 50-task sim is "too few" for the PRM by
measuring how PRM generalization moves as we add training FAMILIES and INSTANCES.

Two curves, both evaluated on the HONEST oracle-labeled subset:

1. family-count curve: train the strong PRM on K randomly-chosen train families
   (K = 1..n_train_families, averaged over several random family subsets) and evaluate
   pairwise/rank on the FIXED unseen-chain held-out set (rce_privesc + leak_login,
   never in PRM train at any K). A still-rising curve at K=max => structurally
   data-starved (more families would help). A plateau => task count is NOT the binding
   constraint and the wall is elsewhere (the abstraction / real-target gap).

2. instance-density curve: train on ALL train families but only M instances per family
   (M = 1..max) and evaluate on unseen-instance + unseen-chain. Separates "need more
   topology families" from "need more token variations of the same families".

Oracle (DQN) labels are held fixed (already in the prm_samples jsonl); only the PRM's
training data is subsetted. So this isolates PRM data sufficiency, not oracle quality.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import argparse
import json
import sys
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from honest_eval import majority_floor, pairwise_within_group  # noqa: E402
from train_prm_baseline import read_jsonl  # noqa: E402
from train_prm_strong import extract_features  # noqa: E402
from task_split import get_split  # noqa: E402
from web_attack_sim.tasks import bundled_task_paths, load_task_config  # noqa: E402

UNSEEN_CHAIN_FAMILIES = {"rce_privesc", "leak_login"}


def stem(path_str: str) -> str:
    return Path(str(path_str)).stem


def family_map() -> dict[str, str]:
    fam: dict[str, str] = {}
    for p in bundled_task_paths():
        fam[p.stem] = load_task_config(p).get("family", "?")
    return fam


def row_family(row: dict[str, Any], fam: dict[str, str]) -> str:
    return fam.get(stem(row.get("task_path", "")), "?")


def fit_eval(train_subset: list[dict[str, Any]], held_rows: list[dict[str, Any]],
             eval_idx_groups: dict[str, list[int]], seed: int, eps: float) -> dict[str, Any]:
    """Fit score+rank on the subset, return pairwise/rank on each eval group."""
    if not train_subset:
        return {g: {"n": 0} for g in eval_idx_groups}
    vec = DictVectorizer(sparse=True)
    x_tr = vec.fit_transform([extract_features(r) for r in train_subset]).toarray()
    x_he = vec.transform([extract_features(r) for r in held_rows]).toarray()
    y_score = np.asarray([float(r["score"]) for r in train_subset], dtype=np.float32)
    y_rank = [str(r["rank_label"]) for r in train_subset]

    score = HistGradientBoostingRegressor(random_state=seed, max_iter=300, learning_rate=0.08)
    score.fit(x_tr, y_score)
    score_pred = list(np.clip(score.predict(x_he), 0.0, 1.0))

    rank_pred = None
    if len(set(y_rank)) >= 2:
        rank = HistGradientBoostingClassifier(random_state=seed, max_iter=300, learning_rate=0.08)
        rank.fit(x_tr, y_rank)
        rank_pred = list(rank.predict(x_he))

    out: dict[str, Any] = {}
    for g, idxs in eval_idx_groups.items():
        if not idxs:
            out[g] = {"n": 0}
            continue
        pair = pairwise_within_group(held_rows, score_pred, idxs, eps)
        rec = {"n": len(idxs), "pairwise_accuracy": pair["pairwise_accuracy"], "pairs": pair["pairs"]}
        if rank_pred is not None:
            y_true = [str(held_rows[i]["rank_label"]) for i in idxs]
            rec["rank_accuracy"] = float(np.mean([int(rank_pred[i] == held_rows[i]["rank_label"]) for i in idxs]))
            rec["majority_floor"] = majority_floor(y_true)
        out[g] = rec
    return out


def summarize(runs: list[dict[str, Any]], group: str, metric: str) -> dict[str, float] | None:
    vals = [r[group][metric] for r in runs if r.get(group, {}).get(metric) is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "min": float(min(vals)), "max": float(max(vals))}


def main() -> None:
    parser = argparse.ArgumentParser(description="PRM data-scaling learning curves (family count + instance density).")
    parser.add_argument("--train-input", type=Path, default=ROOT / "outputs" / "prm_samples_train.jsonl")
    parser.add_argument("--heldout-input", type=Path, default=ROOT / "outputs" / "prm_samples_heldout.jsonl")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "prm_learning_curve.json")
    parser.add_argument("--repeats", type=int, default=6, help="Random family-subset draws per K (averaged).")
    parser.add_argument("--pairwise-eps", type=float, default=1e-6)
    args = parser.parse_args()

    fam = family_map()
    train_rows = read_jsonl(args.train_input)
    held_rows = read_jsonl(args.heldout_input)

    split = get_split()
    instance_ids = {stem(p) for p in split["heldout_instance"]}

    # Held-out oracle-labeled eval groups (fixed across all curve points).
    oracle_all = [i for i, r in enumerate(held_rows) if r.get("label_source") == "oracle"]
    oracle_chain = [i for i in oracle_all if row_family(held_rows[i], fam) in UNSEEN_CHAIN_FAMILIES]
    oracle_instance = [i for i in oracle_all if stem(held_rows[i].get("task_path", "")) in instance_ids]
    groups = {"oracle_all": oracle_all, "oracle_unseen_chain": oracle_chain, "oracle_unseen_instance": oracle_instance}

    # Group train rows by family and by task instance.
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_instances: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in train_rows:
        f = row_family(r, fam)
        by_family[f].append(r)
        family_instances[f][stem(r.get("task_path", ""))].append(r)
    train_families = sorted(by_family)

    rng = np.random.default_rng(0)

    # --- Curve 1: family count ---
    family_curve = []
    for k in range(1, len(train_families) + 1):
        runs = []
        row_counts = []
        for rep in range(args.repeats):
            chosen = list(rng.choice(train_families, size=k, replace=False))
            subset = [r for f in chosen for r in by_family[f]]
            row_counts.append(len(subset))
            runs.append(fit_eval(subset, held_rows, groups, seed=rep, eps=args.pairwise_eps))
        point = {"k_families": k, "n_train_rows_mean": float(np.mean(row_counts))}
        for g in groups:
            point[g] = {
                "pairwise_accuracy": summarize(runs, g, "pairwise_accuracy"),
                "rank_accuracy": summarize(runs, g, "rank_accuracy"),
            }
        family_curve.append(point)

    # --- Curve 2: instance density (all families, M instances each) ---
    max_inst = max(len(v) for v in family_instances.values())
    instance_curve = []
    for m in range(1, max_inst + 1):
        subset = []
        for f, insts in family_instances.items():
            for tid in sorted(insts)[:m]:
                subset.extend(insts[tid])
        run = fit_eval(subset, held_rows, groups, seed=0, eps=args.pairwise_eps)
        instance_curve.append({
            "m_instances_per_family": m,
            "n_train_rows": len(subset),
            **{g: {"pairwise_accuracy": run[g].get("pairwise_accuracy"), "rank_accuracy": run[g].get("rank_accuracy"), "n": run[g].get("n")} for g in groups},
        })

    # Full-data reference (all train families, all instances).
    full = fit_eval(train_rows, held_rows, groups, seed=0, eps=args.pairwise_eps)

    report = {
        "n_train_families": len(train_families),
        "train_families": train_families,
        "unseen_chain_families": sorted(UNSEEN_CHAIN_FAMILIES),
        "oracle_subset_sizes": {g: len(idx) for g, idx in groups.items()},
        "repeats_per_k": args.repeats,
        "family_count_curve": family_curve,
        "instance_density_curve": instance_curve,
        "full_data_reference": full,
        "note": (
            "All metrics are pairwise/rank accuracy on the oracle-labeled held-out subset. "
            "family_count_curve evaluates on the FIXED unseen-chain set at every K, averaged over "
            "random family draws. If oracle_unseen_chain pairwise keeps rising through the last K, "
            "the PRM is structurally data-starved (more families would help). If it plateaus early, "
            "task count is not the binding constraint."
        ),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(s):
        return f"{s['mean']:.3f}±{s['std']:.3f}" if s else "  n/a  "

    print(f"train families: {train_families}")
    print(f"oracle subset sizes: {report['oracle_subset_sizes']}")
    print("\n== family-count curve (pairwise on fixed held-out groups) ==")
    print(f"{'K':>2}  {'all':>14}  {'unseen_chain':>14}  {'unseen_inst':>14}")
    for p in family_curve:
        print(f"{p['k_families']:>2}  {fmt(p['oracle_all']['pairwise_accuracy']):>14}  "
              f"{fmt(p['oracle_unseen_chain']['pairwise_accuracy']):>14}  {fmt(p['oracle_unseen_instance']['pairwise_accuracy']):>14}")
    print("\n== instance-density curve (M instances/family, pairwise) ==")
    print(f"{'M':>2}  {'rows':>5}  {'all':>8}  {'chain':>8}  {'inst':>8}")
    for p in instance_curve:
        def g(x):
            v = x.get("pairwise_accuracy")
            return f"{v:.3f}" if v is not None else " n/a "
        print(f"{p['m_instances_per_family']:>2}  {p['n_train_rows']:>5}  {g(p['oracle_all']):>8}  {g(p['oracle_unseen_chain']):>8}  {g(p['oracle_unseen_instance']):>8}")
    fa = full["oracle_all"]["pairwise_accuracy"]
    fc = full["oracle_unseen_chain"]["pairwise_accuracy"]
    print(f"\nfull-data reference: all={fa:.3f} chain={fc:.3f}")


if __name__ == "__main__":
    main()
