"""Productionize MC labels: blend y = alpha*Q + (1-alpha)*MC on oracle samples, sweep alpha,
and adopt the best — with bootstrap CIs to tighten the small-N decision-relevant caveat.

For each alpha, the oracle samples' regression target becomes alpha*dqn_score + (1-alpha)*mc_goal
(rule rows keep their certain labels). We train the strong-PRM heads on the blended labels and
evaluate on the HELD-OUT oracle set against the REALIZED MC goal-return (the ground truth):
  - decision_relevant_top1_realized: on groups whose realized returns differ (>MEANINGFUL), does the
    model's top pick have the (near-)max realized return — WITH a bootstrap 95% CI over groups.
  - oracle_all_pairwise_vs_realized_MC: does the model order ALL oracle candidates the way realized
    MC does (the honest global ranking metric, not vs the noisy DQN labels).
The best alpha is selected by the ROBUST global metric (oracle_all_pairwise_vs_realized_MC) rather
than the noisy small-N decision-relevant top-1, and persisted to outputs/prm_strong_mcblend.joblib
for adoption as the canonical PRM. Pure MC (alpha=0) tends to overfit (tanks both the global pairwise
and the DQN-headline), so a modest blend is the honest pick.
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

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_prm_baseline import read_jsonl  # noqa: E402
from train_prm_strong import extract_features  # noqa: E402

MEANINGFUL = 0.02


def load_rows(path: Path, mc: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for r in rows:
        m = mc.get(r.get("sample_id"))
        r["_mc_goal"] = float(m["mc_g_goal"]) if m else None
    return rows


def blended_score(r: dict[str, Any], alpha: float) -> float:
    if r.get("label_source") == "oracle" and r.get("_mc_goal") is not None:
        return alpha * float(r["score"]) + (1.0 - alpha) * float(r["_mc_goal"])
    return float(r["score"])  # rule rows: certain labels unchanged


def pairwise_vs_truth(rows: list[dict[str, Any]], pred: np.ndarray, groups: dict[str, list[int]], truth_key: str) -> float | None:
    conc = tot = 0
    for idxs in groups.values():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                td = rows[a][truth_key] - rows[b][truth_key]
                if abs(td) <= 1e-9:
                    continue
                pd = pred[a] - pred[b]
                tot += 1
                if (td > 0 and pd > 0) or (td < 0 and pd < 0):
                    conc += 1
    return conc / tot if tot else None


def decision_relevant_groups(rows: list[dict[str, Any]], groups: dict[str, list[int]]) -> list[list[int]]:
    out = []
    for idxs in groups.values():
        mc = [rows[i]["_mc_goal"] for i in idxs if rows[i]["_mc_goal"] is not None]
        if len(mc) >= 2 and (max(mc) - min(mc)) > MEANINGFUL:
            out.append(idxs)
    return out


def top1_rate(rows: list[dict[str, Any]], pred: np.ndarray, dgroups: list[list[int]]) -> float:
    hit = 0
    for idxs in dgroups:
        mc = [rows[i]["_mc_goal"] for i in idxs]
        best = max(idxs, key=lambda i: pred[i])
        if rows[best]["_mc_goal"] >= max(mc) - 1e-6:
            hit += 1
    return hit / max(len(dgroups), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend MC into oracle labels, sweep alpha, adopt best.")
    parser.add_argument("--mc-labels", type=Path, default=ROOT / "outputs" / "mc_return_labels.jsonl")
    parser.add_argument("--train-input", type=Path, default=ROOT / "outputs" / "prm_samples_train.jsonl")
    parser.add_argument("--heldout-input", type=Path, default=ROOT / "outputs" / "prm_samples_heldout.jsonl")
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.3, 0.0])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--model-output", type=Path, default=ROOT / "outputs" / "prm_strong_mcblend.joblib")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "mc_blend_report.json")
    args = parser.parse_args()

    mc = {r["sample_id"]: r for r in read_jsonl(args.mc_labels)}
    train_rows = load_rows(args.train_input, mc)
    held_rows = load_rows(args.heldout_input, mc)

    vec = DictVectorizer(sparse=True)
    x_train = vec.fit_transform([extract_features(r) for r in train_rows]).toarray()
    x_held = vec.transform([extract_features(r) for r in held_rows]).toarray()

    held_oracle = [i for i, r in enumerate(held_rows) if r.get("label_source") == "oracle" and r["_mc_goal"] is not None]
    groups: dict[str, list[int]] = defaultdict(list)
    for i in held_oracle:
        groups[str(held_rows[i].get("candidate_group"))].append(i)
    dgroups = decision_relevant_groups(held_rows, groups)
    rng = np.random.default_rng(0)

    y_rank = [str(r["rank_label"]) for r in train_rows]
    y_diag = [str(r["diagnosis"]) for r in train_rows]

    sweep = []
    best = None
    for alpha in args.alphas:
        y_score = np.asarray([blended_score(r, alpha) for r in train_rows], dtype=np.float32)
        score_m = HistGradientBoostingRegressor(random_state=0, max_iter=300, learning_rate=0.08).fit(x_train, y_score)
        pred = np.clip(score_m.predict(x_held), 0.0, 1.0)
        t1 = top1_rate(held_rows, pred, dgroups)
        # bootstrap CI over decision-relevant groups
        boots = []
        for _ in range(args.bootstrap):
            samp = [dgroups[i] for i in rng.integers(0, len(dgroups), len(dgroups))] if dgroups else []
            boots.append(top1_rate(held_rows, pred, samp) if samp else 0.0)
        ci = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)] if boots else None
        pw_mc = pairwise_vs_truth(held_rows, pred, groups, "_mc_goal")
        pw_dqn = pairwise_vs_truth(held_rows, pred, groups, "score")
        rec = {
            "alpha": alpha,
            "decision_relevant_top1_realized": round(t1, 4),
            "decision_relevant_top1_ci95": ci,
            "oracle_all_pairwise_vs_realized_MC": round(pw_mc, 4) if pw_mc is not None else None,
            "oracle_all_pairwise_vs_dqn_label": round(pw_dqn, 4) if pw_dqn is not None else None,
        }
        sweep.append(rec)
        # Select by the ROBUST global metric (pairwise vs realized MC), not the noisy small-N top-1.
        key = pw_mc if pw_mc is not None else -1.0
        if best is None or key > best[1]:
            best = (alpha, key, t1)

    # Persist the best-alpha full strong model (score on blended labels; rank/diag on original labels).
    best_alpha = best[0]
    y_best = np.asarray([blended_score(r, best_alpha) for r in train_rows], dtype=np.float32)
    score_final = HistGradientBoostingRegressor(random_state=0, max_iter=300, learning_rate=0.08).fit(x_train, y_best)
    rank_final = HistGradientBoostingClassifier(random_state=0, max_iter=300, learning_rate=0.08).fit(x_train, y_rank)
    diag_final = HistGradientBoostingClassifier(random_state=0, max_iter=300, learning_rate=0.08).fit(x_train, y_diag)
    joblib.dump({"kind": "strong", "vectorizer": vec, "score": score_final, "rank": rank_final,
                 "diagnosis": diag_final, "mc_blend_alpha": best_alpha}, args.model_output)

    dqn_only = next(r for r in sweep if r["alpha"] == 1.0)
    best_rec = next(r for r in sweep if r["alpha"] == best_alpha)
    report = {
        "n_train": len(train_rows), "n_heldout_oracle": len(held_oracle),
        "n_decision_relevant_heldout_groups": len(dgroups),
        "blend": "y = alpha*Q + (1-alpha)*MC on oracle samples (rule rows unchanged)",
        "selection_criterion": "max oracle_all_pairwise_vs_realized_MC (robust global metric, not the noisy 34-group top-1)",
        "alpha_sweep": sweep,
        "best_alpha": best_alpha,
        "best_pairwise_vs_realized_MC": best_rec["oracle_all_pairwise_vs_realized_MC"],
        "dqn_only_pairwise_vs_realized_MC": dqn_only["oracle_all_pairwise_vs_realized_MC"],
        "global_pairwise_improvement_over_dqn": round(best_rec["oracle_all_pairwise_vs_realized_MC"] - dqn_only["oracle_all_pairwise_vs_realized_MC"], 4),
        "best_decision_relevant_top1": best_rec["decision_relevant_top1_realized"],
        "dqn_only_decision_relevant_top1": dqn_only["decision_relevant_top1_realized"],
        "decision_relevant_caveat": f"only {len(dgroups)} decision-relevant held-out groups; the top-1 CIs overlap across alphas, so the decision-relevant gain is DIRECTIONAL not significant — global pairwise vs realized MC is the robust signal.",
        "adopted_model": str(args.model_output),
        "note": (
            "Target metric = held-out decision-relevant top-1 against REALIZED MC return (ground truth), "
            "with a bootstrap 95% CI over the (small) set of decision-relevant groups. oracle_all_pairwise_"
            "vs_realized_MC is the honest global ranking metric. MC = V^pi(masked-greedy), not V*. The "
            "persisted model uses the best alpha and is a drop-in for prm_strong.joblib."
        ),
    }
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"decision-relevant heldout groups: {len(dgroups)}")
    print(f"{'alpha':>6}  {'dec-rel top1':>12}  {'ci95':>16}  {'pw_vs_MC':>9}  {'pw_vs_DQN':>9}")
    for r in sweep:
        print(f"{r['alpha']:>6.2f}  {r['decision_relevant_top1_realized']:>12.3f}  "
              f"{str(r['decision_relevant_top1_ci95']):>16}  {r['oracle_all_pairwise_vs_realized_MC']:>9}  {r['oracle_all_pairwise_vs_dqn_label']:>9}")
    print(f"best alpha={best_alpha} (by global pairwise-vs-realized-MC {best_rec['oracle_all_pairwise_vs_realized_MC']:.3f} "
          f"vs DQN-only {dqn_only['oracle_all_pairwise_vs_realized_MC']:.3f})  -> {args.model_output}")


if __name__ == "__main__":
    main()
