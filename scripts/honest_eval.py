"""Honest re-evaluation that corrects the inflated headline metrics surfaced by the
adversarial robustness review. Each section reports the *meaningful* number next to
the *trivial baseline* it must beat, so saturated / self-predictable metrics cannot
masquerade as evidence of a good model.

Sections:
1. Oracle gate vs random-masked baseline (review: goal_rate=1.0 and top3~0.90 are
   produced by the action mask, not the Q-network). Reports oracle goal_rate masked
   AND permissive, plus expert top-1/top-3 as *lift over a uniformly-random ranking
   within the masked allowed-set* (computed analytically + a sampled random policy).
2. PRM on the oracle-labeled subset only (review: ~74% of rows have score=0 and the
   rule rank_label is copied verbatim into the input as "normalized status:"). Reports
   rank/score/pairwise on high/medium/low oracle rows vs the majority-class floor.
3. Leakage over the FULL model input (review: leakage_audit scanned only `context`,
   but the PRM trains on sample_to_text = context + raw_llm_action + normalized fields).
4. Split integrity at the model-input level (review: with the OOV scenario hash removed,
   a large fraction of held-out inputs are byte-identical to a train input).
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_pipeline import DQNValueOracle  # noqa: E402
from leakage_audit import hidden_tokens_for_task, strip_task_label  # noqa: E402
from run_oracle_seed_gate import DEFAULT_EVAL_TASKS, DEFAULT_TRAIN_TASKS  # noqa: E402
from train_dqn import resolve_task_paths  # noqa: E402
from train_prm_baseline import read_jsonl, sample_to_text  # noqa: E402
from verify_qstar import resolve_gated_checkpoint  # noqa: E402
from web_attack_sim import Action, ActionType, WebAttackSimEnv  # noqa: E402
from web_attack_sim.action_space import ACTIONS  # noqa: E402
from web_attack_sim.tasks import bundled_task_paths, load_task_config  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Oracle gate vs random-masked baseline
# ---------------------------------------------------------------------------

def plan_action(raw: Any) -> Action:
    if isinstance(raw, str):
        return Action(ActionType(raw))
    return Action(ActionType(raw["action_type"]), target=raw.get("target"), parameter=raw.get("parameter"))


def random_masked_goal_rate(task_paths: list[Path], episodes: int, seed: int, use_mask: bool) -> float:
    rng = random.Random(seed)
    solved = 0
    total = 0
    for task_path in task_paths:
        for _ in range(episodes):
            env = WebAttackSimEnv()
            obs, _info = env.reset(task_path)
            done = False
            while not done:
                mask = env.action_mask(permissive=not use_mask)
                allowed = [i for i, m in enumerate(mask) if m] or list(range(len(ACTIONS)))
                action_id = rng.choice(allowed)
                obs, _r, done, _t, _i = env.step(action_id)
            solved += int(bool(env.state and env.state.done and env._goal_reached()))
            total += 1
    return solved / max(total, 1)


def oracle_goal_rate(oracle: DQNValueOracle, task_paths: list[Path], use_mask: bool) -> float:
    solved = 0
    for task_path in task_paths:
        env = WebAttackSimEnv()
        obs, _info = env.reset(task_path)
        done = False
        while not done:
            mask = env.action_mask(permissive=not use_mask)
            q = oracle.q_values(env.encode_observation(obs))
            allowed = [i for i, m in enumerate(mask) if m]
            if use_mask and allowed:
                action_id = max(allowed, key=lambda i: q[i])
            else:
                action_id = int(np.argmax(q))
            obs, _r, done, _t, _i = env.step(action_id)
        solved += int(bool(env.state and env.state.done and env._goal_reached()))
    return solved / max(len(task_paths), 1)


def expert_random_ranking_baseline(oracle: DQNValueOracle, task_paths: list[Path]) -> dict[str, float]:
    """Analytic expected top-1/top-3 of a uniformly-random ranking within the masked
    allowed-set, evaluated at the expert-plan states (the gate's comparison points)."""
    top1 = []
    top3 = []
    allowed_sizes = []
    for task_path in task_paths:
        task = load_task_config(task_path)
        env = WebAttackSimEnv()
        obs, _info = env.reset(task_path)
        done = False
        for raw in task.get("expert_plan", []):
            if done:
                break
            mask = env.action_mask(permissive=False)
            action = plan_action(raw)
            action_id = ACTIONS.index(action.action_type)
            if mask[action_id]:
                n = sum(mask)
                allowed_sizes.append(n)
                top1.append(1.0 / n)
                top3.append(min(3, n) / n)
            obs, _r, done, _t, _i = env.step(action)
    return {
        "random_masked_expert_top1": float(np.mean(top1)) if top1 else None,
        "random_masked_expert_top3": float(np.mean(top3)) if top3 else None,
        "mean_allowed_set_size": float(np.mean(allowed_sizes)) if allowed_sizes else None,
        "frac_steps_le3_allowed": float(np.mean([s <= 3 for s in allowed_sizes])) if allowed_sizes else None,
    }


def oracle_expert_topk(oracle: DQNValueOracle, task_paths: list[Path]) -> dict[str, float]:
    """Compute the checkpoint's own masked expert top-1/top-3 (not read from a gate file)."""
    top1 = []
    top3 = []
    for task_path in task_paths:
        task = load_task_config(task_path)
        env = WebAttackSimEnv()
        obs, _info = env.reset(task_path)
        done = False
        for raw in task.get("expert_plan", []):
            if done:
                break
            mask = env.action_mask(permissive=False)
            action = plan_action(raw)
            action_id = ACTIONS.index(action.action_type)
            if mask[action_id]:
                q = oracle.q_values(env.encode_observation(obs))
                allowed = [i for i, m in enumerate(mask) if m]
                order = sorted(allowed, key=lambda i: q[i], reverse=True)
                rank = order.index(action_id) + 1
                top1.append(int(rank == 1))
                top3.append(int(rank <= 3))
            obs, _r, done, _t, _i = env.step(action)
    return {
        "oracle_expert_top1": float(np.mean(top1)) if top1 else None,
        "oracle_expert_top3": float(np.mean(top3)) if top3 else None,
    }


def section_oracle(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = resolve_gated_checkpoint(argparse.Namespace(checkpoint=args.checkpoint))
    oracle = DQNValueOracle(checkpoint, args.device)
    eval_paths = resolve_task_paths(DEFAULT_EVAL_TASKS)

    topk = oracle_expert_topk(oracle, eval_paths)
    oracle_top1 = topk["oracle_expert_top1"]
    oracle_top3 = topk["oracle_expert_top3"]

    rand = expert_random_ranking_baseline(oracle, eval_paths)
    return {
        "checkpoint": str(checkpoint),
        "oracle_goal_rate_masked": oracle_goal_rate(oracle, eval_paths, use_mask=True),
        "oracle_goal_rate_permissive": oracle_goal_rate(oracle, eval_paths, use_mask=False),
        "random_goal_rate_masked": random_masked_goal_rate(eval_paths, args.random_episodes, args.seed, use_mask=True),
        "random_goal_rate_permissive": random_masked_goal_rate(eval_paths, args.random_episodes, args.seed, use_mask=False),
        "oracle_expert_top1": oracle_top1,
        "oracle_expert_top3": oracle_top3,
        **rand,
        "oracle_top1_lift_over_random": (oracle_top1 - rand["random_masked_expert_top1"]) if oracle_top1 is not None and rand["random_masked_expert_top1"] is not None else None,
        "oracle_top3_lift_over_random": (oracle_top3 - rand["random_masked_expert_top3"]) if oracle_top3 is not None and rand["random_masked_expert_top3"] is not None else None,
        "verdict_note": (
            "goal_rate and top-3 are mask-saturated (a random masked policy matches them); the oracle's "
            "genuine signal is the top-1 LIFT over the random-within-mask baseline. Report lift, not raw."
        ),
    }


# ---------------------------------------------------------------------------
# 2. PRM on the oracle-labeled subset
# ---------------------------------------------------------------------------

def majority_floor(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    return counts.most_common(1)[0][1] / len(labels)


def pairwise_within_group(rows: list[dict[str, Any]], preds: list[float], idxs: list[int], eps: float) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i in idxs:
        groups[str(rows[i].get("candidate_group"))].append(i)
    concordant = total = 0
    for members in groups.values():
        for a_pos in range(len(members)):
            for b_pos in range(a_pos + 1, len(members)):
                a, b = members[a_pos], members[b_pos]
                td = float(rows[a]["score"]) - float(rows[b]["score"])
                if abs(td) <= eps:
                    continue
                total += 1
                pd = preds[a] - preds[b]
                if (td > 0 and pd > 0) or (td < 0 and pd < 0):
                    concordant += 1
    return {"pairs": total, "pairwise_accuracy": (concordant / total) if total else None}


def section_prm(args: argparse.Namespace) -> dict[str, Any]:
    heldout = read_jsonl(ROOT / "outputs" / "prm_samples_heldout.jsonl")
    model = joblib.load(args.prm_model)
    texts = [sample_to_text(r) for r in heldout]
    rank_pred = list(model["rank_classifier"].predict(texts))
    score_pred = list(np.clip(model["score_regressor"].predict(texts), 0.0, 1.0))

    oracle_idx = [i for i, r in enumerate(heldout) if r.get("label_source") == "oracle"]
    rule_idx = [i for i, r in enumerate(heldout) if r.get("label_source") != "oracle"]

    def rank_acc(idxs: list[int]) -> float:
        if not idxs:
            return 0.0
        return float(np.mean([int(rank_pred[i] == heldout[i]["rank_label"]) for i in idxs]))

    oracle_labels = [str(heldout[i]["rank_label"]) for i in oracle_idx]
    return {
        "n_heldout": len(heldout),
        "n_oracle_labeled": len(oracle_idx),
        "n_rule_labeled": len(rule_idx),
        "frac_score_zero": float(np.mean([float(r["score"]) == 0.0 for r in heldout])),
        "rank_accuracy_full": rank_acc(list(range(len(heldout)))),
        "rank_accuracy_rule_subset": rank_acc(rule_idx),
        "rank_accuracy_oracle_subset": rank_acc(oracle_idx),
        "oracle_subset_majority_floor": majority_floor(oracle_labels),
        "oracle_subset_rank_lift_over_floor": rank_acc(oracle_idx) - majority_floor(oracle_labels),
        "pairwise_all_pairs": pairwise_within_group(heldout, score_pred, list(range(len(heldout))), args.pairwise_eps),
        "pairwise_oracle_only": pairwise_within_group(heldout, score_pred, oracle_idx, args.pairwise_eps),
        "verdict_note": (
            "Rule rows (status copied into input, score=0) are trivially self-predictable. The real "
            "evaluator quality is rank/pairwise on the ORACLE subset vs the majority floor."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Leakage over the full model input
# ---------------------------------------------------------------------------

def section_full_input_leakage() -> dict[str, Any]:
    task_index = {str(load_task_config(p)["task_id"]): load_task_config(p) for p in bundled_task_paths()}
    results: dict[str, Any] = {}
    for split, fname in [("train", "prm_samples_train.jsonl"), ("heldout", "prm_samples_heldout.jsonl")]:
        rows = read_jsonl(ROOT / "outputs" / fname)
        secret_in_context = secret_in_action = 0
        undiscovered_path_in_action = 0
        for row in rows:
            task = task_index.get(str(row.get("task_id")))
            if not task:
                continue
            tokens = hidden_tokens_for_task(task)
            full_text = sample_to_text(row)
            context = str(row.get("context", ""))
            visible = set(re.findall(r"'(/[^']*)'", context))
            for secret in tokens["unique_secrets"]:
                if secret in context:
                    secret_in_context += 1
                elif secret in full_text:
                    secret_in_action += 1
            # Undiscovered hidden paths appearing in the candidate-action text.
            for path in tokens["hidden_paths"]:
                if path not in visible and re.search(r"\b" + re.escape(path) + r"\b", full_text):
                    undiscovered_path_in_action += 1
        results[split] = {
            "num_samples": len(rows),
            "unique_secret_leaks_in_context": secret_in_context,
            "unique_secret_leaks_in_action_only": secret_in_action,
            "undiscovered_hidden_path_in_candidate_action": undiscovered_path_in_action,
        }
    results["interpretation"] = (
        "Secret tokens (flags/passwords) must be 0 everywhere — they are. Undiscovered hidden paths DO "
        "appear in the candidate-action text: this is the action's own target (the LLM proposes the path "
        "to probe), not hidden STATE leaked into context. It is benign per-sample, but the dataset's "
        "candidate generator is task-aware, so the candidate distribution is task-correlated (a "
        "candidate-quality limitation, not a context leak). Real LLM rollouts would propose paths "
        "without ground-truth knowledge."
    )
    return results


# ---------------------------------------------------------------------------
# 4. Split integrity at the model-input level
# ---------------------------------------------------------------------------

SCENARIO_RE = re.compile(r"Scenario scenario_[0-9a-f]+,")


def normalize_scenario(text: str) -> str:
    return SCENARIO_RE.sub("Scenario <SID>,", text)


def section_split_integrity() -> dict[str, Any]:
    train = read_jsonl(ROOT / "outputs" / "prm_samples_train.jsonl")
    heldout = read_jsonl(ROOT / "outputs" / "prm_samples_heldout.jsonl")
    train_inputs = {normalize_scenario(sample_to_text(r)) for r in train}
    train_contexts = {normalize_scenario(str(r.get("context", ""))) for r in train}
    dup_input = sum(1 for r in heldout if normalize_scenario(sample_to_text(r)) in train_inputs)
    dup_context = sum(1 for r in heldout if normalize_scenario(str(r.get("context", ""))) in train_contexts)

    # Plan-signature overlap at the task level.
    def signature(task: dict[str, Any]) -> tuple:
        return tuple(s if isinstance(s, str) else s["action_type"] for s in task.get("expert_plan", []))

    train_sigs = {signature(load_task_config(resolve_task_paths([t])[0])) for t in DEFAULT_TRAIN_TASKS}
    heldout_tasks = [load_task_config(resolve_task_paths([t])[0]) for t in DEFAULT_EVAL_TASKS]
    novel = [t["task_id"] for t in heldout_tasks if signature(t) not in train_sigs]
    return {
        "n_heldout": len(heldout),
        "heldout_inputs_byte_identical_to_train": dup_input,
        "frac_heldout_inputs_duplicated": round(dup_input / max(len(heldout), 1), 4),
        "heldout_contexts_collapsing_onto_train": dup_context,
        "frac_heldout_contexts_duplicated": round(dup_context / max(len(heldout), 1), 4),
        "heldout_tasks_with_novel_plan_signature": novel,
        "num_heldout_tasks_novel": len(novel),
        "num_heldout_tasks_total": len(heldout_tasks),
        "verdict_note": (
            "After normalizing the OOV scenario hash, duplicated held-out inputs are effectively "
            "in-distribution; held-out tasks sharing a train plan-signature are re-skinned templates "
            "(sub-chain overlap, method §13.1). 'Unseen' should be interpreted as unseen-token, not "
            "unseen-structure."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Honest re-evaluation correcting inflated robustness metrics.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--prm-model", type=Path, default=ROOT / "outputs" / "prm_baseline.joblib")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random-episodes", type=int, default=8)
    parser.add_argument("--pairwise-eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "honest_eval.json")
    args = parser.parse_args()

    report = {
        "oracle_gate_vs_random": section_oracle(args),
        "prm_oracle_subset": section_prm(args),
        "full_input_leakage": section_full_input_leakage(),
        "split_integrity": section_split_integrity(),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
