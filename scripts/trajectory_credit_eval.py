"""Trajectory-level failure credit-assignment for the Pentest-PRM (error-trajectory gap #1).

The PRM labels are per-step (process reward). This evaluates whether those per-step scores do
TRAJECTORY-level credit assignment: when an early wrong action derails the chain and the episode
fails, does the PRM (a) prefer the correct action over the derailing one AT THE FORK (root-cause
prevention), and (b) have per-step scores that track the realized Monte-Carlo return-to-go (so the
derail step is scored low and the doomed suffix is recognized)?

Synthetic derailment (deterministic, on-simulator): for each task with expert_plan length >= 4,
at the fork step k = n//2 substitute a PREMATURE GOAL-GRAB (the plan's final goal action attempted
early) — a realistic LLM mistake that fails its precondition and skips the real step k, breaking the
chain so the expert tail can no longer reach the goal. We then roll the derailed trajectory, score
each action with the persisted strong PRM, and compare its scores to the env's realized return-to-go.

Metrics (no oracle q-values used as features):
  - root_cause_prevention_rate: at the fork, fraction of tasks where PRM(correct a_k) > PRM(derail).
  - mean Spearman(per-step PRM score, return-to-go): does the score anticipate the outcome?
  - derail_failure_rate: sanity that the injected error actually breaks the chain.
  - derail_step_score_drop: mean PRM score on the correct pre-fork steps minus the score at the derail.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import argparse
import json
import sys
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_pipeline import context_from_observation  # noqa: E402
from train_prm_strong import extract_features  # noqa: E402
from web_attack_sim import WebAttackSimEnv  # noqa: E402
from web_attack_sim.tasks import bundled_task_paths, load_task_config  # noqa: E402

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

GAMMA = 0.98


def to_action(step: Any) -> dict[str, Any]:
    if isinstance(step, str):
        return {"action_type": step, "target": None, "parameter": None}
    return {"action_type": step["action_type"], "target": step.get("target"), "parameter": step.get("parameter")}


GOOD_RANKS = {"high", "medium"}


def _features(model: dict[str, Any], context: str, action: dict[str, Any]):
    na = {"action_type": action["action_type"], "status": "valid",
          "target": action.get("target"), "parameter": action.get("parameter")}
    sample = {"context": context, "raw_llm_action": "", "normalized_action": na, "normalizer_confidence": 1.0}
    return model["vectorizer"].transform([extract_features(sample)]).toarray()


def prm_score(model: dict[str, Any], context: str, action: dict[str, Any]) -> float:
    return float(np.clip(model["score"].predict(_features(model, context, action))[0], 0.0, 1.0))


def prm_rank(model: dict[str, Any], context: str, action: dict[str, Any]) -> str:
    return str(model["rank"].predict(_features(model, context, action))[0])


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    if spearmanr is not None:
        rho = spearmanr(a, b).correlation
        return None if rho != rho else float(rho)
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ar, br)[0, 1])


def returns_to_go(rewards: list[float]) -> list[float]:
    g = [0.0] * len(rewards)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + GAMMA * acc
        g[t] = acc
    return g


def run_task(task_path: Path, model: dict[str, Any]) -> dict[str, Any] | None:
    task = load_task_config(task_path)
    plan = [to_action(s) for s in task.get("expert_plan", [])]
    n = len(plan)
    if n < 4:
        return None
    k = n // 2
    derail = dict(plan[-1])  # the goal-grab action, attempted prematurely at the fork

    env = WebAttackSimEnv()
    obs, info = env.reset(task_path)
    task_id = str(info["task_id"])
    trace: list[dict[str, Any]] = []
    derail_plan = plan[:k] + [derail] + plan[k + 1:]

    scores: list[float] = []
    rewards: list[float] = []
    fork_correct = fork_derail = None
    fork_rank_correct = fork_rank_derail = None

    for t, action in enumerate(derail_plan):
        context = context_from_observation(task_id, t, obs, trace)
        if t == k:  # at the fork, score BOTH the correct step-k action and the derail at the same state
            fork_correct = prm_score(model, context, plan[k])
            fork_derail = prm_score(model, context, derail)
            fork_rank_correct = prm_rank(model, context, plan[k])
            fork_rank_derail = prm_rank(model, context, derail)
        scores.append(prm_score(model, context, action))
        obs, reward, done, _trunc, step_info = env.step(action)
        rewards.append(float(reward))
        trace.append({"action": {"action_type": action["action_type"]}, "feedback": step_info["feedback"]})
        if done:
            break

    goal = bool(env.state and env.state.done and env._goal_reached())
    g = returns_to_go(rewards)
    pre_fork = scores[:k]
    return {
        "task_id": task_id,
        "family": task.get("family"),
        "n": n,
        "fork_k": k,
        "goal_reached": goal,
        "fork_correct_preferred": bool(fork_correct is not None and fork_derail is not None and fork_correct > fork_derail),
        "fork_correct_score": fork_correct,
        "fork_derail_score": fork_derail,
        # rank-head gate: correct predicted good AND derail predicted as an error (not high/medium)
        "fork_rankhead_prevented": bool(fork_rank_correct in GOOD_RANKS and fork_rank_derail not in GOOD_RANKS),
        "fork_rank_correct": fork_rank_correct,
        "fork_rank_derail": fork_rank_derail,
        "score_return_spearman": spearman(scores, g),
        "derail_step_score_drop": (float(np.mean(pre_fork)) - scores[k]) if pre_fork and k < len(scores) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory-level failure credit-assignment eval for the PRM.")
    parser.add_argument("--model", type=Path, default=ROOT / "outputs" / "prm_strong.joblib")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "trajectory_credit_eval.json")
    args = parser.parse_args()

    model = joblib.load(args.model)
    if model.get("kind") != "strong":
        raise SystemExit("expected the strong PRM joblib (run train_prm_strong.py first)")

    per_task = [r for r in (run_task(p, model) for p in bundled_task_paths()) if r is not None]
    spearmans = [r["score_return_spearman"] for r in per_task if r["score_return_spearman"] is not None]
    drops = [r["derail_step_score_drop"] for r in per_task if r["derail_step_score_drop"] is not None]

    report = {
        "n_tasks": len(per_task),
        "gamma": GAMMA,
        "derailment": "fork step k=n//2 replaced by a premature goal-grab (skips the real step k, breaks the chain)",
        "root_cause_prevention_rate_score": round(float(np.mean([int(r["fork_correct_preferred"]) for r in per_task])), 4),
        "root_cause_prevention_rate_rankhead": round(float(np.mean([int(r["fork_rankhead_prevented"]) for r in per_task])), 4),
        "derail_failure_rate": round(float(np.mean([int(not r["goal_reached"]) for r in per_task])), 4),
        "mean_fork_correct_score": round(float(np.mean([r["fork_correct_score"] for r in per_task])), 4),
        "mean_fork_derail_score": round(float(np.mean([r["fork_derail_score"] for r in per_task])), 4),
        "mean_score_return_spearman": round(float(np.mean(spearmans)), 4) if spearmans else None,
        "frac_positive_spearman": round(float(np.mean([int(s > 0) for s in spearmans])), 4) if spearmans else None,
        "mean_derail_step_score_drop": round(float(np.mean(drops)), 4) if drops else None,
        "per_task": per_task,
        "note": (
            "root_cause_prevention_rate = at the fork, does the PRM rank the correct action above the "
            "derailing premature goal-grab (would the PRM have prevented the root-cause error). "
            "mean_score_return_spearman = do per-step PRM scores track the realized return-to-go (credit "
            "assignment). derail_failure_rate confirms the injected error actually breaks the chain."
        ),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"tasks={report['n_tasks']}  derail_failure_rate={report['derail_failure_rate']:.3f}")
    print(f"root_cause_prevention (score: correct>derail)   = {report['root_cause_prevention_rate_score']:.3f}")
    print(f"root_cause_prevention (rank-head: derail flagged) = {report['root_cause_prevention_rate_rankhead']:.3f}")
    print(f"  mean fork score: correct={report['mean_fork_correct_score']:.3f}  derail={report['mean_fork_derail_score']:.3f}")
    print(f"mean Spearman(PRM score, return-to-go) = {report['mean_score_return_spearman']}  (frac positive {report['frac_positive_spearman']})")
    print(f"mean derail-step score drop = {report['mean_derail_step_score_drop']}")


if __name__ == "__main__":
    main()
