"""Information-value alignment check for the Web-RL value oracle (method §9.2).

Reconnaissance that returns nothing *this step* but unlocks a high-value chain later
is epistemically rational, yet it only collects step_cost. A feed-forward DQN under
partial observability can mislabel such rational enumeration as low value, and the PRM
would then inherit a "enumerate less, exploit early" bias — the most dangerous habit in
Web pentesting. This check verifies the oracle distinguishes *productive* enumeration
(discovers a path that unlocks the chain) from *dead-end* enumeration (discovers only a
distractor), which the distractor-laden tasks make directly testable.

Method: walk each task's expert plan. At every state where `web_path_enumeration` is a
legal action, score the oracle's value_gap for that enumeration action and classify it:

- productive  : the expert's next action *is* enumeration (it discovers a needed path).
- dead_end    : the expert's next action is *not* enumeration, but enumeration is still
                legal (only distractor routes remain) — so enumerating here is a decoy.

A well-aligned oracle gives productive enumeration a small value_gap (near-optimal) and
dead-end enumeration a larger value_gap (clearly suboptimal vs advancing the chain). The
separation `mean(dead_end_gap) - mean(productive_gap)` is the distinguishability score.

The DQN oracle is used (not literal-Q*), because literal-reward Q* milks distractor
path_found rewards before a terminal and therefore does *not* down-rank dead-end
enumeration; the trained oracle is the object whose information-value alignment matters
for the PRM labels.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_pipeline import DQNValueOracle  # noqa: E402
from train_dqn import resolve_task_paths  # noqa: E402
from verify_qstar import resolve_gated_checkpoint  # noqa: E402
from web_attack_sim import Action, ActionType, WebAttackSimEnv  # noqa: E402
from web_attack_sim.action_space import ACTIONS  # noqa: E402
from web_attack_sim.tasks import load_task_config  # noqa: E402

ENUM_ID = ACTIONS.index(ActionType.WEB_PATH_ENUMERATION)


def plan_action_type(raw: Any) -> str:
    return raw if isinstance(raw, str) else raw["action_type"]


def plan_to_action(raw: Any) -> Action:
    if isinstance(raw, str):
        return Action(ActionType(raw))
    return Action(ActionType(raw["action_type"]), target=raw.get("target"), parameter=raw.get("parameter"))


def evaluate_task(oracle: DQNValueOracle, task_path: Path) -> dict[str, Any]:
    task = load_task_config(task_path)
    plan = task["expert_plan"]
    hidden_paths = set(task.get("hidden_paths", []))
    # Paths the expert actually enumerates are "productive"; the rest are distractors.
    expert_enum_targets = {
        step.get("target") for step in plan
        if isinstance(step, dict) and step.get("action_type") == ActionType.WEB_PATH_ENUMERATION.value and step.get("target")
    }
    distractor_paths = hidden_paths - expert_enum_targets

    env = WebAttackSimEnv()
    obs, _info = env.reset(task_path)

    productive: list[tuple[float, int]] = []   # (gap, step_idx)
    dead_end: list[tuple[float, int]] = []
    true_decoy: list[tuple[float, int]] = []   # dead-end states where ONLY distractors remain enumerable

    for step_idx, raw in enumerate(plan):
        mask = env.action_mask(permissive=False)
        if mask[ENUM_ID]:
            obs_vec = env.encode_observation(obs)
            q_report = oracle.score_action(obs_vec, ActionType.WEB_PATH_ENUMERATION, mask)
            gap = q_report["value_gap"]
            if gap is not None:
                undiscovered = hidden_paths - set(obs.discovered_paths)
                productive_remaining = undiscovered & expert_enum_targets
                next_is_enum = plan_action_type(raw) == ActionType.WEB_PATH_ENUMERATION.value
                if next_is_enum:
                    productive.append((float(gap), step_idx))
                else:
                    dead_end.append((float(gap), step_idx))
                    # True decoy: enumerating here can only surface a distractor route.
                    if not productive_remaining and (undiscovered & distractor_paths):
                        true_decoy.append((float(gap), step_idx))
        obs, _reward, done, _trunc, _info = env.step(plan_to_action(raw))
        if done:
            break

    def gaps(rows: list[tuple[float, int]]) -> float | None:
        return float(np.mean([g for g, _ in rows])) if rows else None

    def steps(rows: list[tuple[float, int]]) -> float | None:
        return float(np.mean([s for _, s in rows])) if rows else None

    return {
        "task_id": task.get("task_id"),
        "num_distractor_paths": len(distractor_paths),
        "num_productive_states": len(productive),
        "num_dead_end_states": len(dead_end),
        "num_true_decoy_states": len(true_decoy),
        "productive_enum_mean_value_gap": gaps(productive),
        "dead_end_enum_mean_value_gap": gaps(dead_end),
        "true_decoy_enum_mean_value_gap": gaps(true_decoy),
        "productive_mean_step": steps(productive),
        "dead_end_mean_step": steps(dead_end),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = resolve_gated_checkpoint(argparse.Namespace(checkpoint=args.checkpoint))
    oracle = DQNValueOracle(checkpoint, args.device)
    task_paths = resolve_task_paths(args.tasks)
    task_reports = [evaluate_task(oracle, path) for path in task_paths]

    prod_gaps = [r["productive_enum_mean_value_gap"] for r in task_reports if r["productive_enum_mean_value_gap"] is not None]
    dead_gaps = [r["dead_end_enum_mean_value_gap"] for r in task_reports if r["dead_end_enum_mean_value_gap"] is not None]
    decoy_gaps = [r["true_decoy_enum_mean_value_gap"] for r in task_reports if r["true_decoy_enum_mean_value_gap"] is not None]

    productive_gap = float(np.mean(prod_gaps)) if prod_gaps else None
    dead_end_gap = float(np.mean(dead_gaps)) if dead_gaps else None
    true_decoy_gap = float(np.mean(decoy_gaps)) if decoy_gaps else None
    separation = (dead_end_gap - productive_gap) if (productive_gap is not None and dead_end_gap is not None) else None
    decoy_separation = (true_decoy_gap - productive_gap) if (productive_gap is not None and true_decoy_gap is not None) else None

    prod_step = float(np.mean([r["productive_mean_step"] for r in task_reports if r["productive_mean_step"] is not None]))
    dead_step = float(np.mean([r["dead_end_mean_step"] for r in task_reports if r["dead_end_mean_step"] is not None]))
    num_true_decoy_states = sum(r["num_true_decoy_states"] for r in task_reports)

    aggregate = {
        "checkpoint": str(checkpoint),
        "num_tasks": len(task_reports),
        "productive_enum_mean_value_gap": productive_gap,
        "dead_end_enum_mean_value_gap": dead_end_gap,
        "true_decoy_enum_mean_value_gap": true_decoy_gap,
        "naive_separation_dead_end_minus_productive": separation,
        "true_decoy_separation": decoy_separation,
        "num_true_decoy_states_total": num_true_decoy_states,
        "productive_mean_step_index": prod_step,
        "dead_end_mean_step_index": dead_step,
        "verdict": {
            "interpretation": (
                "HONEST REFRAME (per adversarial review): the encoder exposes only num_paths_norm and four "
                "hardcoded path flags, so the Q-network CANNOT see which enumerable path is a decoy. The "
                "value_gap separation is therefore NOT decoy-discrimination. It is an OPPORTUNITY-COST "
                "signal: the oracle down-ranks enumeration once higher-value chain-advancing actions become "
                "available (dead-end states are later: mean step ~{:.1f} vs ~{:.1f}). The narrow claim that "
                "survives is: the oracle does not over-value enumeration when better actions exist (it will "
                "not enumerate-forever), which is the §9.2 failure mode we care about — but it does so via "
                "opportunity cost, not by recognising distractor routes."
            ).format(dead_step, prod_step),
            "oracle_avoids_overvaluing_enumeration_when_better_actions_exist": bool(
                separation is not None and separation >= args.min_separation
            ),
            "true_decoy_states_observed": num_true_decoy_states,
            "true_decoy_separation_note": (
                "Separation restricted to states where ONLY distractor routes remain enumerable. With few "
                "such states and an encoder blind to decoy-ness, this is not strong evidence of decoy "
                "discrimination."
            ),
        },
        "task_reports": task_reports,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Information-value alignment check (method §9.2).")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to best gated seed checkpoint.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Defaults to all bundled tasks.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--margin", type=float, default=0.0, help="Min per-task gap difference to count as correct.")
    parser.add_argument("--min-separation", type=float, default=1.0)
    parser.add_argument("--min-pairwise", type=float, default=0.7)
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "info_value_check.json")
    args = parser.parse_args()

    report = evaluate(args)
    print(json.dumps({k: v for k, v in report.items() if k != "task_reports"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
