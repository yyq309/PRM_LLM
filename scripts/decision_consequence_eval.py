"""Does tightening the budget make decisions CONSEQUENTIAL? (env improvement #1, cheap probe.)

The MC analysis showed the env is forgiving: forced-action goal recovery ~0.95, so only ~20% of
same-state decisions are outcome-relevant -> scarce decision-relevant eval -> weak fork-prevention.
This sweeps the per-task budget slack (max_steps = len(expert_plan) + slack) and, using the EXISTING
canonical oracle (no retrain), measures for each slack:
  - expert_solvable_rate: does the scripted expert still reach the goal at this budget (sanity).
  - forced_action_goal_recovery_rate: after forcing an arbitrary candidate, can masked-greedy still
    reach the goal (low => decisions are consequential).
  - decision_relevant_group_frac: fraction of same-state candidate groups whose realized MC returns
    differ (max-min goal-return > MEANINGFUL) => the choice actually matters.

A drop in recovery + a rise in decision-relevant fraction as slack shrinks is direct evidence that a
tight-budget "hard mode" yields the consequential decisions the oracle/PRM need — purely env dynamics,
no schema change.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_prm_dataset import select_checkpoint  # noqa: E402
from mc_return_labels import action_from_norm, mc_rollout  # noqa: E402
from demo_pipeline import DQNValueOracle  # noqa: E402
from train_prm_baseline import read_jsonl  # noqa: E402
from web_attack_sim import WebAttackSimEnv, normalize_llm_action  # noqa: E402
from web_attack_sim.tasks import load_task_config  # noqa: E402

MEANINGFUL = 0.02


def task_with_budget(task: dict[str, Any], slack: int | None) -> dict[str, Any]:
    if slack is None:
        return task  # loose: keep the task's own (generous) budget
    t = copy.deepcopy(task)
    t.setdefault("budget", {})["max_steps"] = len(task.get("expert_plan", [])) + slack
    return t


def snapshots(task_cfg: dict[str, Any]) -> tuple[dict[int, Any], int, bool]:
    env = WebAttackSimEnv()
    obs, _i = env.reset(task_cfg)
    snaps: dict[int, Any] = {}
    for t, raw in enumerate(task_cfg.get("expert_trajectory", [])):
        snaps[t] = copy.deepcopy(env.state)
        n = normalize_llm_action(raw)
        if n.action is None:
            break
        obs, _r, done, _t, _i = env.step(n.action)
        if done:
            break
    solved = bool(env.state and env.state.done and env._goal_reached())
    return snaps, env.max_steps, solved


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget-slack consequence probe (env improvement #1).")
    parser.add_argument("--heldout-input", type=Path, default=ROOT / "outputs" / "prm_samples_heldout.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed-gate-report", type=Path, default=ROOT / "outputs" / "oracle_seed_gate.json")
    parser.add_argument("--allow-ungated-oracle", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--slacks", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "decision_consequence_eval.json")
    args = parser.parse_args()

    checkpoint, _g = select_checkpoint(args)
    oracle = DQNValueOracle(checkpoint, args.device)

    oracle_rows = [r for r in read_jsonl(args.heldout_input) if r.get("label_source") == "oracle"]
    by_task_step: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in oracle_rows:
        by_task_step[(str(r.get("task_path")), int(r.get("step", 0)))].append(r)
    task_paths = sorted({ts[0] for ts in by_task_step})
    task_cfgs = {tp: load_task_config(Path(tp)) for tp in task_paths}

    env = WebAttackSimEnv()
    sweep = []
    for slack in [None, *args.slacks]:
        snap_cache: dict[str, tuple[dict[int, Any], int, bool]] = {}
        expert_solved = []
        recoveries = []
        dr_groups = 0
        total_groups = 0
        for (task_path, step), members in by_task_step.items():
            if task_path not in snap_cache:
                cfg = task_with_budget(task_cfgs[task_path], slack)
                snap_cache[task_path] = snapshots(cfg)
                expert_solved.append(int(snap_cache[task_path][2]))
            snaps, max_steps, _ = snap_cache[task_path]
            state = snaps.get(step)
            if state is None:
                continue
            goals = []
            for r in members:
                a = action_from_norm(r.get("normalized_action") or {})
                if a is None:
                    continue
                mc = mc_rollout(env, oracle, state, a, max_steps)
                recoveries.append(int(mc["reached"]))
                goals.append(mc["g_goal"])
            if len(goals) >= 2:
                total_groups += 1
                if max(goals) - min(goals) > MEANINGFUL:
                    dr_groups += 1
        sweep.append({
            "slack": slack if slack is not None else "loose(default)",
            "expert_solvable_rate": round(float(np.mean(expert_solved)), 4) if expert_solved else None,
            "forced_action_goal_recovery_rate": round(float(np.mean(recoveries)), 4) if recoveries else None,
            "decision_relevant_group_frac": round(dr_groups / total_groups, 4) if total_groups else None,
            "n_groups": total_groups,
        })

    report = {
        "checkpoint": str(checkpoint),
        "meaningful_threshold": MEANINGFUL,
        "sweep": sweep,
        "note": (
            "Lower forced_action_goal_recovery_rate + higher decision_relevant_group_frac as slack "
            "shrinks => a tight-budget hard mode makes decisions consequential (the lever works). "
            "expert_solvable_rate must stay high (the scripted optimal must still fit the budget)."
        ),
    }
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'slack':>14}  {'expert_solv':>11}  {'forced_recovery':>15}  {'decision_relevant':>17}  {'n_grp':>5}")
    for s in sweep:
        print(f"{str(s['slack']):>14}  {str(s['expert_solvable_rate']):>11}  "
              f"{str(s['forced_action_goal_recovery_rate']):>15}  {str(s['decision_relevant_group_frac']):>17}  {s['n_groups']:>5}")


if __name__ == "__main__":
    main()
