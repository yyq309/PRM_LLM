from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_dqn import QNetwork, resolve_task_paths  # noqa: E402
from web_attack_sim import Action, ActionType, WebAttackSimEnv  # noqa: E402
from web_attack_sim.tasks import load_task_config  # noqa: E402


class OracleCheckpoint:
    def __init__(self, checkpoint: Path, device: str):
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        payload = self._load_payload(checkpoint)
        model_state = payload["model_state"]
        hidden_sizes = payload.get("metadata", {}).get("hidden_sizes") or infer_hidden_sizes(model_state)

        self.actions = [ActionType(action) for action in payload["actions"]]
        self.q_net = QNetwork(int(payload["obs_dim"]), int(payload["num_actions"]), list(hidden_sizes)).to(self.device)
        self.q_net.load_state_dict(model_state)
        self.q_net.eval()
        self.metadata = payload.get("metadata", {})

    def q_values(self, obs_vec: list[float]) -> list[float]:
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return [float(value) for value in self.q_net(obs_t).squeeze(0).cpu().tolist()]

    def select_action(self, q_values: list[float], action_mask: list[int]) -> int:
        q = np.asarray(q_values, dtype=np.float32)
        mask = np.asarray(action_mask, dtype=bool)
        if mask.any():
            q = np.where(mask, q, -1e9)
        return int(np.argmax(q).item())

    def q_report(self, obs_vec: list[float], action_mask: list[int]) -> dict[str, Any]:
        q_values = self.q_values(obs_vec)
        allowed_ids = [idx for idx, allowed in enumerate(action_mask) if allowed]
        value_ids = allowed_ids or list(range(len(q_values)))
        greedy_action = max(value_ids, key=lambda idx: q_values[idx])
        v_web = q_values[greedy_action]
        selected_order = sorted(value_ids, key=lambda idx: q_values[idx], reverse=True)
        return {
            "v_web": float(v_web),
            "greedy_action": int(greedy_action),
            "greedy_action_type": self.actions[greedy_action].value,
            "q_values": q_values,
            "top_actions": [
                {"action_id": idx, "action_type": self.actions[idx].value, "q": q_values[idx]}
                for idx in selected_order[:5]
            ],
            "action_values": [
                {
                    "action_id": idx,
                    "action_type": action.value,
                    "allowed": bool(action_mask[idx]),
                    "q": float(q_values[idx]),
                    "value_gap": float(v_web - q_values[idx]) if action_mask[idx] else None,
                }
                for idx, action in enumerate(self.actions)
            ],
        }

    def _load_payload(self, checkpoint: Path) -> dict[str, Any]:
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        try:
            return torch.load(checkpoint, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(checkpoint, map_location=self.device)


def infer_hidden_sizes(model_state: dict[str, torch.Tensor]) -> list[int]:
    weight_keys = sorted(
        (key for key in model_state if key.startswith("net.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    if len(weight_keys) < 2:
        raise ValueError("cannot infer hidden sizes from checkpoint")
    return [int(model_state[key].shape[0]) for key in weight_keys[:-1]]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    oracle = OracleCheckpoint(args.checkpoint, args.device)
    task_paths = resolve_task_paths(args.tasks)
    task_reports: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []

    for task_path in task_paths:
        for episode in range(args.episodes_per_task):
            task_report, task_q_rows = run_episode(
                oracle=oracle,
                task_path=task_path,
                episode=episode,
                use_action_mask=args.use_action_mask,
            )
            task_reports.append(task_report)
            q_rows.extend(task_q_rows)

    expert_reports = [expert_plan_agreement(oracle, task_path, args.use_action_mask) for task_path in task_paths]
    expert_steps = [step for report in expert_reports for step in report["steps"]]
    goal_rate = sum(int(report["goal"]) for report in task_reports) / max(len(task_reports), 1)
    avg_reward = sum(float(report["total_reward"]) for report in task_reports) / max(len(task_reports), 1)
    avg_steps = sum(int(report["steps"]) for report in task_reports) / max(len(task_reports), 1)
    allowed_steps = [step for step in expert_steps if step["expert_allowed"]]
    report = {
        "checkpoint": str(args.checkpoint),
        "use_action_mask": args.use_action_mask,
        "tasks": [str(path) for path in task_paths],
        "episodes": len(task_reports),
        "eval_goal_rate": goal_rate,
        "eval_avg_reward": avg_reward,
        "eval_avg_steps": avg_steps,
        "expert_action_allowed_rate": sum(int(step["expert_allowed"]) for step in expert_steps) / max(len(expert_steps), 1),
        "expert_action_top1_rate": sum(int(step["expert_rank"] == 1) for step in allowed_steps) / max(len(allowed_steps), 1),
        "expert_action_top3_rate": sum(int(step["expert_rank"] is not None and step["expert_rank"] <= 3) for step in allowed_steps) / max(len(allowed_steps), 1),
        "expert_action_avg_value_gap": (
            sum(float(step["expert_value_gap"]) for step in allowed_steps if step["expert_value_gap"] is not None)
            / max(sum(int(step["expert_value_gap"] is not None) for step in allowed_steps), 1)
        ),
        "task_reports": task_reports,
        "expert_plan_reports": expert_reports,
    }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if args.q_report_output:
        args.q_report_output.parent.mkdir(parents=True, exist_ok=True)
        with args.q_report_output.open("w", encoding="utf-8") as f:
            for row in q_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return report


def expert_plan_agreement(oracle: OracleCheckpoint, task_path: Path, use_action_mask: bool) -> dict[str, Any]:
    task_config = load_task_config(task_path)
    expert_plan = task_config.get("expert_plan", [])
    env = WebAttackSimEnv()
    obs, info = env.reset(task_path)
    task_id = str(info["task_id"])
    steps: list[dict[str, Any]] = []

    for step_idx, raw_action in enumerate(expert_plan):
        obs_vec = env.encode_observation(obs)
        action_mask = env.action_mask(permissive=not use_action_mask)
        q_report = oracle.q_report(obs_vec, action_mask)
        action = normalize_plan_action(raw_action)
        action_id = oracle.actions.index(action.action_type)
        ranked_ids = [item["action_id"] for item in q_report["action_values"] if item["allowed"]]
        ranked_ids.sort(key=lambda idx: q_report["q_values"][idx], reverse=True)
        allowed = bool(action_mask[action_id])
        rank = ranked_ids.index(action_id) + 1 if allowed and action_id in ranked_ids else None
        value_gap = float(q_report["v_web"] - q_report["q_values"][action_id]) if allowed else None
        steps.append(
            {
                "step": step_idx,
                "expert_action_type": action.action_type.value,
                "expert_target": action.target,
                "expert_allowed": allowed,
                "expert_rank": rank,
                "expert_q": float(q_report["q_values"][action_id]),
                "v_web": q_report["v_web"],
                "expert_value_gap": value_gap,
                "greedy_action_type": q_report["greedy_action_type"],
                "top_actions": q_report["top_actions"],
            }
        )
        obs, _reward, done, _truncated, _step_info = env.step(action)
        if done:
            break

    allowed_steps = [step for step in steps if step["expert_allowed"]]
    return {
        "task_id": task_id,
        "num_steps": len(steps),
        "expert_goal": bool(env.state and env.state.done and env._goal_reached()),
        "expert_allowed_rate": sum(int(step["expert_allowed"]) for step in steps) / max(len(steps), 1),
        "expert_top1_rate": sum(int(step["expert_rank"] == 1) for step in allowed_steps) / max(len(allowed_steps), 1),
        "expert_top3_rate": sum(int(step["expert_rank"] is not None and step["expert_rank"] <= 3) for step in allowed_steps) / max(len(allowed_steps), 1),
        "expert_avg_value_gap": (
            sum(float(step["expert_value_gap"]) for step in allowed_steps if step["expert_value_gap"] is not None)
            / max(sum(int(step["expert_value_gap"] is not None) for step in allowed_steps), 1)
        ),
        "steps": steps,
    }


def normalize_plan_action(raw_action: str | dict[str, Any]) -> Action:
    if isinstance(raw_action, str):
        return Action(ActionType(raw_action))
    return Action(
        action_type=ActionType(raw_action["action_type"]),
        target=raw_action.get("target"),
        parameter=raw_action.get("parameter"),
    )


def run_episode(
    *,
    oracle: OracleCheckpoint,
    task_path: Path,
    episode: int,
    use_action_mask: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = WebAttackSimEnv()
    obs, info = env.reset(task_path)
    task_id = str(info["task_id"])
    done = False
    total_reward = 0.0
    step = 0
    trace: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []

    while not done:
        obs_before = obs
        obs_vec = env.encode_observation(obs_before)
        action_mask = env.action_mask(permissive=not use_action_mask)
        q_report = oracle.q_report(obs_vec, action_mask)
        action_id = oracle.select_action(q_report["q_values"], action_mask if use_action_mask else [1] * len(oracle.actions))

        q_rows.append(
            {
                "task_id": task_id,
                "episode": episode,
                "step": step,
                "obs": obs_before.to_dict(),
                "obs_vec": obs_vec,
                **q_report,
            }
        )

        obs, reward, done, _truncated, step_info = env.step(action_id)
        total_reward += reward
        selected_q = q_report["q_values"][action_id]
        trace.append(
            {
                "step": step,
                "action_id": action_id,
                "action_type": oracle.actions[action_id].value,
                "allowed": bool(action_mask[action_id]),
                "q_selected": float(selected_q),
                "v_web": q_report["v_web"],
                "value_gap": float(q_report["v_web"] - selected_q) if action_mask[action_id] else None,
                "reward": reward,
                "feedback": step_info["feedback"],
                "done": done,
                "top_actions": q_report["top_actions"],
            }
        )
        step += 1

    return (
        {
            "task_id": task_id,
            "episode": episode,
            "goal": bool(env.state and env.state.done and env._goal_reached()),
            "steps": step,
            "total_reward": round(total_reward, 6),
            "final_observation": obs.to_dict(),
            "trace": trace,
        },
        q_rows,
    )


def default_checkpoint() -> Path:
    return ROOT / "outputs" / "web_dqn.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Web-RL Value Oracle checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--tasks", nargs="*", default=None, help="Task JSON paths, dirs, or ROOT-relative globs.")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.set_defaults(use_action_mask=True)
    parser.add_argument("--no-action-mask", dest="use_action_mask", action="store_false")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "oracle_eval_report.json")
    parser.add_argument("--q-report-output", type=Path, default=ROOT / "outputs" / "oracle_q_gap_report.jsonl")
    args = parser.parse_args()

    report = evaluate(args)
    print(
        json.dumps(
            {
                "checkpoint": report["checkpoint"],
                "use_action_mask": report["use_action_mask"],
                "episodes": report["episodes"],
                "eval_goal_rate": report["eval_goal_rate"],
                "eval_avg_reward": report["eval_avg_reward"],
                "eval_avg_steps": report["eval_avg_steps"],
                "expert_action_top1_rate": report["expert_action_top1_rate"],
                "expert_action_top3_rate": report["expert_action_top3_rate"],
                "expert_action_avg_value_gap": report["expert_action_avg_value_gap"],
                "report_output": str(args.report_output),
                "q_report_output": str(args.q_report_output) if args.q_report_output else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for task in report["task_reports"]:
        print(
            f"{task['task_id']}: goal={task['goal']} "
            f"steps={task['steps']} total_reward={task['total_reward']}"
        )


if __name__ == "__main__":
    main()
