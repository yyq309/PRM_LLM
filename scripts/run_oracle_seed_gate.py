from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from task_split import get_split  # noqa: E402

# De-templated structural-family split (method §13.1). Held-out = unseen instances of
# trained families + WHOLE unseen-chain families (0 plan-signature overlap with train).
_SPLIT = get_split()
DEFAULT_TRAIN_TASKS = _SPLIT["train"]
DEFAULT_EVAL_TASKS = _SPLIT["heldout_all"]
HELDOUT_INSTANCE_TASKS = _SPLIT["heldout_instance"]
HELDOUT_CHAIN_TASKS = _SPLIT["heldout_chain"]

METRIC_KEYS = [
    "eval_goal_rate",
    "eval_avg_reward",
    "eval_avg_steps",
    "expert_action_allowed_rate",
    "expert_action_top1_rate",
    "expert_action_top3_rate",
    "expert_action_avg_value_gap",
]


def run_command(cmd: list[str]) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def seed_passed(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        float(metrics["eval_goal_rate"]) >= args.min_goal_rate
        and float(metrics["expert_action_top3_rate"]) >= args.min_expert_top3_rate
        and float(metrics["expert_action_avg_value_gap"]) <= args.max_expert_avg_gap
    )


def run_seed(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = seed_dir / f"web_dqn_seed_{seed}.pt"
    train_eval_report = checkpoint.with_suffix(".eval.json")
    eval_report = seed_dir / "oracle_eval_heldout.json"
    q_report = seed_dir / "oracle_q_gap_heldout.jsonl"

    train_tasks = args.train_tasks or DEFAULT_TRAIN_TASKS
    eval_tasks = args.eval_tasks or DEFAULT_EVAL_TASKS

    if not args.skip_existing or not checkpoint.exists():
        train_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_dqn.py"),
            "--seed",
            str(seed),
            "--training-steps",
            str(args.training_steps),
            "--eval-episodes",
            str(args.eval_episodes),
            "--log-every",
            str(args.log_every),
            "--device",
            args.device,
            "--output",
            str(checkpoint),
            "--train-tasks",
            *train_tasks,
            "--eval-tasks",
            *eval_tasks,
        ]
        if args.train_no_action_mask:
            train_cmd.append("--no-action-mask")
        if getattr(args, "curriculum", False):
            train_cmd.append("--curriculum")
        run_command(train_cmd)
    else:
        print(f"skip training seed={seed}: {checkpoint} exists", flush=True)

    if not args.skip_existing or not eval_report.exists():
        eval_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_oracle.py"),
            "--checkpoint",
            str(checkpoint),
            "--tasks",
            *eval_tasks,
            "--episodes-per-task",
            str(args.episodes_per_task),
            "--device",
            args.device,
            "--report-output",
            str(eval_report),
            "--q-report-output",
            str(q_report),
        ]
        run_command(eval_cmd)
    else:
        print(f"skip evaluation seed={seed}: {eval_report} exists", flush=True)

    report = read_json(eval_report)
    metrics = {key: report[key] for key in METRIC_KEYS}
    passed = seed_passed(metrics, args)
    return {
        "seed": seed,
        "passed": passed,
        "checkpoint": str(checkpoint),
        "train_eval_report": str(train_eval_report),
        "eval_report": str(eval_report),
        "q_report": str(q_report),
        "metrics": metrics,
    }


def build_aggregate(seed_reports: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    metric_summary = {
        key: numeric_summary([float(report["metrics"][key]) for report in seed_reports])
        for key in METRIC_KEYS
    }
    return {
        "passed": all(bool(report["passed"]) for report in seed_reports),
        "thresholds": {
            "min_goal_rate": args.min_goal_rate,
            "min_expert_top3_rate": args.min_expert_top3_rate,
            "max_expert_avg_value_gap": args.max_expert_avg_gap,
        },
        "training_steps": args.training_steps,
        "train_tasks": args.train_tasks or DEFAULT_TRAIN_TASKS,
        "eval_tasks": args.eval_tasks or DEFAULT_EVAL_TASKS,
        "seeds": [report["seed"] for report in seed_reports],
        "metrics": metric_summary,
        "seed_reports": seed_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed Web-RL oracle gate on the pilot held-out split.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--training-steps", type=int, default=25000)
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per task used by train_dqn.py metadata eval.")
    parser.add_argument("--episodes-per-task", type=int, default=1, help="Episodes per held-out task used by evaluate_oracle.py.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-tasks", nargs="*", default=None, help="Train task JSON paths, ROOT-relative by default.")
    parser.add_argument("--eval-tasks", nargs="*", default=None, help="Held-out eval task JSON paths, ROOT-relative by default.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "seed_gate")
    parser.add_argument("--aggregate-output", type=Path, default=ROOT / "outputs" / "oracle_seed_gate.json")
    parser.add_argument("--min-goal-rate", type=float, default=1.0)
    parser.add_argument("--min-expert-top3-rate", type=float, default=0.8)
    parser.add_argument("--max-expert-avg-gap", type=float, default=3.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--train-no-action-mask",
        action="store_true",
        help="Train the oracle in permissive mode (no strict action mask) so its Q-values are meaningful for all 16 actions.",
    )
    parser.add_argument("--curriculum", action="store_true", help="Easy->medium->hard curriculum (method §12.1).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_reports = [run_seed(seed, args) for seed in args.seeds]
    aggregate = build_aggregate(seed_reports, args)

    args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    with args.aggregate_output.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "passed": aggregate["passed"],
                "seeds": aggregate["seeds"],
                "thresholds": aggregate["thresholds"],
                "metric_min": {key: value["min"] for key, value in aggregate["metrics"].items()},
                "aggregate_output": str(args.aggregate_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
