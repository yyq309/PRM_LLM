from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from collections import Counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_pipeline import (  # noqa: E402
    DQNValueOracle,
    build_prm_sample,
    context_from_observation,
    rank_candidates,
)
from run_oracle_seed_gate import DEFAULT_EVAL_TASKS, DEFAULT_TRAIN_TASKS  # noqa: E402
from train_dqn import resolve_task_paths  # noqa: E402
from web_attack_sim import WebAttackSimEnv, normalize_llm_action  # noqa: E402
from web_attack_sim.tasks import load_task_config  # noqa: E402


LABEL_VERSION = "webattacksim_prm_v1"


def select_checkpoint(args: argparse.Namespace) -> tuple[Path, dict[str, Any] | None]:
    if args.checkpoint:
        checkpoint = args.checkpoint
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        return checkpoint, read_json(args.seed_gate_report) if args.seed_gate_report.exists() else None

    if args.seed_gate_report.exists():
        gate = read_json(args.seed_gate_report)
        # Explicit canonical override: the gate auto-selects by masked top-3, but the
        # canonical oracle may be a different seed chosen for maskless robustness + top-1.
        # Honoring this field keeps bare-command runs (no --checkpoint) reproducible.
        canonical = gate.get("canonical_checkpoint")
        if canonical and gate.get("passed"):
            cp = Path(canonical)
            if not cp.is_absolute():
                cp = ROOT / cp
            return cp, gate
        if gate.get("passed"):
            reports = [report for report in gate.get("seed_reports", []) if report.get("passed")]
            if not reports:
                raise RuntimeError(f"seed gate passed but has no passed seed reports: {args.seed_gate_report}")
            selected = max(
                reports,
                key=lambda report: (
                    float(report["metrics"]["eval_goal_rate"]),
                    float(report["metrics"]["expert_action_top3_rate"]),
                    -float(report["metrics"]["expert_action_avg_value_gap"]),
                    float(report["metrics"]["expert_action_top1_rate"]),
                    float(report["metrics"]["eval_avg_reward"]),
                ),
            )
            return Path(selected["checkpoint"]), gate
        if not args.allow_ungated_oracle:
            raise RuntimeError(f"seed gate did not pass: {args.seed_gate_report}")

    fallback = ROOT / "outputs" / "web_dqn_pilot.pt"
    if fallback.exists() and args.allow_ungated_oracle:
        return fallback, None
    raise RuntimeError(
        "No passed seed-gate checkpoint is available. Run run_oracle_seed_gate.py first "
        "or pass --checkpoint with --allow-ungated-oracle."
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_split(
    *,
    split: str,
    task_paths: list[Path],
    oracle: DQNValueOracle,
    checkpoint: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    candidate_groups: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    cap = args.max_samples_per_split

    for task_path in task_paths:
        task_config = load_task_config(task_path)
        expert_trajectory = task_config.get("expert_trajectory")
        if not expert_trajectory:
            raise RuntimeError(f"task is missing expert_trajectory: {task_path}")

        env = WebAttackSimEnv()
        obs, info = env.reset(task_path)
        task_id = str(info["task_id"])
        total_reward = 0.0
        trace: list[dict[str, Any]] = []

        for step_idx, expert_raw_action in enumerate(expert_trajectory):
            context = context_from_observation(task_id, step_idx, obs, trace)
            candidates = dataset_candidate_actions(expert_raw_action, obs.to_dict(), args.candidates_per_state)
            ranked = rank_candidates(oracle, env, obs.to_dict(), candidates)
            group_id = f"{split}:{task_id}:step_{step_idx:02d}"
            candidate_groups.append(
                {
                    "group_id": group_id,
                    "dataset_split": split,
                    "task_id": task_id,
                    "task_path": str(task_path),
                    "step": step_idx,
                    "context": context,
                    "candidates": ranked,
                }
            )

            for rank_idx, scored in enumerate(ranked, start=1):
                if cap > 0 and len(samples) >= cap:
                    continue
                sample = build_prm_sample(
                    group_id=group_id,
                    rank_idx=rank_idx,
                    context=context,
                    task_id=task_id,
                    step_idx=step_idx,
                    scored=scored,
                )
                sample.update(
                    {
                        "dataset_split": split,
                        "task_path": str(task_path),
                        "oracle_checkpoint": str(checkpoint),
                        "label_version": args.label_version,
                    }
                )
                samples.append(sample)

            normalized_expert = normalize_llm_action(expert_raw_action)
            if normalized_expert.status != "valid" or normalized_expert.action is None:
                raise RuntimeError(f"expert action failed to normalize: {task_id} step={step_idx}: {expert_raw_action}")
            obs, reward, done, _truncated, step_info = env.step(normalized_expert.action)
            total_reward += reward
            trace.append(
                {
                    "step": step_idx,
                    "raw_action": expert_raw_action,
                    "action": normalized_expert.to_dict(),
                    "reward": reward,
                    "feedback": step_info["feedback"],
                    "done": done,
                }
            )
            if done:
                break

        solved = bool(env.state and env.state.done and env._goal_reached())
        task_summaries.append(
            {
                "task_id": task_id,
                "task_path": str(task_path),
                "solved": solved,
                "steps": len(trace),
                "total_reward": round(total_reward, 3),
                "final_observation": obs.to_dict(),
                "trace": trace,
            }
        )
        if not solved:
            raise RuntimeError(f"expert trajectory did not solve {task_id}")

    return {
        "split": split,
        "samples": samples,
        "candidate_groups": candidate_groups,
        "task_summaries": task_summaries,
        "summary": summarize_samples(split, samples, candidate_groups, task_summaries),
    }


def dataset_candidate_actions(expert_raw_action: str, obs: dict[str, Any], max_candidates: int) -> list[str]:
    paths = set(obs["discovered_paths"])
    known_parameters = list(obs["known_parameters"])
    candidates = [
        expert_raw_action,
        "Fingerprint the HTTP server headers and technology stack.",
        "Enumerate web directories to discover more routes.",
        "Use leaked credential to login through /login.",
        "Run whoami through the web shell.",
        "Escalate privileges to root with sudo.",
    ]

    if any(path in paths for path in ["/backup", "/config", "/.git"]):
        candidates.append("Download and inspect the discovered backup or config content for leaked credentials.")
    else:
        candidates.append("Download and inspect the /backup content for leaked credentials.")

    if "/login" in paths:
        candidates.append("Inspect the login form and discover its input fields.")
        candidates.append("Try login with default password admin/admin.")
    else:
        candidates.append("Try login with default password admin/admin before discovering the login path.")

    if "/upload" in paths:
        candidates.append("Inspect the upload form and discover its file input.")
        candidates.append("Upload a PHP web shell through /upload.")
    else:
        candidates.append("Upload a PHP web shell through /upload before discovering the upload path.")

    if known_parameters:
        parameter = known_parameters[0]
        candidates.append(f"Verify SQL injection on {parameter} parameter.")
        candidates.append(f"Exploit the verified vulnerability on {parameter} to dump credentials or obtain a shell.")
    else:
        candidates.append("Verify SQL injection on /item?id parameter.")

    if obs["auth_state"] != "anonymous" or obs["shell_state"] != "none":
        candidates.append("Read the target flag file.")
    else:
        candidates.append("Read the /admin/flag sensitive file now.")

    diagnostic_candidates = [
        "Check JWT token signing issues on the login session.",
        "Try SSRF by requesting cloud metadata through a URL parameter.",
        "Run rm -rf /tmp/app on the target.",
        "Phishing the site administrator for credentials.",
        "Take the next step.",
    ]

    context_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in context_candidates:
            context_candidates.append(candidate)

    diagnostics: list[str] = []
    for candidate in diagnostic_candidates:
        if candidate not in context_candidates and candidate not in diagnostics:
            diagnostics.append(candidate)
    if max_candidates <= 0:
        return context_candidates + diagnostics
    if max_candidates <= len(diagnostics):
        return [expert_raw_action, *diagnostics[: max_candidates - 1]]
    context_limit = max_candidates - len(diagnostics)
    return context_candidates[:context_limit] + diagnostics


def summarize_samples(
    split: str,
    samples: list[dict[str, Any]],
    candidate_groups: list[dict[str, Any]],
    task_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    label_sources = Counter(str(sample["label_source"]) for sample in samples)
    rank_labels = Counter(str(sample["rank_label"]) for sample in samples)
    diagnoses = Counter(str(sample["diagnosis"]) for sample in samples)
    scores = [float(sample["score"]) for sample in samples]
    confidences = [float(sample["label_confidence"]) for sample in samples]
    return {
        "split": split,
        "num_tasks": len(task_summaries),
        "num_solved_tasks": sum(int(task["solved"]) for task in task_summaries),
        "num_candidate_groups": len(candidate_groups),
        "num_samples": len(samples),
        "label_source_counts": dict(sorted(label_sources.items())),
        "rank_label_counts": dict(sorted(rank_labels.items())),
        "diagnosis_counts": dict(sorted(diagnoses.items())),
        "score_mean": round(sum(scores) / max(len(scores), 1), 6),
        "label_confidence_mean": round(sum(confidences) / max(len(confidences), 1), 6),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate gated WebAttackSim PRM label datasets.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed-gate-report", type=Path, default=ROOT / "outputs" / "oracle_seed_gate.json")
    parser.add_argument("--allow-ungated-oracle", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-tasks", nargs="*", default=DEFAULT_TRAIN_TASKS)
    parser.add_argument("--heldout-tasks", nargs="*", default=DEFAULT_EVAL_TASKS)
    parser.add_argument("--candidates-per-state", type=int, default=12)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--label-version", type=str, default=LABEL_VERSION)
    parser.add_argument("--train-output", type=Path, default=ROOT / "outputs" / "prm_samples_train.jsonl")
    parser.add_argument("--heldout-output", type=Path, default=ROOT / "outputs" / "prm_samples_heldout.jsonl")
    parser.add_argument("--ranking-output", type=Path, default=ROOT / "outputs" / "prm_candidate_ranking.json")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "outputs" / "prm_dataset_summary.json")
    args = parser.parse_args()

    checkpoint, seed_gate = select_checkpoint(args)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    oracle = DQNValueOracle(checkpoint, args.device)
    train_paths = resolve_task_paths(args.train_tasks)
    heldout_paths = resolve_task_paths(args.heldout_tasks)
    train = generate_split(split="train", task_paths=train_paths, oracle=oracle, checkpoint=checkpoint, args=args)
    heldout = generate_split(split="heldout", task_paths=heldout_paths, oracle=oracle, checkpoint=checkpoint, args=args)

    write_jsonl(args.train_output, train["samples"])
    write_jsonl(args.heldout_output, heldout["samples"])
    write_json(
        args.ranking_output,
        {
            "label_version": args.label_version,
            "oracle_checkpoint": str(checkpoint),
            "candidate_groups": train["candidate_groups"] + heldout["candidate_groups"],
        },
    )
    summary = {
        "label_version": args.label_version,
        "oracle_checkpoint": str(checkpoint),
        "seed_gate_report": str(args.seed_gate_report) if seed_gate else None,
        "seed_gate_passed": bool(seed_gate.get("passed")) if seed_gate else None,
        "train_output": str(args.train_output),
        "heldout_output": str(args.heldout_output),
        "ranking_output": str(args.ranking_output),
        "splits": {
            "train": train["summary"],
            "heldout": heldout["summary"],
        },
    }
    write_json(args.summary_output, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
