"""Real LLM (DeepSeek V4) rollout reflux for the Pentest-PRM (method §11.3 DAgger-style).

Instead of the scripted candidate generator, this asks DeepSeek to propose the next Web
pentest actions for each observed state, normalizes them through the Action Normalizer, and
labels them with the RL value oracle. The result is an ON-DISTRIBUTION PRM dataset: the
candidate-action distribution is what a real LLM actually proposes, including out-of-schema
and unsafe actions the normalizer must catch. These samples are refluxed into PRM training
so the PRM is aligned to the real LLM's action distribution, not the designer's templates.

The API key is read only from DEEPSEEK_API_KEY (never written/printed).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from deepseek_client import chat, extract_json_array  # noqa: E402
from demo_pipeline import (  # noqa: E402
    DQNValueOracle,
    build_prm_sample,
    context_from_observation,
    score_candidate,
)
from train_dqn import resolve_task_paths  # noqa: E402
from verify_qstar import resolve_gated_checkpoint  # noqa: E402
from web_attack_sim import WebAttackSimEnv, normalize_llm_action  # noqa: E402
from web_attack_sim.tasks import load_task_config  # noqa: E402

SYSTEM_PROMPT = (
    "You are an authorized web penetration tester assessing a single host in an isolated lab. "
    "Given the current OBSERVED state, propose the {k} most useful DISTINCT next actions. "
    "Each action must be a single concrete step (e.g. 'Enumerate web directories to find hidden "
    "paths', 'Inspect the login form inputs', 'Verify SQL injection on the id parameter', 'Use the "
    "leaked admin credential to log in', 'Upload a web shell', 'Read the flag file'). "
    "Output ONLY a JSON array of {k} short action strings, no prose, no numbering."
)


def llm_candidates(context: str, k: int, model: str) -> list[str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(k=k)},
        {"role": "user", "content": f"Observed state:\n{context}\n\nReturn a JSON array of {k} next actions."},
    ]
    try:
        text = chat(messages, model=model, max_tokens=1200, temperature=0.8)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] DeepSeek call failed: {exc}")
        return []
    items = extract_json_array(text)
    return [str(x).strip() for x in items if str(x).strip()][:k]


def rollout(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = resolve_gated_checkpoint(argparse.Namespace(checkpoint=args.checkpoint))
    oracle = DQNValueOracle(checkpoint, args.device)
    task_paths = resolve_task_paths(args.tasks)

    samples: list[dict[str, Any]] = []
    status_counts: Counter = Counter()
    label_source_counts: Counter = Counter()
    states_visited = 0
    llm_calls = 0

    for task_path in task_paths:
        if args.max_states and states_visited >= args.max_states:
            break
        task = load_task_config(task_path)
        traj = task.get("expert_trajectory") or []
        env = WebAttackSimEnv()
        obs, info = env.reset(task_path)
        task_id = str(info["task_id"])
        history: list[dict[str, Any]] = []

        for step_idx, expert_raw in enumerate(traj):
            if args.max_states and states_visited >= args.max_states:
                break
            context = context_from_observation(task_id, step_idx, obs, history)
            cands = llm_candidates(context, args.candidates_per_state, args.model)
            llm_calls += 1
            states_visited += 1
            obs_vec = env.encode_observation()
            action_mask = env.action_mask(permissive=False)
            obs_dict = obs.to_dict()
            group_id = f"llm:{task_id}:step_{step_idx:02d}"
            for rank_idx, raw in enumerate(cands, start=1):
                scored = score_candidate(oracle, obs_vec, obs_dict, action_mask, raw)
                status_counts[scored["normalized_action"]["status"]] += 1
                label_source_counts[scored["label_source"]] += 1
                sample = build_prm_sample(
                    group_id=group_id, rank_idx=rank_idx, context=context,
                    task_id=task_id, step_idx=step_idx, scored=scored,
                )
                sample.update({"dataset_split": "llm_rollout", "task_path": str(task_path),
                               "oracle_checkpoint": str(checkpoint), "llm_generated": True,
                               "llm_model": args.model})
                samples.append(sample)

            # advance along the expert trajectory to reach the next on-policy state
            n = normalize_llm_action(expert_raw)
            if n.status != "valid" or n.action is None:
                break
            obs, _r, done, _t, step_info = env.step(n.action)
            history.append({"step": step_idx, "raw_action": expert_raw, "action": n.to_dict(),
                            "reward": _r, "feedback": step_info["feedback"], "done": done})
            if done:
                break
        print(f"  {task_id}: states so far={states_visited} samples={len(samples)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    summary = {
        "oracle_checkpoint": str(checkpoint),
        "model": args.model,
        "llm_calls": llm_calls,
        "states_visited": states_visited,
        "num_samples": len(samples),
        "candidate_status_distribution": dict(status_counts),
        "label_source_distribution": dict(label_source_counts),
        "output": str(args.output),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek V4 real LLM rollout reflux for the PRM.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Default: all bundled tasks.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--candidates-per-state", type=int, default=5)
    parser.add_argument("--max-states", type=int, default=80, help="Cap total LLM calls (one per state).")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "llm_rollout_samples.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "outputs" / "llm_rollout_summary.json")
    args = parser.parse_args()

    summary = rollout(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
