from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_dqn import QNetwork  # noqa: E402
from web_attack_sim import ActionType, WebAttackSimEnv, normalize_llm_action  # noqa: E402
from web_attack_sim.normalizer import NormalizedAction  # noqa: E402
from web_attack_sim.tasks import bundled_task_paths, load_task_config  # noqa: E402


EXPERT_TRAJECTORIES: dict[str, list[str]] = {
    "web_001_backup_leak": [
        "Enumerate web directories to discover /login.",
        "Enumerate web directories to discover /backup.",
        "Download and inspect the /backup content for leaked credentials.",
        "Enumerate web directories to discover /admin.",
        "Inspect the login form and discover its input fields.",
        "Use the leaked admin credential to login through /login.",
        "Read the /admin/flag sensitive file.",
    ],
    "web_002_sqli_login": [
        "Enumerate web directories to discover /login.",
        "Enumerate web directories to discover /item.",
        "Discover input parameters on the /item endpoint.",
        "Verify SQL injection on /item?id parameter.",
        "Exploit the SQL injection to dump the admin credential.",
        "Enumerate web directories to discover /admin.",
        "Use the known admin credential to login through /login.",
        "Read the /admin/flag sensitive file.",
    ],
    "web_003_upload_shell": [
        "Enumerate web directories to discover /login.",
        "Enumerate web directories to discover /admin.",
        "Inspect the login form and discover its input fields.",
        "Login with default password admin/admin.",
        "Enumerate web directories to discover /upload.",
        "Inspect the upload form and discover its file input.",
        "Upload a PHP web shell through /upload.",
        "Run whoami through the uploaded web shell.",
        "Read the target flag file.",
    ],
}


class DQNValueOracle:
    def __init__(self, checkpoint: Path, device: str):
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        payload = _load_checkpoint(checkpoint, self.device)

        model_state = payload["model_state"]
        obs_dim = int(payload["obs_dim"])
        num_actions = int(payload["num_actions"])
        hidden_sizes = payload.get("metadata", {}).get("hidden_sizes") or _infer_hidden_sizes(model_state)

        self.q_net = QNetwork(obs_dim, num_actions, list(hidden_sizes)).to(self.device)
        self.q_net.load_state_dict(model_state)
        self.q_net.eval()
        self.actions = [ActionType(action) for action in payload.get("actions", [a.value for a in ActionType])]
        self.action_to_id = {action: idx for idx, action in enumerate(self.actions)}

    def score_action(self, obs_vec: list[float], action_type: ActionType, action_mask: list[int]) -> dict[str, Any]:
        action_id = self.action_to_id[action_type]
        q_values = self.q_values(obs_vec)
        allowed_ids = [idx for idx, allowed in enumerate(action_mask) if allowed]
        value_ids = allowed_ids or list(range(len(q_values)))
        order = sorted(value_ids, key=lambda idx: q_values[idx], reverse=True)
        greedy_action = order[0]
        q_selected = float(q_values[action_id])
        v_state = float(q_values[greedy_action])
        selected_allowed = bool(action_mask[action_id])
        value_gap = float(v_state - q_selected) if selected_allowed else None
        return {
            "q_values": q_values,
            "top_actions": [
                {
                    "action_id": idx,
                    "action_type": self.actions[idx].value,
                    "q": float(q_values[idx]),
                }
                for idx in order[:5]
            ],
            "greedy_action": greedy_action,
            "greedy_action_type": self.actions[greedy_action].value,
            "v_web": v_state,
            "selected_action": action_id,
            "selected_action_type": action_type.value,
            "selected_action_allowed": selected_allowed,
            "q_selected": q_selected,
            "value_gap": value_gap,
            "oracle_rank": order.index(action_id) + 1 if selected_allowed else None,
            "num_ranked_actions": len(order),
        }

    def q_values(self, obs_vec: list[float]) -> list[float]:
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return [float(value) for value in self.q_net(obs_t).squeeze(0).cpu().tolist()]


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _infer_hidden_sizes(model_state: dict[str, torch.Tensor]) -> list[int]:
    weight_keys = sorted(
        (key for key in model_state if key.startswith("net.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    if len(weight_keys) < 2:
        raise ValueError("cannot infer QNetwork hidden sizes from checkpoint")
    return [int(model_state[key].shape[0]) for key in weight_keys[:-1]]


def load_checkpoint_eval_report(checkpoint: Path) -> dict[str, Any] | None:
    eval_path = checkpoint.with_suffix(".eval.json")
    if not eval_path.exists():
        return None
    with eval_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    oracle = DQNValueOracle(args.checkpoint, args.device)
    checkpoint_eval = load_checkpoint_eval_report(args.checkpoint)
    oracle_warning = None
    if checkpoint_eval is None:
        oracle_warning = "checkpoint_eval_report_missing"
    elif float(checkpoint_eval.get("eval_goal_rate", 0.0)) < 1.0:
        oracle_warning = "checkpoint_eval_goal_rate_below_formal_oracle_threshold"

    samples: list[dict[str, Any]] = []
    candidate_groups: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []

    for task_path in bundled_task_paths():
        task_config = load_task_config(task_path)
        env = WebAttackSimEnv()
        obs, info = env.reset(task_path)
        task_id = str(info["task_id"])
        expert_trajectory = task_config.get("expert_trajectory") or EXPERT_TRAJECTORIES[task_id]
        total_reward = 0.0
        trace: list[dict[str, Any]] = []

        for step_idx, expert_raw_action in enumerate(expert_trajectory):
            context = context_from_observation(task_id, step_idx, obs, trace)
            candidates = candidate_actions_for_state(expert_raw_action, obs.to_dict())
            ranked = rank_candidates(oracle, env, obs.to_dict(), candidates)
            group_id = f"{task_id}:step_{step_idx:02d}"

            candidate_group = {
                "group_id": group_id,
                "task_id": task_id,
                "step": step_idx,
                "context": context,
                "candidates": ranked,
            }
            candidate_groups.append(candidate_group)

            for rank_idx, scored in enumerate(ranked, start=1):
                if len(samples) >= args.max_samples:
                    break
                samples.append(
                    build_prm_sample(
                        group_id=group_id,
                        rank_idx=rank_idx,
                        context=context,
                        task_id=task_id,
                        step_idx=step_idx,
                        scored=scored,
                    )
                )

            normalized_expert = normalize_llm_action(expert_raw_action)
            if normalized_expert.status != "valid" or normalized_expert.action is None:
                raise RuntimeError(f"expert action failed to normalize: {expert_raw_action}")
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
                "solved": solved,
                "steps": len(trace),
                "total_reward": round(total_reward, 3),
                "final_observation": obs.to_dict(),
                "trace": trace,
            }
        )
        if not solved:
            raise RuntimeError(f"demo expert trajectory did not solve {task_id}")

    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    with args.sample_output.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    args.ranking_output.parent.mkdir(parents=True, exist_ok=True)
    with args.ranking_output.open("w", encoding="utf-8") as f:
        json.dump(candidate_groups, f, ensure_ascii=False, indent=2)

    summary = {
        "checkpoint": str(args.checkpoint),
        "oracle_eval_report": checkpoint_eval,
        "oracle_warning": oracle_warning,
        "sample_output": str(args.sample_output),
        "ranking_output": str(args.ranking_output),
        "num_prm_samples": len(samples),
        "num_candidate_groups": len(candidate_groups),
        "tasks": task_summaries,
    }
    return summary


def rank_candidates(
    oracle: DQNValueOracle,
    env: WebAttackSimEnv,
    obs_dict: dict[str, Any],
    raw_actions: list[str],
) -> list[dict[str, Any]]:
    obs_vec = env.encode_observation()
    action_mask = env.action_mask(permissive=False)
    scored = [score_candidate(oracle, obs_vec, obs_dict, action_mask, raw) for raw in raw_actions]
    ranked = sorted(
        scored,
        key=lambda row: (
            row["score"],
            row.get("q_selected") if row.get("q_selected") is not None else -9999.0,
            row["normalized_action"]["confidence"],
        ),
        reverse=True,
    )
    for idx, row in enumerate(ranked, start=1):
        row["candidate_rank"] = idx
    return ranked


def score_candidate(
    oracle: DQNValueOracle,
    obs_vec: list[float],
    obs_dict: dict[str, Any],
    action_mask: list[int],
    raw_action: str,
) -> dict[str, Any]:
    normalized = normalize_llm_action(raw_action)
    if normalized.status != "valid" or normalized.action is None:
        return {
            "raw_llm_action": raw_action,
            "normalized_action": normalized.to_dict(),
            "score": 0.0,
            "q_selected": None,
            "v_state": None,
            "value_gap": None,
            "rank_label": normalized.status,
            "diagnosis": normalized.reason,
            "natural_language_feedback": feedback_for_diagnosis(normalized.reason),
            "oracle_q_report": None,
            "label_source": "normalizer_rule",
            "normalizer_confidence": normalized.confidence,
            "schema_confidence": schema_confidence_for_status(normalized.status),
            "oracle_label_confidence": None,
            "label_confidence": label_confidence(
                normalizer_confidence=normalized.confidence,
                schema_confidence=schema_confidence_for_status(normalized.status),
                oracle_label_confidence=None,
            ),
        }

    q_report = oracle.score_action(obs_vec, normalized.action.action_type, action_mask)
    if not q_report["selected_action_allowed"]:
        diagnosis = diagnose_disallowed_action(normalized, obs_dict)
        return {
            "raw_llm_action": raw_action,
            "normalized_action": normalized.to_dict(),
            "score": 0.0,
            "q_selected": q_report["q_selected"],
            "v_state": q_report["v_web"],
            "value_gap": None,
            "rank_label": "precondition_missing",
            "diagnosis": diagnosis,
            "natural_language_feedback": feedback_for_diagnosis(diagnosis),
            "oracle_q_report": q_report,
            "label_source": "precondition_rule",
            "normalizer_confidence": normalized.confidence,
            "schema_confidence": 1.0,
            "oracle_label_confidence": None,
            "label_confidence": label_confidence(
                normalizer_confidence=normalized.confidence,
                schema_confidence=1.0,
                oracle_label_confidence=None,
            ),
        }

    rank_label = rank_label_from_oracle(q_report["oracle_rank"], q_report["num_ranked_actions"])
    diagnosis = diagnose_valid_action(normalized, obs_dict, q_report, rank_label)
    score = score_from_gap(q_report["value_gap"], rank_label)
    return {
        "raw_llm_action": raw_action,
        "normalized_action": normalized.to_dict(),
        "score": score,
        "q_selected": q_report["q_selected"],
        "v_state": q_report["v_web"],
        "value_gap": q_report["value_gap"],
        "rank_label": rank_label,
        "diagnosis": diagnosis,
        "natural_language_feedback": feedback_for_diagnosis(diagnosis),
        "oracle_q_report": q_report,
        "label_source": "oracle",
        "normalizer_confidence": normalized.confidence,
        "schema_confidence": 1.0,
        "oracle_label_confidence": oracle_label_confidence(q_report, rank_label),
        "label_confidence": label_confidence(
            normalizer_confidence=normalized.confidence,
            schema_confidence=1.0,
            oracle_label_confidence=oracle_label_confidence(q_report, rank_label),
        ),
    }


def build_prm_sample(
    *,
    group_id: str,
    rank_idx: int,
    context: str,
    task_id: str,
    step_idx: int,
    scored: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": f"{group_id}:candidate_{rank_idx:02d}",
        "task_id": task_id,
        "step": step_idx,
        "candidate_group": group_id,
        "candidate_rank": rank_idx,
        "context": context,
        "raw_llm_action": scored["raw_llm_action"],
        "normalized_action": scored["normalized_action"],
        "score": scored["score"],
        "q_selected": scored["q_selected"],
        "v_state": scored["v_state"],
        "value_gap": scored["value_gap"],
        "rank_label": scored["rank_label"],
        "diagnosis": scored["diagnosis"],
        "natural_language_feedback": scored["natural_language_feedback"],
        "oracle_q_report": scored["oracle_q_report"],
        "label_source": scored["label_source"],
        "normalizer_confidence": scored["normalizer_confidence"],
        "schema_confidence": scored["schema_confidence"],
        "oracle_label_confidence": scored["oracle_label_confidence"],
        "label_confidence": scored["label_confidence"],
    }


def candidate_actions_for_state(expert_raw_action: str, obs: dict[str, Any]) -> list[str]:
    candidates = [expert_raw_action]
    paths = set(obs["discovered_paths"])

    if obs["auth_state"] == "anonymous":
        candidates.append("Read the /admin/flag sensitive file now.")
        candidates.append("Use leaked credential to login through /login.")
    else:
        candidates.append("Use the known admin credential to login through /login again.")

    if obs["shell_state"] == "none":
        candidates.append("Run whoami through the web shell.")
    else:
        candidates.append("Escalate privileges to root with sudo.")

    if "/upload" not in paths:
        candidates.append("Upload a PHP web shell through /upload.")
    else:
        candidates.append("Inspect the upload form and discover its file input.")

    candidates.append("Pivot to another host on the internal network.")
    return list(dict.fromkeys(candidates))[:5]


def opaque_scenario_id(task_id: str) -> str:
    """Deterministic, non-descriptive scenario id for the verbalized PRM context.

    The descriptive task_id (e.g. ``web_002_sqli_login``) names the vulnerability
    family and would leak the scenario type into the PRM input. A stable hash keeps
    a scenario identifier (method §10 schema) without revealing the vuln type, and
    held-out tasks get hashes unseen in training (OOV under TF-IDF), so the PRM
    cannot key off the label on the held-out split.
    """
    digest = hashlib.sha1(str(task_id).encode("utf-8")).hexdigest()[:10]
    return f"scenario_{digest}"


def context_from_observation(task_id: str, step_idx: int, obs: Any, trace: list[dict[str, Any]]) -> str:
    data = obs.to_dict()
    recent = trace[-3:]
    history = [
        {
            "action": item["action"]["action_type"],
            "success": item["feedback"]["success"],
            "event": item["feedback"]["progress_event"],
            "error": item["feedback"]["error_type"],
            "evidence": item["feedback"]["evidence"],
        }
        for item in recent
    ]
    return (
        f"Scenario {opaque_scenario_id(task_id)}, step {step_idx}. "
        f"Known paths: {data['discovered_paths']}. "
        f"Known forms: {data['known_forms']}. "
        f"Known parameters: {data['known_parameters']}. "
        f"Credentials: {data['credentials']}. "
        f"Auth state: {data['auth_state']}. "
        f"Shell state: {data['shell_state']}. "
        f"Verified vulnerabilities: {data['verified_vulnerabilities']}. "
        f"Read files: {data['read_files']}. "
        f"Failed branches: {data['failed_branches']}. "
        f"Remaining budget: {data['remaining_budget']}. "
        f"Recent feedback: {history}."
    )


def rank_label_from_oracle(oracle_rank: int, num_actions: int) -> str:
    if oracle_rank <= max(1, num_actions // 4):
        return "high"
    if oracle_rank <= max(2, int(num_actions * 0.65)):
        return "medium"
    return "low"


def score_from_gap(value_gap: float, rank_label: str) -> float:
    base = 1.0 / (1.0 + max(value_gap, 0.0))
    if rank_label == "medium":
        base *= 0.75
    elif rank_label == "low":
        base *= 0.4
    return round(max(0.0, min(base, 1.0)), 4)


def schema_confidence_for_status(status: str) -> float:
    if status == "valid":
        return 1.0
    if status == "schema_gap":
        return 0.45
    if status == "ambiguous":
        return 0.25
    return 0.0


def oracle_label_confidence(q_report: dict[str, Any], rank_label: str) -> float:
    if q_report["value_gap"] is None:
        return 0.0
    top = q_report["top_actions"]
    if len(top) <= 1:
        margin = 1.0
    else:
        margin = abs(float(top[0]["q"]) - float(top[1]["q"]))
    margin_score = min(margin / 3.0, 1.0)
    if rank_label == "high":
        return round(0.65 + 0.35 * margin_score, 4)
    if rank_label == "medium":
        return round(0.45 + 0.25 * margin_score, 4)
    return round(0.35 + 0.2 * margin_score, 4)


def label_confidence(
    *,
    normalizer_confidence: float,
    schema_confidence: float,
    oracle_label_confidence: float | None,
) -> float:
    oracle_component = 1.0 if oracle_label_confidence is None else oracle_label_confidence
    return round(normalizer_confidence * schema_confidence * oracle_component, 4)


def diagnose_valid_action(
    normalized: NormalizedAction,
    obs: dict[str, Any],
    q_report: dict[str, Any],
    rank_label: str,
) -> str:
    assert normalized.action is not None
    action_type = normalized.action.action_type
    paths = set(obs["discovered_paths"])

    if rank_label == "high" and q_report["value_gap"] <= 1e-5:
        return "high_value_action"
    if action_type == ActionType.CREDENTIAL_USE and not obs["credentials"]:
        return "credential_missing"
    if action_type in {ActionType.AUTH_ATTEMPT, ActionType.CREDENTIAL_USE} and "/login" not in paths:
        return "precondition_missing"
    if action_type == ActionType.COMMAND_EXECUTION and obs["shell_state"] == "none":
        return "precondition_missing"
    if action_type == ActionType.FILE_UPLOAD_ATTEMPT and "/upload" not in paths:
        return "insufficient_enumeration"
    if action_type == ActionType.PRIVILEGE_ESCALATION and obs["shell_state"] == "none":
        return "premature_privilege_escalation"
    if action_type == ActionType.SENSITIVE_FILE_READ and obs["auth_state"] == "anonymous" and obs["shell_state"] == "none":
        return "premature_goal_access"
    if action_type == ActionType.VULNERABILITY_CHECK and not obs["known_parameters"]:
        return "insufficient_input_discovery"
    if action_type == ActionType.CONTENT_RETRIEVAL and len(paths) <= 1:
        return "insufficient_enumeration"
    if rank_label == "low":
        return "low_oracle_value"
    if rank_label == "medium":
        return "medium_value_action"
    return "high_value_action"


def diagnose_disallowed_action(normalized: NormalizedAction, obs: dict[str, Any]) -> str:
    assert normalized.action is not None
    action_type = normalized.action.action_type
    paths = set(obs["discovered_paths"])

    if action_type == ActionType.CREDENTIAL_USE and not obs["credentials"]:
        return "credential_missing"
    if action_type in {ActionType.AUTH_ATTEMPT, ActionType.CREDENTIAL_USE} and "/login" not in paths:
        return "precondition_missing"
    if action_type == ActionType.AUTH_ATTEMPT and obs["auth_state"] == "admin":
        return "duplicate_action"
    if action_type == ActionType.CREDENTIAL_USE and obs["auth_state"] == "admin":
        return "duplicate_action"
    if action_type == ActionType.COMMAND_EXECUTION and obs["shell_state"] == "none":
        return "precondition_missing"
    if action_type == ActionType.FILE_UPLOAD_ATTEMPT and "/upload" not in paths:
        return "insufficient_enumeration"
    if action_type == ActionType.FILE_UPLOAD_ATTEMPT and obs["auth_state"] != "admin":
        return "auth_state_mismatch"
    if action_type == ActionType.PRIVILEGE_ESCALATION and obs["shell_state"] == "none":
        return "premature_privilege_escalation"
    if action_type == ActionType.SENSITIVE_FILE_READ and obs["auth_state"] == "anonymous" and obs["shell_state"] == "none":
        return "premature_goal_access"
    if action_type == ActionType.VULNERABILITY_CHECK and not obs["known_parameters"]:
        return "insufficient_input_discovery"
    if action_type in {ActionType.WEB_PATH_ENUMERATION, ActionType.INPUT_DISCOVERY, ActionType.CONTENT_RETRIEVAL}:
        return "no_new_information"
    return "precondition_missing"


def feedback_for_diagnosis(diagnosis: str) -> str:
    messages = {
        "high_value_action": "This action matches the current high-value Web attack path.",
        "medium_value_action": "This action is valid, but another action has higher long-term value in this state.",
        "low_oracle_value": "The value oracle ranks this action low for the current observable state.",
        "credential_missing": "Credentials have not been discovered yet, so credential use is premature.",
        "precondition_missing": "A required foothold, path, form, or session is missing before this action can help.",
        "insufficient_enumeration": "Enumerate the relevant Web surface before trying this action.",
        "premature_privilege_escalation": "Privilege escalation requires a shell foothold first.",
        "premature_goal_access": "The goal file is not reachable from the current anonymous/no-shell state.",
        "insufficient_input_discovery": "Discover parameters or forms before verifying input-driven vulnerabilities.",
        "auth_state_mismatch": "This action requires a stronger authenticated state.",
        "duplicate_action": "This action repeats a state transition that has already been completed.",
        "empty_action": "The action text is empty and cannot be normalized.",
        "out_of_scope_or_destructive_action": "The action is unsafe or outside the single-host Web scope.",
        "outside_single_host_web_scope": "This action belongs to another domain and is unsupported by the abstract Web simulator.",
        "action_intent_is_too_vague": "The action is too vague to map to a concrete abstract Web action.",
        "no_matching_abstract_web_action": "No supported abstract Web action matches this text.",
        "unsupported_in_scope": "The action is in the Web domain, but the current abstract action schema does not cover it yet.",
    }
    return messages.get(diagnosis, "The action needs review before it can be used as a PRM training label.")


def default_checkpoint() -> Path:
    masked = ROOT / "outputs" / "web_dqn_masked.pt"
    if masked.exists():
        return masked
    full = ROOT / "outputs" / "web_dqn.pt"
    if full.exists():
        return full
    return ROOT / "outputs" / "web_dqn_smoke.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-one WebAttackSim demo pipeline.")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--sample-output", type=Path, default=ROOT / "outputs" / "demo_prm_samples.jsonl")
    parser.add_argument("--ranking-output", type=Path, default=ROOT / "outputs" / "demo_candidate_ranking.json")
    args = parser.parse_args()

    summary = run_demo(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, ensure_ascii=False, indent=2))
    for task in summary["tasks"]:
        print(
            f"{task['task_id']}: solved={task['solved']} "
            f"steps={task['steps']} total_reward={task['total_reward']}"
        )


if __name__ == "__main__":
    main()
