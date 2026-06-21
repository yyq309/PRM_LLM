from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys
from typing import Any

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_pipeline import DQNValueOracle, context_from_observation  # noqa: E402
from generate_prm_dataset import select_checkpoint  # noqa: E402
from task_split import get_split  # noqa: E402
from train_dqn import resolve_task_paths  # noqa: E402
from train_prm_strong import extract_features  # noqa: E402

# Default closed-loop eval set = the structural-split held-out tasks (method §13.1):
# task_split.json heldout_all = 20 tasks (10 unseen-instance + 10 unseen-chain).
HELDOUT_ALL_TASKS = get_split()["heldout_all"]
from train_prm_baseline import sample_to_text  # noqa: E402
from web_attack_sim import WebAttackSimEnv, normalize_llm_action  # noqa: E402
from web_attack_sim.action_space import ActionType  # noqa: E402
from web_attack_sim.tasks import load_task_config  # noqa: E402


RANK_PRIOR = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.25,
    "precondition_missing": 0.0,
    "schema_gap": 0.0,
    "unsafe": 0.0,
    "outside_single_host_web_scope": 0.0,
    "ambiguous": 0.0,
}


def policy_candidate_actions(obs: dict[str, Any], max_candidates: int, include_diagnostics: bool) -> list[str]:
    paths = [path for path in obs["discovered_paths"] if path != "/"]
    known_parameters = list(obs["known_parameters"])
    candidates = [
        "Fingerprint the HTTP server headers and technology stack.",
        "Enumerate web directories to discover more routes.",
        "Submit the final report.",
    ]

    for path in paths:
        candidates.append(f"Download and inspect {path} content for leaked credentials.")
        candidates.append(f"Inspect the form or input fields on {path}.")

    if any(path in paths for path in ["/login", "/admin/login"]):
        candidates.append("Inspect the login form and discover its input fields.")
        candidates.append("Try login with default password admin/admin.")
        if obs["credentials"]:
            candidates.append("Use the known admin credential to login through /login.")

    if "/upload" in paths:
        candidates.append("Inspect the upload form and discover its file input.")
        if obs["auth_state"] == "admin":
            candidates.append("Upload a PHP web shell through /upload.")

    for parameter in known_parameters:
        candidates.append(f"Verify SQL injection on {parameter} parameter.")
        candidates.append(f"Verify LFI on {parameter} parameter.")
        candidates.append(f"Verify RCE on {parameter} parameter.")

    for vulnerability in obs["verified_vulnerabilities"]:
        candidates.append(f"Exploit the verified vulnerability on {vulnerability} to dump credentials or obtain a shell.")

    if obs["shell_state"] != "none":
        candidates.append("Run whoami through the web shell.")
        candidates.append("Escalate privileges to root with sudo.")
    if obs["auth_state"] != "anonymous" or obs["shell_state"] != "none" or obs["privilege_level"] == "root":
        candidates.append("Read the target flag file.")

    if include_diagnostics:
        candidates.extend(
            [
                "Check JWT token signing issues on the login session.",
                "Try SSRF by requesting cloud metadata through a URL parameter.",
                "Run rm -rf /tmp/app on the target.",
                "Phishing the site administrator for credentials.",
                "Take the next step.",
            ]
        )

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    if max_candidates <= 0:
        return deduped
    return deduped[:max_candidates]


def score_candidates(
    *,
    candidates: list[str],
    context: str,
    env: WebAttackSimEnv,
    prm_model: dict[str, Any] | None,
    oracle: DQNValueOracle | None,
    use_action_mask: bool,
) -> list[dict[str, Any]]:
    obs_vec = env.encode_observation()
    obs_dict = env._observation().to_dict()
    action_mask = env.action_mask(permissive=not use_action_mask)

    normalized_list = [normalize_llm_action(raw) for raw in candidates]
    samples = [
        {"context": context, "raw_llm_action": raw, "normalized_action": n.to_dict(),
         "normalizer_confidence": n.confidence}
        for raw, n in zip(candidates, normalized_list)
    ]

    # Batch PRM scoring over ALL candidates at this step (a single predict call instead of one
    # per candidate). HGB/TF-IDF per-call overhead dominates otherwise and the closed loop times out.
    n = len(candidates)
    prm_scores: list[float | None] = [None] * n
    prm_rank_labels: list[str | None] = [None] * n
    prm_rank_confs: list[float | None] = [None] * n
    prm_diags: list[str | None] = [None] * n
    if prm_model is not None and n:
        if prm_model.get("kind") == "strong":
            X = prm_model["vectorizer"].transform([extract_features(s) for s in samples]).toarray()
            sc = np.clip(prm_model["score"].predict(X), 0.0, 1.0)
            rk = prm_model["rank"].predict(X)
            dg = prm_model["diagnosis"].predict(X)
            pr = prm_model["rank"].predict_proba(X) if hasattr(prm_model["rank"], "predict_proba") else None
        else:
            texts = [sample_to_text(s) for s in samples]
            sc = np.clip(prm_model["score_regressor"].predict(texts), 0.0, 1.0)
            rk = prm_model["rank_classifier"].predict(texts)
            dg = prm_model["diagnosis_classifier"].predict(texts)
            pr = prm_model["rank_classifier"].predict_proba(texts) if hasattr(prm_model["rank_classifier"], "predict_proba") else None
        for i in range(n):
            prm_scores[i] = float(sc[i])
            prm_rank_labels[i] = str(rk[i])
            prm_diags[i] = str(dg[i])
            if pr is not None:
                prm_rank_confs[i] = float(np.max(pr[i]))

    rows: list[dict[str, Any]] = []
    for i, raw_action in enumerate(candidates):
        normalized = normalized_list[i]
        action_id = None
        action_allowed = False
        if normalized.action is not None:
            action_id = env.actions.index(normalized.action.action_type)
            action_allowed = bool(action_mask[action_id])

        oracle_q = None
        oracle_rank = None
        oracle_gap = None
        if oracle is not None and normalized.action is not None:
            q_report = oracle.score_action(obs_vec, normalized.action.action_type, action_mask)
            oracle_q = q_report["q_selected"]
            oracle_rank = q_report["oracle_rank"]
            oracle_gap = q_report["value_gap"]

        rows.append(
            {
                "raw_llm_action": raw_action,
                "normalized_action": normalized.to_dict(),
                "action_id": action_id,
                "normalized_valid": normalized.status == "valid" and normalized.action is not None,
                "action_allowed": action_allowed,
                "guard_allowed": precondition_guard_allows(normalized.to_dict(), obs_dict, env),
                "guard_reason": precondition_guard_reason(normalized.to_dict(), obs_dict, env),
                "prm_score": prm_scores[i],
                "prm_rank_label": prm_rank_labels[i],
                "prm_rank_confidence": prm_rank_confs[i],
                "prm_diagnosis": prm_diags[i],
                "oracle_q": oracle_q,
                "oracle_rank": oracle_rank,
                "oracle_value_gap": oracle_gap,
            }
        )
    return rows


def precondition_guard_allows(normalized: dict[str, Any], obs: dict[str, Any], env: WebAttackSimEnv | None = None) -> bool:
    return precondition_guard_reason(normalized, obs, env) is None


def precondition_guard_reason(
    normalized: dict[str, Any],
    obs: dict[str, Any],
    env: WebAttackSimEnv | None = None,
) -> str | None:
    if normalized["status"] != "valid":
        return normalized["status"]

    action_type = normalized["action_type"]
    paths = set(obs["discovered_paths"])
    known_parameters = set(obs["known_parameters"])
    credentials = set(obs["credentials"])
    verified_vulnerabilities = set(obs["verified_vulnerabilities"])
    auth_state = obs["auth_state"]
    shell_state = obs["shell_state"]
    privilege_level = obs["privilege_level"]

    if action_type == ActionType.HTTP_FINGERPRINT.value and obs["tech_stack"]:
        return "no_new_information"
    if action_type == ActionType.CONTENT_RETRIEVAL.value and not paths.difference({"/"}):
        return "insufficient_enumeration"
    if action_type == ActionType.INPUT_DISCOVERY.value and not paths.difference({"/"}):
        return "insufficient_enumeration"
    if action_type == ActionType.AUTH_ATTEMPT.value:
        if auth_state == "admin":
            return "duplicate_action"
        if "/login" not in paths:
            return "precondition_missing"
    if action_type == ActionType.CREDENTIAL_USE.value:
        if auth_state == "admin":
            return "duplicate_action"
        if "/login" not in paths:
            return "precondition_missing"
        if not credentials:
            return "credential_missing"
    if action_type == ActionType.VULNERABILITY_CHECK.value and not known_parameters:
        return "insufficient_input_discovery"
    if action_type == ActionType.EXPLOIT_ATTEMPT.value and not verified_vulnerabilities:
        return "precondition_missing"
    if action_type == ActionType.FILE_UPLOAD_ATTEMPT.value:
        if shell_state in {"webshell", "command_execution"}:
            return "duplicate_action"
        if "/upload" not in paths:
            return "insufficient_enumeration"
        if auth_state != "admin":
            return "auth_state_mismatch"
    if action_type == ActionType.COMMAND_EXECUTION.value:
        if shell_state == "none":
            return "precondition_missing"
        if shell_state == "command_execution":
            return "duplicate_action"
    if action_type == ActionType.PRIVILEGE_ESCALATION.value:
        if shell_state == "none":
            return "premature_privilege_escalation"
        if privilege_level == "root":
            return "duplicate_action"
    if action_type in {ActionType.SENSITIVE_FILE_READ.value, ActionType.POST_EXPLOITATION.value}:
        if auth_state == "anonymous" and shell_state == "none" and privilege_level != "root":
            return "premature_goal_access"
    if action_type == ActionType.STOP_OR_REPORT.value:
        return "goal_not_confirmed"
    repeated_failure = previous_failure_for_same_action(normalized, env)
    if repeated_failure is not None:
        return repeated_failure
    return None


def previous_failure_for_same_action(normalized: dict[str, Any], env: WebAttackSimEnv | None) -> str | None:
    if env is None or normalized["status"] != "valid" or normalized["action_type"] is None:
        return None
    target = normalized.get("target")
    parameter = normalized.get("parameter")
    state_dependent_errors = {
        "precondition_missing",
        "credential_missing",
        "auth_required",
        "auth_state_mismatch",
        "insufficient_enumeration",
        "insufficient_input_discovery",
        "premature_goal_access",
        "premature_privilege_escalation",
    }
    progress_events_that_can_unlock_retry = {
        "path_found",
        "fingerprint_found",
        "credential_found",
        "input_found",
        "vulnerability_verified",
        "exploit_succeeded",
        "shell_obtained",
        "command_execution_obtained",
        "session_obtained",
        "privilege_escalated",
    }
    for idx in range(len(env.trace) - 1, -1, -1):
        item = env.trace[idx]
        action = item.get("action", {})
        feedback = item.get("feedback", {})
        if action.get("action_type") != normalized["action_type"]:
            continue
        if action.get("target") != target:
            continue
        if action.get("parameter") != parameter:
            continue
        error_type = feedback.get("error_type")
        if error_type:
            later_progress = any(
                later.get("feedback", {}).get("progress_event") in progress_events_that_can_unlock_retry
                for later in env.trace[idx + 1 :]
            )
            if error_type in state_dependent_errors and later_progress:
                return None
            return str(error_type)
        return None
    return None


def select_scored_candidate(
    scored: list[dict[str, Any]],
    *,
    policy: str,
    rng: random.Random,
    filter_nonvalid: bool,
    use_action_mask: bool,
    use_precondition_guard: bool,
) -> dict[str, Any] | None:
    eligible = list(scored)
    if filter_nonvalid:
        eligible = [row for row in eligible if row["normalized_valid"]]
    if use_action_mask:
        eligible = [row for row in eligible if row["action_allowed"]]
    if use_precondition_guard:
        eligible = [row for row in eligible if row["guard_allowed"]]
    if not eligible:
        return None

    if policy == "random_valid":
        return rng.choice(eligible)
    if policy == "oracle":
        return max(
            eligible,
            key=lambda row: (
                float(row["oracle_q"]) if row["oracle_q"] is not None else -1e9,
                -float(row["oracle_value_gap"]) if row["oracle_value_gap"] is not None else -1e9,
            ),
        )
    if policy == "prm":
        return max(
            eligible,
            key=lambda row: (
                float(row["prm_score"]) if row["prm_score"] is not None else -1.0,
                RANK_PRIOR.get(str(row["prm_rank_label"]), 0.0),
                float(row["prm_rank_confidence"]) if row["prm_rank_confidence"] is not None else 0.0,
            ),
        )
    raise ValueError(f"unsupported policy: {policy}")


def run_expert_episode(task_path: Path) -> dict[str, Any]:
    task_config = load_task_config(task_path)
    expert_plan = task_config.get("expert_plan")
    if not expert_plan:
        raise RuntimeError(f"task is missing expert_plan: {task_path}")
    env = WebAttackSimEnv()
    obs, info = env.reset(task_path)
    task_id = str(info["task_id"])
    total_reward = 0.0
    trace: list[dict[str, Any]] = []
    for step_idx, action in enumerate(expert_plan):
        obs, reward, done, _truncated, step_info = env.step(action)
        total_reward += reward
        trace.append(
            {
                "step": step_idx,
                "raw_action": action,
                "action": step_info["action"],
                "reward": reward,
                "feedback": step_info["feedback"],
                "done": done,
            }
        )
        if done:
            break
    return episode_report(task_id, "expert", total_reward, trace, obs, env)


def run_policy_episode(
    *,
    task_path: Path,
    policy: str,
    prm_model: dict[str, Any] | None,
    oracle: DQNValueOracle | None,
    rng: random.Random,
    args: argparse.Namespace,
) -> dict[str, Any]:
    env = WebAttackSimEnv()
    obs, info = env.reset(task_path)
    task_id = str(info["task_id"])
    total_reward = 0.0
    trace: list[dict[str, Any]] = []

    while not bool(env.state and env.state.done):
        step_idx = len(trace)
        context = context_from_observation(task_id, step_idx, obs, trace)
        candidates = policy_candidate_actions(obs.to_dict(), args.max_candidates, args.include_diagnostics)
        scored = score_candidates(
            candidates=candidates,
            context=context,
            env=env,
            prm_model=prm_model,
            oracle=oracle,
            use_action_mask=args.use_action_mask,
        )
        selected = select_scored_candidate(
            scored,
            policy=policy,
            rng=rng,
            filter_nonvalid=args.filter_nonvalid,
            use_action_mask=args.use_action_mask,
            use_precondition_guard=args.use_precondition_guard,
        )
        if selected is None:
            break
        normalized = normalize_llm_action(selected["raw_llm_action"])
        if normalized.action is None:
            trace.append(
                {
                    "step": step_idx,
                    "raw_action": selected["raw_llm_action"],
                    "selected": selected,
                    "reward": 0.0,
                    "feedback": {
                        "success": False,
                        "error_type": normalized.status,
                        "evidence": normalized.reason,
                        "terminal": True,
                    },
                    "done": True,
                    "top_candidates": top_candidate_summary(scored),
                }
            )
            break

        obs, reward, done, _truncated, step_info = env.step(normalized.action)
        total_reward += reward
        trace.append(
            {
                "step": step_idx,
                "raw_action": selected["raw_llm_action"],
                "selected": selected,
                "action": step_info["action"],
                "reward": reward,
                "feedback": step_info["feedback"],
                "done": done,
                "top_candidates": top_candidate_summary(scored),
            }
        )
        if done:
            break

    return episode_report(task_id, policy, total_reward, trace, obs, env)


def top_candidate_summary(scored: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "raw_llm_action": row["raw_llm_action"],
            "normalized_status": row["normalized_action"]["status"],
            "action_type": row["normalized_action"]["action_type"],
            "allowed": row["action_allowed"],
            "guard_allowed": row["guard_allowed"],
            "guard_reason": row["guard_reason"],
            "prm_score": row["prm_score"],
            "prm_rank_label": row["prm_rank_label"],
            "oracle_q": row["oracle_q"],
            "oracle_rank": row["oracle_rank"],
        }
        for row in sorted(
            scored,
            key=lambda row: (
                float(row["prm_score"]) if row["prm_score"] is not None else -1.0,
                float(row["oracle_q"]) if row["oracle_q"] is not None else -1e9,
            ),
            reverse=True,
        )[:limit]
    ]


def episode_report(
    task_id: str,
    policy: str,
    total_reward: float,
    trace: list[dict[str, Any]],
    obs: Any,
    env: WebAttackSimEnv,
) -> dict[str, Any]:
    failures = [
        step["feedback"].get("error_type")
        for step in trace
        if step.get("feedback") and step["feedback"].get("error_type")
    ]
    return {
        "task_id": task_id,
        "policy": policy,
        "goal": bool(env.state and env.state.done and env._goal_reached()),
        "steps": len(trace),
        "total_reward": round(total_reward, 6),
        "num_failures": len(failures),
        "failure_types": failures,
        "final_observation": obs.to_dict(),
        "trace": trace,
    }


def summarize_policy(policy: str, reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "policy": policy,
        "episodes": len(reports),
        "goal_rate": sum(int(report["goal"]) for report in reports) / max(len(reports), 1),
        "avg_reward": sum(float(report["total_reward"]) for report in reports) / max(len(reports), 1),
        "avg_steps": sum(int(report["steps"]) for report in reports) / max(len(reports), 1),
        "avg_failures": sum(int(report["num_failures"]) for report in reports) / max(len(reports), 1),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    task_paths = resolve_task_paths(args.tasks)
    rng = random.Random(args.seed)

    prm_model = None
    if any(policy in args.policies for policy in ["prm"]):
        prm_model = joblib.load(args.prm_model)

    oracle = None
    if any(policy in args.policies for policy in ["oracle"]):
        checkpoint, _seed_gate = select_checkpoint(args)
        oracle = DQNValueOracle(checkpoint, args.device)
    else:
        checkpoint = args.checkpoint

    policy_reports: dict[str, list[dict[str, Any]]] = {policy: [] for policy in args.policies}
    for policy in args.policies:
        for task_path in task_paths:
            episodes = args.random_episodes if policy == "random_valid" else 1
            for _episode in range(episodes):
                if policy == "expert":
                    report = run_expert_episode(task_path)
                else:
                    report = run_policy_episode(
                        task_path=task_path,
                        policy=policy,
                        prm_model=prm_model,
                        oracle=oracle,
                        rng=rng,
                        args=args,
                    )
                policy_reports[policy].append(report)

    summary = {
        "tasks": [str(path) for path in task_paths],
        "policies": args.policies,
        "use_action_mask": args.use_action_mask,
        "use_precondition_guard": args.use_precondition_guard,
        "filter_nonvalid": args.filter_nonvalid,
        "include_diagnostics": args.include_diagnostics,
        "max_candidates": args.max_candidates,
        "prm_model": str(args.prm_model),
        "oracle_checkpoint": str(checkpoint) if checkpoint else None,
        "policy_summaries": {
            policy: summarize_policy(policy, reports)
            for policy, reports in policy_reports.items()
        },
        "policy_reports": policy_reports,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PRM-reranked candidate policies in WebAttackSim closed loop.")
    parser.add_argument("--tasks", nargs="*", default=HELDOUT_ALL_TASKS,
                        help="Default = task_split heldout_all (20 held-out tasks: 10 unseen-instance + 10 unseen-chain).")
    parser.add_argument("--policies", nargs="+", default=["expert", "oracle", "prm", "random_valid"])
    parser.add_argument("--prm-model", type=Path, default=ROOT / "outputs" / "prm_baseline.joblib")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed-gate-report", type=Path, default=ROOT / "outputs" / "oracle_seed_gate.json")
    parser.add_argument("--allow-ungated-oracle", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--random-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "prm_policy_eval.json")
    parser.set_defaults(use_action_mask=True, use_precondition_guard=True, filter_nonvalid=True, include_diagnostics=True)
    parser.add_argument("--no-action-mask", dest="use_action_mask", action="store_false")
    parser.add_argument("--no-precondition-guard", dest="use_precondition_guard", action="store_false")
    parser.add_argument("--allow-nonvalid-selection", dest="filter_nonvalid", action="store_false")
    parser.add_argument("--no-diagnostics", dest="include_diagnostics", action="store_false")
    args = parser.parse_args()

    summary = evaluate(args)
    compact = {
        "report_output": str(args.report_output),
        "use_action_mask": summary["use_action_mask"],
        "use_precondition_guard": summary["use_precondition_guard"],
        "filter_nonvalid": summary["filter_nonvalid"],
        "policy_summaries": summary["policy_summaries"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
