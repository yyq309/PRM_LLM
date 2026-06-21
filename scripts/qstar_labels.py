"""Exact Q* labels for simple fully-observable templates (method §12.1).

The method says: for fully-observable simple templates, use the exact value-iteration Q*
directly as labels (it is provably optimal), narrowing the DQN's job to the part it alone
can do (partial observability + unseen-instance generalization). This both makes the easy
labels exact AND quantifies "what RL adds" by reporting where the trained DQN disagrees
with Q*.

To avoid the decoy-milking degeneracy (literal Q* milks distractor path_found rewards before
a terminal), Q* labels are computed on DISTRACTOR-FREE clean templates, where reaching the
goal leaves no un-collected reward and Q* is well-behaved. At every state along the optimal
chain we emit Q*(s,a) and value_gap*(s,a) for each allowed action, and compare the DQN
oracle's ranking/value_gap against Q* (rank-bucket agreement + value_gap MAE).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_pipeline import DQNValueOracle, rank_label_from_oracle  # noqa: E402
from verify_qstar import QStarSolver, resolve_gated_checkpoint  # noqa: E402
from web_attack_sim import WebAttackSimEnv  # noqa: E402
from web_attack_sim.action_space import ACTIONS  # noqa: E402

# Clean (distractor-free) fully-observable simple templates.
CLEAN_TEMPLATES = [
    {
        "task_id": "clean_leak_file",
        "target": "http://target:80", "service_surface": ["http:80"],
        "initial_observation": {"open_services": ["http:80"], "discovered_paths": ["/"]},
        "hidden_paths": ["/backup"], "technologies": ["php"], "forms": {}, "parameters": {},
        "credentials": {}, "auth": {}, "vulnerabilities": {},
        "leaks": {"/backup": {"files": ["flag"]}},
        "files": {"flag": {"path": "/backup/flag.txt", "flag": "F"}},
        "goal": {"type": "read_file", "file": "flag"}, "budget": {"max_steps": 8},
        "expert_plan": [
            {"action_type": "web_path_enumeration", "target": "/backup"},
            {"action_type": "content_retrieval", "target": "/backup"},
        ],
    },
    {
        "task_id": "clean_default_pw",
        "target": "http://target:80", "service_surface": ["http:80"],
        "initial_observation": {"open_services": ["http:80"], "discovered_paths": ["/"]},
        "hidden_paths": ["/login", "/admin"], "technologies": ["php"],
        "forms": {"/login": ["username", "password"]}, "parameters": {},
        "credentials": {"admin": {"username": "admin", "password": "admin", "role": "admin", "weak": True}},
        "auth": {"login_path": "/login", "valid_credential": "admin", "role": "admin"},
        "vulnerabilities": {}, "leaks": {},
        "files": {"admin_flag": {"path": "/admin/flag", "requires_auth": "admin", "flag": "F"}},
        "goal": {"type": "read_file", "file": "admin_flag"}, "budget": {"max_steps": 10},
        "expert_plan": [
            {"action_type": "web_path_enumeration", "target": "/login"},
            "input_discovery", "auth_attempt",
            {"action_type": "web_path_enumeration", "target": "/admin"},
            "sensitive_file_read",
        ],
    },
    {
        "task_id": "clean_sqli_login",
        "target": "http://target:80", "service_surface": ["http:80"],
        "initial_observation": {"open_services": ["http:80"], "discovered_paths": ["/"]},
        "hidden_paths": ["/item", "/login", "/admin"], "technologies": ["php"],
        "forms": {"/login": ["username", "password"]},
        "parameters": {"/item": [{"name": "id", "vulnerability": "sqli1"}]},
        "credentials": {"admin": {"username": "admin", "password": "dumped", "role": "admin", "weak": False}},
        "auth": {"login_path": "/login", "valid_credential": "admin", "role": "admin"},
        "vulnerabilities": {"sqli1": {"type": "sqli", "target": "/item?id", "requires": ["parameter_found:/item?id"], "effects": {"credentials": ["admin"]}}},
        "leaks": {}, "files": {"admin_flag": {"path": "/admin/flag", "requires_auth": "admin", "flag": "F"}},
        "goal": {"type": "read_file", "file": "admin_flag"}, "budget": {"max_steps": 12},
        "expert_plan": [
            {"action_type": "web_path_enumeration", "target": "/item"},
            {"action_type": "input_discovery", "target": "/item"},
            {"action_type": "vulnerability_check", "target": "/item?id", "parameter": "id"},
            {"action_type": "exploit_attempt", "target": "/item?id", "parameter": "id"},
            {"action_type": "web_path_enumeration", "target": "/login"},
            {"action_type": "web_path_enumeration", "target": "/admin"},
            {"action_type": "credential_use", "target": "/login"},
            {"action_type": "sensitive_file_read", "target": "/admin/flag"},
        ],
    },
]


def plan_to_int(raw) -> int:
    name = raw if isinstance(raw, str) else raw["action_type"]
    return [a.value for a in ACTIONS].index(name)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = resolve_gated_checkpoint(argparse.Namespace(checkpoint=args.checkpoint))
    oracle = DQNValueOracle(checkpoint, args.device)

    labels: list[dict[str, Any]] = []
    rank_agree = 0
    rank_total = 0
    gap_abs_err: list[float] = []
    task_reports = []

    for task in CLEAN_TEMPLATES:
        solver = QStarSolver(task, gamma=args.gamma)
        solver.solve(max_states=200000)
        env = WebAttackSimEnv()
        obs, _info = env.reset(task)
        per_task_agree = per_task_total = 0

        for step_idx, raw in enumerate(task["expert_plan"]):
            from verify_qstar import state_key  # local import to reuse the canonical key

            key = state_key(env.state)
            qstar = solver.q_cache.get(key)
            if qstar is None:
                solver._compute(env.state, max_states=200000)
                qstar = solver.q_cache[key]
            mask = env.action_mask(permissive=False)
            allowed = [i for i, m in enumerate(mask) if m]
            if not allowed:
                env.step(plan_to_int(raw))
                continue

            vstar = max(qstar[i] for i in allowed)
            dqn_q = oracle.q_values(env.encode_observation(obs))
            v_dqn = max(dqn_q[i] for i in allowed)
            order_star = sorted(allowed, key=lambda i: qstar[i], reverse=True)
            order_dqn = sorted(allowed, key=lambda i: dqn_q[i], reverse=True)

            for a in allowed:
                gap_star = vstar - qstar[a]
                gap_dqn = v_dqn - dqn_q[a]
                rank_star = rank_label_from_oracle(order_star.index(a) + 1, len(allowed))
                rank_dqn = rank_label_from_oracle(order_dqn.index(a) + 1, len(allowed))
                rank_total += 1
                per_task_total += 1
                if rank_star == rank_dqn:
                    rank_agree += 1
                    per_task_agree += 1
                gap_abs_err.append(abs(gap_star - gap_dqn))
                labels.append({
                    "task_id": task["task_id"], "step": step_idx,
                    "action_type": ACTIONS[a].value,
                    "qstar_value_gap": round(gap_star, 4),
                    "qstar_rank_label": rank_star,
                    "dqn_value_gap": round(gap_dqn, 4),
                    "dqn_rank_label": rank_dqn,
                    "label_source": "qstar_exact",
                })
            env.step(plan_to_int(raw))

        task_reports.append({
            "task_id": task["task_id"],
            "dqn_qstar_rank_agreement": round(per_task_agree / max(per_task_total, 1), 4),
            "states_scored": per_task_total,
        })

    report = {
        "checkpoint": str(checkpoint),
        "num_clean_templates": len(CLEAN_TEMPLATES),
        "num_qstar_labels": len(labels),
        "dqn_qstar_rank_agreement": round(rank_agree / max(rank_total, 1), 4),
        "dqn_qstar_value_gap_MAE": round(float(np.mean(gap_abs_err)), 4) if gap_abs_err else None,
        "per_task": task_reports,
        "interpretation": (
            "Q* is the exact optimal label on these clean fully-observable templates. The rank "
            "agreement and value_gap MAE quantify how far the trained DQN's labels deviate from "
            "exact optimal on simple tasks; for these templates the Q* labels can REPLACE the DQN "
            "labels (provably correct), narrowing the DQN's role to partial-observability and "
            "unseen-instance generalization on the harder tasks."
        ),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    with args.labels_output.open("w", encoding="utf-8") as f:
        for row in labels:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact Q* labels for simple templates (method §12.1).")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "qstar_labels_report.json")
    parser.add_argument("--labels-output", type=Path, default=ROOT / "outputs" / "qstar_labels.jsonl")
    args = parser.parse_args()

    report = evaluate(args)
    print(json.dumps({k: v for k, v in report.items() if k != "interpretation"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
