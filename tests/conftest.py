"""Shared pytest fixtures and path setup for the WebAttackSim test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402


# A self-contained leak -> credential -> login -> read chain (medium).
SAMPLE_TASK = {
    "task_id": "test_leak_login",
    "target": "http://target:80",
    "service_surface": ["http:80"],
    "initial_observation": {"open_services": ["http:80"], "discovered_paths": ["/"]},
    "hidden_paths": ["/backup", "/login", "/admin"],
    "technologies": ["php", "nginx"],
    "forms": {"/login": ["username", "password"]},
    "parameters": {},
    "leaks": {"/backup": {"credentials": ["admin"], "files": ["cfg"]}},
    "credentials": {"admin": {"username": "admin", "password": "s3cr3t_pw", "role": "admin", "weak": False}},
    "auth": {"login_path": "/login", "valid_credential": "admin", "role": "admin"},
    "vulnerabilities": {},
    "files": {"admin_flag": {"path": "/admin/flag", "requires_auth": "admin", "flag": "FLAG{unit_test}"}},
    "goal": {"type": "read_file", "file": "admin_flag"},
    "budget": {"max_steps": 20},
    "expert_plan": [
        {"action_type": "web_path_enumeration", "target": "/backup"},
        {"action_type": "content_retrieval", "target": "/backup"},
        {"action_type": "web_path_enumeration", "target": "/login"},
        "input_discovery",
        "credential_use",
        {"action_type": "web_path_enumeration", "target": "/admin"},
        "sensitive_file_read",
    ],
    "expert_trajectory": [
        "Enumerate web directories to discover /backup.",
        "Download and inspect the /backup content for leaked credentials.",
        "Enumerate web directories to discover /login.",
        "Inspect the login form and discover its input fields.",
        "Use the leaked credential to login through /login.",
        "Enumerate web directories to discover /admin.",
        "Read the /admin/flag sensitive file.",
    ],
}

# RCE -> shell -> privilege escalation -> root (for the privesc duplicate-fix test).
PRIVESC_TASK = {
    "task_id": "test_rce_privesc",
    "target": "http://target:80",
    "service_surface": ["http:80"],
    "initial_observation": {"open_services": ["http:80"], "discovered_paths": ["/"]},
    "hidden_paths": ["/tools"],
    "technologies": ["nodejs"],
    "forms": {},
    "parameters": {"/tools": [{"name": "cmd", "vulnerability": "rce1"}]},
    "leaks": {},
    "credentials": {},
    "auth": {},
    "vulnerabilities": {"rce1": {"type": "rce", "target": "/tools?cmd", "requires": ["parameter_found:/tools?cmd"], "effects": {"shell": True}}},
    "privilege_escalation": {"available": True, "target_privilege": "root"},
    "files": {"root_flag": {"path": "/root/root.txt", "requires_shell": True, "requires_privilege": "root", "flag": "FLAG{root}"}},
    "goal": {"type": "read_file", "file": "root_flag"},
    "budget": {"max_steps": 20},
    "expert_plan": [
        {"action_type": "web_path_enumeration", "target": "/tools"},
        "input_discovery",
        {"action_type": "vulnerability_check", "target": "/tools?cmd", "parameter": "cmd"},
        "exploit_attempt",
        "command_execution",
        "privilege_escalation",
        "sensitive_file_read",
    ],
}


@pytest.fixture
def sample_task() -> dict:
    return dict(SAMPLE_TASK)


@pytest.fixture
def privesc_task() -> dict:
    return dict(PRIVESC_TASK)


@pytest.fixture
def env(sample_task):
    from web_attack_sim import WebAttackSimEnv

    e = WebAttackSimEnv()
    e.reset(sample_task)
    return e
