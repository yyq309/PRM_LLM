"""De-templated programmatic WebAttackSim task generator (method §12.1 / §13.1).

Earlier the generated tasks were re-skins of a few hand-authored templates: held-out
instances shared byte-identical expert_plan signatures with training, so "unseen"
generalization was overstated (an adversarial review confirmed this). This version:

- Diversifies tokens per instance (paths, parameters, credentials, tech, flags, budget)
  from large pools via a per-instance RNG, so no two instances are byte-identical.
- Tags each task with a `family` (the chain TOPOLOGY = expert_plan action-type sequence)
  and `difficulty`, so `task_split.py` can hold out WHOLE families (genuine unseen-chain)
  in addition to unseen instances of trained families.
- Interleaves a budget-consuming distractor BEFORE the first productive path, so a
  no-target enumerator wastes budget on a decoy (the expert uses targeted enumeration and
  avoids it) — making distractors actually matter rather than trivially avoided.

Every task is self-verified solvable by both expert_plan and expert_trajectory before
being written. The abstract 16-action schema is frozen.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web_attack_sim import WebAttackSimEnv, normalize_llm_action  # noqa: E402
from web_attack_sim.action_space import ActionType  # noqa: E402

TASKS_DIR = ROOT / "tasks"

LOGIN_PATHS = ["/login", "/signin", "/auth", "/account/login", "/user/login", "/portal", "/admin-login", "/session"]
ADMIN_PATHS = ["/admin", "/dashboard", "/manage", "/console", "/panel", "/backend", "/cp", "/administrator"]
UPLOAD_PATHS = ["/upload", "/files/upload", "/media/upload", "/import", "/attach", "/uploads/new"]
LEAK_PATHS = ["/backup", "/config", "/.env", "/db.bak", "/.git", "/export", "/dump.sql", "/site.bak", "/.svn", "/conf.old"]
PARAM_ENDPOINTS = ["/search", "/product", "/view", "/item", "/download", "/api/v1/data", "/report", "/page", "/news", "/doc"]
PARAM_NAMES = ["q", "id", "pid", "file", "page", "tpl", "name", "uid", "doc", "cat"]
CRED_IDS = ["admin", "root", "webadmin", "svc", "operator", "dbadmin", "manager", "sysadmin"]
DISTRACTOR_POOL = ["/images", "/css", "/static", "/assets", "/js", "/about", "/contact", "/robots.txt", "/blog", "/help", "/news-archive", "/favicon.ico", "/docs", "/terms", "/legal"]
TECH_POOL = [
    ["php", "apache", "mysql"], ["python", "flask", "sqlite"], ["php", "nginx", "mysql"],
    ["nodejs", "express", "mongodb"], ["php", "nginx", "linux"], ["java", "tomcat", "mysql"],
    ["ruby", "rails", "postgres"], ["go", "caddy", "redis"], ["dotnet", "iis", "mssql"],
]


def pick(rng: random.Random, pool: list, used: set) -> Any:
    choices = [c for c in pool if c not in used]
    value = rng.choice(choices if choices else pool)
    used.add(value)
    return value


def base(task_id: str, hidden: list[str], tech: list[str], family: str, difficulty: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "family": family,
        "difficulty": difficulty,
        "target": "http://target:80",
        "service_surface": ["http:80"],
        "initial_observation": {"open_services": ["http:80"], "discovered_paths": ["/"]},
        "hidden_paths": hidden,
        "technologies": tech,
        "forms": {},
        "parameters": {},
        "leaks": {},
        "credentials": {},
        "auth": {},
        "vulnerabilities": {},
        "files": {},
    }


def interleave_distractors(rng: random.Random, productive: list[str], n_before: int, n_after: int) -> tuple[list[str], int]:
    used = set(productive)
    decoys_before = [pick(rng, DISTRACTOR_POOL, used) for _ in range(n_before)]
    decoys_after = [pick(rng, DISTRACTOR_POOL, used) for _ in range(n_after)]
    # A budget-consuming decoy sits BEFORE the first productive path.
    hidden = decoys_before + productive + decoys_after
    return hidden, n_before + n_after


# Hard mode (env improvement #1): tight budget so wasted/wrong steps actually risk the goal,
# making same-state decisions CONSEQUENTIAL. None = loose default (backward compatible). A probe
# (decision_consequence_eval.py) showed slack=2 ~doubles the decision-relevant fraction (0.23->0.42)
# while keeping the scripted expert 100% solvable.
_HARD_SLACK: int | None = None


def budget(steps: int, n_decoys: int) -> dict[str, int]:
    if _HARD_SLACK is not None:
        return {"max_steps": steps + _HARD_SLACK}
    return {"max_steps": steps + 3 * n_decoys + 12}


# ---------------------------------------------------------------------------
# Family builders (each returns a de-templated instance with a topology family)
# ---------------------------------------------------------------------------

def f_leak_file(rng: random.Random, idx: int) -> dict[str, Any]:
    used: set = set()
    leak = pick(rng, LEAK_PATHS, used)
    flag_id = f"flag_{idx}"
    hidden, nd = interleave_distractors(rng, [leak], 1, 1)
    t = base(f"gen_{idx:03d}_leak_file", hidden, rng.choice(TECH_POOL), "leak_file", "easy")
    t["leaks"] = {leak: {"files": [flag_id]}}
    t["files"] = {flag_id: {"path": f"{leak}/{flag_id}.txt", "flag": f"FLAG{{gen_{idx}_leakfile}}"}}
    t["goal"] = {"type": "read_file", "file": flag_id}
    t["budget"] = budget(2, nd)
    t["expert_plan"] = [
        {"action_type": "web_path_enumeration", "target": leak},
        {"action_type": "content_retrieval", "target": leak},
    ]
    t["expert_trajectory"] = [
        f"Enumerate web directories to discover {leak}.",
        f"Download and inspect the {leak} content for leaked data.",
    ]
    return t


def f_default_pw(rng: random.Random, idx: int) -> dict[str, Any]:
    used: set = set()
    login = pick(rng, LOGIN_PATHS, used)
    admin = pick(rng, ADMIN_PATHS, used)
    pw = rng.choice(["admin", "password", "admin123", "root", "letmein", "changeme"])
    hidden, nd = interleave_distractors(rng, [login, admin], 1, 1)
    t = base(f"gen_{idx:03d}_default_pw", hidden, rng.choice(TECH_POOL), "default_pw", "easy")
    t["forms"] = {login: ["username", "password"]}
    t["credentials"] = {"admin": {"username": "admin", "password": pw, "role": "admin", "weak": True}}
    t["auth"] = {"login_path": login, "valid_credential": "admin", "role": "admin"}
    t["files"] = {"admin_flag": {"path": f"{admin}/flag", "requires_auth": "admin", "flag": f"FLAG{{gen_{idx}_defpw}}"}}
    t["goal"] = {"type": "read_file", "file": "admin_flag"}
    t["budget"] = budget(5, nd)
    t["expert_plan"] = [
        {"action_type": "web_path_enumeration", "target": login},
        {"action_type": "input_discovery", "target": login},
        "auth_attempt",
        {"action_type": "web_path_enumeration", "target": admin},
        "sensitive_file_read",
    ]
    t["expert_trajectory"] = [
        f"Enumerate web directories to discover {login}.",
        "Inspect the login form and discover its input fields.",
        f"Login with default password admin/{pw}.",
        f"Enumerate web directories to discover {admin}.",
        "Read the admin flag sensitive file.",
    ]
    return t


def f_leak_login(rng: random.Random, idx: int) -> dict[str, Any]:
    used: set = set()
    leak = pick(rng, LEAK_PATHS, used)
    login = pick(rng, LOGIN_PATHS, used)
    admin = pick(rng, ADMIN_PATHS, used)
    cred = pick(rng, CRED_IDS, used)
    hidden, nd = interleave_distractors(rng, [leak, login, admin], 1, 1)
    t = base(f"gen_{idx:03d}_leak_login", hidden, rng.choice(TECH_POOL), "leak_login", "medium")
    t["forms"] = {login: ["username", "password"]}
    t["leaks"] = {leak: {"credentials": [cred], "files": [f"{cred}_cfg"]}}
    t["credentials"] = {cred: {"username": cred, "password": f"{cred}_{idx}_pw", "role": "admin", "weak": False}}
    t["auth"] = {"login_path": login, "valid_credential": cred, "role": "admin"}
    t["files"] = {"admin_flag": {"path": f"{admin}/flag", "requires_auth": "admin", "flag": f"FLAG{{gen_{idx}_leaklogin}}"}}
    t["goal"] = {"type": "read_file", "file": "admin_flag"}
    t["budget"] = budget(7, nd)
    t["expert_plan"] = [
        {"action_type": "web_path_enumeration", "target": leak},
        {"action_type": "content_retrieval", "target": leak},
        {"action_type": "web_path_enumeration", "target": login},
        "input_discovery",
        "credential_use",
        {"action_type": "web_path_enumeration", "target": admin},
        "sensitive_file_read",
    ]
    t["expert_trajectory"] = [
        f"Enumerate web directories to discover {leak}.",
        f"Download and inspect the {leak} content for leaked credentials.",
        f"Enumerate web directories to discover {login}.",
        "Inspect the login form and discover its input fields.",
        f"Use the leaked credential to login through {login}.",
        f"Enumerate web directories to discover {admin}.",
        "Read the admin flag sensitive file.",
    ]
    return t


def f_injection_login(rng: random.Random, idx: int, vuln_kind: str) -> dict[str, Any]:
    used: set = set()
    endpoint = pick(rng, PARAM_ENDPOINTS, used)
    param = pick(rng, PARAM_NAMES, used)
    login = pick(rng, LOGIN_PATHS, used)
    admin = pick(rng, ADMIN_PATHS, used)
    vuln_id = f"{vuln_kind}_{param}_{idx}"
    hidden, nd = interleave_distractors(rng, [endpoint, login, admin], 1, 1)
    t = base(f"gen_{idx:03d}_{vuln_kind}_login", hidden, rng.choice(TECH_POOL), "injection_login", "medium")
    t["forms"] = {login: ["username", "password"]}
    t["parameters"] = {endpoint: [{"name": param, "vulnerability": vuln_id}]}
    t["credentials"] = {"admin": {"username": "admin", "password": f"dump_{idx}_pw", "role": "admin", "weak": False}}
    t["auth"] = {"login_path": login, "valid_credential": "admin", "role": "admin"}
    t["vulnerabilities"] = {vuln_id: {"type": vuln_kind, "target": f"{endpoint}?{param}", "requires": [f"parameter_found:{endpoint}?{param}"], "effects": {"credentials": ["admin"]}}}
    t["files"] = {"admin_flag": {"path": f"{admin}/flag", "requires_auth": "admin", "flag": f"FLAG{{gen_{idx}_{vuln_kind}}}"}}
    t["goal"] = {"type": "read_file", "file": "admin_flag"}
    t["budget"] = budget(8, nd)
    verify = "Verify SQL injection on" if vuln_kind == "sqli" else "Verify LFI on"
    exploit = "Exploit the SQL injection to dump the admin credential." if vuln_kind == "sqli" else "Exploit the LFI to dump the admin credential."
    t["expert_plan"] = [
        {"action_type": "web_path_enumeration", "target": endpoint},
        {"action_type": "input_discovery", "target": endpoint},
        {"action_type": "vulnerability_check", "target": f"{endpoint}?{param}", "parameter": param},
        {"action_type": "exploit_attempt", "target": f"{endpoint}?{param}", "parameter": param},
        {"action_type": "web_path_enumeration", "target": login},
        {"action_type": "web_path_enumeration", "target": admin},
        {"action_type": "credential_use", "target": login},
        {"action_type": "sensitive_file_read", "target": f"{admin}/flag"},
    ]
    t["expert_trajectory"] = [
        f"Enumerate web directories to discover {endpoint}.",
        f"Discover input parameters on the {endpoint} endpoint.",
        f"{verify} {endpoint}?{param} parameter.",
        exploit,
        f"Enumerate web directories to discover {login}.",
        f"Enumerate web directories to discover {admin}.",
        f"Use the known admin credential to login through {login}.",
        "Read the admin flag sensitive file.",
    ]
    return t


def f_rce(rng: random.Random, idx: int, privesc: bool) -> dict[str, Any]:
    used: set = set()
    endpoint = pick(rng, PARAM_ENDPOINTS, used)
    param = pick(rng, PARAM_NAMES, used)
    vuln_id = f"rce_{param}_{idx}"
    family = "rce_privesc" if privesc else "rce_shell"
    hidden, nd = interleave_distractors(rng, [endpoint], 1, 1)
    t = base(f"gen_{idx:03d}_{family}", hidden, rng.choice(TECH_POOL), family, "hard")
    t["parameters"] = {endpoint: [{"name": param, "vulnerability": vuln_id}]}
    t["vulnerabilities"] = {vuln_id: {"type": "rce", "target": f"{endpoint}?{param}", "requires": [f"parameter_found:{endpoint}?{param}"], "effects": {"shell": True}}}
    if privesc:
        t["privilege_escalation"] = {"available": True, "target_privilege": "root"}
        t["files"] = {"root_flag": {"path": "/root/root.txt", "requires_shell": True, "requires_privilege": "root", "flag": f"FLAG{{gen_{idx}_rcepriv}}"}}
        t["goal"] = {"type": "read_file", "file": "root_flag"}
        tail_plan, tail_traj, steps = ["command_execution", "privilege_escalation", "sensitive_file_read"], ["Run whoami through the web shell.", "Escalate privileges to root with a local SUID helper.", "Read the root flag file."], 7
    else:
        t["files"] = {"user_flag": {"path": "/home/web/flag.txt", "requires_shell": True, "flag": f"FLAG{{gen_{idx}_rceshell}}"}}
        t["goal"] = {"type": "read_file", "file": "user_flag"}
        tail_plan, tail_traj, steps = ["command_execution", "sensitive_file_read"], ["Run whoami through the web shell.", "Read the target flag file."], 6
    t["budget"] = budget(steps, nd)
    t["expert_plan"] = [
        {"action_type": "web_path_enumeration", "target": endpoint},
        "input_discovery",
        {"action_type": "vulnerability_check", "target": f"{endpoint}?{param}", "parameter": param},
        "exploit_attempt",
        *tail_plan,
    ]
    t["expert_trajectory"] = [
        f"Enumerate web directories to discover {endpoint}.",
        f"Discover input parameters on the {endpoint} endpoint.",
        f"Verify RCE on {endpoint}?{param} parameter.",
        "Exploit the RCE to obtain a web shell.",
        *tail_traj,
    ]
    return t


def f_upload(rng: random.Random, idx: int, via: str, privesc: bool) -> dict[str, Any]:
    used: set = set()
    login = pick(rng, LOGIN_PATHS, used)
    upload = pick(rng, UPLOAD_PATHS, used)
    weak = via == "default"
    family = f"upload_{via}_{'privesc' if privesc else 'shell'}"
    productive = [login, upload]
    leak = None
    if via == "leak":
        leak = pick(rng, LEAK_PATHS, used)
        productive = [leak] + productive
    hidden, nd = interleave_distractors(rng, productive, 1, 1)
    t = base(f"gen_{idx:03d}_{family}", hidden, rng.choice(TECH_POOL), family, "hard")
    t["forms"] = {login: ["username", "password"], upload: ["file"]}
    t["credentials"] = {"admin": {"username": "admin", "password": "admin" if weak else f"up_{idx}_pw", "role": "admin", "weak": weak}}
    t["auth"] = {"login_path": login, "valid_credential": "admin", "role": "admin"}
    if via == "leak":
        t["leaks"] = {leak: {"credentials": ["admin"]}}
    t["upload"] = {"path": upload, "requires_auth": "admin", "vulnerable": True, "shell_on_upload": True}
    if privesc:
        t["privilege_escalation"] = {"available": True, "target_privilege": "root"}
        t["files"] = {"root_flag": {"path": "/root/root.txt", "requires_shell": True, "requires_privilege": "root", "flag": f"FLAG{{gen_{idx}_uppriv}}"}}
        t["goal"] = {"type": "read_file", "file": "root_flag"}
    else:
        t["files"] = {"user_flag": {"path": "/home/web/flag.txt", "requires_shell": True, "flag": f"FLAG{{gen_{idx}_upshell}}"}}
        t["goal"] = {"type": "read_file", "file": "user_flag"}

    plan: list[Any] = []
    traj: list[str] = []
    if via == "leak":
        plan += [{"action_type": "web_path_enumeration", "target": leak}, {"action_type": "content_retrieval", "target": leak},
                 {"action_type": "web_path_enumeration", "target": login}, "input_discovery", "credential_use"]
        traj += [f"Enumerate web directories to discover {leak}.", f"Download and inspect the {leak} content for leaked credentials.",
                 f"Enumerate web directories to discover {login}.", "Inspect the login form and discover its input fields.",
                 f"Use the leaked credential to login through {login}."]
    else:
        plan += [{"action_type": "web_path_enumeration", "target": login}, "input_discovery", "auth_attempt"]
        traj += [f"Enumerate web directories to discover {login}.", "Inspect the login form and discover its input fields.",
                 "Login with default password admin/admin."]
    plan += [{"action_type": "web_path_enumeration", "target": upload}, "input_discovery", "file_upload_attempt", "command_execution"]
    traj += [f"Enumerate web directories to discover {upload}.", "Inspect the upload form and discover its file input.",
             f"Upload a PHP web shell through {upload}.", "Run whoami through the uploaded web shell."]
    if privesc:
        plan += ["privilege_escalation", "sensitive_file_read"]
        traj += ["Escalate privileges to root with a local SUID helper.", "Read the root flag file."]
    else:
        plan += ["sensitive_file_read"]
        traj += ["Read the target flag file."]
    t["budget"] = budget(len(plan), nd)
    t["expert_plan"] = plan
    t["expert_trajectory"] = traj
    return t


# ---------------------------------------------------------------------------
# NEW structurally-novel families (method §13.1 coverage extension). These exploit
# the env's vuln `requires` preconditions (auth_state / vulnerability_verified) that
# none of the 9 original families use, producing expert_plan signatures that are
# genuinely new combinations of the SAME frozen 16 primitives. Each maps to a real
# VulnHub chain type and is held out as unseen-chain to test structural generalization.
# ---------------------------------------------------------------------------

def f_authed_injection(rng: random.Random, idx: int) -> dict[str, Any]:
    """Weak login -> AUTH-GATED injection -> shell -> flag (real: authenticated RCE).

    The injection vuln `requires: auth_state:admin`, so the auth step is forced ON the
    critical path before the exploit chain — distinct from rce_shell (no auth prefix).
    """
    used: set = set()
    login = pick(rng, LOGIN_PATHS, used)
    endpoint = pick(rng, PARAM_ENDPOINTS, used)
    param = pick(rng, PARAM_NAMES, used)
    pw = rng.choice(["admin", "password", "admin123", "root", "letmein", "changeme"])
    vuln_id = f"authed_rce_{param}_{idx}"
    hidden, nd = interleave_distractors(rng, [login, endpoint], 1, 1)
    t = base(f"gen_{idx:03d}_authed_injection", hidden, rng.choice(TECH_POOL), "authed_injection", "hard")
    t["forms"] = {login: ["username", "password"]}
    t["credentials"] = {"admin": {"username": "admin", "password": pw, "role": "admin", "weak": True}}
    t["auth"] = {"login_path": login, "valid_credential": "admin", "role": "admin"}
    t["parameters"] = {endpoint: [{"name": param, "vulnerability": vuln_id}]}
    t["vulnerabilities"] = {vuln_id: {
        "type": "rce", "target": f"{endpoint}?{param}",
        "requires": [f"parameter_found:{endpoint}?{param}", "auth_state:admin"],
        "effects": {"shell": True},
    }}
    t["files"] = {"user_flag": {"path": "/home/web/flag.txt", "requires_shell": True, "flag": f"FLAG{{gen_{idx}_authedinj}}"}}
    t["goal"] = {"type": "read_file", "file": "user_flag"}
    t["budget"] = budget(9, nd)
    t["expert_plan"] = [
        {"action_type": "web_path_enumeration", "target": login},
        {"action_type": "input_discovery", "target": login},
        "auth_attempt",
        {"action_type": "web_path_enumeration", "target": endpoint},
        {"action_type": "input_discovery", "target": endpoint},
        {"action_type": "vulnerability_check", "target": f"{endpoint}?{param}", "parameter": param},
        {"action_type": "exploit_attempt", "target": f"{endpoint}?{param}", "parameter": param},
        "command_execution",
        "sensitive_file_read",
    ]
    t["expert_trajectory"] = [
        f"Enumerate web directories to discover {login}.",
        "Inspect the login form and discover its input fields.",
        f"Login with default password admin/{pw}.",
        f"Enumerate web directories to discover {endpoint}.",
        f"Discover input parameters on the {endpoint} endpoint.",
        f"Verify RCE on {endpoint}?{param} parameter.",
        f"Exploit the RCE on {endpoint}?{param} to obtain a web shell.",
        "Run whoami through the web shell.",
        "Read the target flag file.",
    ]
    return t


def f_chained_exploit(rng: random.Random, idx: int) -> dict[str, Any]:
    """Two-stage exploit: SQLi -> service credential, then a SECOND vuln that
    `requires: vulnerability_verified:v1` -> shell (real: chained SQLi -> authenticated/secondary RCE).

    Signature has two vulnerability_check/exploit_attempt pairs — unlike any single-exploit family.
    """
    used: set = set()
    endpoint = pick(rng, PARAM_ENDPOINTS, used)
    p1 = pick(rng, PARAM_NAMES, used)
    p2 = pick(rng, PARAM_NAMES, used)
    cred = pick(rng, CRED_IDS, used)
    v1 = f"sqli_{p1}_{idx}"
    v2 = f"rce_{p2}_{idx}"
    hidden, nd = interleave_distractors(rng, [endpoint], 1, 1)
    t = base(f"gen_{idx:03d}_chained_exploit", hidden, rng.choice(TECH_POOL), "chained_exploit", "hard")
    t["parameters"] = {endpoint: [{"name": p1, "vulnerability": v1}, {"name": p2, "vulnerability": v2}]}
    t["credentials"] = {cred: {"username": cred, "password": f"{cred}_{idx}_pw", "role": "user", "weak": False}}
    t["vulnerabilities"] = {
        v1: {"type": "sqli", "target": f"{endpoint}?{p1}",
             "requires": [f"parameter_found:{endpoint}?{p1}"], "effects": {"credentials": [cred]}},
        v2: {"type": "rce", "target": f"{endpoint}?{p2}",
             "requires": [f"parameter_found:{endpoint}?{p2}", f"vulnerability_verified:{v1}"], "effects": {"shell": True}},
    }
    t["files"] = {"user_flag": {"path": "/home/web/flag.txt", "requires_shell": True, "flag": f"FLAG{{gen_{idx}_chained}}"}}
    t["goal"] = {"type": "read_file", "file": "user_flag"}
    t["budget"] = budget(8, nd)
    t["expert_plan"] = [
        {"action_type": "web_path_enumeration", "target": endpoint},
        {"action_type": "input_discovery", "target": endpoint},
        {"action_type": "vulnerability_check", "target": f"{endpoint}?{p1}", "parameter": p1},
        {"action_type": "exploit_attempt", "target": f"{endpoint}?{p1}", "parameter": p1},
        {"action_type": "vulnerability_check", "target": f"{endpoint}?{p2}", "parameter": p2},
        {"action_type": "exploit_attempt", "target": f"{endpoint}?{p2}", "parameter": p2},
        "command_execution",
        "sensitive_file_read",
    ]
    t["expert_trajectory"] = [
        f"Enumerate web directories to discover {endpoint}.",
        f"Discover input parameters on the {endpoint} endpoint.",
        f"Verify SQL injection on {endpoint}?{p1} parameter.",
        f"Exploit the SQL injection on {endpoint}?{p1} to dump a service credential.",
        f"Verify RCE on {endpoint}?{p2} parameter.",
        f"Exploit the RCE on {endpoint}?{p2} to obtain a web shell.",
        "Run whoami through the web shell.",
        "Read the target flag file.",
    ]
    return t


def f_leak_authed_privesc(rng: random.Random, idx: int) -> dict[str, Any]:
    """Leak admin creds -> authenticated RCE -> privesc -> root (real: leaked-cred-to-root).

    A 12-step chain mixing a leak hop, an auth-gated exploit, and privesc — the longest
    and most heterogeneous topology, distinct from every original family signature.
    """
    used: set = set()
    leak = pick(rng, LEAK_PATHS, used)
    login = pick(rng, LOGIN_PATHS, used)
    endpoint = pick(rng, PARAM_ENDPOINTS, used)
    param = pick(rng, PARAM_NAMES, used)
    cred = pick(rng, CRED_IDS, used)
    vuln_id = f"authed_rce_{param}_{idx}"
    hidden, nd = interleave_distractors(rng, [leak, login, endpoint], 1, 1)
    t = base(f"gen_{idx:03d}_leak_authed_privesc", hidden, rng.choice(TECH_POOL), "leak_authed_privesc", "hard")
    t["forms"] = {login: ["username", "password"]}
    t["leaks"] = {leak: {"credentials": [cred]}}
    t["credentials"] = {cred: {"username": cred, "password": f"{cred}_{idx}_pw", "role": "admin", "weak": False}}
    t["auth"] = {"login_path": login, "valid_credential": cred, "role": "admin"}
    t["parameters"] = {endpoint: [{"name": param, "vulnerability": vuln_id}]}
    t["vulnerabilities"] = {vuln_id: {
        "type": "rce", "target": f"{endpoint}?{param}",
        "requires": [f"parameter_found:{endpoint}?{param}", "auth_state:admin"],
        "effects": {"shell": True},
    }}
    t["privilege_escalation"] = {"available": True, "target_privilege": "root"}
    t["files"] = {"root_flag": {"path": "/root/root.txt", "requires_shell": True, "requires_privilege": "root", "flag": f"FLAG{{gen_{idx}_leakauthedpriv}}"}}
    t["goal"] = {"type": "read_file", "file": "root_flag"}
    t["budget"] = budget(12, nd)
    t["expert_plan"] = [
        {"action_type": "web_path_enumeration", "target": leak},
        {"action_type": "content_retrieval", "target": leak},
        {"action_type": "web_path_enumeration", "target": login},
        {"action_type": "input_discovery", "target": login},
        {"action_type": "credential_use", "target": login},
        {"action_type": "web_path_enumeration", "target": endpoint},
        {"action_type": "input_discovery", "target": endpoint},
        {"action_type": "vulnerability_check", "target": f"{endpoint}?{param}", "parameter": param},
        {"action_type": "exploit_attempt", "target": f"{endpoint}?{param}", "parameter": param},
        "command_execution",
        "privilege_escalation",
        "sensitive_file_read",
    ]
    t["expert_trajectory"] = [
        f"Enumerate web directories to discover {leak}.",
        f"Download and inspect the {leak} content for leaked credentials.",
        f"Enumerate web directories to discover {login}.",
        "Inspect the login form and discover its input fields.",
        f"Use the leaked credential to login through {login}.",
        f"Enumerate web directories to discover {endpoint}.",
        f"Discover input parameters on the {endpoint} endpoint.",
        f"Verify RCE on {endpoint}?{param} parameter.",
        f"Exploit the RCE on {endpoint}?{param} to obtain a web shell.",
        "Run whoami through the web shell.",
        "Escalate privileges to root with a local SUID helper.",
        "Read the root flag file.",
    ]
    return t


def generate_all(instances_per_family: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    idx = 200
    families = [
        lambda i: f_leak_file(random.Random(i), i),
        lambda i: f_default_pw(random.Random(i), i),
        lambda i: f_leak_login(random.Random(i), i),
        lambda i: f_injection_login(random.Random(i), i, "sqli"),
        lambda i: f_injection_login(random.Random(i), i, "lfi"),
        lambda i: f_rce(random.Random(i), i, False),
        lambda i: f_rce(random.Random(i), i, True),
        lambda i: f_upload(random.Random(i), i, "default", False),
        lambda i: f_upload(random.Random(i), i, "leak", False),
        lambda i: f_upload(random.Random(i), i, "default", True),
        lambda i: f_authed_injection(random.Random(i), i),
        lambda i: f_chained_exploit(random.Random(i), i),
        lambda i: f_leak_authed_privesc(random.Random(i), i),
    ]
    for builder in families:
        for _ in range(instances_per_family):
            tasks.append(builder(idx))
            idx += 1
    return tasks


def verify(task: dict[str, Any]) -> tuple[bool, str]:
    for mode in ("plan", "traj"):
        env = WebAttackSimEnv()
        env.reset(task)
        done = False
        seq = task["expert_plan"] if mode == "plan" else task["expert_trajectory"]
        for step in seq:
            if done:
                break
            if mode == "traj":
                n = normalize_llm_action(step)
                if n.status != "valid" or n.action is None:
                    return False, f"traj normalize fail: {step!r} -> {n.status}"
                action = n.action
            else:
                action = step
            _o, _r, done, _t, _i = env.step(action)
        if not (env.state and env.state.done and env._goal_reached()):
            return False, f"{mode} did not reach goal"
    return True, ""


def main() -> None:
    global _HARD_SLACK
    parser = argparse.ArgumentParser(description="Generate de-templated, family-tagged WebAttackSim tasks.")
    parser.add_argument("--instances-per-family", type=int, default=5)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--clean-generated", action="store_true", help="Delete previous gen_*/web_1xx generated tasks first.")
    parser.add_argument("--loose", action="store_true",
                        help="Loose (legacy) budgets = plan_len + 3*decoys + 12. DEFAULT is hard mode (env "
                             "improvement #1): tight budget so same-state decisions are consequential, which makes "
                             "the oracle informative where decisions matter (Spearman vs realized return -0.005->+0.37).")
    parser.add_argument("--hard-slack", type=int, default=2, help="Steps of budget above the expert plan length (hard mode).")
    args = parser.parse_args()

    hard = not args.loose
    out_dir = TASKS_DIR
    if hard:
        _HARD_SLACK = args.hard_slack

    tasks = generate_all(args.instances_per_family)
    if hard:
        for t in tasks:
            t["hard_mode"] = True
    failures = []
    for t in tasks:
        ok, msg = verify(t)
        if not ok:
            failures.append((t["task_id"], msg))
    print(f"generated {len(tasks)} tasks across {len(set(t['family'] for t in tasks))} families")
    for t in tasks[:3]:
        print(f"  sample {t['task_id']} family={t['family']} diff={t['difficulty']} budget={t['budget']['max_steps']} hidden={t['hidden_paths']}")
    if failures:
        for tid, msg in failures:
            print(f"  FAIL {tid}: {msg}")
        raise SystemExit(f"{len(failures)} tasks failed verification")
    print(f"All {len(tasks)} tasks verified solvable.")

    if args.write:
        if args.clean_generated:
            for old in out_dir.glob("gen_*.json"):
                old.unlink()
            for old in out_dir.glob("*_easy.json"):
                old.unlink()
            for prefix in ["backup_flag_leak", "config_flag_leak", "export_flag_leak", "swp_flag_leak",
                           "default_admin_pw", "weak_password123", "env_leak_login", "bak_leak_login",
                           "sqli_search_login", "sqli_product_login", "sqli_api_login", "lfi_view_login",
                           "lfi_include_login", "rce_ping_shell", "rce_exec_privesc", "upload_default_shell",
                           "upload_leak_shell", "upload_default_privesc"]:
                p = out_dir / f"{prefix}.json"
                if p.exists():
                    p.unlink()
        for t in tasks:
            (out_dir / f"{t['task_id']}.json").write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(tasks)} tasks to {out_dir}")


if __name__ == "__main__":
    main()
