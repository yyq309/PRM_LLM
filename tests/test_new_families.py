"""Tests for the 3 structurally-novel chain families (coverage extension, method §13.1).

These families (authed_injection, chained_exploit, leak_authed_privesc) are built from
the SAME frozen 16 primitives but exploit the env's vuln `requires` preconditions
(auth_state / vulnerability_verified) to form expert_plan signatures that no original
family has. The tests assert: (1) they exist, (2) their topology signatures are genuinely
new vs every original family, (3) they exercise the new precondition mechanics, and
(4) the expert trajectory normalizes + solves end-to-end.
"""

import pytest

from task_split import plan_signature
from web_attack_sim import WebAttackSimEnv, normalize_llm_action
from web_attack_sim.action_space import ActionType
from web_attack_sim.tasks import bundled_task_paths, load_task_config

NEW_FAMILIES = {"authed_injection", "chained_exploit", "leak_authed_privesc"}
ORIGINAL_FAMILIES = {
    "leak_file", "default_pw", "leak_login", "injection_login", "rce_shell",
    "rce_privesc", "upload_default_shell", "upload_leak_shell", "upload_default_privesc",
}


def _tasks_by_family():
    by_family = {}
    for p in bundled_task_paths():
        t = load_task_config(p)
        by_family.setdefault(t.get("family", "?"), []).append(t)
    return by_family


def test_new_families_present():
    fams = set(_tasks_by_family())
    assert NEW_FAMILIES <= fams, f"missing new families: {NEW_FAMILIES - fams}"


def test_new_family_signatures_are_novel():
    by_family = _tasks_by_family()
    original_sigs = {plan_signature(t) for fam in ORIGINAL_FAMILIES for t in by_family.get(fam, [])}
    for fam in NEW_FAMILIES:
        for t in by_family[fam]:
            assert plan_signature(t) not in original_sigs, f"{fam} signature collides with an original family"


def test_authed_injection_requires_auth_before_exploit():
    # the injection vuln must gate on auth_state, making the auth step structurally forced
    t = _tasks_by_family()["authed_injection"][0]
    vuln = next(iter(t["vulnerabilities"].values()))
    assert any(r.startswith("auth_state:") for r in vuln["requires"])


def test_chained_exploit_second_vuln_requires_first():
    t = _tasks_by_family()["chained_exploit"][0]
    requires = [r for v in t["vulnerabilities"].values() for r in v["requires"]]
    assert any(r.startswith("vulnerability_verified:") for r in requires)
    # two distinct vulnerabilities (double exploit chain)
    assert len(t["vulnerabilities"]) == 2


def test_leak_authed_privesc_is_longest_and_reaches_root():
    t = _tasks_by_family()["leak_authed_privesc"][0]
    assert t["goal"] == {"type": "read_file", "file": "root_flag"}
    assert t["privilege_escalation"]["target_privilege"] == "root"
    assert len(t["expert_plan"]) >= 12


@pytest.mark.parametrize("family", sorted(NEW_FAMILIES))
def test_new_family_trajectory_normalizes_and_solves(family):
    t = _tasks_by_family()[family][0]
    env = WebAttackSimEnv()
    env.reset(t)
    done = False
    for raw in t["expert_trajectory"]:
        if done:
            break
        n = normalize_llm_action(raw)
        assert n.status == "valid" and n.action is not None, f"{family}: {raw!r} -> {n.status}"
        _o, _r, done, _tr, _i = env.step(n.action)
    assert env.state and env.state.done and env._goal_reached(), f"{family} trajectory did not reach goal"


def test_new_families_use_only_frozen_16_actions():
    valid = {a.value for a in ActionType}
    for fam in NEW_FAMILIES:
        for t in _tasks_by_family()[fam]:
            for step in t["expert_plan"]:
                at = step if isinstance(step, str) else step["action_type"]
                assert at in valid, f"{fam} uses out-of-schema action {at}"
