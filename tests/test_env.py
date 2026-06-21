"""Tests for the WebAttackSim environment dynamics (method §4, §7)."""

import pytest

from web_attack_sim import WebAttackSimEnv, normalize_llm_action
from web_attack_sim.action_space import ACTIONS, ActionType


def test_reset_exposes_plan_b_minimal_observation(env):
    obs = env._observation()
    assert obs.discovered_paths == ["/"]
    assert obs.auth_state == "anonymous"
    assert obs.shell_state == "none"
    assert obs.credentials == []
    assert obs.remaining_budget == 20


def test_expert_plan_solves_sample_task(sample_task):
    e = WebAttackSimEnv()
    e.reset(sample_task)
    done = False
    for action in sample_task["expert_plan"]:
        _o, _r, done, _t, _i = e.step(action)
    assert done and e.state.done and e._goal_reached()


def test_expert_trajectory_solves_via_normalizer(sample_task):
    e = WebAttackSimEnv()
    e.reset(sample_task)
    done = False
    for raw in sample_task["expert_trajectory"]:
        n = normalize_llm_action(raw)
        assert n.status == "valid", f"{raw!r} -> {n.status}"
        _o, _r, done, _t, _i = e.step(n.action)
    assert done and e._goal_reached()


def test_premature_credential_use_is_precondition_missing(env):
    # No login path discovered, no credential yet.
    _o, reward, _d, _t, info = env.step("credential_use")
    assert info["feedback"]["error_type"] == "precondition_missing"
    assert reward < 0


def test_web_path_enumeration_discovers_in_list_order(sample_task):
    e = WebAttackSimEnv()
    e.reset(sample_task)
    _o, _r, _d, _t, info = e.step("web_path_enumeration")  # no target -> first hidden path
    assert "/backup" in info["feedback"]["discovered_items"]


def test_goal_reached_terminates_episode(sample_task):
    e = WebAttackSimEnv()
    e.reset(sample_task)
    for action in sample_task["expert_plan"][:-1]:
        e.step(action)
    assert not e.state.done
    _o, _r, done, _t, info = e.step(sample_task["expert_plan"][-1])
    assert done
    assert "goal_reached" in info["feedback"]["discovered_items"]


def test_step_after_done_is_invalid_terminal(sample_task):
    e = WebAttackSimEnv()
    e.reset(sample_task)
    for action in sample_task["expert_plan"]:
        e.step(action)
    _o, _r, done, _t, info = e.step("web_path_enumeration")
    assert done
    assert info["feedback"]["error_type"] == "invalid_action"


def test_budget_exhaustion_terminates(sample_task):
    t = dict(sample_task)
    t["budget"] = {"max_steps": 3}
    e = WebAttackSimEnv()
    e.reset(t)
    done = False
    steps = 0
    while not done and steps < 10:
        _o, _r, done, _t, _i = e.step("service_enumeration")  # no-op, burns budget
        steps += 1
    assert done and steps <= 3


def test_permissive_mask_all_allowed_strict_mask_subset(env):
    permissive = env.action_mask(permissive=True)
    strict = env.action_mask(permissive=False)
    assert permissive == [1] * len(ACTIONS)
    assert sum(strict) <= sum(permissive)
    assert all(s <= p for s, p in zip(strict, permissive))


def test_privilege_escalation_duplicate_action_at_target(privesc_task):
    e = WebAttackSimEnv()
    e.reset(privesc_task)
    # drive to root via the expert plan up to and including the first privesc
    for action in privesc_task["expert_plan"][:6]:  # ... command_execution, privilege_escalation
        e.step(action)
    assert e.state.privilege_level == "root"
    # a SECOND privilege_escalation must be a duplicate_action, not a re-grant
    _o, reward, _d, _t, info = e.step("privilege_escalation")
    assert info["feedback"]["error_type"] == "duplicate_action"
    assert reward < 0
    assert e.state.privilege_level == "root"


def test_privesc_excluded_by_strict_mask_after_root(privesc_task):
    e = WebAttackSimEnv()
    e.reset(privesc_task)
    for action in privesc_task["expert_plan"][:6]:
        e.step(action)
    mask = e.action_mask(permissive=False)
    assert mask[ACTIONS.index(ActionType.PRIVILEGE_ESCALATION)] == 0


def test_step_accepts_int_str_and_dict_actions(sample_task):
    for action in (2, "web_path_enumeration", {"action_type": "web_path_enumeration", "target": "/backup"}):
        e = WebAttackSimEnv()
        e.reset(sample_task)
        _o, _r, _d, _t, info = e.step(action)
        assert info["action"]["action_type"] == "web_path_enumeration"


def test_invalid_action_index_raises(env):
    with pytest.raises((IndexError, ValueError, KeyError)):
        env.step(999)
