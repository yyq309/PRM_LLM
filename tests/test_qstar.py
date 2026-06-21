"""Tests for exact Q* value iteration (method §5.1)."""

import copy

from verify_qstar import QStarSolver, state_key, step_from
from web_attack_sim import WebAttackSimEnv


TINY_TASK = {
    "task_id": "qstar_tiny",
    "target": "http://target:80",
    "service_surface": ["http:80"],
    "initial_observation": {"open_services": ["http:80"], "discovered_paths": ["/"]},
    "hidden_paths": ["/backup"],
    "technologies": ["php"],
    "forms": {}, "parameters": {}, "credentials": {}, "auth": {}, "vulnerabilities": {},
    "leaks": {"/backup": {"files": ["flag"]}},
    "files": {"flag": {"path": "/backup/flag.txt", "flag": "FLAG{tiny}"}},
    "goal": {"type": "read_file", "file": "flag"},
    "budget": {"max_steps": 6},
    "expert_plan": [
        {"action_type": "web_path_enumeration", "target": "/backup"},
        {"action_type": "content_retrieval", "target": "/backup"},
    ],
}


def test_qstar_solver_runs_and_vstar_positive():
    solver = QStarSolver(TINY_TASK, gamma=0.98)
    v0, num_states = solver.solve(max_states=100000)
    assert num_states > 1
    # optimal: enum(+1.9) then content (credential_found +4 + goal +12 - 0.1 = 15.9), discounted
    assert v0 > 10.0


def test_qstar_goal_mode_is_discounted_goal_reward_only():
    # goal-aligned reward = +1 only on the goal-reaching transition; optimal = gamma^(steps-to-goal).
    # The tiny task reaches the goal in 2 steps (enum -> content), so V0_goal = gamma^1 = 0.98.
    full = QStarSolver(TINY_TASK, gamma=0.98, reward_mode="full")
    vf, _ = full.solve(max_states=100000)
    goal = QStarSolver(TINY_TASK, gamma=0.98, reward_mode="goal")
    vg, _ = goal.solve(max_states=100000)
    assert 0.9 < vg <= 1.0
    # full reward (discovery + goal) is much larger in magnitude than the unit goal reward
    assert vf > vg + 5.0


def test_state_key_ignores_failed_history():
    e = WebAttackSimEnv()
    e.reset(TINY_TASK)
    s1 = copy.deepcopy(e.state)
    s2 = copy.deepcopy(e.state)
    s2.failed_actions = ["precondition_missing"]
    s2.failed_branches = {"precondition_missing": 1}
    assert state_key(s1) == state_key(s2)


def test_step_from_is_deterministic():
    e = WebAttackSimEnv()
    e.reset(TINY_TASK)
    start = copy.deepcopy(e.state)
    a, ra, da = step_from(e, start, 2)
    b, rb, db = step_from(e, start, 2)
    assert ra == rb and da == db
    assert state_key(a) == state_key(b)


def test_obs_vec_is_history_invariant():
    solver = QStarSolver(TINY_TASK, gamma=0.98)
    e = WebAttackSimEnv()
    e.reset(TINY_TASK)
    s1 = copy.deepcopy(e.state)
    s2 = copy.deepcopy(e.state)
    s2.failed_actions = ["x", "y", "z"]
    s2.failed_branches = {"x": 3}
    assert solver.obs_vec(s1) == solver.obs_vec(s2)
