"""Tests for the hard-mode (tight-budget, consequential-decisions) generator option."""

import importlib

from web_attack_sim import WebAttackSimEnv


def test_hard_budget_is_tight_and_loose_is_default():
    gt = importlib.import_module("generate_tasks")
    try:
        gt._HARD_SLACK = None
        loose = gt.budget(8, 2)["max_steps"]  # 8 + 3*2 + 12 = 26
        assert loose == 26
        gt._HARD_SLACK = 2
        hard = gt.budget(8, 2)["max_steps"]  # 8 + 2 = 10
        assert hard == 10
        assert hard < loose
    finally:
        gt._HARD_SLACK = None  # never leak global state into other tests


def test_hard_tasks_still_solvable_under_tight_budget():
    gt = importlib.import_module("generate_tasks")
    import random as _r
    try:
        gt._HARD_SLACK = 2
        # one representative from a long family (leak_authed_privesc, 12-step plan)
        t = gt.f_leak_authed_privesc(_r.Random(260), 260)
        t["hard_mode"] = True
        assert t["budget"]["max_steps"] == len(t["expert_plan"]) + 2
        ok, msg = gt.verify(t)
        assert ok, msg
    finally:
        gt._HARD_SLACK = None


def test_canonical_tasks_are_hard():
    # hard mode is now the DEFAULT: every bundled task carries hard_mode=true and a tight budget.
    from web_attack_sim.tasks import bundled_task_paths, load_task_config
    tasks = [load_task_config(p) for p in bundled_task_paths()]
    assert all(t.get("hard_mode") for t in tasks)
    # tight: budget is plan_len + small slack, far below the legacy loose budget
    for t in tasks:
        assert t["budget"]["max_steps"] <= len(t["expert_plan"]) + 4


def test_env_honors_declared_budget_either_way():
    # the env code is unchanged: it faithfully honors whatever budget the task declares
    # (hard is a generated task-config property, not an env behavior switch).
    gt = importlib.import_module("generate_tasks")
    import random as _r
    try:
        gt._HARD_SLACK = None
        loose = gt.f_default_pw(_r.Random(205), 205)
        gt._HARD_SLACK = 2
        hard = gt.f_default_pw(_r.Random(205), 205)
    finally:
        gt._HARD_SLACK = None
    env = WebAttackSimEnv()
    env.reset(loose)
    loose_steps = env.max_steps
    env.reset(hard)
    assert env.max_steps == hard["budget"]["max_steps"] < loose_steps
