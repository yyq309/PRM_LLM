"""Every bundled task must be solvable by BOTH its expert_plan and expert_trajectory."""

import pytest

from web_attack_sim import WebAttackSimEnv, normalize_llm_action
from web_attack_sim.schemas import Action
from web_attack_sim.action_space import ActionType
from web_attack_sim.tasks import bundled_task_paths, load_task_config


TASKS = [(p.name, load_task_config(p)) for p in bundled_task_paths()]


def _plan_action(raw):
    if isinstance(raw, str):
        return Action(ActionType(raw))
    return Action(ActionType(raw["action_type"]), target=raw.get("target"), parameter=raw.get("parameter"))


def test_task_set_nonempty():
    assert len(TASKS) >= 10


@pytest.mark.parametrize("name,task", TASKS)
def test_expert_plan_solves(name, task):
    e = WebAttackSimEnv()
    e.reset(task)
    done = False
    for raw in task["expert_plan"]:
        if done:
            break
        _o, _r, done, _t, _i = e.step(_plan_action(raw))
    assert e.state.done and e._goal_reached(), f"{name}: expert_plan did not reach goal"


@pytest.mark.parametrize("name,task", TASKS)
def test_expert_trajectory_solves(name, task):
    e = WebAttackSimEnv()
    e.reset(task)
    done = False
    for raw in task["expert_trajectory"]:
        if done:
            break
        n = normalize_llm_action(raw)
        assert n.status == "valid", f"{name}: {raw!r} -> {n.status}"
        _o, _r, done, _t, _i = e.step(n.action)
    assert e.state.done and e._goal_reached(), f"{name}: expert_trajectory did not reach goal"


@pytest.mark.parametrize("name,task", TASKS)
def test_generated_tasks_carry_family_and_difficulty(name, task):
    # de-templated generated tasks carry topology family + difficulty tags
    if task["task_id"].startswith("gen_"):
        assert task.get("family")
        assert task.get("difficulty") in {"easy", "medium", "hard"}
