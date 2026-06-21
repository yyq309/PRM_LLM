"""Tests for the Monte-Carlo return label helpers."""

from pathlib import Path

from mc_return_labels import action_from_norm, snapshot_expert_states, spearman
from web_attack_sim.action_space import ActionType


def test_action_from_norm():
    a = action_from_norm({"action_type": "exploit_attempt", "target": "/x?id", "parameter": "id"})
    assert a is not None and a.action_type == ActionType.EXPLOIT_ATTEMPT and a.target == "/x?id"
    assert action_from_norm({"action_type": None}) is None


def test_spearman_basic():
    assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == 1.0
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # constant -> undefined


def test_snapshot_expert_states_one_per_step():
    # use a bundled task; snapshots align with the expert trajectory length
    task_path = Path("tasks/gen_200_leak_file.json")
    snaps, max_steps = snapshot_expert_states(task_path)
    assert max_steps > 0
    assert set(snaps).issuperset({0, 1})  # at least the first couple of pre-step states
    # the step-0 snapshot is the initial state (only the root path discovered)
    assert "/" in snaps[0].discovered_paths
