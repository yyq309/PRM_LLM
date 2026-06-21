"""Tests for the trajectory-level credit-assignment helpers."""

from trajectory_credit_eval import GAMMA, returns_to_go, spearman, to_action


def test_to_action_normalizes_str_and_dict():
    assert to_action("auth_attempt") == {"action_type": "auth_attempt", "target": None, "parameter": None}
    a = to_action({"action_type": "exploit_attempt", "target": "/x?id", "parameter": "id"})
    assert a["action_type"] == "exploit_attempt" and a["target"] == "/x?id"


def test_returns_to_go_discounting():
    # rewards [0, 0, 1] with gamma -> g0 = gamma^2, g1 = gamma, g2 = 1
    g = returns_to_go([0.0, 0.0, 1.0])
    assert abs(g[2] - 1.0) < 1e-9
    assert abs(g[1] - GAMMA) < 1e-9
    assert abs(g[0] - GAMMA * GAMMA) < 1e-9


def test_returns_to_go_is_monotone_blame_to_failure():
    # a late negative reward (failure) pulls earlier return-to-go down
    g_ok = returns_to_go([1.0, 1.0, 12.0])
    g_fail = returns_to_go([1.0, 1.0, -3.0])
    assert g_fail[0] < g_ok[0]


def test_spearman_perfect_and_degenerate():
    assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == 1.0
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # zero variance -> None
