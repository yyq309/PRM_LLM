"""Tests for the MC-blend label productionization helpers."""

import numpy as np

from mc_blend_train import blended_score, decision_relevant_groups, top1_rate


def test_blended_score_oracle_vs_rule():
    oracle = {"label_source": "oracle", "score": 1.0, "_mc_goal": 0.0}
    # alpha=0.3 -> 0.3*1 + 0.7*0 = 0.3
    assert abs(blended_score(oracle, 0.3) - 0.3) < 1e-9
    # alpha=1 -> pure DQN
    assert blended_score(oracle, 1.0) == 1.0
    # rule rows keep their certain label regardless of alpha
    rule = {"label_source": "precondition_rule", "score": 0.0, "_mc_goal": None}
    assert blended_score(rule, 0.3) == 0.0


def test_decision_relevant_groups_threshold():
    rows = [
        {"_mc_goal": 0.9}, {"_mc_goal": 0.2},   # group g1: range 0.7 -> relevant
        {"_mc_goal": 0.80}, {"_mc_goal": 0.79},  # group g2: range 0.01 -> flat
    ]
    groups = {"g1": [0, 1], "g2": [2, 3]}
    dg = decision_relevant_groups(rows, groups)
    assert [0, 1] in dg and [2, 3] not in dg


def test_top1_rate_picks_realized_best():
    rows = [{"_mc_goal": 0.9}, {"_mc_goal": 0.2}]
    dgroups = [[0, 1]]
    # prediction ranks index 0 highest -> matches realized best -> rate 1.0
    assert top1_rate(rows, np.array([0.8, 0.1]), dgroups) == 1.0
    # prediction ranks index 1 highest -> misses -> rate 0.0
    assert top1_rate(rows, np.array([0.1, 0.8]), dgroups) == 0.0
