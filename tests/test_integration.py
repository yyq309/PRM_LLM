"""Integration tests: demo-pipeline scoring helpers + a tiny end-to-end DQN train/score."""

import numpy as np
import pytest

from demo_pipeline import (
    label_confidence,
    rank_label_from_oracle,
    schema_confidence_for_status,
    score_from_gap,
)


def test_rank_label_buckets():
    assert rank_label_from_oracle(1, 12) == "high"
    assert rank_label_from_oracle(12, 12) == "low"


def test_score_from_gap_monotonic():
    # smaller value_gap -> higher process score
    assert score_from_gap(0.0, "high") > score_from_gap(2.0, "high")
    assert score_from_gap(0.0, "high") >= score_from_gap(0.0, "low")


def test_schema_confidence_levels():
    assert schema_confidence_for_status("valid") == 1.0
    assert schema_confidence_for_status("schema_gap") == 0.45
    assert schema_confidence_for_status("unsafe") == 0.0


def test_label_confidence_is_product():
    c = label_confidence(normalizer_confidence=0.8, schema_confidence=1.0, oracle_label_confidence=0.5)
    assert abs(c - 0.4) < 1e-6


@pytest.mark.slow
def test_tiny_dqn_train_and_score(sample_task):
    from train_dqn import WebDQNAgent

    agent = WebDQNAgent(
        seed=0, lr=1e-3, gamma=0.98, batch_size=16, replay_size=2000,
        hidden_sizes=[32, 32], target_update_freq=50, device="cpu",
        mask_invalid_actions=True, task_paths=[sample_task],
    )
    for _ in range(40):
        agent.train_episode(epsilon=0.3)

    obs, _info = agent.env.reset(sample_task)
    vec = np.asarray(agent.env.encode_observation(obs), dtype=np.float32)
    report = agent.score_action(vec, action_id=2, action_mask=agent.current_action_mask())
    assert len(report["q_values"]) == 16
    assert report["selected_action_type"] == "web_path_enumeration"
    assert report["value_gap"] is not None  # action 2 is allowed at the start
    assert report["v_web"] >= report["q_selected"] - 1e-6


@pytest.mark.slow
def test_tiny_dqn_can_learn_to_solve(sample_task):
    from train_dqn import WebDQNAgent

    agent = WebDQNAgent(
        seed=0, lr=1e-3, gamma=0.98, batch_size=32, replay_size=5000,
        hidden_sizes=[64, 64], target_update_freq=100, device="cpu",
        mask_invalid_actions=True, task_paths=[sample_task],
    )
    for _ in range(150):
        agent.train_episode(epsilon=0.2)
    result = agent.eval_episode(task_path=sample_task, epsilon=0.0)
    assert result.goal  # masked DQN should solve this simple task after training
