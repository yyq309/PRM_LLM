"""Tests for the reward table and reward computation (method §8)."""

from web_attack_sim.reward import DEFAULT_REWARDS, reward_for_feedback
from web_attack_sim.schemas import StepFeedback


def test_method_reward_values():
    assert DEFAULT_REWARDS["step_cost"] == -0.1
    assert DEFAULT_REWARDS["path_found"] == 2.0
    assert DEFAULT_REWARDS["input_found"] == 2.0
    assert DEFAULT_REWARDS["credential_found"] == 4.0
    assert DEFAULT_REWARDS["vulnerability_verified"] == 4.0
    assert DEFAULT_REWARDS["session_obtained"] == 5.0
    assert DEFAULT_REWARDS["exploit_succeeded"] == 6.0
    assert DEFAULT_REWARDS["shell_obtained"] == 8.0
    assert DEFAULT_REWARDS["sensitive_file_read"] == 8.0
    assert DEFAULT_REWARDS["privilege_escalated"] == 8.0
    assert DEFAULT_REWARDS["goal_reached"] == 12.0
    for negative in ["unsupported_action", "unsafe_action", "invalid_action"]:
        assert DEFAULT_REWARDS[negative] == -3.0


def test_progress_event_reward_includes_step_cost():
    fb = StepFeedback(success=True, progress_event="path_found")
    assert reward_for_feedback(fb) == -0.1 + 2.0


def test_error_event_is_penalised():
    fb = StepFeedback(success=False, error_type="precondition_missing")
    assert reward_for_feedback(fb) == -0.1 + DEFAULT_REWARDS["precondition_missing"]


def test_terminal_success_adds_goal_bonus_once():
    # A non-stop action that triggers the goal: progress_event is its own event, plus the
    # terminal goal bonus once (no double count of goal_reached).
    fb = StepFeedback(success=True, progress_event="sensitive_file_read", terminal=True)
    assert reward_for_feedback(fb) == -0.1 + 8.0 + 12.0
