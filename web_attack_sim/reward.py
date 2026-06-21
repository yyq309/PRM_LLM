DEFAULT_REWARDS = {
    "step_cost": -0.1,
    "service_found": 1.0,
    "fingerprint_found": 1.5,
    "path_found": 2.0,
    "input_found": 2.0,
    "credential_found": 4.0,
    "vulnerability_verified": 4.0,
    "exploit_succeeded": 6.0,
    "session_obtained": 5.0,
    "file_written": 3.0,
    "command_execution_obtained": 4.0,
    "shell_obtained": 8.0,
    "sensitive_file_read": 8.0,
    "privilege_escalated": 8.0,
    "goal_reached": 12.0,
    "no_new_information": -0.3,
    "duplicate_action": -1.0,
    "precondition_missing": -1.5,
    "auth_required": -1.5,
    "credential_invalid": -1.0,
    "vulnerability_not_present": -1.0,
    "unsupported_action": -3.0,
    "unsafe_action": -3.0,
    "invalid_action": -3.0,
}


def reward_for_feedback(feedback, rewards: dict[str, float] | None = None) -> float:
    table = rewards or DEFAULT_REWARDS
    reward = table["step_cost"]
    if feedback.progress_event:
        reward += table.get(feedback.progress_event, 0.0)
    if feedback.error_type:
        reward += table.get(feedback.error_type, 0.0)
    if feedback.terminal and feedback.success:
        reward += table.get("goal_reached", 0.0)
    return reward
