"""Tests for the fixed 16-action abstract Web action space (method §6)."""

from web_attack_sim.action_space import ACTIONS, ACTION_TO_ID, ActionType


EXPECTED_ORDER = [
    "service_enumeration", "http_fingerprint", "web_path_enumeration", "content_retrieval",
    "input_discovery", "form_interaction", "auth_attempt", "credential_use",
    "vulnerability_check", "exploit_attempt", "file_upload_attempt", "command_execution",
    "sensitive_file_read", "privilege_escalation", "post_exploitation", "stop_or_report",
]


def test_exactly_sixteen_actions():
    assert len(ACTIONS) == 16


def test_action_order_and_ids_match_method():
    assert [a.value for a in ACTIONS] == EXPECTED_ORDER
    for idx, action in enumerate(ACTIONS):
        assert ACTION_TO_ID[action] == idx


def test_action_to_id_is_bijective():
    assert len(set(ACTION_TO_ID.values())) == 16
    assert set(ACTION_TO_ID) == set(ActionType)
