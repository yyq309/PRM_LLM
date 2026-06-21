"""Tests for the Action / Observation / StepFeedback schemas (method §5, §7)."""

from web_attack_sim.action_space import ActionType
from web_attack_sim.schemas import Action, Observation, StepFeedback


def test_action_defaults():
    a = Action(ActionType.WEB_PATH_ENUMERATION)
    assert a.target is None and a.parameter is None


def test_stepfeedback_to_dict_has_all_abstract_fields():
    fb = StepFeedback(success=True, progress_event="path_found", evidence="x")
    d = fb.to_dict()
    for field in [
        "success", "progress_event", "new_observation", "discovered_items", "evidence",
        "auth_change", "foothold_change", "privilege_change", "file_change", "error_type",
        "cost", "terminal",
    ]:
        assert field in d
    assert d["success"] is True and d["progress_event"] == "path_found"


def test_observation_to_dict_roundtrip():
    obs = Observation(
        target_known=True, service_known=True, open_services=["http:80"], base_url_known=True,
        discovered_paths=["/"], known_forms=[], known_parameters=[], tech_stack=[],
        suspected_vulnerabilities=[], verified_vulnerabilities=[], credentials=[],
        auth_state="anonymous", shell_state="none", privilege_level="none", read_files=[],
        failed_actions=[], failed_branches={}, remaining_budget=20,
    )
    d = obs.to_dict()
    assert d["auth_state"] == "anonymous" and d["remaining_budget"] == 20
    assert isinstance(d["discovered_paths"], list)
