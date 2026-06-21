"""Tests for the Action Normalizer (method §6, §6.1) including the form-phrase pitfall."""

import pytest

from web_attack_sim import normalize_llm_action
from web_attack_sim.action_space import ActionType


@pytest.mark.parametrize("text,expected", [
    ("Enumerate web directories to discover /backup.", ActionType.WEB_PATH_ENUMERATION),
    ("Run gobuster against the target", ActionType.WEB_PATH_ENUMERATION),
    ("Download and inspect the /backup content for leaked credentials.", ActionType.CONTENT_RETRIEVAL),
    ("Inspect the login form and discover its input fields.", ActionType.INPUT_DISCOVERY),
    ("Verify SQL injection on /item?id parameter.", ActionType.VULNERABILITY_CHECK),
    ("Exploit the SQL injection to dump the admin credential.", ActionType.EXPLOIT_ATTEMPT),
    ("Use the leaked credential to login through /login.", ActionType.CREDENTIAL_USE),
    ("Login with default password admin/admin.", ActionType.AUTH_ATTEMPT),
    ("Upload a PHP web shell through /upload.", ActionType.FILE_UPLOAD_ATTEMPT),
    ("Run whoami through the uploaded web shell.", ActionType.COMMAND_EXECUTION),
    ("Escalate privileges to root with a local SUID helper.", ActionType.PRIVILEGE_ESCALATION),
    ("Read the /admin/flag sensitive file.", ActionType.SENSITIVE_FILE_READ),
    ("Fingerprint the HTTP server headers and technology stack.", ActionType.HTTP_FINGERPRINT),
])
def test_valid_action_mapping(text, expected):
    n = normalize_llm_action(text)
    assert n.status == "valid", f"{text!r} -> {n.status}/{n.reason}"
    assert n.action.action_type == expected


def test_upload_form_phrase_is_input_discovery_not_upload():
    # "upload form" is an input-discovery phrase; only the path form triggers an actual upload.
    n = normalize_llm_action("Inspect the upload form and discover its file input.")
    assert n.action.action_type == ActionType.INPUT_DISCOVERY


def test_credential_use_with_path_not_misread_as_input_discovery():
    n = normalize_llm_action("Use the leaked credential to login through /signin.")
    assert n.action.action_type == ActionType.CREDENTIAL_USE
    assert n.action.target == "/signin"


def test_target_and_parameter_extraction():
    n = normalize_llm_action("Verify SQL injection on /item?id parameter.")
    assert n.action.target == "/item?id"
    # parameter extraction is loose (grabs the query token); the env matches on target,
    # so the imprecise parameter does not affect solvability.
    assert n.action.parameter is not None and "id" in n.action.parameter


def test_explicit_parameter_keyword_extraction():
    n = normalize_llm_action("Discover input parameters; param: uid is interesting")
    assert n.action.parameter == "uid"


@pytest.mark.parametrize("text,status", [
    ("", "invalid"),
    ("rm -rf / on the production server", "unsafe"),
    ("Pivot to another host on the internal network via lateral movement", "unsafe"),
    ("Check JWT token signing weaknesses", "schema_gap"),
    ("Try SSRF against cloud metadata", "schema_gap"),
    ("Phishing the administrator for credentials", "outside_single_host_web_scope"),
    ("just continue", "ambiguous"),
])
def test_non_valid_categories(text, status):
    n = normalize_llm_action(text)
    assert n.status == status, f"{text!r} -> {n.status}"


def test_non_valid_actions_have_no_action_object():
    n = normalize_llm_action("ddos the server")
    assert n.action is None
    assert n.confidence > 0


# --- regression tests for the §10.1-benchmark-surfaced fixes ---

def test_short_keyword_word_boundary_not_substring():
    # "rce" must not match inside "source"/"force"; "lfi"/"sqli" likewise word-bounded.
    assert normalize_llm_action("Retrieve the page source of /config").action.action_type == ActionType.CONTENT_RETRIEVAL
    assert normalize_llm_action("Run gobuster directory brute force").action.action_type == ActionType.WEB_PATH_ENUMERATION


def test_post_exploitation_not_misread_as_exploit():
    n = normalize_llm_action("Perform post-exploitation system enumeration")
    assert n.action.action_type == ActionType.POST_EXPLOITATION


def test_ssh_is_outside_single_host_web_scope():
    n = normalize_llm_action("Brute force the SSH service")
    assert n.status == "outside_single_host_web_scope"


def test_explicit_rce_keyword_still_matches_on_boundary():
    n = normalize_llm_action("Verify RCE on the cmd parameter")
    assert n.action.action_type == ActionType.VULNERABILITY_CHECK
