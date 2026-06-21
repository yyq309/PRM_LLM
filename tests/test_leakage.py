"""Tests for the leakage-audit helpers (method §10.1)."""

from leakage_audit import (
    CONTEXT_LABELS,
    contains_token,
    hidden_tokens_for_task,
    mask_field,
    strip_task_label,
    structural_leak_test,
    visible_paths,
)

TASK = {
    "task_id": "web_002_sqli_login",
    "hidden_paths": ["/login", "/item", "/admin"],
    "credentials": {"admin": {"password": "sqli_dumped_pw"}, "guest": {"password": "admin"}},
    "files": {"flag": {"path": "/admin/flag", "flag": "FLAG{secret_value}"}},
}

CLEAN_CONTEXT = (
    "Scenario scenario_abc123, step 1. Known paths: ['/', '/login']. Known forms: []. "
    "Known parameters: []. Credentials: []. Auth state: anonymous. Shell state: none. "
    "Verified vulnerabilities: []. Read files: []. Failed branches: {}. Remaining budget: 20. "
    "Recent feedback: []."
)


def test_hidden_tokens_classification():
    tokens = hidden_tokens_for_task(TASK)
    assert "sqli_dumped_pw" in tokens["unique_secrets"]
    assert "FLAG{secret_value}" in tokens["unique_secrets"]
    assert "admin" in tokens["dictionary_secrets"]  # dictionary-word password, not a unique secret
    assert "/admin/flag" in tokens["file_paths"]


def test_mask_field_replaces_only_target_segment():
    masked = mask_field(CLEAN_CONTEXT, "Auth state:")
    assert "[MASKED]" in masked
    assert "Known paths: ['/', '/login']" in masked  # other fields untouched


def test_strip_task_label_removes_scenario_and_id():
    body = strip_task_label(CLEAN_CONTEXT, "web_002_sqli_login")
    assert not body.startswith("Scenario")
    assert "web_002_sqli_login" not in body


def test_contains_token_word_boundary():
    assert contains_token("the admin user", "admin")
    assert not contains_token("administrator panel", "admin")  # boundary-aware
    assert contains_token("flag FLAG{x}", "FLAG{x}")  # non-word token -> substring


def test_visible_paths_parses_known_paths():
    assert visible_paths(CLEAN_CONTEXT) == {"/", "/login"}


def test_structural_leak_test_clean_is_zero():
    rows = [{"sample_id": "s1", "task_id": "web_002_sqli_login", "context": CLEAN_CONTEXT}]
    report = structural_leak_test(rows, {"web_002_sqli_login": TASK})
    assert report["total_leaks"] == 0
    assert report["passed"]


def test_structural_leak_test_detects_planted_secret():
    leaky = CLEAN_CONTEXT + " (debug FLAG{secret_value})"
    rows = [{"sample_id": "s1", "task_id": "web_002_sqli_login", "context": leaky}]
    report = structural_leak_test(rows, {"web_002_sqli_login": TASK})
    assert report["unique_secret_leaks"] >= 1
    assert not report["passed"]


def test_all_context_labels_present_in_a_real_context():
    for label in CONTEXT_LABELS:
        assert label in CLEAN_CONTEXT
