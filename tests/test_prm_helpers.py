"""Tests for PRM feature extraction and evaluation helpers (method §10, §11.2/§11.3)."""

import numpy as np

from honest_eval import majority_floor, pairwise_within_group
from train_prm_strong import expected_calibration_error, extract_features


CONTEXT = (
    "Scenario scenario_abc, step 3. Known paths: ['/', '/login', '/backup']. "
    "Known forms: ['form:/login']. Known parameters: ['/item?id']. Credentials: ['admin']. "
    "Auth state: admin. Shell state: none. Verified vulnerabilities: ['sqli']. "
    "Read files: []. Failed branches: {'precondition_missing': 1}. Remaining budget: 14. "
    "Recent feedback: []."
)


def test_extract_features_parses_state():
    sample = {
        "context": CONTEXT,
        "normalized_action": {"action_type": "credential_use", "status": "valid", "target": "/login", "parameter": None},
        "normalizer_confidence": 0.85,
    }
    f = extract_features(sample)
    assert f["num_paths"] == 3
    assert f["num_forms"] == 1
    assert f["num_parameters"] == 1
    assert f["num_credentials"] == 1
    assert f["num_verified_vulns"] == 1
    assert f["auth_state"] == "admin"
    assert f["shell_state"] == "none"
    assert f["remaining_budget"] == 14
    assert f["action_type"] == "credential_use"
    assert f["has_target"] == 1 and f["has_parameter"] == 0


def test_majority_floor():
    assert majority_floor(["high", "high", "low"]) == 2 / 3
    assert majority_floor([]) == 0.0


def test_pairwise_ranking_accuracy_perfect_and_inverted():
    rows = [
        {"candidate_group": "g", "score": 0.9},
        {"candidate_group": "g", "score": 0.5},
        {"candidate_group": "g", "score": 0.1},
    ]
    # predictions in the same order -> perfect
    perfect = pairwise_within_group(rows, [0.9, 0.5, 0.1], [0, 1, 2], eps=1e-6)
    assert perfect["pairwise_accuracy"] == 1.0
    # fully inverted predictions -> 0
    inverted = pairwise_within_group(rows, [0.1, 0.5, 0.9], [0, 1, 2], eps=1e-6)
    assert inverted["pairwise_accuracy"] == 0.0


def test_pairwise_only_within_group():
    rows = [
        {"candidate_group": "g1", "score": 0.9},
        {"candidate_group": "g2", "score": 0.1},
    ]
    res = pairwise_within_group(rows, [0.9, 0.1], [0, 1], eps=1e-6)
    assert res["pairs"] == 0  # different groups -> no comparable pairs


def test_expected_calibration_error_bounds():
    conf = np.array([0.9, 0.8, 0.6, 0.55])
    correct = np.array([1.0, 1.0, 0.0, 1.0])
    ece = expected_calibration_error(conf, correct, bins=5)
    assert 0.0 <= ece <= 1.0
