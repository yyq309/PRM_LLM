"""Tests for the observation encoder (method §5 + enriched action-relevance features)."""

from web_attack_sim import WebAttackSimEnv
from web_attack_sim.encoder import FEATURE_NAMES, encode_observation


def test_feature_vector_length_matches_names(env):
    obs = env._observation()
    vec = encode_observation(obs, max_budget=env.max_steps)
    assert len(vec) == len(FEATURE_NAMES) == 34


def test_enriched_action_relevance_features_present():
    for name in [
        "has_unused_credentials", "has_known_form", "has_known_parameter",
        "has_verified_unexploited_vuln", "has_foothold", "has_access_goal_unread",
    ]:
        assert name in FEATURE_NAMES


def test_initial_state_features_are_minimal(env):
    obs = env._observation()
    vec = dict(zip(FEATURE_NAMES, encode_observation(obs, max_budget=env.max_steps)))
    assert vec["service_known"] == 1.0
    assert vec["auth_anonymous"] == 1.0 and vec["auth_admin"] == 0.0
    assert vec["shell_none"] == 1.0
    assert vec["has_unused_credentials"] == 0.0
    assert vec["has_foothold"] == 0.0


def test_unused_credentials_feature_fires_after_leak(sample_task):
    e = WebAttackSimEnv()
    e.reset(sample_task)
    e.step({"action_type": "web_path_enumeration", "target": "/backup"})
    e.step({"action_type": "content_retrieval", "target": "/backup"})
    obs = e._observation()
    vec = dict(zip(FEATURE_NAMES, encode_observation(obs, max_budget=e.max_steps)))
    assert obs.credentials  # admin credential discovered
    assert vec["has_unused_credentials"] == 1.0  # discovered but not yet used (still anonymous)


def test_encoding_is_deterministic(env):
    obs = env._observation()
    a = encode_observation(obs, max_budget=env.max_steps)
    b = encode_observation(obs, max_budget=env.max_steps)
    assert a == b
