"""Tests for the DeepSeek client (offline parsing + optional live healthcheck)."""

import os

import pytest

from deepseek_client import extract_json_array


def test_extract_plain_array():
    assert extract_json_array('["a", "b", "c"]') == ["a", "b", "c"]


def test_extract_fenced_array():
    text = 'Here are the actions:\n```json\n["enumerate", "login"]\n```\nDone.'
    assert extract_json_array(text) == ["enumerate", "login"]


def test_extract_array_embedded_in_prose():
    text = 'I suggest: ["a", "b"] as next steps.'
    assert extract_json_array(text) == ["a", "b"]


def test_extract_malformed_returns_empty():
    assert extract_json_array("no array here") == []
    assert extract_json_array("[broken, json") == []


def test_extract_non_array_json_returns_empty():
    assert extract_json_array('{"a": 1}') == []


@pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set")
def test_live_healthcheck():
    from deepseek_client import healthcheck

    result = healthcheck()
    assert result["ok"]
