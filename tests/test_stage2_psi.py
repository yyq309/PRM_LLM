"""Tests for the Stage-2 ψ coverage layer (stage2/psi.py) and its held-out evaluation.

Locks in the two properties that make the layer trustworthy:
  1. it recovers in-scope false-rejects (the +accuracy), and
  2. its out-guard never maps an out-of-abstraction primitive to a web action (false-accept == 0).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage2.psi import EnhancedNormalizer  # noqa: E402
from stage2.eval_psi import _load_benchmark, _load_fixture_pairs, _score, _baseline  # noqa: E402
from web_attack_sim import normalize_llm_action  # noqa: E402


@pytest.fixture(scope="module")
def enh():
    return EnhancedNormalizer()


# ---- out-guard: non-web primitives must stay out (no false-accept) ----------------------
OUT_INTENTS = [
    "Crack the captured NTLM hash offline with hashcat",
    "Use john the ripper to recover the password from the shadow hash",
    "Switch to the user account with su using the cracked password",
    "Run the DirtyCow kernel exploit to get root",
    "Exploit the SMB stack buffer overflow to spawn a shell",
    "Run the remote overflow exploit against the FTP service",
    "Pivot through this host to reach the internal network",
]


@pytest.mark.parametrize("intent", OUT_INTENTS)
def test_out_guard_keeps_non_web_out(enh, intent):
    r = enh.normalize(intent)
    assert r.status != "valid", f"out-of-abstraction intent wrongly mapped valid: {intent} -> {r.action}"
    assert r.action is None


def test_out_guard_overrides_stage1_valid_exploit(enh):
    # Stage-1 maps "...exploit..." to exploit_attempt; the out-guard must override for a binary overflow.
    intent = "Exploit the SMB stack buffer overflow to spawn a shell"
    assert normalize_llm_action(intent).status == "valid"  # Stage-1 false-accepts it
    assert enh.normalize(intent).status == "schema_gap"     # enhanced corrects it
    assert "binary_service_exploit" in enh.normalize(intent).reason


# ---- recovery: in-scope false-rejects get the right action -----------------------------
RECOVER_CASES = [
    ("Read the password hash file left in the user's directory", "sensitive_file_read"),
    ("Cat the user flag file in the home directory", "sensitive_file_read"),
    ("Verify the application is vulnerable to SQL injection", "vulnerability_check"),
    ("Confirm the id parameter is injectable", "vulnerability_check"),
    ("Run id to check which user the shell runs as", "command_execution"),
    ("Brute-force the login with a password list", "auth_attempt"),
    ("Enumerate the system for privilege-escalation vectors", "post_exploitation"),
]


@pytest.mark.parametrize("intent,expected", RECOVER_CASES)
def test_recovery_maps_false_rejects(enh, intent, expected):
    r = enh.normalize(intent)
    assert r.status == "valid", f"expected recovery for: {intent} (got {r.status})"
    assert r.action.action_type.value == expected


def test_trusts_stage1_valid(enh):
    # A clean in-scope intent Stage-1 already handles must be passed through unchanged.
    intent = "Run a port scan with nmap to enumerate open services"
    base = normalize_llm_action(intent)
    assert base.status == "valid"
    out = enh.normalize(intent)
    assert out.status == "valid"
    assert out.action.action_type == base.action.action_type


# ---- end-to-end: held-out generalization + zero false-accept ---------------------------
def test_heldout_fixtures_improve_and_no_false_accept(enh):
    pairs = _load_fixture_pairs(ROOT / "stage2" / "walkthroughs")
    base = _score(pairs, _baseline)
    high = _score(pairs, enh.normalize)
    # enhanced lifts accuracy materially on the HELD-OUT fixtures...
    assert high["accuracy"] >= base["accuracy"] + 0.15
    assert high["accuracy"] >= 0.7
    # ...without ever mapping an out-of-abstraction step to a web action.
    assert high["false_accept"] == 0
    assert base["false_accept"] == 0


def test_benchmark_no_false_accept(enh):
    pairs = _load_benchmark(ROOT / "stage2" / "psi_benchmark.jsonl")
    high = _score(pairs, enh.normalize)
    assert high["false_accept"] == 0
