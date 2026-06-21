"""Tests for the Stage-2 real-lab adapter (φ/ψ/η) and the Phase-1 offline replay harness."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from stage2.phi import Phi  # noqa: E402
from stage2.fixtures import FixtureError, validate_walkthrough, load_walkthrough  # noqa: E402
from stage2.eta import LiveExecutor, ReplayExecutor, eta_command, ETA_TEMPLATES  # noqa: E402
from stage2.replay import replay_one  # noqa: E402
from web_attack_sim.action_space import ACTIONS, ActionType  # noqa: E402
from web_attack_sim.schemas import Action, Observation  # noqa: E402


# ---- φ parsers -----------------------------------------------------------------------
def test_phi_nmap_extracts_http_service_and_tech():
    phi = Phi()
    out = phi.ingest("nmap", "80/tcp open http Apache httpd 2.2.22 ((Debian))\n22/tcp open ssh OpenSSH")
    assert "http:80" in out["open_services"]
    assert "apache" in phi.state.tech_stack


def test_phi_gobuster_extracts_paths():
    phi = Phi()
    out = phi.ingest("gobuster", "/admin (Status: 403)\n/login (Status: 200)\n/backup (Status: 301)")
    for p in ("/admin", "/login", "/backup"):
        assert p in phi.state.discovered_paths
    assert set(out["discovered_paths"]) >= {"/admin", "/login", "/backup"}


def test_phi_whatweb_tech():
    phi = Phi()
    phi.ingest("whatweb", "http://x [200] Apache[2.2], PHP[5.4], Drupal 7, MySQL")
    assert {"apache", "php", "drupal", "mysql"} <= phi.state.tech_stack


def test_phi_exploit_yields_shell():
    phi = Phi()
    phi.ingest("exploit", "Target vulnerable. Spawned PHP web shell.", meta={"vuln_id": "rce1", "yields_shell": True})
    assert phi.state.shell_state == "webshell"
    assert "rce1" in phi.state.verified_vulnerabilities


def test_phi_command_then_privesc():
    phi = Phi()
    phi.ingest("shell", "uid=33(www-data) gid=33(www-data)")
    assert phi.state.shell_state == "command_execution"
    assert phi.state.privilege_level == "web_user"
    phi.ingest("privesc", "# whoami\nroot", meta={"rooted": True})
    assert phi.state.privilege_level == "root"


def test_phi_login_success_sets_auth():
    phi = Phi()
    phi.ingest("login", "HTTP/1.1 302 Found\nLocation: /admin\nSet-Cookie: session=abc", meta={"role": "admin"})
    assert phi.state.auth_state == "admin"


def test_phi_login_failure_no_auth():
    phi = Phi()
    phi.ingest("login", "Invalid username or password. Login failed.")
    assert phi.state.auth_state == "anonymous"


def test_phi_sqlmap_dumps_credentials():
    phi = Phi()
    out = phi.ingest("sqlmap", "parameter 'id' is vulnerable\nDatabase: app\n| admin | 5f4dcc3b5aa765 |",
                     meta={"vuln_id": "sqli_id"})
    assert "sqli_id" in phi.state.verified_vulnerabilities
    assert phi.state.credentials  # at least one dumped credential
    assert out.get("verified_vulnerabilities") == ["sqli_id"]


def test_phi_observation_is_valid():
    phi = Phi()
    phi.ingest("nmap", "80/tcp open http Apache")
    obs = phi.observation()
    assert isinstance(obs, Observation)
    d = obs.to_dict()
    assert d["service_known"] is True
    assert "http:80" in d["open_services"]


def test_phi_unknown_tool_is_recorded_not_crashed():
    phi = Phi()
    res = phi.ingest("nonexistent-tool", "whatever")
    assert res.get("_unrouted") is True
    assert phi.state.unparsed_outputs


# ---- fixture schema ------------------------------------------------------------------
def _minimal_fixture(**overrides):
    step = {
        "actor_intent": "Scan the target with nmap.",
        "tool": "nmap",
        "tool_output": "80/tcp open http Apache",
        "reference_abstract_action": "service_enumeration",
        "reference_state_after": {"open_services": ["http:80"]},
    }
    step.update(overrides.pop("step0", {}))
    data = {"box": "T", "source": "test", "abstract_family": "rce_privesc", "steps": [step]}
    data.update(overrides)
    return data


def test_fixture_valid_passes():
    validate_walkthrough(_minimal_fixture())


def test_fixture_bad_action_rejected():
    with pytest.raises(FixtureError):
        validate_walkthrough(_minimal_fixture(step0={"reference_abstract_action": "hack_the_planet"}))


def test_fixture_out_of_abstraction_requires_reason():
    with pytest.raises(FixtureError):
        validate_walkthrough(_minimal_fixture(step0={"reference_abstract_action": "out_of_abstraction"}))


def test_fixture_out_of_abstraction_with_reason_ok():
    validate_walkthrough(_minimal_fixture(step0={
        "reference_abstract_action": "out_of_abstraction",
        "out_of_abstraction_reason": "ssh login is outside the web schema",
    }))


def test_fixture_missing_top_key_rejected():
    bad = _minimal_fixture()
    del bad["abstract_family"]
    with pytest.raises(FixtureError):
        validate_walkthrough(bad)


def test_fixture_unknown_state_field_rejected():
    with pytest.raises(FixtureError):
        validate_walkthrough(_minimal_fixture(step0={"reference_state_after": {"not_a_field": 1}}))


# ---- η executors ---------------------------------------------------------------------
def test_eta_templates_cover_all_16_actions():
    assert set(ETA_TEMPLATES) == set(ACTIONS)


def test_eta_command_renders():
    cmd = eta_command(Action(ActionType.WEB_PATH_ENUMERATION), target="http://t")
    assert "gobuster" in cmd and "http://t" in cmd


def test_live_executor_refuses_without_authorization():
    ex = LiveExecutor("http://dvwa.lab", confirmed_isolated=False)
    with pytest.raises(PermissionError):
        ex.run(Action(ActionType.SERVICE_ENUMERATION))


def test_live_executor_refuses_even_with_flag_but_no_env(monkeypatch):
    monkeypatch.delenv("STAGE2_LIVE_AUTHORIZED", raising=False)
    ex = LiveExecutor("http://dvwa.lab", confirmed_isolated=True)
    with pytest.raises(PermissionError):
        ex.run(Action(ActionType.SERVICE_ENUMERATION))


def test_replay_executor_returns_recorded_output():
    steps = [
        {"tool": "nmap", "tool_output": "80/tcp open http"},
        {"tool": "gobuster", "tool_output": "/admin (Status: 200)"},
    ]
    ex = ReplayExecutor(steps)
    r = ex.run(Action(ActionType.WEB_PATH_ENUMERATION))
    assert "/admin" in r.output
    assert r.executed is False


# ---- replay harness on the bundled DC-1 reference fixture ----------------------------
def test_replay_dc1_reference_metrics():
    path = ROOT / "stage2" / "walkthroughs" / "dc-1.json"
    data = load_walkthrough(path)
    box = replay_one(path, data, prm=None)
    assert box["box"] == "DC-1"
    assert 0.0 <= box["out_of_abstraction_rate"] <= 1.0
    assert box["n_out_of_abstraction"] >= 1  # the john hash-crack step
    # φ should reconstruct (near) all hand-labeled state fields for this clean web chain
    assert box["phi"]["field_recall"] >= 0.9
    # ψ should map a clear majority of the in-abstraction operator intents
    assert box["psi"]["accuracy"] >= 0.6
