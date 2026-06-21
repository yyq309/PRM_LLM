"""Tests for the Stage-2 Phase-2 readiness stack: safety gate, η executors, engagement loop,
preflight. The single most important property under test: NOTHING executes live without explicit
authorization, and the target/command gate refuses anything outside an isolated lab.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from stage2.safety import (is_lab_target, command_allowed, AuthorizationGate, AuditLog,  # noqa: E402
                           LIVE_ENV_VAR, LIVE_CONFIRM, KILL_SWITCH_ENV)
from stage2.eta import (eta_command, eta_ctx_from_target, load_target, ETA_TEMPLATES, ETA_TOOL,  # noqa: E402
                        DryRunExecutor, LiveExecutor)
from web_attack_sim.action_space import ACTIONS, ActionType
from web_attack_sim.schemas import Action


# ---- safety: target scoping ------------------------------------------------------------
@pytest.mark.parametrize("target,lab", [
    ("http://127.0.0.1:8080", True),
    ("http://10.10.10.5", True),
    ("http://192.168.56.101", True),
    ("http://dvwa.lab", True),
    ("http://target.local", True),
    ("http://8.8.8.8", False),
    ("http://example.com", False),
    ("https://scanme.nmap.org", False),
])
def test_lab_target_scoping(target, lab):
    assert is_lab_target(target) is lab


def test_command_gate_refusals():
    assert command_allowed("nmap -sV http://dvwa.lab", "http://dvwa.lab")[0]
    # public target
    assert not command_allowed("nmap -sV 8.8.8.8", "http://8.8.8.8")[0]
    # binary not on allow-list
    assert not command_allowed("telnet dvwa.lab 23", "http://dvwa.lab")[0]
    # destructive token
    assert not command_allowed("curl http://dvwa.lab; rm -rf /", "http://dvwa.lab")[0]
    # command aimed at a different host than the in-scope target
    assert not command_allowed("curl http://dvwa.lab", "http://other.lab")[0]


# ---- safety: authorization gate deny-by-default ----------------------------------------
def test_gate_denies_without_env(monkeypatch):
    monkeypatch.delenv(LIVE_ENV_VAR, raising=False)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    assert AuthorizationGate(confirmed_isolated=True).authorized() is False


def test_gate_denies_without_confirmed_isolated(monkeypatch):
    monkeypatch.setenv(LIVE_ENV_VAR, LIVE_CONFIRM)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    assert AuthorizationGate(confirmed_isolated=False).authorized() is False


def test_gate_authorizes_only_with_both(monkeypatch):
    monkeypatch.setenv(LIVE_ENV_VAR, LIVE_CONFIRM)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    assert AuthorizationGate(confirmed_isolated=True).authorized() is True


def test_kill_switch_overrides(monkeypatch, tmp_path):
    sentinel = tmp_path / "stop"
    sentinel.write_text("x", encoding="utf-8")
    monkeypatch.setenv(LIVE_ENV_VAR, LIVE_CONFIRM)
    monkeypatch.setenv(KILL_SWITCH_ENV, str(sentinel))
    assert AuthorizationGate(confirmed_isolated=True).authorized() is False


# ---- η: templates + executors ----------------------------------------------------------
def test_eta_templates_complete():
    for a in ACTIONS:
        assert a in ETA_TEMPLATES and a in ETA_TOOL


def test_eta_renders_target():
    dv = load_target(ROOT / "stage2" / "targets" / "dvwa.json")
    ctx = eta_ctx_from_target(dv)
    cmd = eta_command(Action(action_type=ActionType.WEB_PATH_ENUMERATION), target=dv["target"], **ctx)
    assert "dvwa.lab" in cmd and "gobuster" in cmd


def test_live_executor_refuses_without_auth(monkeypatch):
    monkeypatch.delenv(LIVE_ENV_VAR, raising=False)
    dv = load_target(ROOT / "stage2" / "targets" / "dvwa.json")
    le = LiveExecutor(dv["target"], confirmed_isolated=True, **eta_ctx_from_target(dv))
    with pytest.raises(PermissionError):
        le.run(Action(action_type=ActionType.SERVICE_ENUMERATION))


def test_live_executor_refuses_public_target_even_if_authorized(monkeypatch):
    monkeypatch.setenv(LIVE_ENV_VAR, LIVE_CONFIRM)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    le = LiveExecutor("http://example.com", confirmed_isolated=True)
    with pytest.raises(PermissionError):
        le.run(Action(action_type=ActionType.SERVICE_ENUMERATION))


def test_dry_run_executes_nothing_but_logs(tmp_path):
    dv = load_target(ROOT / "stage2" / "targets" / "dvwa.json")
    audit = AuditLog(tmp_path / "a.jsonl")
    dr = DryRunExecutor(dv["target"], audit=audit, **eta_ctx_from_target(dv))
    res = dr.run(Action(action_type=ActionType.SERVICE_ENUMERATION))
    assert res.executed is False
    assert audit.records and audit.records[0]["executed"] is False
    assert "dvwa.lab" in audit.records[0]["command"]


# ---- engagement: offline loop runs, gated, bounded -------------------------------------
def test_engagement_dryrun_runs_offline():
    import joblib
    from stage2.engagement import run_engagement, StateProposer
    from stage2.eta import DryRunExecutor as DRE
    prm = joblib.load(ROOT / "outputs" / "prm_strong.joblib")
    dv = load_target(ROOT / "stage2" / "targets" / "dvwa.json")
    audit = AuditLog(ROOT / "outputs" / "_test_engagement_audit.jsonl")
    ex = DRE(dv["target"], audit=audit, **eta_ctx_from_target(dv))
    out = run_engagement(dv, prm=prm, executor=ex, proposer=StateProposer(), mode="prm", budget=6, audit=audit)
    assert out["steps_taken"] <= 6
    assert out["stop_reason"] in {"budget_exhausted", "goal_reached", "no_available_action",
                                  "kill_switch", "no_progress_stuck"}
    (ROOT / "outputs" / "_test_engagement_audit.jsonl").unlink(missing_ok=True)


def test_caching_proposer_shares_candidates_and_saves_calls():
    """CachingProposer must return the SAME candidate set for an identical (context, obs) (so paired
    arms share it) and only call the inner proposer once per unique state (common random numbers)."""
    from stage2.engagement import CachingProposer

    class _Counter:
        def __init__(self):
            self.calls = 0

        def propose(self, context, obs):
            self.calls += 1
            return [f"candidate from call {self.calls}"]

    inner = _Counter()
    cp = CachingProposer(inner)
    a = cp.propose("ctx-A", {"shell_state": "none"})
    b = cp.propose("ctx-A", {"shell_state": "none"})   # identical state -> cache HIT, no new call
    c = cp.propose("ctx-B", {"shell_state": "none"})   # different state -> MISS, fresh call
    assert a == b                                      # shared candidate set across arms
    assert inner.calls == 2                            # only 2 unique states -> 2 calls (not 3)
    assert cp.hits == 1 and cp.misses == 2
    assert cp.cache_stats()["llm_calls_saved"] == 1
    # mutating a returned list must not corrupt the cache (callers may shuffle in place)
    a.append("mutated")
    assert "mutated" not in cp.propose("ctx-A", {"shell_state": "none"})


def test_engagement_patience_circuit_breaker():
    """Cross-type oscillation (a DIFFERENT no-progress action each step, so the per-type exhaustion
    guard never fires) must still stop early via the GLOBAL no-progress circuit-breaker
    (stop_reason='no_progress_stuck'); patience=0 disables it."""
    import joblib
    from stage2.engagement import run_engagement
    from stage2.eta import DryRunExecutor as DRE

    class _OscillatingProposer:  # a new action TYPE each step -> per-type counter never reaches 2
        _ACTS = ["enumerate web directories", "fingerprint the web server", "scan for open services",
                 "discover input parameters", "retrieve the index page content"]

        def __init__(self):
            self.i = 0

        def propose(self, context, obs):
            a = self._ACTS[self.i % len(self._ACTS)]
            self.i += 1
            return [a]

    prm = joblib.load(ROOT / "outputs" / "prm_strong.joblib")
    dv = load_target(ROOT / "stage2" / "targets" / "dvwa.json")
    audit = AuditLog(ROOT / "outputs" / "_test_patience_audit.jsonl")
    ex = DRE(dv["target"], audit=audit, **eta_ctx_from_target(dv))
    out = run_engagement(dv, prm=prm, executor=ex, proposer=_OscillatingProposer(), mode="llm_only",
                         budget=20, audit=audit, permissive_guard=True, patience=3)
    assert out["stop_reason"] == "no_progress_stuck"
    assert out["steps_taken"] <= 4  # stops ~patience steps in, well under budget=20
    # patience=0 disables the breaker -> it does NOT stop via no_progress_stuck
    ex2 = DRE(dv["target"], audit=audit, **eta_ctx_from_target(dv))
    out0 = run_engagement(dv, prm=prm, executor=ex2, proposer=_OscillatingProposer(), mode="llm_only",
                          budget=20, audit=audit, permissive_guard=True, patience=0)
    assert out0["stop_reason"] != "no_progress_stuck"
    (ROOT / "outputs" / "_test_patience_audit.jsonl").unlink(missing_ok=True)


# ---- per-target exploit recipes (the box-specific payloads η needs for a real CVE) ------
def test_eta_recipe_overrides_generic_template():
    from stage2.eta import eta_recipes_from_target
    t = load_target(ROOT / "stage2" / "targets" / "thinkphp-5-rce.json")
    rec = eta_recipes_from_target(t)
    cmd = eta_command(Action(action_type=ActionType.EXPLOIT_ATTEMPT), target=t["target"],
                      recipes=rec, **eta_ctx_from_target(t))
    # the ThinkPHP invokefunction payload, not the generic sqlmap template
    assert "invokefunction" in cmd and "curl" in cmd and "sqlmap" not in cmd
    # curl globbing must be disabled (the vars[1][] brackets) and the payload host-scoped
    assert "-g" in cmd.split()
    from stage2.safety import command_allowed
    assert command_allowed(cmd, t["target"])[0]


# ---- φ robustness on real messy output: CSS/HTML must NOT be read as credentials --------
def test_phi_does_not_read_css_as_credentials():
    from stage2.phi import Phi
    phi = Phi()
    html = ("<style>a:hover{color:#2E5CD5;} .x{font-size:14px; table-layout:fixed;}</style>"
            "<h1>System Error</h1>")
    phi.ingest("curl", html, target="http://x.lab")
    assert phi.state.credentials == set(), f"CSS mis-parsed as creds: {phi.state.credentials}"


def test_phi_still_reads_real_leaked_credentials():
    from stage2.phi import Phi
    phi = Phi()
    phi.ingest("curl", "DB_PASSWORD = 'R@v3nSecurity'\nelliot:ER28-0652", target="http://x.lab",
               meta={"path": "/config"})
    assert any("ER28-0652" in c or "R@v3nSecurity" in c for c in phi.state.credentials)


# ---- live smoke: gated, refuses without authorization -----------------------------------
def test_live_smoke_module_importable_and_gated(monkeypatch):
    monkeypatch.delenv("STAGE2_LIVE_AUTHORIZED", raising=False)
    from stage2.safety import AuthorizationGate
    assert AuthorizationGate(confirmed_isolated=True).authorized() is False
    import stage2.live_smoke as ls
    assert hasattr(ls, "SEQUENCE") and len(ls.SEQUENCE) >= 4


# ---- target-aware proposer + permissive guard (the live autonomous A/B path) -----------
def test_target_aware_proposer_surfaces_box_exploit():
    from stage2.engagement import TargetAwareProposer
    from web_attack_sim import normalize_llm_action
    t = load_target(ROOT / "stage2" / "targets" / "thinkphp-5-rce.json")
    cands = TargetAwareProposer(t).propose("", {})
    mapped = {normalize_llm_action(c).action.action_type.value for c in cands
              if normalize_llm_action(c).action}
    # the abstract StateProposer never proposes exploit_attempt for this box; the target-aware one does
    assert "exploit_attempt" in mapped
    assert {"command_execution", "sensitive_file_read"} & mapped


def test_engagement_dryrun_target_proposer_permissive_runs():
    import joblib
    from stage2.engagement import run_engagement, TargetAwareProposer
    from stage2.eta import DryRunExecutor as DRE, eta_recipes_from_target
    prm = joblib.load(ROOT / "outputs" / "prm_strong.joblib")
    t = load_target(ROOT / "stage2" / "targets" / "thinkphp-5-rce.json")
    audit = AuditLog(ROOT / "outputs" / "_test_tp_audit.jsonl")
    ex = DRE(t["target"], audit=audit, recipes=eta_recipes_from_target(t), **eta_ctx_from_target(t))
    out = run_engagement(t, prm=prm, executor=ex, proposer=TargetAwareProposer(t), mode="prm",
                         budget=6, audit=audit, permissive_guard=True)
    assert out["steps_taken"] <= 6
    (ROOT / "outputs" / "_test_tp_audit.jsonl").unlink(missing_ok=True)


# ---- preflight: offline-ready, and the live items remain BLOCKED ------------------------
def test_preflight_offline_ready(monkeypatch):
    monkeypatch.delenv(LIVE_ENV_VAR, raising=False)
    from stage2.preflight import run_checks
    r = run_checks()
    assert r["offline_ready"] is True
    assert r["operator_blocked"]  # live prerequisites are surfaced, not auto-satisfied
    # the safety-deny check must be among the passing checks
    assert any(c["name"].startswith("safety gate denies") and c["ok"] for c in r["checks"])
