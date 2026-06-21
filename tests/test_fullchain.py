"""Tests for the Stage-2 full-chain VM additions (2026-06-21):
safety allow-list (ssh/sshpass/smbclient), phi root/local-enum credit, the env reward-decay (G2/H) flag,
and the full_vm dual-transport descriptors render a host-scoped command.
"""
from pathlib import Path

from web_attack_sim.action_space import ActionType
from web_attack_sim.env import WebAttackSimEnv
from web_attack_sim.schemas import Action, StepFeedback
from stage2.safety import ALLOWED_BINARIES, command_allowed
from stage2.phi import Phi
from stage2.eta import eta_command, load_target

ROOT = Path(__file__).resolve().parents[1]


# ---- safety allow-list (full-VM ssh/smb transport) ----
def test_ssh_smb_in_allowlist():
    assert {"ssh", "sshpass", "smbclient"} <= ALLOWED_BINARIES


def test_gate_accepts_lab_ssh_refuses_public():
    ok, _ = command_allowed("sshpass -p x ssh ted@192.168.52.132 id", "http://192.168.52.132")
    assert ok
    ok, _ = command_allowed("sshpass -p x ssh ted@8.8.8.8 id", "http://8.8.8.8")
    assert not ok
    ok, _ = command_allowed("smbclient -L //192.168.52.131/ -N", "http://192.168.52.131")
    assert ok


# ---- phi root + local-enum credit ----
def test_phi_credits_euid0_as_root():
    p = Phi()
    p._content_credit("uid=33(www-data) gid=33(www-data) euid=0(root) groups=0(root),33(www-data)")
    assert p.state.privilege_level == "root"
    assert p.state.shell_state == "command_execution"


def test_phi_credits_lone_root_line():
    p = Phi()
    p._content_credit("root\n[{\"command\":\"settings\"}]")
    assert p.state.privilege_level == "root"


def test_phi_credits_local_privesc_surface():
    p = Phi()
    p._content_credit("user ALL=(ALL) NOPASSWD: /usr/bin/python")
    assert "local_privesc_surface" in p.state.verified_vulnerabilities
    p2 = Phi()
    p2._content_credit("-rwsr-xr-x 1 root root 88744 /usr/bin/find")
    assert "local_privesc_surface" in p2.state.verified_vulnerabilities


# ---- env reward-decay (G2/H) ----
def test_env_decay_recon_reward_default_off():
    assert WebAttackSimEnv().decay_recon_reward is False
    assert WebAttackSimEnv(decay_recon_reward=True).decay_recon_reward is True


def _reset_any_task(env):
    task = sorted((ROOT / "tasks").glob("*.json"))[0]
    env.reset(task)


def test_env_recon_bonus_zeroed_post_foothold():
    fb = StepFeedback(success=True, progress_event="path_found")
    act = Action(action_type=ActionType.WEB_PATH_ENUMERATION)
    # default env: post-foothold recon STILL earns the +2.0 path_found bonus
    base = WebAttackSimEnv()
    _reset_any_task(base)
    base.state.shell_state = "command_execution"
    _o, r_base, *_ = base._finalize_step(act, fb)
    # reward-fixed env: the bonus is removed once a foothold exists
    fix = WebAttackSimEnv(decay_recon_reward=True)
    _reset_any_task(fix)
    fix.state.shell_state = "command_execution"
    _o, r_fix, *_ = fix._finalize_step(act, StepFeedback(success=True, progress_event="path_found"))
    assert r_base > r_fix  # the +path_found bonus was removed post-foothold
    assert abs((r_base - r_fix) - 2.0) < 1e-6  # exactly the path_found reward


def test_env_recon_bonus_kept_pre_foothold():
    # before a foothold, recon is genuinely useful -> the bonus is NOT removed even with the flag on
    fix = WebAttackSimEnv(decay_recon_reward=True)
    _reset_any_task(fix)
    fix.state.shell_state = "none"
    act = Action(action_type=ActionType.WEB_PATH_ENUMERATION)
    _o, r, *_ = fix._finalize_step(act, StepFeedback(success=True, progress_event="path_found"))
    assert r > 0  # step_cost(-0.1) + path_found(+2.0) kept


# ---- full_vm descriptors render a host-scoped command (gate requirement) ----
def test_fullvm_descriptors_render_host_scoped():
    for name in ("vulnhub-dc-1", "vulnhub-toppo-1"):
        d = load_target(ROOT / "stage2" / "targets" / f"{name}.json")
        assert d["kind"] == "full_vm"
        recipes = d["eta_recipes"]
        for atype in (ActionType.PRIVILEGE_ESCALATION, ActionType.COMMAND_EXECUTION):
            cmd = eta_command(Action(action_type=atype), target=d["target"], recipes=recipes)
            host = d["vm"]["ip"]
            ok, reason = command_allowed(cmd, d["target"])
            assert host in cmd, f"{name}/{atype}: host not in command"
            assert ok, f"{name}/{atype}: gate refused: {reason}"
