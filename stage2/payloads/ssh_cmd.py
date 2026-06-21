#!/usr/bin/env python
"""One-shot SSH command runner for the gated full-chain lab (SSH-foothold boxes: Toppo / Symfonos).

The Windows host has no sshpass/plink, but paramiko is installed. This is operator-maintained eta (eta)
plumbing for the local lab VMs; the credentials live in the target descriptor's eta_fill and are NOT
fed to the proposer / PRM / phi-context (the agent must discover them from observable web output).

Usage: ssh_cmd.py <host> <user> <password> <command>
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import paramiko  # noqa: E402

host, user, pw, cmd = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect(host, username=user, password=pw, timeout=15, allow_agent=False, look_for_keys=False)
    _in, out, err = c.exec_command(cmd, timeout=40)
    sys.stdout.write(out.read().decode("utf-8", "replace") + err.read().decode("utf-8", "replace"))
except Exception as e:  # noqa: BLE001
    sys.stdout.write(f"[ssh_cmd] {type(e).__name__}: {e}")
finally:
    try:
        c.close()
    except Exception:  # noqa: BLE001
        pass
