"""Phase-0 safety harness for Stage-2 live execution.

Everything that could touch a real target funnels through here. The defaults are *deny*: nothing
runs unless an operator has explicitly authorized an owned, isolated lab AND every command passes
the gate (allow-listed binary, in-scope target, no obvious destructiveness). This module performs
NO execution itself — it decides whether a command MAY run and records what was proposed/run.

Pieces:
  * `is_lab_target` / `target_host` — only private/loopback/.lab hosts are in scope; public IPs and
    public hostnames are refused, so an authorized run cannot escape onto the internet.
  * `command_allowed` — binary allow-list + a destructive-token denylist (rm -rf, mkfs, shutdown…).
  * `AuthorizationGate` — the single source of truth for "may we execute live": env confirmation
    string + an explicit `confirmed_isolated` flag + an optional kill-switch sentinel file.
  * `AuditLog` — append-only JSONL of every proposed and executed command (kept even in dry-run).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import ipaddress
import json
import os
import re
import shlex
from urllib.parse import urlparse

# The exact env confirmation string. Chosen to be unambiguous and un-guessable-by-accident.
LIVE_ENV_VAR = "STAGE2_LIVE_AUTHORIZED"
LIVE_CONFIRM = "i-own-this-isolated-authorized-lab"
# A sentinel file the operator can `touch` to immediately halt all live execution.
KILL_SWITCH_ENV = "STAGE2_KILL_SWITCH"

# Tools η may invoke. Anything not here is refused even under full authorization.
ALLOWED_BINARIES = {
    "nmap", "whatweb", "wappalyzer", "nikto", "httpx", "gobuster", "dirb", "dirsearch", "ffuf",
    "feroxbuster", "wfuzz", "curl", "wget", "sqlmap", "hydra", "nc", "ncat",
    # exploitation/PoC launchers and shell helpers — still target-scoped + denylisted below
    "python", "python3", "bash", "sh",
}

# Tokens that must never appear in a rendered command, even for an allowed binary on an in-scope
# target. Defense-in-depth against a mis-rendered template doing damage to the lab host itself.
DESTRUCTIVE_TOKENS = (
    "rm -rf", "rm -fr", ":(){", "mkfs", "dd if=", "shutdown", "reboot", "halt", "poweroff",
    "> /dev/sd", "wipefs", "fdisk", "userdel", "passwd -d", "chmod -R 000 /", "chown -R", "init 0",
)


def target_host(target: str) -> str | None:
    """Extract the host from a target URL or bare host:port. None if unparseable."""
    t = (target or "").strip()
    if not t:
        return None
    if "://" not in t:
        t = "http://" + t
    host = urlparse(t).hostname
    return host


def is_lab_target(target: str) -> bool:
    """True only for clearly-lab targets: loopback, RFC1918/private, link-local, or *.lab/.local/
    .test/.localhost names. Public IPs and public hostnames are out of scope and refused."""
    host = target_host(target)
    if not host:
        return False
    low = host.lower()
    if low in {"localhost"} or low.endswith((".lab", ".local", ".test", ".localhost", ".vulnhub")):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # a non-IP hostname that is not an explicit lab suffix -> treat as out of scope
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


def command_allowed(command: str, target: str) -> tuple[bool, str]:
    """Gate one rendered command string. Returns (allowed, reason)."""
    if not is_lab_target(target):
        return False, f"target {target!r} is not an isolated-lab host (only private/loopback/.lab in scope)"
    low = command.lower()
    for bad in DESTRUCTIVE_TOKENS:
        if bad in low:
            return False, f"destructive token {bad!r} present in command"
    try:
        parts = shlex.split(command)
    except ValueError:
        return False, "command is not safely tokenizable"
    if not parts:
        return False, "empty command"
    # find the first real binary token (skip env-style VAR=val prefixes)
    binary = next((p for p in parts if "=" not in p.split("/")[0]), parts[0])
    binary = binary.split("/")[-1]
    if binary not in ALLOWED_BINARIES:
        return False, f"binary {binary!r} is not in the η allow-list"
    # the target host must appear in the command (defense against a command aimed elsewhere)
    host = target_host(target)
    if host and host not in command:
        return False, f"target host {host!r} not referenced by the command (out-of-scope guard)"
    return True, "ok"


def kill_switch_engaged() -> bool:
    path = os.environ.get(KILL_SWITCH_ENV)
    return bool(path) and Path(path).exists()


@dataclass
class AuthorizationGate:
    """Single source of truth for whether live execution may proceed.

    deny-by-default: requires the env confirmation string AND an explicit confirmed_isolated flag
    the caller can only set after manually verifying an owned/snapshotted/isolated lab. A kill
    switch sentinel file (STAGE2_KILL_SWITCH) overrides everything to off."""

    confirmed_isolated: bool = False

    def authorized(self) -> bool:
        if kill_switch_engaged():
            return False
        return os.environ.get(LIVE_ENV_VAR) == LIVE_CONFIRM and self.confirmed_isolated

    def reason_blocked(self) -> str:
        if kill_switch_engaged():
            return f"kill switch engaged ({KILL_SWITCH_ENV} sentinel present)"
        if os.environ.get(LIVE_ENV_VAR) != LIVE_CONFIRM:
            return f"env {LIVE_ENV_VAR} is not set to the confirmation string"
        if not self.confirmed_isolated:
            return "confirmed_isolated=False (operator has not asserted an owned, isolated lab)"
        return ""

    def require(self, target: str, command: str) -> None:
        """Raise PermissionError unless live execution of `command` against `target` is allowed."""
        if not self.authorized():
            raise PermissionError("Stage-2 live execution not authorized: " + self.reason_blocked())
        ok, why = command_allowed(command, target)
        if not ok:
            raise PermissionError(f"command refused by safety gate: {why}")


@dataclass
class AuditLog:
    """Append-only JSONL audit of every proposed/executed command. Written even in dry-run."""

    path: Path
    records: list[dict] = field(default_factory=list)

    def record(self, **fields) -> None:
        self.records.append(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(fields, ensure_ascii=False) + "\n")
