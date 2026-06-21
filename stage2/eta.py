"""eta (η): abstract action -> concrete tool command, plus gated executors.

η is the inverse of φ: it turns an abstract `Action` the PRM/agent chose into the concrete
command a pentester would run. Phase-2 wires η to a sandbox; Phase-1 (here) only needs the
command *templates* and an OFFLINE `ReplayExecutor` that returns pre-recorded fixture output —
so the closed loop can be exercised end-to-end WITHOUT touching a network.

SAFETY: `LiveExecutor` is the ONLY path that would run a real command, and it refuses to do so
unless an explicit, unambiguous authorization env var is set AND the caller passes
`confirmed_isolated=True`. By default nothing runs. There is intentionally no convenience flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import time

from web_attack_sim.action_space import ActionType
from web_attack_sim.schemas import Action
from stage2.safety import AuthorizationGate, AuditLog


# Concrete command TEMPLATES, one per frozen abstract action. {target}/{path}/{param}/{file}
# are filled from the Action + session context. These are documentation/Phase-2 scaffolding;
# Phase-1 never fills-and-runs them.
ETA_TEMPLATES: dict[ActionType, str] = {
    ActionType.SERVICE_ENUMERATION: "nmap -sV -p- {target}",
    ActionType.HTTP_FINGERPRINT: "whatweb {target}",
    ActionType.WEB_PATH_ENUMERATION: "gobuster dir -u {target} -w {wordlist}",
    ActionType.CONTENT_RETRIEVAL: "curl -s {target}{path}",
    ActionType.INPUT_DISCOVERY: "curl -s {target}{path}   # parse <form>/<input name=...>",
    ActionType.FORM_INTERACTION: "curl -s -X POST {target}{path} -d '{data}'",
    ActionType.AUTH_ATTEMPT: "curl -s -c cookies.txt -d 'username={user}&password={password}' {target}{login}",
    ActionType.CREDENTIAL_USE: "curl -s -c cookies.txt -d 'username={user}&password={password}' {target}{login}",
    ActionType.VULNERABILITY_CHECK: "sqlmap -u '{target}{path}?{param}=1' --batch --level 2",
    ActionType.EXPLOIT_ATTEMPT: "sqlmap -u '{target}{path}?{param}=1' --batch --dump   # or the matched CVE PoC",
    ActionType.FILE_UPLOAD_ATTEMPT: "curl -s -b cookies.txt -F 'file=@shell.php' {target}{path}",
    ActionType.COMMAND_EXECUTION: "curl -s '{target}{webshell}?cmd=id'",
    ActionType.SENSITIVE_FILE_READ: "curl -s '{target}{webshell}?cmd=cat+{file}'",
    ActionType.PRIVILEGE_ESCALATION: "# run the matched local privesc PoC (SUID/sudo/cron) inside the established shell",
    ActionType.POST_EXPLOITATION: "# enumerate locally: id; sudo -l; find / -perm -4000 -type f 2>/dev/null",
    ActionType.STOP_OR_REPORT: "# goal reached — stop and report",
}

# η route key (the φ tool key whose output an executed η command would produce). Lets the
# ReplayExecutor look up the matching recorded fixture step.
ETA_TOOL: dict[ActionType, str] = {
    ActionType.SERVICE_ENUMERATION: "nmap",
    ActionType.HTTP_FINGERPRINT: "whatweb",
    ActionType.WEB_PATH_ENUMERATION: "gobuster",
    ActionType.CONTENT_RETRIEVAL: "curl",
    ActionType.INPUT_DISCOVERY: "curl",
    ActionType.FORM_INTERACTION: "curl",
    ActionType.AUTH_ATTEMPT: "login",
    ActionType.CREDENTIAL_USE: "login",
    ActionType.VULNERABILITY_CHECK: "sqlmap",
    ActionType.EXPLOIT_ATTEMPT: "exploit",
    ActionType.FILE_UPLOAD_ATTEMPT: "upload",
    ActionType.COMMAND_EXECUTION: "shell",
    ActionType.SENSITIVE_FILE_READ: "cat",
    ActionType.PRIVILEGE_ESCALATION: "privesc",
    ActionType.POST_EXPLOITATION: "shell",
    ActionType.STOP_OR_REPORT: "command",
}


def load_target(path: str | Path) -> dict:
    """Load a target descriptor (stage2/targets/*.json)."""
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def eta_ctx_from_target(descriptor: dict) -> dict:
    """Build the eta_command fill-context from a target descriptor's eta_fill (the `target` itself is
    passed separately to the executors / eta_command, so it is NOT included here)."""
    return dict(descriptor.get("eta_fill", {}))


def eta_recipes_from_target(descriptor: dict) -> dict:
    """Per-target command overrides keyed by action_type value (e.g. a CVE-specific exploit payload
    that the generic ETA_TEMPLATES cannot express). Empty for most targets."""
    return dict(descriptor.get("eta_recipes", {}))


def eta_command(action: Action, *, target: str = "http://TARGET", recipes: dict | None = None, **ctx) -> str:
    """Render the concrete command template for an abstract action (no execution). A per-target
    `recipes` override (keyed by action_type value) takes precedence over the generic template.

    Uses targeted `{key}` replacement, NOT str.format, so a recipe carrying a real payload full of
    literal `{`/`}` (e.g. a Struts2 OGNL expression) is left intact instead of being mis-parsed."""
    tmpl = (recipes or {}).get(action.action_type.value) or ETA_TEMPLATES[action.action_type]
    fields = {"target": target, "path": action.target or "", "param": action.parameter or "",
              "wordlist": "common.txt", "data": "", "user": "USER", "password": "PASS",
              "login": "/login", "webshell": "/shell.php", "file": "/root/root.txt"}
    fields.update({k: v for k, v in ctx.items() if v is not None})
    out = tmpl
    for k, v in fields.items():
        out = out.replace("{" + k + "}", str(v))
    return out


@dataclass
class ExecResult:
    tool: str
    output: str
    executed: bool


class ReplayExecutor:
    """OFFLINE executor: returns recorded fixture output for the chosen abstract action.

    Matches the action's η-tool against the fixture's remaining steps in order. This drives the
    closed loop over a recorded walkthrough WITHOUT running anything. It is the Phase-1 stand-in
    for a sandbox.
    """

    def __init__(self, steps: list[dict]):
        self._steps = list(steps)
        self._cursor = 0

    def run(self, action: Action, **_ctx) -> ExecResult:
        want = ETA_TOOL[action.action_type]
        # advance to the next recorded step whose tool route matches this action's η-tool
        from stage2.phi import TOOL_ALIASES
        want_route = TOOL_ALIASES.get(want, want)
        for j in range(self._cursor, len(self._steps)):
            step = self._steps[j]
            step_route = TOOL_ALIASES.get(str(step.get("tool", "")).lower())
            if step_route == want_route:
                self._cursor = j + 1
                return ExecResult(tool=step["tool"], output=step.get("tool_output", ""), executed=False)
        return ExecResult(tool=want, output="", executed=False)


class DryRunExecutor:
    """Renders + audit-logs the η command for the chosen action but executes NOTHING. Lets the full
    engagement loop be exercised offline (preflight) when there is no fixture to replay."""

    def __init__(self, target: str = "http://target.lab", audit: AuditLog | None = None,
                 recipes: dict | None = None, **ctx):
        self.target = target
        self.audit = audit
        self.recipes = recipes or {}
        self.ctx = ctx

    def run(self, action: Action, **ctx) -> ExecResult:
        merged = {**self.ctx, **ctx}
        merged.pop("target", None)
        cmd = eta_command(action, target=self.target, recipes=self.recipes, **merged)
        if self.audit:
            self.audit.record(mode="dry_run", action=action.action_type.value, command=cmd, executed=False)
        return ExecResult(tool=ETA_TOOL[action.action_type], output="", executed=False)


class LiveExecutor:
    """Phase-2 sandbox executor. FULLY GATED: every command goes through `AuthorizationGate.require`
    (env confirmation string + confirmed_isolated + lab-target + allow-list + destructive denylist)
    before it can run. Without authorization every call raises — nothing executes.

    It runs the rendered η command with subprocess (NO shell, tokenised args), a hard timeout, and
    captures stdout/stderr, recording everything to the audit log. The operator is still responsible
    for the surrounding isolation (network-confined, snapshot/restore) — this class only refuses to
    run anywhere that is not a verified lab target.
    """

    def __init__(self, target: str, *, confirmed_isolated: bool = False,
                 audit: AuditLog | None = None, timeout: int = 120,
                 gate: AuthorizationGate | None = None, recipes: dict | None = None, **ctx):
        self.target = target
        self.gate = gate or AuthorizationGate(confirmed_isolated=confirmed_isolated)
        self.audit = audit or AuditLog(Path("outputs") / "stage2_engagement_audit.jsonl")
        self.timeout = timeout
        self.recipes = recipes or {}
        self.ctx = ctx

    def run(self, action: Action, **ctx) -> ExecResult:
        merged = {**self.ctx, **ctx}
        merged.pop("target", None)
        command = eta_command(action, target=self.target, recipes=self.recipes, **merged)
        # comment-only / empty template (privesc/post-exploit/stop placeholders with no per-target
        # recipe) — nothing concrete to run, treat as a no-op rather than feeding "#" to the gate.
        if not command.strip() or command.lstrip().startswith("#"):
            self.audit.record(mode="live_noop", action=action.action_type.value, command=command, executed=False)
            return ExecResult(tool=ETA_TOOL[action.action_type], output="", executed=False)
        # deny-by-default: raises unless authorized AND the command passes the safety gate
        self.gate.require(self.target, command)
        self.audit.record(mode="live", action=action.action_type.value, command=command, executed=True)
        t0 = time.monotonic()
        try:
            # capture BYTES (not text=True): real tool output is not guaranteed to be the Windows
            # locale encoding (GBK here), so decode UTF-8 with replacement ourselves.
            proc = subprocess.run(shlex.split(command), capture_output=True, timeout=self.timeout)
            out = (proc.stdout or b"").decode("utf-8", "replace") + (proc.stderr or b"").decode("utf-8", "replace")
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            out, rc = "[eta] command timed out", 124
        except FileNotFoundError:
            # tool not installed on the runner (e.g. no nmap/gobuster) — degrade gracefully, do not crash
            out, rc = "[eta] tool not installed on runner", 127
        self.audit.record(mode="live_result", action=action.action_type.value,
                          returncode=rc, duration_s=round(time.monotonic() - t0, 2),
                          output_head=out[:500])
        return ExecResult(tool=ETA_TOOL[action.action_type], output=out, executed=True)
