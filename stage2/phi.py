"""phi (φ): real tool output -> AbstractWebState.

φ ingests the raw stdout of the real recon/exploit tools a pentester runs against a single
host and accumulates them into the abstract `Observation` the Stage-1 oracle + PRM consume.
It is intentionally *heuristic and lossy* — real tool output is messy and φ will not
reconstruct every field. Phase 1 MEASURES that loss (`phi_mapping_accuracy`) rather than
hiding it.

φ does NOT execute anything. It only parses text that was captured elsewhere (offline
replay) or — in a future authorized Phase 2 — produced by `stage2.eta` under a sandbox.

Each `parse_*` returns a dict of the abstract facts it extracted from one tool output, AND
mutates the accumulator. The dict is what `stage2.replay` compares against the per-step
hand-labeled `reference_state_after` to score mapping accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from web_attack_sim.schemas import Observation


# ---- tool dispatch keys -------------------------------------------------------------
# A fixture step names the tool that produced its output; φ routes on this key. Aliases let
# fixtures use whatever real tool name fits the box (gobuster/ffuf/dirb are one route).
TOOL_ALIASES = {
    "nmap": "nmap",
    "rustscan": "nmap",
    "masscan": "nmap",
    "whatweb": "whatweb",
    "wappalyzer": "whatweb",
    "nikto": "whatweb",
    "httpx": "whatweb",
    "gobuster": "dirscan",
    "dirb": "dirscan",
    "dirsearch": "dirscan",
    "ffuf": "dirscan",
    "feroxbuster": "dirscan",
    "wfuzz": "dirscan",
    "curl": "http",
    "wget": "http",
    "http": "http",
    "browser": "http",
    "view-source": "http",
    "sqlmap": "sqlmap",
    "exploit": "exploit",
    "metasploit": "exploit",
    "msfconsole": "exploit",
    "msf": "exploit",
    "searchsploit": "exploit",
    "python": "exploit",
    "login": "login",
    "hydra": "login",  # credential submission result (in-web-form only)
    "upload": "upload",
    "shell": "command",
    "command": "command",
    "nc": "command",
    "sudo": "privesc",
    "privesc": "privesc",
    "find-suid": "privesc",
    "cat": "fileread",
    "fileread": "fileread",
}


@dataclass
class AbstractStateAccumulator:
    """Mutable mirror of the env RuntimeState, populated from real tool output by φ."""

    open_services: set[str] = field(default_factory=set)
    tech_stack: set[str] = field(default_factory=set)
    discovered_paths: set[str] = field(default_factory=lambda: {"/"})
    known_forms: set[str] = field(default_factory=set)
    known_parameters: set[str] = field(default_factory=set)
    suspected_vulnerabilities: set[str] = field(default_factory=set)
    verified_vulnerabilities: set[str] = field(default_factory=set)
    credentials: set[str] = field(default_factory=set)
    auth_state: str = "anonymous"
    shell_state: str = "none"
    privilege_level: str = "none"
    read_files: set[str] = field(default_factory=set)
    # φ telemetry: tool outputs it could not extract anything abstract from.
    unparsed_outputs: list[str] = field(default_factory=list)

    def observation(self, remaining_budget: int = 12) -> Observation:
        return Observation(
            target_known=True,
            service_known=bool(self.open_services),
            open_services=sorted(self.open_services),
            base_url_known=True,
            discovered_paths=sorted(self.discovered_paths),
            known_forms=sorted(self.known_forms),
            known_parameters=sorted(self.known_parameters),
            tech_stack=sorted(self.tech_stack),
            suspected_vulnerabilities=sorted(self.suspected_vulnerabilities),
            verified_vulnerabilities=sorted(self.verified_vulnerabilities),
            credentials=sorted(self.credentials),
            auth_state=self.auth_state,
            shell_state=self.shell_state,
            privilege_level=self.privilege_level,
            read_files=sorted(self.read_files),
            failed_actions=[],
            failed_branches={},
            remaining_budget=remaining_budget,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "open_services": sorted(self.open_services),
            "tech_stack": sorted(self.tech_stack),
            "discovered_paths": sorted(self.discovered_paths),
            "known_forms": sorted(self.known_forms),
            "known_parameters": sorted(self.known_parameters),
            "verified_vulnerabilities": sorted(self.verified_vulnerabilities),
            "credentials": sorted(self.credentials),
            "auth_state": self.auth_state,
            "shell_state": self.shell_state,
            "privilege_level": self.privilege_level,
            "read_files": sorted(self.read_files),
        }


# ---- regexes (module-level, compiled once) ------------------------------------------
_NMAP_PORT_RE = re.compile(r"^(\d{1,5})/(tcp|udp)\s+open\s+([A-Za-z0-9._-]+)(?:\s+(.*))?$", re.MULTILINE)
_DIR_STATUS_RE = re.compile(r"(/[A-Za-z0-9._~%/-]+)\b[^\n]*?(?:Status:\s*|\s)(\d{3})\b")
_FFUF_RE = re.compile(r"(/[A-Za-z0-9._~%/-]+)\s.*?\[Status:\s*(\d{3})", re.IGNORECASE)
_PARAM_INPUT_RE = re.compile(r"<input[^>]*\bname\s*=\s*[\"']([A-Za-z_][\w-]*)[\"']", re.IGNORECASE)
_QUERY_PARAM_RE = re.compile(r"[?&]([A-Za-z_][\w-]*)=")
_TECH_TOKENS = {
    "apache": "apache", "nginx": "nginx", "iis": "iis", "php": "php", "mysql": "mysql",
    "mariadb": "mysql", "wordpress": "wordpress", "drupal": "drupal", "joomla": "joomla",
    "tomcat": "tomcat", "python": "python", "node": "nodejs", "express": "nodejs",
    "openssl": "openssl", "phpmyadmin": "phpmyadmin",
    # web frameworks — the CVE-identifying detail a real LLM proposer needs (kept observable, not
    # the hidden task ground truth). Surfaced to the proposer context, NOT used as a PRM feature.
    "thinkphp": "thinkphp", "laravel": "laravel", "django": "django", "flask": "flask",
    "spring": "spring", "struts": "struts", "jenkins": "jenkins", "gitlab": "gitlab",
    "phpmailer": "phpmailer", "werkzeug": "python",
}
# Lines that are HTML/CSS markup, not credential leaks — skipped before the loose user:pass scan.
_CSS_HTML_LINE = re.compile(r"[<>{}]|:\s*(hover|both|fixed|center|middle|none|block|auto)\b|"
                            r"\b(font-size|table-layout|word-wrap|last-child|first-child|border|margin|padding)\b|"
                            r"\d(px|em|rem|pt)\b|;\s*$|style=", re.IGNORECASE)
# any /etc/passwd line: name:x:uid:gid:... — matches root:x:0:0 AND _apt:x:100:65534:... (PHP exec()
# returns only the LAST line of a multi-line read, so a root-only check misses a genuine file read).
_PASSWD_LINE = re.compile(r"[a-z_][\w.-]*:[^:\n]*:\d+:\d+:", re.IGNORECASE)
_CRED_PATTERNS = [
    re.compile(r"DB_PASSWORD['\"]?\s*[,:=]\s*['\"]([^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"\bpassword\s*[:=]\s*['\"]?([A-Za-z0-9!@#$%^&*_.-]{3,})['\"]?", re.IGNORECASE),
    re.compile(r"\b([A-Za-z0-9_.-]+):([A-Za-z0-9!@#$%^&*_.-]{3,})\b"),  # user:pass
    re.compile(r"\busername\s*[:=]\s*['\"]?([A-Za-z0-9_.-]+)", re.IGNORECASE),
]


class Phi:
    """Stateful φ adapter. `ingest()` routes one tool output to the right parser."""

    def __init__(self, remaining_budget: int = 12) -> None:
        self.state = AbstractStateAccumulator()
        self.remaining_budget = remaining_budget

    # public API -----------------------------------------------------------------
    def ingest(self, tool: str, output: str, *, target: str | None = None,
               meta: dict | None = None) -> dict[str, object]:
        route = TOOL_ALIASES.get((tool or "").strip().lower())
        if route is None:
            self.state.unparsed_outputs.append(f"{tool}: unknown tool")
            return {"_unrouted": True, "tool": tool}
        parser = getattr(self, f"_parse_{route}")
        extracted = parser(output or "", target=target, meta=meta or {})
        # tool-agnostic content credit: strong outcome signals in ANY output upgrade state, so a result
        # returned via the "wrong" action (e.g. /etc/passwd via an exploit action, a flag via a content
        # fetch, uid= via vuln_check) is still recorded — kills a large class of false no-progress steps.
        content = self._content_credit(output or "")
        if content:
            extracted = {**extracted, **content}
        if not extracted:
            self.state.unparsed_outputs.append(f"{tool}: no abstract facts extracted")
        return extracted

    def _content_credit(self, output: str) -> dict[str, object]:
        """Monotone, tool-independent credit from raw output content. Only UPGRADES state; never
        downgrades. Conservative signals (id-format uid=, passwd lines, flag{}) so recon output does
        not false-positive."""
        out: dict[str, object] = {}
        low = output.lower()
        m = re.search(r"uid=(\d+)\([a-z0-9_.-]+\)", low)  # real RCE output (id/whoami) == code exec
        if m:
            if self.state.shell_state != "command_execution":
                self.state.shell_state = "command_execution"
                out["shell_state"] = "command_execution"
            if m.group(1) == "0":
                if self.state.privilege_level != "root":
                    self.state.privilege_level = "root"; out["privilege_level"] = "root"
            elif self.state.privilege_level == "none":
                self.state.privilege_level = "web_user"
        if _PASSWD_LINE.search(output) and "etc_passwd" not in self.state.read_files:
            self.state.read_files.add("etc_passwd"); out["read_files"] = sorted(self.state.read_files)
        if re.search(r"(?i)flag\{[^}\n]{1,80}\}", output) and "flag" not in self.state.read_files:
            self.state.read_files.add("flag"); out["read_files"] = sorted(self.state.read_files)
        return out

    def observation(self) -> Observation:
        return self.state.observation(self.remaining_budget)

    # parsers --------------------------------------------------------------------
    def _parse_nmap(self, output: str, *, target=None, meta=None) -> dict[str, object]:
        services, techs = [], []
        for port, _proto, service, version in _NMAP_PORT_RE.findall(output):
            svc = service.lower()
            label = f"{'http' if svc in {'http', 'http-proxy'} else ('https' if svc == 'https' or svc == 'ssl/http' else svc)}:{port}"
            services.append(label)
            self.state.open_services.add(label)
            for token, norm in _TECH_TOKENS.items():
                if token in (version or "").lower():
                    techs.append(norm)
                    self.state.tech_stack.add(norm)
        return {"open_services": sorted(set(services)), "tech_stack": sorted(set(techs))} if services else {}

    def _parse_whatweb(self, output: str, *, target=None, meta=None) -> dict[str, object]:
        techs = []
        low = output.lower()
        for token, norm in _TECH_TOKENS.items():
            if token in low:
                techs.append(norm)
                self.state.tech_stack.add(norm)
        return {"tech_stack": sorted(set(techs))} if techs else {}

    def _parse_dirscan(self, output: str, *, target=None, meta=None) -> dict[str, object]:
        found = set()
        for path, status in _DIR_STATUS_RE.findall(output) + _FFUF_RE.findall(output):
            if status.startswith(("2", "3")) and status not in {"302"} or status in {"200", "301", "401", "403"}:
                found.add(path.rstrip("/") or "/")
        # plain path listings (one path per line, e.g. dirb summary)
        for line in output.splitlines():
            m = re.match(r"\s*(?:==>\s*DIRECTORY:\s*)?(https?://[^/]+)?(/[A-Za-z0-9._~%/-]+)\s*$", line)
            if m and not m.group(2).endswith((".js", ".css", ".png", ".jpg", ".ico")):
                found.add(m.group(2).rstrip("/") or "/")
        self.state.discovered_paths.update(found)
        return {"discovered_paths": sorted(found)} if found else {}

    def _parse_whatweb_alias(self, *a, **k):  # pragma: no cover - alias safety
        return self._parse_whatweb(*a, **k)

    def _parse_http(self, output: str, *, target=None, meta=None) -> dict[str, object]:
        extracted: dict[str, object] = {}
        # credentials leaked in returned content (config/backup files)
        creds = self._scan_credentials(output)
        if creds:
            self.state.credentials.update(creds)
            extracted["credentials"] = sorted(creds)
        # forms / parameters visible in returned HTML
        inputs = {m for m in _PARAM_INPUT_RE.findall(output)}
        path = target or (meta or {}).get("path")
        if inputs and path:
            self.state.known_forms.add(f"form:{path}")
            for name in inputs:
                self.state.known_parameters.add(f"{path}?{name}")
            extracted["known_forms"] = [f"form:{path}"]
            extracted["known_parameters"] = sorted(f"{path}?{n}" for n in inputs)
        # explicit file read marker
        fid = (meta or {}).get("file_id")
        if fid:
            self.state.read_files.add(fid)
            extracted["read_files"] = [fid]
        return extracted

    def _parse_login(self, output: str, *, target=None, meta=None) -> dict[str, object]:
        low = output.lower()
        success = any(s in low for s in (
            "set-cookie", "location: /admin", "welcome", "dashboard", "302 found",
            "logged in", "authentication successful",
        )) and not any(f in low for f in ("invalid", "incorrect", "failed", "denied"))
        if not success:
            return {}
        role = (meta or {}).get("role", "admin")
        self.state.auth_state = role
        return {"auth_state": role}

    def _parse_sqlmap(self, output: str, *, target=None, meta=None) -> dict[str, object]:
        extracted: dict[str, object] = {}
        low = output.lower()
        _sqli_err = ("xpath syntax error", "you have an error in your sql", "sql syntax",
                     "updatexml", "extractvalue", "warning: mysql", "mysql_fetch", "sqlstate")
        if ("is vulnerable" in low or "injectable" in low or ("parameter '" in low and "vulnerab" in low)
                or any(s in low for s in _sqli_err)):
            vid = (meta or {}).get("vuln_id", "sqli")
            self.state.verified_vulnerabilities.add(vid)
            self.state.suspected_vulnerabilities.add(vid)
            extracted["verified_vulnerabilities"] = [vid]
        creds = self._scan_credentials(output)
        # sqlmap dumped table rows: "admin | 5f4dcc3b..." style
        for m in re.findall(r"\|\s*([A-Za-z0-9_.-]+)\s*\|\s*([A-Za-z0-9$./=+]{6,})\s*\|", output):
            creds.add(f"{m[0]}:{m[1][:12]}")
        if creds:
            self.state.credentials.update(creds)
            extracted["credentials"] = sorted(creds)
        if self._detect_shell(low, meta):
            extracted["shell_state"] = self.state.shell_state
        return extracted

    def _parse_exploit(self, output: str, *, target=None, meta=None) -> dict[str, object]:
        """Generic exploit launcher (metasploit / custom PoC). Marks the vuln verified, captures
        any leaked credentials, and flips shell_state when the tool reports a foothold."""
        extracted: dict[str, object] = {}
        low = output.lower()
        vid = (meta or {}).get("vuln_id")
        if vid and any(s in low for s in ("vulnerable", "exploit", "injectable", "session", "shell", "meterpreter")):
            self.state.verified_vulnerabilities.add(vid)
            self.state.suspected_vulnerabilities.add(vid)
            extracted["verified_vulnerabilities"] = [vid]
        creds = self._scan_credentials(output)
        if creds:
            self.state.credentials.update(creds)
            extracted["credentials"] = sorted(creds)
        if self._detect_shell(low, meta):
            extracted["shell_state"] = self.state.shell_state
        return extracted

    def _detect_shell(self, low: str, meta: dict | None) -> bool:
        """A foothold was obtained: explicit fixture hint OR a tool-reported shell signal."""
        signal = any(s in low for s in (
            "web shell", "webshell", "meterpreter session", "command execution confirmed",
            "command shell session", "spawned", "shell opened", "reverse shell",
        )) or re.search(r"uid=\d+\(", low) is not None  # real RCE output (id/whoami) == code execution
        if (meta or {}).get("yields_shell") or signal:
            if self.state.shell_state == "none":
                self.state.shell_state = "webshell"
            if self.state.privilege_level == "none":
                self.state.privilege_level = "web_user"
            return True
        return False

    def _parse_upload(self, output: str, *, target=None, meta=None) -> dict[str, object]:
        low = output.lower()
        if any(s in low for s in ("uploaded", "200 ok", "file written", "shell.php", "moved to",
                                  "ok - deployed", "deployed application")):
            if (meta or {}).get("yields_shell", True):
                self.state.shell_state = "webshell"
                if self.state.privilege_level == "none":
                    self.state.privilege_level = "web_user"
                return {"shell_state": "webshell"}
        return {}

    def _parse_command(self, output: str, *, target=None, meta=None) -> dict[str, object]:
        low = output.strip().lower()
        if not low:
            return {}
        # whoami / id output
        user = None
        m = re.search(r"uid=\d+\(([A-Za-z0-9_-]+)\)", low) or re.match(r"^([a-z0-9_-]+)\s*$", low)
        if m:
            user = m.group(1)
        if user is None and low in {"www-data", "apache", "nginx", "root", "nobody"}:
            user = low
        if user is None:
            return {}
        if self.state.shell_state in {"none"}:
            self.state.shell_state = "command_execution"
        elif self.state.shell_state == "webshell":
            self.state.shell_state = "command_execution"
        if user == "root":
            self.state.privilege_level = "root"
        elif self.state.privilege_level == "none":
            self.state.privilege_level = "web_user"
        return {"shell_state": self.state.shell_state, "privilege_level": self.state.privilege_level, "user": user}

    def _parse_privesc(self, output: str, *, target=None, meta=None) -> dict[str, object]:
        low = output.lower()
        rooted = any(s in low for s in ("uid=0(root)", "# whoami\nroot", "\nroot\n")) or low.strip() == "root"
        # sudo -l revealing a usable escalation, then confirmed root
        if rooted or (meta or {}).get("rooted"):
            self.state.privilege_level = "root"
            if self.state.shell_state == "none":
                self.state.shell_state = "command_execution"
            return {"privilege_level": "root"}
        return {}

    def _parse_fileread(self, output: str, *, target=None, meta=None) -> dict[str, object]:
        fid = (meta or {}).get("file_id")
        extracted: dict[str, object] = {}
        if fid:
            self.state.read_files.add(fid)
            extracted["read_files"] = [fid]
        # flag content read directly
        if re.search(r"(FLAG\{|flag\{|[0-9a-f]{32})", output) and not fid:
            self.state.read_files.add("flag")
            extracted["read_files"] = ["flag"]
        # /etc/passwd reveals users (credentials-adjacent), record file read. Match ANY passwd line
        # (name:x:uid:gid:...), not just root — a PHP exec() RCE returns only the LAST line of output
        # (e.g. `_apt:x:100:65534:...` on Drupalgeddon2), so a root-only check misses a genuine read.
        if not fid and _PASSWD_LINE.search(output):
            self.state.read_files.add("etc_passwd")
            extracted.setdefault("read_files", []).append("etc_passwd")
        return extracted

    # helpers --------------------------------------------------------------------
    def _scan_credentials(self, text: str) -> set[str]:
        creds: set[str] = set()
        # Real tool output is messy: HTML error pages / CSS blocks have `prop:value` pairs (a:hover,
        # font-size:14px) that the loose user:pass regex mis-reads as credentials. Skip lines that
        # look like CSS/HTML markup before credential scanning (live-target robustness, found on the
        # ThinkPHP error page). Credential leaks (config dumps, DB rows) are not on markup lines.
        for line in text.splitlines():
            if _CSS_HTML_LINE.search(line):
                continue
            for pat in _CRED_PATTERNS:
                for m in pat.findall(line):
                    token = m if isinstance(m, str) else ":".join(p for p in m if p)
                    token = token.strip()
                    if token and not token.startswith(("http", "//", "www")) and len(token) <= 64:
                        creds.add(f"cred:{token[:32]}")
        return creds
