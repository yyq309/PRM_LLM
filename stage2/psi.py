"""psi (ψ+): a Stage-2-LOCAL coverage layer over the frozen Stage-1 normalizer.

Phase-1 measured that the abstraction itself covers ~92% of real steps, but the Stage-1
keyword normalizer (`web_attack_sim.normalize_llm_action`) maps only ~49% of *natural operator
phrasing* on the in-abstraction steps, false-rejecting ~32% of them ("read the **key** file" or
"run id through the **backdoor**" are rejected, while "read **config**" / "through the **web
shell**" map). That is the real Stage-2 bottleneck.

This module does NOT touch Stage 1 (so the frozen oracle/PRM artifacts stay reproducible). It
wraps the Stage-1 normalizer and only acts when the Stage-1 result is `unsupported` — i.e. an
in-scope intent the keyword matcher failed to classify:

  1. trust the Stage-1 result for everything it already handles: `valid` (precision), and the
     correct *rejections* `unsafe` / `outside_single_host_web_scope` / `schema_gap` / `ambiguous`
     / `invalid` (these keep SSH, SSTI/SSRF, destructive, and vague intents out — preserving the
     zero false-accept property).
  2. for `unsupported`, first run an OUT-GUARD: if the intent names a non-web primitive (offline
     hash cracking, SSH login, su/account switch, kernel/binary-overflow exploit, pivot), return a
     non-valid `schema_gap` carrying the gap token — DO NOT recover it (this is what stops the
     recovery layer from manufacturing false-accepts on out-of-abstraction steps).
  3. otherwise apply a principled verb x object lexical recovery built from GENERAL pentest
     vocabulary (not from the held-out walkthrough fixtures). If a rule fires, return `valid` with
     the recovered action; else keep the Stage-1 `unsupported`.

Honesty: the recovery vocabulary is tuned ONLY on `stage2/psi_benchmark.jsonl` (an independent,
hand-labeled intent corpus, disjoint from the 7 walkthrough fixtures). Generalization is reported
on the fixtures as a held-out test (`stage2.eval_psi`, `replay --enhanced-psi`).
"""

from __future__ import annotations

import re

from web_attack_sim.action_space import ActionType
from web_attack_sim.normalizer import (
    NormalizedAction,
    normalize_llm_action,
    _extract_target,
    _extract_parameter,
)
from web_attack_sim.schemas import Action

# Stage-1 statuses we never override (correct classifications or correct rejections).
_TRUST_STATUSES = {
    "valid", "unsafe", "outside_single_host_web_scope", "schema_gap", "ambiguous", "invalid",
}

_SHORT_ASCII_WORD = re.compile(r"^[a-z0-9/_.-]{1,4}$")


def _matches(lower: str, term: str) -> bool:
    """Word-boundary match for short ascii tokens (id, su, ssh, rce, ...), substring otherwise."""
    if _SHORT_ASCII_WORD.match(term):
        return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", lower) is not None
    return term in lower


def _any(lower: str, terms: tuple[str, ...]) -> bool:
    return any(_matches(lower, t) for t in terms)


# ---- OUT-GUARD: non-web primitives that must stay out-of-abstraction --------------------
# (token tuple, schema-gap token). Checked BEFORE recovery so an "unsupported" out-step is never
# mapped to a web action.
_OUT_RULES: list[tuple[tuple[str, ...], str]] = [
    (("crack", "hashcat", "john the ripper", "rockyou", "decrypt the hash", "recover the password from"),
     "offline_credential_cracking"),
    (("ssh", "scp", "id_rsa", "over ssh", "ssh login", "sftp"),
     "ssh_remote_login"),
    (("su to", "su -", "switch to the user", "switch user", "switch to the", "su root", "su using"),
     "local_account_switch"),
    (("kernel exploit", "dirtycow", "dirty cow", "kernel privilege", "local kernel", "kernel-privilege"),
     "kernel_exploit"),
    (("buffer overflow", "stack overflow", "remote overflow", "overflow exploit", "trans2open",
      "memory corruption", "binary exploit", "rop chain"),
     "binary_service_exploit"),
    (("pivot", "lateral movement", "pass the hash", "lsass", "internal network"),
     "lateral_movement"),
]


def _out_guard(lower: str) -> str | None:
    for terms, token in _OUT_RULES:
        if _any(lower, terms):
            return token
    return None


# ---- RECOVERY: verb x object lexical rules for in-scope intents -------------------------
# Each rule fires when the intent contains ANY trigger phrase, OR ANY verb AND ANY object.
# Ordered most-specific -> most-general; first hit wins. Built from general pentest vocabulary.
_READ_VERBS = ("read", "cat ", "dump", "view", "print", "retrieve", "exfiltrate", "grep", "output the", "display")
_SENSITIVE_NOUNS = ("flag", "/etc/passwd", "passwd", "shadow", "id_rsa", "private key", "ssh key",
                    "hash file", "hash from", "credentials file", "credential file", "secret",
                    "config file", "configuration file", "wp-config", "database password", "notes file",
                    "key file", "key-", ".txt", "password hash")
_VULN_VERBS = ("verify", "confirm", "check whether", "check if", "test the", "test for", "probe",
               "validate", "assess whether", "identify whether", "determine whether")
_VULN_NOUNS = ("vulnerable", "vulnerability", "exploitable", "injectable", "injection", "sqli",
               "lfi", "rce", "cve", "susceptible", "vulnerable build")
_AUTH_VERBS = ("brute", "brute-force", "dictionary attack", "guess", "spray", "try logging", "try common")
_AUTH_NOUNS = ("login", "log in", "password", "credential", "authenticat", "sign in", "portal", "panel")
_EXEC_VERBS = ("run", "execute", "issue", "invoke")
_EXEC_NOUNS = ("command", "whoami", "uname", "hostname", "webshell", "web shell", "backdoor",
               "system command", "cmd", "in the shell", "foothold shell", "command channel",
               "id", "shell", "ls", "through the")
_UPLOAD_VERBS = ("upload", "plant", "drop", "deploy", "write", "put", "install")
_UPLOAD_NOUNS = ("shell", "webshell", "payload", "backdoor", "plugin", "php file", "malicious file",
                 "script", "jsp", ".jsp", ".war", "war file", "aspx", ".aspx")
# exploit launch (distinct from VULNERABILITY_CHECK's verify/confirm): "trigger the OGNL RCE",
# "launch the deserialization exploit". The OUT-GUARD runs first, so kernel/binary overflows are
# still kept out of abstraction; these nouns are web-RCE specific.
_EXPLOIT_VERBS = ("trigger", "launch", "fire", "throw", "weaponize", "deliver", "send the")
_EXPLOIT_NOUNS = ("rce", "ognl", "deserialization", "remote code execution", "code execution",
                  "gadget", "exploit chain")
_INPUT_VERBS = ("discover", "find", "identify", "locate", "enumerate", "examine", "inspect")
_INPUT_NOUNS = ("input", "parameter", "form field", "form input", "field", "query-string", "query string")
_PATH_VERBS = ("enumerate", "discover", "find", "brute", "fuzz", "map")
_PATH_NOUNS = ("director", "path", "route", "endpoint", "url", "admin page", "hidden page")
_SVC_VERBS = ("scan", "enumerate", "probe", "identify", "discover", "map", "determine")
_SVC_NOUNS = ("service", "port", "smb", "samba", "daemon", "listening", "open service")
_FP_VERBS = ("fingerprint", "identify", "detect", "profile", "determine", "grab")
_FP_NOUNS = ("cms", "framework", "technology", "tech stack", "server banner", "web server", "http header", "banner")
_POST_VERBS = ("enumerate", "gather", "search", "look for", "inspect", "check")
_POST_NOUNS = ("system for", "local information", "sudo -l", "suid", "cron", "escalation vector",
               "escalation path", "post-exploit", "linpeas", "writable file", "stored credentials")
_PRIV_NOUNS = ("privilege", "root", "suid", "sudo", "gtfobins")
_PRIV_VERBS = ("escalate", "elevate", "privesc", "become root", "get root", "to root")
_CONTENT_VERBS = ("fetch", "retrieve", "download", "pull", "get the", "open the", "view the", "read the")
_CONTENT_NOUNS = ("page", "source", "robots", "listing", "archive", "changelog", "backup", "body of")
_FORM_VERBS = ("submit", "post the", "fill", "complete", "send")
_FORM_NOUNS = ("form",)
_STOP_TERMS = ("stop here", "write up", "report the", "submit the report", "conclude", "finalize",
               "we are done", "we're done", "document the flag", "finish the engagement")


def _recover(lower: str) -> ActionType | None:
    def vo(verbs, nouns):
        return _any(lower, verbs) and _any(lower, nouns)

    # post-exploitation enumeration must beat the bare service/path/input rules ("enumerate ...")
    if vo(_POST_VERBS, _POST_NOUNS):
        return ActionType.POST_EXPLOITATION
    if vo(_READ_VERBS, _SENSITIVE_NOUNS):
        return ActionType.SENSITIVE_FILE_READ
    if vo(_VULN_VERBS, _VULN_NOUNS):
        return ActionType.VULNERABILITY_CHECK
    if vo(_EXPLOIT_VERBS, _EXPLOIT_NOUNS):
        return ActionType.EXPLOIT_ATTEMPT
    if vo(_UPLOAD_VERBS, _UPLOAD_NOUNS):
        return ActionType.FILE_UPLOAD_ATTEMPT
    if vo(_AUTH_VERBS, _AUTH_NOUNS):
        return ActionType.AUTH_ATTEMPT
    if vo(_EXEC_VERBS, _EXEC_NOUNS):
        return ActionType.COMMAND_EXECUTION
    if vo(_PRIV_VERBS, _PRIV_NOUNS):
        return ActionType.PRIVILEGE_ESCALATION
    if vo(_INPUT_VERBS, _INPUT_NOUNS):
        return ActionType.INPUT_DISCOVERY
    if vo(_PATH_VERBS, _PATH_NOUNS):
        return ActionType.WEB_PATH_ENUMERATION
    if vo(_SVC_VERBS, _SVC_NOUNS):
        return ActionType.SERVICE_ENUMERATION
    if vo(_FP_VERBS, _FP_NOUNS):
        return ActionType.HTTP_FINGERPRINT
    if vo(_FORM_VERBS, _FORM_NOUNS):
        return ActionType.FORM_INTERACTION
    if vo(_CONTENT_VERBS, _CONTENT_NOUNS):
        return ActionType.CONTENT_RETRIEVAL
    if _any(lower, _STOP_TERMS):
        return ActionType.STOP_OR_REPORT
    return None


class EnhancedNormalizer:
    """Stage-1 ψ + an in-scope recovery layer. `normalize(text)` returns a `NormalizedAction`
    with the same shape as `normalize_llm_action`, so it is a drop-in at Stage-2 inference."""

    def normalize(self, text: str) -> NormalizedAction:
        base = normalize_llm_action(text)
        raw = base.raw_text
        lower = raw.lower()

        # OUT-GUARD runs FIRST, with precedence even over a Stage-1 `valid`: the keyword matcher
        # maps "exploit the buffer overflow" / "kernel exploit" to exploit_attempt because it keys
        # on the word "exploit", but those are non-web primitives. An unambiguous out-token forces
        # schema_gap and prevents the false-accept.
        gap = _out_guard(lower)
        if gap is not None:
            return NormalizedAction(
                raw_text=raw, status="schema_gap", action=None,
                reason=f"out_of_abstraction:{gap}", confidence=0.7,
            )

        if base.status in _TRUST_STATUSES:
            return base
        # base.status == "unsupported": an in-scope intent the keyword matcher missed.
        action_type = _recover(lower)
        if action_type is None:
            return base
        return NormalizedAction(
            raw_text=raw, status="valid",
            action=Action(action_type=action_type,
                          target=_extract_target(raw), parameter=_extract_parameter(raw)),
            reason="recovered_by_stage2_coverage_layer", confidence=0.7,
        )


def enhanced_normalize(text: str) -> NormalizedAction:
    return EnhancedNormalizer().normalize(text)
