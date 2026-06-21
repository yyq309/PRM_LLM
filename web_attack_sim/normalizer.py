from __future__ import annotations

from dataclasses import dataclass
import re

from .action_space import ActionType
from .schemas import Action


PATH_RE = re.compile(r"(/[A-Za-z0-9._~!$&'()*+,;=:@%/-]+(?:\?[A-Za-z0-9_=&.%:-]+)?)")
PARAM_RE = re.compile(r"(?:\bparam(?:eter)?\b|参数)\s*[:=]?\s*([A-Za-z_][\w-]*)", re.IGNORECASE)


@dataclass(frozen=True)
class NormalizedAction:
    raw_text: str
    status: str
    action: Action | None
    reason: str
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_text": self.raw_text,
            "status": self.status,
            "action_type": self.action.action_type.value if self.action else None,
            "target": self.action.target if self.action else None,
            "parameter": self.action.parameter if self.action else None,
            "reason": self.reason,
            "confidence": self.confidence,
        }


def normalize_llm_action(text: str) -> NormalizedAction:
    """Map LLM-style Web pentest intent into the abstract Web action space."""
    raw = " ".join(text.strip().split())
    lower = raw.lower()
    target = _extract_target(raw)
    parameter = _extract_parameter(raw)

    if not raw:
        return _non_valid(raw, "invalid", "empty_action", 1.0)

    if _has_any(lower, UNSAFE_TERMS):
        return _non_valid(raw, "unsafe", "out_of_scope_or_destructive_action", 0.95)

    if _has_any(lower, SCHEMA_GAP_TERMS):
        return _non_valid(raw, "schema_gap", "unsupported_in_scope", 0.7)

    if _has_any(lower, OUTSIDE_SCOPE_TERMS):
        return _non_valid(raw, "outside_single_host_web_scope", "outside_single_host_web_scope", 0.9)

    if len(raw) < 6 or _has_any(lower, AMBIGUOUS_TERMS):
        return _non_valid(raw, "ambiguous", "action_intent_is_too_vague", 0.75)

    action_type = _classify(lower)
    if action_type is None:
        return _non_valid(raw, "unsupported", "no_matching_abstract_web_action", 0.55)

    return NormalizedAction(
        raw_text=raw,
        status="valid",
        action=Action(action_type=action_type, target=target, parameter=parameter),
        reason="matched_abstract_web_action",
        confidence=0.85,
    )


def _classify(lower: str) -> ActionType | None:
    path_intent_terms = (
        "enumerate web directories",
        "enumerate directories",
        "enumerate directory",
        "enumerate path",
        "discover path",
        "discover /",
        "find hidden",
        "目录枚举",
        "目录扫描",
        "路径枚举",
        "路径扫描",
        "发现路径",
        "发现目录",
    )
    input_terms = ("parameter", "参数", "input", "form", "表单", "输入点", "输入框")
    if _has_any(lower, path_intent_terms) and not _has_any(lower, input_terms):
        return ActionType.WEB_PATH_ENUMERATION

    input_intent_terms = (
        "discover input",
        "discover parameter",
        "discover parameters",
        "find parameter",
        "find parameters",
        "enumerate parameter",
        "enumerate parameters",
        "inspect form",
        "login form",
        "upload form",
        "input field",
        "input fields",
        "输入点",
        "参数",
        "表单",
    )
    vulnerability_intent_terms = (
        "verify",
        "check",
        "test",
        "sqli",
        "sql injection",
        "lfi",
        "rce",
        "漏洞验证",
        "验证漏洞",
        "测试注入",
    )
    if _has_any(lower, input_intent_terms) and not _has_any(lower, vulnerability_intent_terms):
        return ActionType.INPUT_DISCOVERY

    # post-exploitation must be caught before the bare "exploit" gate ("post-exploitation"
    # contains "exploit").
    if _has_any(lower, ("post exploitation", "post-exploitation", "后渗透", "collect loot", "收集战利品")):
        return ActionType.POST_EXPLOITATION

    if _has_any(lower, ("exploit", "利用漏洞")):
        return ActionType.EXPLOIT_ATTEMPT

    ordered_rules: list[tuple[ActionType, tuple[str, ...]]] = [
        (
            ActionType.SENSITIVE_FILE_READ,
            (
                "read flag",
                "flag file",
                "sensitive file",
                "read /etc/passwd",
                "读取 flag",
                "读取flag",
                "读取敏感",
                "读取文件",
                "读 flag",
            ),
        ),
        (
            ActionType.PRIVILEGE_ESCALATION,
            (
                "privilege escalation",
                "privesc",
                "escalate privilege",
                "escalate to root",
                "sudo",
                "suid",
                "提权",
                "root 权限",
                "提升权限",
            ),
        ),
        (
            ActionType.COMMAND_EXECUTION,
            (
                "run command",
                "execute command",
                "command execution",
                "whoami",
                "uname",
                "命令执行",
                "执行命令",
                "运行命令",
            ),
        ),
        (
            ActionType.FILE_UPLOAD_ATTEMPT,
            (
                "upload",
                "web shell",
                "webshell",
                "php shell",
                "上传",
                "文件上传",
                "上传 shell",
                "上传webshell",
            ),
        ),
        (
            ActionType.EXPLOIT_ATTEMPT,
            (
                "exploit",
                "dump credential",
                "dump password",
                "sqlmap --dump",
                "利用漏洞",
                "漏洞利用",
                "导出凭据",
                "dump 凭据",
                "dump 密码",
            ),
        ),
        (
            ActionType.VULNERABILITY_CHECK,
            (
                "verify sqli",
                "check sqli",
                "test sqli",
                "sql injection",
                "sql 注入",
                "sqli",
                "注入",
                "lfi",
                "rce",
                "vulnerability check",
                "漏洞验证",
                "验证漏洞",
                "测试注入",
                "验证注入",
                "检查漏洞",
            ),
        ),
        (
            ActionType.CONTENT_RETRIEVAL,
            (
                "download",
                "retrieve",
                "curl",
                "view source",
                "read page",
                "open /backup",
                "backup",
                "config",
                "读取页面",
                "下载",
                "查看源码",
                "查看备份",
                "读取备份",
            ),
        ),
        (
            ActionType.CREDENTIAL_USE,
            (
                "use credential",
                "known credential",
                "leaked credential",
                "leaked admin credential",
                "credential to login",
                "login with known",
                "use leaked",
                "使用凭据",
                "使用泄露凭据",
                "泄露凭据",
                "已知凭据",
                "用凭据登录",
            ),
        ),
        (
            ActionType.AUTH_ATTEMPT,
            (
                "default password",
                "weak password",
                "admin/admin",
                "admin admin",
                "try login",
                "login as",
                "brute force default",
                "默认口令",
                "弱口令",
                "尝试登录",
                "默认密码",
                "弱密码",
            ),
        ),
        (
            ActionType.INPUT_DISCOVERY,
            (
                "discover input",
                "discover parameter",
                "find parameter",
                "enumerate parameter",
                "find form",
                "inspect form",
                "login form",
                "upload form",
                "input field",
                "input fields",
                "表单",
                "参数",
                "输入点",
                "输入框",
                "发现参数",
                "枚举参数",
                "检查表单",
            ),
        ),
        (
            ActionType.FORM_INTERACTION,
            (
                "submit form",
                "interact with form",
                "post form",
                "fill form",
                "提交表单",
                "填写表单",
                "表单交互",
            ),
        ),
        (
            ActionType.WEB_PATH_ENUMERATION,
            (
                "dirsearch",
                "gobuster",
                "dirb",
                "enumerate path",
                "enumerate directory",
                "enumerate directories",
                "enumerate web directories",
                "discover path",
                "find hidden",
                "web path",
                "directory enumeration",
                "目录枚举",
                "目录扫描",
                "路径枚举",
                "路径扫描",
                "发现路径",
                "发现目录",
            ),
        ),
        (
            ActionType.HTTP_FINGERPRINT,
            (
                "fingerprint",
                "whatweb",
                "wappalyzer",
                "technology stack",
                "server banner",
                "http header",
                "技术栈",
                "指纹",
                "服务指纹",
                "识别框架",
            ),
        ),
        (
            ActionType.SERVICE_ENUMERATION,
            (
                "port scan",
                "nmap",
                "service enumeration",
                "open service",
                "scan port",
                "端口扫描",
                "服务枚举",
                "扫描端口",
                "开放服务",
            ),
        ),
        (
            ActionType.POST_EXPLOITATION,
            (
                "post exploitation",
                "post-exploitation",
                "collect loot",
                "enumerate system",
                "后渗透",
                "系统枚举",
                "收集战利品",
            ),
        ),
        (
            ActionType.STOP_OR_REPORT,
            (
                "stop",
                "report",
                "submit report",
                "finish",
                "停止",
                "报告",
                "提交",
                "结束",
            ),
        ),
    ]
    for action_type, terms in ordered_rules:
        if _has_any(lower, terms):
            return action_type
    return None


def _extract_target(raw: str) -> str | None:
    match = PATH_RE.search(raw)
    if not match:
        return None
    return match.group(1).rstrip(".,;)")


def _extract_parameter(raw: str) -> str | None:
    match = PARAM_RE.search(raw)
    if match:
        return match.group(1)
    if "?" in raw:
        _, _, query = raw.partition("?")
        name, _, _value = query.partition("=")
        if name:
            return name.strip(".,;)")
    return None


_SHORT_ASCII_WORD = re.compile(r"^[a-z0-9]{1,4}$")


def _term_matches(lower: str, term: str) -> bool:
    # Short ASCII keywords (rce, lfi, sqli, xxe, dos, ssh, ...) must match on word
    # boundaries, otherwise they fire inside common words ("rce" in "source"/"force",
    # "dos" in "windows"), corrupting the normalization. Phrases and CJK terms keep
    # plain substring matching.
    if _SHORT_ASCII_WORD.match(term):
        return re.search(r"\b" + re.escape(term) + r"\b", lower) is not None
    return term in lower


def _has_any(lower: str, terms: tuple[str, ...]) -> bool:
    return any(_term_matches(lower, term) for term in terms)


def _non_valid(raw: str, status: str, reason: str, confidence: float) -> NormalizedAction:
    return NormalizedAction(raw_text=raw, status=status, action=None, reason=reason, confidence=confidence)


UNSAFE_TERMS = (
    "delete production",
    "rm -rf",
    "wipe",
    "ddos",
    "dos attack",
    "destroy",
    "exfiltrate real",
    "attack another host",
    "pivot to another host",
    "lateral movement",
    "横向移动",
    "攻击其他主机",
    "内网横向",
    "破坏",
)

SCHEMA_GAP_TERMS = (
    "jwt",
    "csrf",
    "ssrf",
    "ssti",
    "template injection",
    "server-side request forgery",
    "deserialization",
    "xxe",
    "graphql introspection",
)

OUTSIDE_SCOPE_TERMS = (
    "phishing",
    "social engineer",
    "wifi",
    "bluetooth",
    "kerberos",
    "active directory",
    "domain controller",
    "ssh brute force",
    "ssh",
    "钓鱼",
    "社工",
    "域控",
    "无线",
)

AMBIGUOUS_TERMS = (
    "do something",
    "try harder",
    "continue",
    "keep going",
    "next step",
    "look around",
    "继续看看",
    "随便试试",
    "下一步",
)
