from enum import Enum


class ActionType(str, Enum):
    SERVICE_ENUMERATION = "service_enumeration"
    HTTP_FINGERPRINT = "http_fingerprint"
    WEB_PATH_ENUMERATION = "web_path_enumeration"
    CONTENT_RETRIEVAL = "content_retrieval"
    INPUT_DISCOVERY = "input_discovery"
    FORM_INTERACTION = "form_interaction"
    AUTH_ATTEMPT = "auth_attempt"
    CREDENTIAL_USE = "credential_use"
    VULNERABILITY_CHECK = "vulnerability_check"
    EXPLOIT_ATTEMPT = "exploit_attempt"
    FILE_UPLOAD_ATTEMPT = "file_upload_attempt"
    COMMAND_EXECUTION = "command_execution"
    SENSITIVE_FILE_READ = "sensitive_file_read"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    POST_EXPLOITATION = "post_exploitation"
    STOP_OR_REPORT = "stop_or_report"


ACTIONS = [
    ActionType.SERVICE_ENUMERATION,
    ActionType.HTTP_FINGERPRINT,
    ActionType.WEB_PATH_ENUMERATION,
    ActionType.CONTENT_RETRIEVAL,
    ActionType.INPUT_DISCOVERY,
    ActionType.FORM_INTERACTION,
    ActionType.AUTH_ATTEMPT,
    ActionType.CREDENTIAL_USE,
    ActionType.VULNERABILITY_CHECK,
    ActionType.EXPLOIT_ATTEMPT,
    ActionType.FILE_UPLOAD_ATTEMPT,
    ActionType.COMMAND_EXECUTION,
    ActionType.SENSITIVE_FILE_READ,
    ActionType.PRIVILEGE_ESCALATION,
    ActionType.POST_EXPLOITATION,
    ActionType.STOP_OR_REPORT,
]


ACTION_TO_ID = {action: idx for idx, action in enumerate(ACTIONS)}

