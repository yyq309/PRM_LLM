from .schemas import Observation


FEATURE_NAMES = [
    "service_known",
    "open_http",
    "tech_known",
    "num_tech_norm",
    "num_paths_norm",
    "path_login_found",
    "path_admin_found",
    "path_backup_found",
    "path_upload_found",
    "num_forms_norm",
    "num_parameters_norm",
    "has_suspected_vuln",
    "has_verified_vuln",
    "num_credentials_norm",
    "auth_anonymous",
    "auth_user",
    "auth_admin",
    "shell_none",
    "shell_webshell",
    "shell_command_execution",
    "priv_none",
    "priv_web_user",
    "priv_system_user",
    "priv_root",
    "num_read_files_norm",
    "failed_action_count_norm",
    "failed_branch_count_norm",
    "remaining_budget_norm",
    # Observable action-relevance signals: let a maskless (permissive) oracle learn which
    # actions are currently productive, instead of relying on an external strict mask.
    "has_unused_credentials",
    "has_known_form",
    "has_known_parameter",
    "has_verified_unexploited_vuln",
    "has_foothold",
    "has_access_goal_unread",
]


def encode_observation(obs: Observation, max_budget: int = 20) -> list[float]:
    paths = set(obs.discovered_paths)
    forms = set(obs.known_forms)
    parameters = set(obs.known_parameters)
    failed_branch_count = sum(obs.failed_branches.values())

    return [
        float(obs.service_known),
        float(any("http" in service for service in obs.open_services)),
        float(bool(obs.tech_stack)),
        min(len(obs.tech_stack) / 5.0, 1.0),
        min(len(paths) / 10.0, 1.0),
        float("/login" in paths),
        float("/admin" in paths),
        float("/backup" in paths),
        float("/upload" in paths),
        min(len(forms) / 5.0, 1.0),
        min(len(parameters) / 8.0, 1.0),
        float(bool(obs.suspected_vulnerabilities)),
        float(bool(obs.verified_vulnerabilities)),
        min(len(obs.credentials) / 5.0, 1.0),
        float(obs.auth_state == "anonymous"),
        float(obs.auth_state == "user"),
        float(obs.auth_state == "admin"),
        float(obs.shell_state == "none"),
        float(obs.shell_state == "webshell"),
        float(obs.shell_state == "command_execution"),
        float(obs.privilege_level == "none"),
        float(obs.privilege_level == "web_user"),
        float(obs.privilege_level == "system_user"),
        float(obs.privilege_level == "root"),
        min(len(obs.read_files) / 5.0, 1.0),
        min(len(obs.failed_actions) / 10.0, 1.0),
        min(failed_branch_count / 10.0, 1.0),
        max(min(obs.remaining_budget / max(max_budget, 1), 1.0), 0.0),
        # Observable action-relevance signals (all derived from visible state).
        float(bool(obs.credentials) and obs.auth_state != "admin"),
        float(bool(obs.known_forms)),
        float(bool(obs.known_parameters)),
        float(bool(obs.verified_vulnerabilities) and obs.shell_state == "none"),
        float(obs.shell_state != "none"),
        float((obs.auth_state != "anonymous" or obs.shell_state != "none") and not obs.read_files),
    ]

