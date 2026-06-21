from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .action_space import ACTIONS, ActionType
from .encoder import encode_observation
from .reward import DEFAULT_REWARDS, reward_for_feedback
from .schemas import Action, Observation, StepFeedback
from .tasks import load_task_config


@dataclass
class RuntimeState:
    task: dict[str, Any]
    discovered_services: set[str] = field(default_factory=set)
    discovered_paths: set[str] = field(default_factory=set)
    tech_stack: set[str] = field(default_factory=set)
    known_forms: set[str] = field(default_factory=set)
    known_parameters: set[str] = field(default_factory=set)
    suspected_vulnerabilities: set[str] = field(default_factory=set)
    verified_vulnerabilities: set[str] = field(default_factory=set)
    credentials: set[str] = field(default_factory=set)
    auth_state: str = "anonymous"
    shell_state: str = "none"
    privilege_level: str = "none"
    read_files: set[str] = field(default_factory=set)
    attempted_actions: list[str] = field(default_factory=list)
    failed_actions: list[str] = field(default_factory=list)
    failed_branches: dict[str, int] = field(default_factory=dict)
    remaining_budget: int = 20
    done: bool = False


class WebAttackSimEnv:
    """A minimal Gym-style single-host Web penetration-testing simulator."""

    def __init__(self, rewards: dict[str, float] | None = None):
        self.actions = list(ACTIONS)
        self.rewards = rewards or DEFAULT_REWARDS
        self.state: RuntimeState | None = None
        self.max_steps = 20
        self.trace: list[dict[str, Any]] = []

    def reset(self, task: str | Path | dict[str, Any]) -> tuple[Observation, dict[str, Any]]:
        task_config = load_task_config(task)
        budget = task_config.get("budget", {})
        self.max_steps = int(budget.get("max_steps", 20))
        initial = task_config.get("initial_observation", {})
        self.state = RuntimeState(
            task=task_config,
            discovered_services=set(initial.get("open_services", ["http:80"])),
            discovered_paths=set(initial.get("discovered_paths", ["/"])),
            remaining_budget=self.max_steps,
        )
        self.trace = []
        obs = self._observation()
        return obs, {
            "task_id": task_config.get("task_id"),
            "action_mask": self.action_mask(permissive=True),
        }

    def step(self, action: int | str | Action | dict[str, Any]) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        self._require_state()
        assert self.state is not None

        normalized = self._normalize_action(action)
        if self.state.done:
            feedback = StepFeedback(
                success=False,
                error_type="invalid_action",
                evidence="episode already terminated",
                terminal=True,
            )
            return self._finalize_step(normalized, feedback)

        self.state.remaining_budget -= 1
        action_key = self._action_key(normalized)
        self.state.attempted_actions.append(action_key)

        handler = {
            ActionType.SERVICE_ENUMERATION: self._handle_service_enumeration,
            ActionType.HTTP_FINGERPRINT: self._handle_http_fingerprint,
            ActionType.WEB_PATH_ENUMERATION: self._handle_web_path_enumeration,
            ActionType.CONTENT_RETRIEVAL: self._handle_content_retrieval,
            ActionType.INPUT_DISCOVERY: self._handle_input_discovery,
            ActionType.FORM_INTERACTION: self._handle_form_interaction,
            ActionType.AUTH_ATTEMPT: self._handle_auth_attempt,
            ActionType.CREDENTIAL_USE: self._handle_credential_use,
            ActionType.VULNERABILITY_CHECK: self._handle_vulnerability_check,
            ActionType.EXPLOIT_ATTEMPT: self._handle_exploit_attempt,
            ActionType.FILE_UPLOAD_ATTEMPT: self._handle_file_upload_attempt,
            ActionType.COMMAND_EXECUTION: self._handle_command_execution,
            ActionType.SENSITIVE_FILE_READ: self._handle_sensitive_file_read,
            ActionType.PRIVILEGE_ESCALATION: self._handle_privilege_escalation,
            ActionType.POST_EXPLOITATION: self._handle_post_exploitation,
            ActionType.STOP_OR_REPORT: self._handle_stop_or_report,
        }[normalized.action_type]

        feedback = handler(normalized)
        if self.state.remaining_budget <= 0:
            self.state.done = True
            feedback.terminal = True
        return self._finalize_step(normalized, feedback)

    def action_mask(self, permissive: bool = True) -> list[int]:
        self._require_state()
        if permissive:
            return [1 for _ in self.actions]

        assert self.state is not None
        paths = self.state.discovered_paths
        mask = []
        for action in self.actions:
            allowed = True
            if action == ActionType.SERVICE_ENUMERATION:
                services = set(self.state.task.get("service_surface", ["http:80"]))
                allowed = bool(services - self.state.discovered_services)
            elif action == ActionType.HTTP_FINGERPRINT:
                techs = set(self.state.task.get("technologies", []))
                allowed = bool(techs - self.state.tech_stack)
            elif action == ActionType.WEB_PATH_ENUMERATION:
                hidden = set(self.state.task.get("hidden_paths", []))
                allowed = bool(hidden - paths)
            elif action == ActionType.CONTENT_RETRIEVAL:
                allowed = self._choose_path_with_unclaimed_leak(None) is not None
            elif action == ActionType.INPUT_DISCOVERY:
                allowed = self._has_discoverable_input()
            elif action == ActionType.FORM_INTERACTION:
                allowed = False
            elif action == ActionType.AUTH_ATTEMPT:
                valid_id = self.state.task.get("auth", {}).get("valid_credential")
                credential = self.state.task.get("credentials", {}).get(valid_id, {})
                allowed = self.state.auth_state != "admin" and self._login_path() in paths and bool(credential.get("weak"))
            elif action == ActionType.CREDENTIAL_USE:
                allowed = self.state.auth_state != "admin" and bool(self.state.credentials) and self._login_path() in paths
            elif action == ActionType.VULNERABILITY_CHECK:
                allowed = self._has_unverified_checkable_vulnerability_with_new_effect()
            elif action == ActionType.EXPLOIT_ATTEMPT:
                allowed = self._choose_verified_vulnerability(None) is not None and self._verified_vulnerability_has_new_effect()
            elif action == ActionType.FILE_UPLOAD_ATTEMPT:
                upload = self.state.task.get("upload", {})
                allowed = (
                    self.state.shell_state not in {"webshell", "command_execution"}
                    and upload.get("path") in paths
                    and self._auth_satisfies(upload.get("requires_auth"))
                    and bool(upload.get("vulnerable"))
                )
            elif action == ActionType.COMMAND_EXECUTION:
                allowed = self.state.shell_state == "webshell"
            elif action == ActionType.SENSITIVE_FILE_READ:
                allowed = self._choose_readable_file(None) is not None
            elif action == ActionType.PRIVILEGE_ESCALATION:
                privesc = self.state.task.get("privilege_escalation", {})
                allowed = (
                    self.state.shell_state in {"webshell", "command_execution"}
                    and bool(privesc.get("available", False))
                    and self.state.privilege_level != privesc.get("target_privilege", "root")
                )
            elif action == ActionType.POST_EXPLOITATION:
                allowed = self._choose_readable_file(None) is not None
            elif action == ActionType.STOP_OR_REPORT:
                allowed = self._goal_reached()
            mask.append(int(allowed))
        return mask

    def encode_observation(self, obs: Observation | None = None) -> list[float]:
        return encode_observation(obs or self._observation(), max_budget=self.max_steps)

    def _finalize_step(self, action: Action, feedback: StepFeedback) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        assert self.state is not None
        self._check_goal(feedback)
        if feedback.error_type:
            self._record_failure(feedback.error_type)
        reward = reward_for_feedback(feedback, self.rewards)
        obs = self._observation()
        info = {
            "action": {
                "action_type": action.action_type.value,
                "target": action.target,
                "parameter": action.parameter,
            },
            "feedback": feedback.to_dict(),
            "action_mask": self.action_mask(permissive=True),
        }
        self.trace.append({"observation": obs.to_dict(), "reward": reward, **info})
        return obs, reward, self.state.done, False, info

    def _normalize_action(self, action: int | str | Action | dict[str, Any]) -> Action:
        if isinstance(action, Action):
            return action
        if isinstance(action, int):
            return Action(self.actions[action])
        if isinstance(action, str):
            return Action(ActionType(action))
        if isinstance(action, dict):
            return Action(
                action_type=ActionType(action["action_type"]),
                target=action.get("target"),
                parameter=action.get("parameter"),
            )
        raise TypeError(f"unsupported action: {action!r}")

    def _observation(self) -> Observation:
        self._require_state()
        assert self.state is not None
        return Observation(
            target_known=True,
            service_known=bool(self.state.discovered_services),
            open_services=sorted(self.state.discovered_services),
            base_url_known=True,
            discovered_paths=sorted(self.state.discovered_paths),
            known_forms=sorted(self.state.known_forms),
            known_parameters=sorted(self.state.known_parameters),
            tech_stack=sorted(self.state.tech_stack),
            suspected_vulnerabilities=sorted(self.state.suspected_vulnerabilities),
            verified_vulnerabilities=sorted(self.state.verified_vulnerabilities),
            credentials=sorted(self.state.credentials),
            auth_state=self.state.auth_state,
            shell_state=self.state.shell_state,
            privilege_level=self.state.privilege_level,
            read_files=sorted(self.state.read_files),
            failed_actions=list(self.state.failed_actions),
            failed_branches=dict(self.state.failed_branches),
            remaining_budget=self.state.remaining_budget,
        )

    def _handle_service_enumeration(self, action: Action) -> StepFeedback:
        assert self.state is not None
        services = set(self.state.task.get("service_surface", ["http:80"]))
        new_services = sorted(services - self.state.discovered_services)
        if new_services:
            self.state.discovered_services.update(new_services)
            return StepFeedback(True, "service_found", discovered_items=new_services, evidence="new service discovered")
        return StepFeedback(False, error_type="no_new_information", evidence="no undiscovered service remains")

    def _handle_http_fingerprint(self, action: Action) -> StepFeedback:
        assert self.state is not None
        techs = set(self.state.task.get("technologies", []))
        new_techs = sorted(techs - self.state.tech_stack)
        if new_techs:
            self.state.tech_stack.update(new_techs)
            return StepFeedback(True, "fingerprint_found", discovered_items=new_techs, evidence="web fingerprinting exposed technology stack")
        return StepFeedback(False, error_type="no_new_information", evidence="technology stack already known or unavailable")

    def _handle_web_path_enumeration(self, action: Action) -> StepFeedback:
        assert self.state is not None
        hidden = self.state.task.get("hidden_paths", [])
        if action.target and action.target in hidden and action.target not in self.state.discovered_paths:
            next_path = action.target
        else:
            next_path = next((path for path in hidden if path not in self.state.discovered_paths), None)
        if next_path:
            self.state.discovered_paths.add(next_path)
            return StepFeedback(True, "path_found", discovered_items=[next_path], evidence=f"path discovered: {next_path}")
        return StepFeedback(False, error_type="no_new_information", evidence="path enumeration found no new route")

    def _handle_content_retrieval(self, action: Action) -> StepFeedback:
        assert self.state is not None
        path = self._choose_path_with_unclaimed_leak(action.target)
        if not path:
            return StepFeedback(False, error_type="no_new_information", evidence="no discovered path contains new readable content")
        if path not in self.state.discovered_paths:
            return StepFeedback(False, error_type="precondition_missing", evidence=f"path not discovered: {path}")

        leak = self.state.task.get("leaks", {}).get(path, {})
        discovered: list[str] = []
        for credential_id in leak.get("credentials", []):
            if credential_id not in self.state.credentials:
                self.state.credentials.add(credential_id)
                discovered.append(f"credential:{credential_id}")
        for file_id in leak.get("files", []):
            if file_id not in self.state.read_files:
                self.state.read_files.add(file_id)
                discovered.append(f"file:{file_id}")

        if discovered:
            return StepFeedback(True, "credential_found", discovered_items=discovered, evidence=f"content retrieval exposed {', '.join(discovered)}")
        return StepFeedback(False, error_type="no_new_information", evidence=f"retrieved {path}, but no new useful content was found")

    def _handle_input_discovery(self, action: Action) -> StepFeedback:
        assert self.state is not None
        discovered: list[str] = []
        for path, fields in self.state.task.get("forms", {}).items():
            if path in self.state.discovered_paths:
                form_id = f"form:{path}"
                if form_id not in self.state.known_forms:
                    self.state.known_forms.add(form_id)
                    discovered.append(form_id)
        for endpoint, specs in self.state.task.get("parameters", {}).items():
            if endpoint in self.state.discovered_paths:
                for spec in specs:
                    parameter_id = f"{endpoint}?{spec['name']}"
                    if parameter_id not in self.state.known_parameters:
                        self.state.known_parameters.add(parameter_id)
                        discovered.append(f"parameter:{parameter_id}")
        if discovered:
            return StepFeedback(True, "input_found", discovered_items=discovered, evidence="input discovery found forms or parameters")
        return StepFeedback(False, error_type="no_new_information", evidence="no new forms or parameters found")

    def _handle_form_interaction(self, action: Action) -> StepFeedback:
        assert self.state is not None
        if self.state.known_forms:
            return StepFeedback(False, error_type="no_new_information", evidence="form interaction produced no new information")
        return StepFeedback(False, error_type="precondition_missing", evidence="no form has been discovered")

    def _handle_auth_attempt(self, action: Action) -> StepFeedback:
        assert self.state is not None
        if self.state.auth_state == "admin":
            return StepFeedback(False, error_type="duplicate_action", evidence="already authenticated as admin")
        login_path = self._login_path()
        if login_path not in self.state.discovered_paths:
            return StepFeedback(False, error_type="precondition_missing", evidence="login path has not been discovered")
        valid_id = self.state.task.get("auth", {}).get("valid_credential")
        credential = self.state.task.get("credentials", {}).get(valid_id, {})
        if credential.get("weak"):
            old = self.state.auth_state
            self.state.auth_state = credential.get("role", "user")
            return StepFeedback(True, "session_obtained", auth_change=f"{old}->{self.state.auth_state}", evidence="weak/default credential login succeeded")
        return StepFeedback(False, error_type="credential_invalid", evidence="default or guessed credentials failed")

    def _handle_credential_use(self, action: Action) -> StepFeedback:
        assert self.state is not None
        if self.state.auth_state == "admin":
            return StepFeedback(False, error_type="duplicate_action", evidence="already authenticated as admin")
        login_path = self._login_path()
        if login_path not in self.state.discovered_paths:
            return StepFeedback(False, error_type="precondition_missing", evidence="login path has not been discovered")
        valid_id = self.state.task.get("auth", {}).get("valid_credential")
        if valid_id not in self.state.credentials:
            return StepFeedback(False, error_type="precondition_missing", evidence="valid credential has not been discovered")
        old = self.state.auth_state
        role = self.state.task.get("auth", {}).get("role", "user")
        self.state.auth_state = role
        return StepFeedback(True, "session_obtained", auth_change=f"{old}->{role}", evidence="known credential produced an authenticated session")

    def _handle_vulnerability_check(self, action: Action) -> StepFeedback:
        assert self.state is not None
        vuln_id = self._choose_checkable_vulnerability(action.target)
        if not vuln_id:
            return StepFeedback(False, error_type="precondition_missing", evidence="no discovered input or path can be checked for vulnerability")
        if vuln_id in self.state.verified_vulnerabilities:
            return StepFeedback(False, error_type="duplicate_action", evidence=f"vulnerability already verified: {vuln_id}")
        self.state.suspected_vulnerabilities.add(vuln_id)
        self.state.verified_vulnerabilities.add(vuln_id)
        return StepFeedback(True, "vulnerability_verified", discovered_items=[vuln_id], evidence=f"vulnerability verified: {vuln_id}")

    def _handle_exploit_attempt(self, action: Action) -> StepFeedback:
        assert self.state is not None
        vuln_id = self._choose_verified_vulnerability(action.target)
        if not vuln_id:
            return StepFeedback(False, error_type="precondition_missing", evidence="no verified vulnerability is available for exploitation")
        vuln = self.state.task.get("vulnerabilities", {}).get(vuln_id, {})
        effects = vuln.get("effects", {})
        discovered: list[str] = []
        for credential_id in effects.get("credentials", []):
            if credential_id not in self.state.credentials:
                self.state.credentials.add(credential_id)
                discovered.append(f"credential:{credential_id}")
        if effects.get("shell"):
            if self.state.shell_state in {"webshell", "command_execution"}:
                return StepFeedback(False, error_type="duplicate_action", evidence=f"shell already obtained from {vuln_id}")
            old = self.state.shell_state
            self.state.shell_state = "webshell"
            self.state.privilege_level = "web_user"
            return StepFeedback(True, "shell_obtained", discovered_items=discovered + ["webshell"], foothold_change=f"{old}->webshell", evidence=f"exploit succeeded: {vuln_id}")
        if discovered:
            return StepFeedback(True, "exploit_succeeded", discovered_items=discovered, evidence=f"exploit produced {', '.join(discovered)}")
        return StepFeedback(False, error_type="duplicate_action", evidence=f"exploit produced no new state change: {vuln_id}")

    def _handle_file_upload_attempt(self, action: Action) -> StepFeedback:
        assert self.state is not None
        upload = self.state.task.get("upload", {})
        path = upload.get("path")
        if not path or path not in self.state.discovered_paths:
            return StepFeedback(False, error_type="precondition_missing", evidence="upload path has not been discovered")
        if not self._auth_satisfies(upload.get("requires_auth")):
            return StepFeedback(False, error_type="auth_required", evidence="upload requires a stronger authenticated state")
        if not upload.get("vulnerable"):
            return StepFeedback(False, error_type="vulnerability_not_present", evidence="upload control rejected the payload")
        if upload.get("shell_on_upload"):
            if self.state.shell_state in {"webshell", "command_execution"}:
                return StepFeedback(False, error_type="duplicate_action", evidence="webshell already obtained")
            old = self.state.shell_state
            self.state.shell_state = "webshell"
            self.state.privilege_level = "web_user"
            return StepFeedback(True, "shell_obtained", discovered_items=["webshell"], foothold_change=f"{old}->webshell", evidence="uploaded webshell is reachable")
        return StepFeedback(True, "file_written", discovered_items=[path], evidence="file upload succeeded")

    def _handle_command_execution(self, action: Action) -> StepFeedback:
        assert self.state is not None
        if self.state.shell_state not in {"webshell", "command_execution"}:
            return StepFeedback(False, error_type="precondition_missing", evidence="no shell or command execution foothold is available")
        if self.state.shell_state == "command_execution":
            return StepFeedback(False, error_type="duplicate_action", evidence="command execution already confirmed")
        old = self.state.shell_state
        self.state.shell_state = "command_execution"
        self.state.privilege_level = "web_user"
        return StepFeedback(True, "command_execution_obtained", foothold_change=f"{old}->command_execution", evidence="command execution confirmed")

    def _handle_sensitive_file_read(self, action: Action) -> StepFeedback:
        assert self.state is not None
        file_id = self._choose_readable_file(action.target)
        if not file_id:
            return StepFeedback(False, error_type="precondition_missing", evidence="no sensitive file is currently readable")
        self.state.read_files.add(file_id)
        return StepFeedback(True, "sensitive_file_read", discovered_items=[file_id], evidence=f"sensitive file read: {file_id}")

    def _handle_privilege_escalation(self, action: Action) -> StepFeedback:
        assert self.state is not None
        privesc = self.state.task.get("privilege_escalation", {})
        if self.state.shell_state not in {"webshell", "command_execution"}:
            return StepFeedback(False, error_type="precondition_missing", evidence="privilege escalation requires a shell foothold")
        if not privesc.get("available", False):
            return StepFeedback(False, error_type="vulnerability_not_present", evidence="no local privilege escalation path is configured")
        target_privilege = privesc.get("target_privilege", "root")
        if self.state.privilege_level == target_privilege:
            return StepFeedback(False, error_type="duplicate_action", evidence=f"already at target privilege level: {target_privilege}")
        old = self.state.privilege_level
        self.state.privilege_level = target_privilege
        return StepFeedback(True, "privilege_escalated", privilege_change=f"{old}->{self.state.privilege_level}", evidence="local privilege escalation succeeded")

    def _handle_post_exploitation(self, action: Action) -> StepFeedback:
        feedback = self._handle_sensitive_file_read(action)
        if feedback.success:
            return feedback
        return StepFeedback(False, error_type="precondition_missing", evidence="post-exploitation goal is not currently reachable")

    def _handle_stop_or_report(self, action: Action) -> StepFeedback:
        assert self.state is not None
        self.state.done = True
        if self._goal_reached():
            return StepFeedback(True, "goal_reached", evidence="goal already reached; report submitted", terminal=True)
        return StepFeedback(False, error_type="no_new_information", evidence="stopped before reaching the goal", terminal=True)

    def _choose_path_with_unclaimed_leak(self, target: str | None) -> str | None:
        assert self.state is not None
        leaks = self.state.task.get("leaks", {})
        if target:
            return target
        for path, leak in leaks.items():
            if path not in self.state.discovered_paths:
                continue
            if any(c not in self.state.credentials for c in leak.get("credentials", [])):
                return path
            if any(f not in self.state.read_files for f in leak.get("files", [])):
                return path
        return None

    def _choose_checkable_vulnerability(self, target: str | None) -> str | None:
        assert self.state is not None
        vulnerabilities = self.state.task.get("vulnerabilities", {})
        for vuln_id, vuln in vulnerabilities.items():
            if target and target not in {vuln_id, vuln.get("target")}:
                continue
            if self._preconditions_met(vuln.get("requires", [])):
                return vuln_id
        return None

    def _choose_verified_vulnerability(self, target: str | None) -> str | None:
        assert self.state is not None
        vulnerabilities = self.state.task.get("vulnerabilities", {})
        for vuln_id in self.state.verified_vulnerabilities:
            vuln = vulnerabilities.get(vuln_id, {})
            if not target or target in {vuln_id, vuln.get("target")}:
                return vuln_id
        return None

    def _choose_readable_file(self, target: str | None) -> str | None:
        assert self.state is not None
        files = self.state.task.get("files", {})
        for file_id, spec in files.items():
            if target and target not in {file_id, spec.get("path")}:
                continue
            if file_id in self.state.read_files:
                continue
            if self._file_readable(spec):
                return file_id
        return None

    def _file_readable(self, spec: dict[str, Any]) -> bool:
        requires_auth = spec.get("requires_auth")
        requires_shell = spec.get("requires_shell", False)
        requires_privilege = spec.get("requires_privilege")
        if requires_auth and not self._auth_satisfies(requires_auth):
            return False
        if requires_shell and self.state.shell_state not in {"webshell", "command_execution"}:  # type: ignore[union-attr]
            return False
        if requires_privilege and self.state.privilege_level != requires_privilege:  # type: ignore[union-attr]
            return False
        return True

    def _has_discoverable_input(self) -> bool:
        assert self.state is not None
        for path in self.state.task.get("forms", {}):
            if path in self.state.discovered_paths and f"form:{path}" not in self.state.known_forms:
                return True
        for endpoint, specs in self.state.task.get("parameters", {}).items():
            if endpoint not in self.state.discovered_paths:
                continue
            for spec in specs:
                if f"{endpoint}?{spec['name']}" not in self.state.known_parameters:
                    return True
        return False

    def _verified_vulnerability_has_new_effect(self) -> bool:
        assert self.state is not None
        for vuln_id in self.state.verified_vulnerabilities:
            if self._vulnerability_has_new_effect(vuln_id):
                return True
        return False

    def _has_unverified_checkable_vulnerability_with_new_effect(self) -> bool:
        assert self.state is not None
        for vuln_id, vuln in self.state.task.get("vulnerabilities", {}).items():
            if vuln_id in self.state.verified_vulnerabilities:
                continue
            if not self._preconditions_met(vuln.get("requires", [])):
                continue
            if self._vulnerability_has_new_effect(vuln_id):
                return True
        return False

    def _vulnerability_has_new_effect(self, vuln_id: str) -> bool:
        assert self.state is not None
        effects = self.state.task.get("vulnerabilities", {}).get(vuln_id, {}).get("effects", {})
        if any(credential_id not in self.state.credentials for credential_id in effects.get("credentials", [])):
            return True
        if effects.get("shell") and self.state.shell_state not in {"webshell", "command_execution"}:
            return True
        return not effects

    def _check_goal(self, feedback: StepFeedback) -> None:
        assert self.state is not None
        if self._goal_reached():
            self.state.done = True
            feedback.terminal = True
            if "goal_reached" not in feedback.discovered_items:
                feedback.discovered_items.append("goal_reached")

    def _goal_reached(self) -> bool:
        assert self.state is not None
        goal = self.state.task.get("goal", {})
        if goal.get("type") == "read_file":
            return goal.get("file") in self.state.read_files
        if goal.get("type") == "shell":
            return self.state.shell_state in {"webshell", "command_execution"}
        if goal.get("type") == "privilege":
            return self.state.privilege_level == goal.get("level", "root")
        return False

    def _preconditions_met(self, preconditions: list[str]) -> bool:
        assert self.state is not None
        for condition in preconditions:
            kind, _, value = condition.partition(":")
            if kind == "parameter_found" and value not in self.state.known_parameters:
                return False
            if kind == "path_found" and value not in self.state.discovered_paths:
                return False
            if kind == "auth_state" and self.state.auth_state != value:
                return False
            if kind == "vulnerability_verified" and value not in self.state.verified_vulnerabilities:
                return False
        return True

    def _auth_satisfies(self, required: str | None) -> bool:
        if not required:
            return True
        assert self.state is not None
        if required == "admin":
            return self.state.auth_state == "admin"
        if required == "user":
            return self.state.auth_state in {"user", "admin"}
        return self.state.auth_state == required

    def _login_path(self) -> str:
        assert self.state is not None
        return self.state.task.get("auth", {}).get("login_path", "/login")

    def _action_key(self, action: Action) -> str:
        return ":".join(
            part for part in [action.action_type.value, action.target or "", action.parameter or ""] if part
        )

    def _record_failure(self, error_type: str) -> None:
        assert self.state is not None
        self.state.failed_actions.append(error_type)
        self.state.failed_branches[error_type] = self.state.failed_branches.get(error_type, 0) + 1

    def _require_state(self) -> None:
        if self.state is None:
            raise RuntimeError("env.reset(task) must be called before using the environment")
