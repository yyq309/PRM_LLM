from dataclasses import dataclass, field
from typing import Any

from .action_space import ActionType


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    target: str | None = None
    parameter: str | None = None


@dataclass
class StepFeedback:
    success: bool
    progress_event: str | None = None
    new_observation: str | None = None
    discovered_items: list[str] = field(default_factory=list)
    evidence: str | None = None
    auth_change: str | None = None
    foothold_change: str | None = None
    privilege_change: str | None = None
    file_change: str | None = None
    error_type: str | None = None
    cost: float = 1.0
    terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "progress_event": self.progress_event,
            "new_observation": self.new_observation,
            "discovered_items": list(self.discovered_items),
            "evidence": self.evidence,
            "auth_change": self.auth_change,
            "foothold_change": self.foothold_change,
            "privilege_change": self.privilege_change,
            "file_change": self.file_change,
            "error_type": self.error_type,
            "cost": self.cost,
            "terminal": self.terminal,
        }


@dataclass
class Observation:
    target_known: bool
    service_known: bool
    open_services: list[str]
    base_url_known: bool
    discovered_paths: list[str]
    known_forms: list[str]
    known_parameters: list[str]
    tech_stack: list[str]
    suspected_vulnerabilities: list[str]
    verified_vulnerabilities: list[str]
    credentials: list[str]
    auth_state: str
    shell_state: str
    privilege_level: str
    read_files: list[str]
    failed_actions: list[str]
    failed_branches: dict[str, int]
    remaining_budget: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_known": self.target_known,
            "service_known": self.service_known,
            "open_services": list(self.open_services),
            "base_url_known": self.base_url_known,
            "discovered_paths": list(self.discovered_paths),
            "known_forms": list(self.known_forms),
            "known_parameters": list(self.known_parameters),
            "tech_stack": list(self.tech_stack),
            "suspected_vulnerabilities": list(self.suspected_vulnerabilities),
            "verified_vulnerabilities": list(self.verified_vulnerabilities),
            "credentials": list(self.credentials),
            "auth_state": self.auth_state,
            "shell_state": self.shell_state,
            "privilege_level": self.privilege_level,
            "read_files": list(self.read_files),
            "failed_actions": list(self.failed_actions),
            "failed_branches": dict(self.failed_branches),
            "remaining_budget": self.remaining_budget,
        }

