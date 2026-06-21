"""Walkthrough fixture schema + loader/validator for Stage-2 Phase-1 offline replay.

A *walkthrough fixture* is a recorded (or representative) successful attack chain against ONE
real single-host box, expressed as an ordered list of steps. Each step carries:
  * what the operator did, in natural language (`actor_intent`)        -> ψ input
  * the real tool that produced output (`tool`) + its raw `tool_output` -> φ input
  * a HAND-LABELED ground-truth abstract action (`reference_abstract_action`, one of the 16
    frozen actions, or the literal "out_of_abstraction") and, when applicable, why it falls
    outside the schema + which extension would cover it.
  * the abstract-state fields this step should establish (`reference_state_after`) -> φ accuracy.
  * an optional `candidate_pool` of alternative action texts (first = the good one) -> PRM rerank.

The hand-labels are the ground truth; ψ/φ accuracy and the abstraction gap are measured AGAINST
them. Fixtures are author-constructed from public write-up structure unless captured live — the
`source` field must say which. No fixture executes anything.
"""

from __future__ import annotations

from pathlib import Path
import json

from web_attack_sim.action_space import ActionType

VALID_ACTIONS = {a.value for a in ActionType}
OUT_OF_ABSTRACTION = "out_of_abstraction"

# Abstract-state fields a step may assert in `reference_state_after`.
STATE_FIELDS = {
    "open_services", "tech_stack", "discovered_paths", "known_forms", "known_parameters",
    "suspected_vulnerabilities", "verified_vulnerabilities", "credentials",
    "auth_state", "shell_state", "privilege_level", "read_files",
}
SCALAR_FIELDS = {"auth_state", "shell_state", "privilege_level"}

REQUIRED_TOP = {"box", "source", "abstract_family", "steps"}
REQUIRED_STEP = {"actor_intent", "tool", "tool_output", "reference_abstract_action"}


class FixtureError(ValueError):
    pass


def load_walkthrough(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    validate_walkthrough(data, source=str(p))
    return data


def validate_walkthrough(data: dict, *, source: str = "<dict>") -> None:
    missing = REQUIRED_TOP - set(data)
    if missing:
        raise FixtureError(f"{source}: missing top-level keys {sorted(missing)}")
    if not isinstance(data["steps"], list) or not data["steps"]:
        raise FixtureError(f"{source}: 'steps' must be a non-empty list")
    for i, step in enumerate(data["steps"]):
        _validate_step(step, i, source)


def _validate_step(step: dict, i: int, source: str) -> None:
    missing = REQUIRED_STEP - set(step)
    if missing:
        raise FixtureError(f"{source} step {i}: missing keys {sorted(missing)}")
    ref = step["reference_abstract_action"]
    if ref != OUT_OF_ABSTRACTION and ref not in VALID_ACTIONS:
        raise FixtureError(
            f"{source} step {i}: reference_abstract_action '{ref}' is not one of the 16 actions "
            f"nor '{OUT_OF_ABSTRACTION}'"
        )
    if ref == OUT_OF_ABSTRACTION and not step.get("out_of_abstraction_reason"):
        raise FixtureError(f"{source} step {i}: out_of_abstraction steps require 'out_of_abstraction_reason'")
    rsa = step.get("reference_state_after", {})
    if not isinstance(rsa, dict):
        raise FixtureError(f"{source} step {i}: reference_state_after must be an object")
    for field in rsa:
        if field not in STATE_FIELDS:
            raise FixtureError(f"{source} step {i}: unknown state field '{field}' in reference_state_after")
    pool = step.get("candidate_pool")
    if pool is not None and (not isinstance(pool, list) or len(pool) < 2):
        raise FixtureError(f"{source} step {i}: candidate_pool must be a list of >=2 action texts")


def iter_fixtures(directory: str | Path) -> list[tuple[Path, dict]]:
    d = Path(directory)
    out = []
    for p in sorted(d.glob("*.json")):
        out.append((p, load_walkthrough(p)))
    return out
