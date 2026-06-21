"""Coverage / diversity audit for the WebAttackSim task set (method §12.1).

Makes "the simulator evaluates a diverse range of situations" an auditable grid
rather than a feeling. Each task is classified along four axes and reported as
per-cell counts so coverage gaps are explicit:

    vulnerability class x attack surface x chain depth x difficulty

The classification is derived structurally from each task_config (its declared
vulnerabilities, forms/parameters/upload surface, expert_plan length, and the
privilege-escalation / shell / leak structure), so it stays in sync with the
templates without a hand-maintained label table.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import json
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web_attack_sim.tasks import bundled_task_paths, load_task_config  # noqa: E402


VULN_CLASSES = [
    "backup_or_config_leak",
    "default_or_weak_password",
    "sqli",
    "lfi",
    "rce",
    "file_upload",
    "host_privilege_escalation",
]
ATTACK_SURFACES = ["path", "form", "parameter", "upload"]
DIFFICULTIES = ["easy", "medium", "hard"]


def vuln_types(task: dict[str, Any]) -> set[str]:
    return {str(v.get("type", "")).lower() for v in task.get("vulnerabilities", {}).values()}


def has_weak_credential(task: dict[str, Any]) -> bool:
    return any(bool(c.get("weak")) for c in task.get("credentials", {}).values())


def has_privesc(task: dict[str, Any]) -> bool:
    return bool(task.get("privilege_escalation", {}).get("available", False))


def has_leak(task: dict[str, Any]) -> bool:
    return bool(task.get("leaks"))


def task_vuln_classes(task: dict[str, Any]) -> list[str]:
    types = vuln_types(task)
    classes: list[str] = []
    if has_leak(task) and not has_weak_credential(task) and not types:
        classes.append("backup_or_config_leak")
    if has_weak_credential(task):
        classes.append("default_or_weak_password")
    if "sqli" in types:
        classes.append("sqli")
    if "lfi" in types:
        classes.append("lfi")
    if any("rce" in t for t in types):
        classes.append("rce")
    if "upload" in task:
        classes.append("file_upload")
    if has_privesc(task):
        classes.append("host_privilege_escalation")
    # Fall back to leak family if nothing else matched (e.g. pure path -> file read).
    if not classes and has_leak(task):
        classes.append("backup_or_config_leak")
    return classes


def task_attack_surfaces(task: dict[str, Any]) -> list[str]:
    surfaces = ["path"]  # every task requires path enumeration
    if task.get("forms"):
        surfaces.append("form")
    if task.get("parameters"):
        surfaces.append("parameter")
    if "upload" in task:
        surfaces.append("upload")
    return surfaces


def chain_depth(task: dict[str, Any]) -> int:
    return len(task.get("expert_plan", []))


def needs_shell(task: dict[str, Any]) -> bool:
    if task.get("goal", {}).get("type") in {"shell", "privilege"}:
        return True
    for spec in task.get("files", {}).values():
        if spec.get("requires_shell") or spec.get("requires_privilege"):
            return True
    if task.get("upload", {}).get("shell_on_upload"):
        return True
    return any(v.get("effects", {}).get("shell") for v in task.get("vulnerabilities", {}).values())


def difficulty(task: dict[str, Any]) -> str:
    """Capability-based difficulty matching the method §12.1 / plan difficulty table.

    hard   = requires a shell foothold or local privilege escalation
    medium = requires obtaining a credential via a vuln (SQLi/LFI) or a content leak,
             then authenticating
    easy   = direct sensitive-file leak (no auth) or default/weak-password login
    """
    if has_privesc(task) or needs_shell(task):
        return "hard"
    types = vuln_types(task)
    if "sqli" in types or "lfi" in types:
        return "medium"
    leak_gives_credential = any(leak.get("credentials") for leak in task.get("leaks", {}).values())
    if leak_gives_credential:
        return "medium"
    return "easy"


def distractor_paths(task: dict[str, Any]) -> int:
    """Hidden paths that are not referenced by the expert plan = decoy attack surface."""
    used = {str(step.get("target")) for step in task.get("expert_plan", []) if isinstance(step, dict)}
    return sum(1 for path in task.get("hidden_paths", []) if path not in used)


def plan_signature(task: dict[str, Any]) -> tuple[str, ...]:
    """Topology signature = the expert_plan action-type sequence (structural identity)."""
    return tuple(s if isinstance(s, str) else s.get("action_type", "?") for s in task.get("expert_plan", []))


def audit(task_paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    vuln_counter: Counter[str] = Counter()
    surface_counter: Counter[str] = Counter()
    difficulty_counter: Counter[str] = Counter()
    depth_counter: Counter[int] = Counter()
    family_counter: Counter[str] = Counter()
    family_difficulty: dict[str, Counter[str]] = {}
    signatures: set[tuple[str, ...]] = set()
    grid: dict[str, dict[str, int]] = {vc: {d: 0 for d in DIFFICULTIES} for vc in VULN_CLASSES}

    for task_path in task_paths:
        task = load_task_config(task_path)
        classes = task_vuln_classes(task)
        surfaces = task_attack_surfaces(task)
        diff = difficulty(task)
        depth = chain_depth(task)
        family = str(task.get("family", "untagged"))
        sig = plan_signature(task)

        vuln_counter.update(classes)
        surface_counter.update(surfaces)
        difficulty_counter.update([diff])
        depth_counter.update([depth])
        family_counter.update([family])
        family_difficulty.setdefault(family, Counter()).update([diff])
        signatures.add(sig)
        for vc in classes:
            grid[vc][diff] += 1

        rows.append(
            {
                "task_id": task.get("task_id"),
                "family": family,
                "topology_signature": list(sig),
                "vuln_classes": classes,
                "attack_surfaces": surfaces,
                "chain_depth": depth,
                "difficulty": diff,
                "distractor_paths": distractor_paths(task),
            }
        )

    empty_cells = [
        {"vuln_class": vc, "difficulty": d}
        for vc in VULN_CLASSES
        for d in DIFFICULTIES
        if grid[vc][d] == 0
    ]
    total_cells = len(VULN_CLASSES) * len(DIFFICULTIES)
    filled_cells = total_cells - len(empty_cells)

    return {
        "num_tasks": len(rows),
        "num_families": len(family_counter),
        "num_distinct_topology_signatures": len(signatures),
        "family_counts": dict(sorted(family_counter.items())),
        "family_x_difficulty": {fam: dict(counts) for fam, counts in sorted(family_difficulty.items())},
        "vuln_class_counts": {vc: vuln_counter.get(vc, 0) for vc in VULN_CLASSES},
        "attack_surface_counts": {s: surface_counter.get(s, 0) for s in ATTACK_SURFACES},
        "difficulty_counts": {d: difficulty_counter.get(d, 0) for d in DIFFICULTIES},
        "chain_depth_histogram": dict(sorted(depth_counter.items())),
        "tasks_with_distractors": sum(1 for row in rows if row["distractor_paths"] > 0),
        "vuln_class_x_difficulty_grid": grid,
        "coverage": {
            "filled_cells": filled_cells,
            "total_cells": total_cells,
            "fill_rate": round(filled_cells / total_cells, 3),
            "empty_cells": empty_cells,
        },
        "tasks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit WebAttackSim task diversity along the method §12.1 axes.")
    parser.add_argument("--report-output", type=Path, default=ROOT / "outputs" / "coverage_audit.json")
    args = parser.parse_args()

    report = audit(bundled_task_paths())
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"tasks: {report['num_tasks']}    families: {report['num_families']}    "
          f"distinct topology signatures: {report['num_distinct_topology_signatures']}    "
          f"coverage fill rate: {report['coverage']['fill_rate']:.2f} "
          f"({report['coverage']['filled_cells']}/{report['coverage']['total_cells']} cells)")
    print(f"family counts: {report['family_counts']}")
    print(f"difficulty: {report['difficulty_counts']}")
    print(f"attack surfaces: {report['attack_surface_counts']}")
    print(f"chain depth histogram: {report['chain_depth_histogram']}")
    print(f"tasks with distractor paths: {report['tasks_with_distractors']}")
    print("\nvuln_class x difficulty grid:")
    print(f"  {'vuln_class':28s} {'easy':>5s} {'medium':>7s} {'hard':>5s}  total")
    for vc in VULN_CLASSES:
        cells = report["vuln_class_x_difficulty_grid"][vc]
        total = sum(cells.values())
        print(f"  {vc:28s} {cells['easy']:>5d} {cells['medium']:>7d} {cells['hard']:>5d}  {total:>5d}")
    if report["coverage"]["empty_cells"]:
        print(f"\nempty cells ({len(report['coverage']['empty_cells'])}): targets for expansion")
        for cell in report["coverage"]["empty_cells"]:
            print(f"  - {cell['vuln_class']} @ {cell['difficulty']}")
    print(f"\nreport: {args.report_output}")


if __name__ == "__main__":
    main()
