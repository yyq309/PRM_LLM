"""Structural-family train/held-out split (method §13.1).

Splits the de-templated task set so the held-out set is genuinely unseen at two levels:

- unseen_chain: WHOLE topology families held out (their expert_plan action-type signature
  never appears in train), while every primitive action in those families still appears in
  some train family — so the oracle can transfer primitives but the *combination* is new.
- unseen_instance: one instance per TRAIN family held out (same topology, different tokens).

This replaces the earlier templated split where held-out instances shared byte-identical
plan signatures with training. The split is computed from the task `family` tags and the
plan signatures, and written to outputs/task_split.json for auditing.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
from collections import defaultdict
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web_attack_sim.tasks import bundled_task_paths, load_task_config  # noqa: E402

# Families held out entirely (unseen vulnerability chain). Chosen so every primitive in
# them still appears in a TRAIN family (rce/exploit via rce_shell, privesc via
# upload_default_privesc, content/cred_use via leak_file/injection_login/upload_leak_shell).
UNSEEN_CHAIN_FAMILIES = ["rce_privesc", "leak_login"]


def plan_signature(task: dict[str, Any]) -> tuple:
    return tuple(s if isinstance(s, str) else s["action_type"] for s in task.get("expert_plan", []))


def compute_split() -> dict[str, Any]:
    by_family: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    rel: dict[str, str] = {}
    for path in bundled_task_paths():
        task = load_task_config(path)
        family = task.get("family", "untagged")
        relpath = "tasks/" + path.name
        rel[task["task_id"]] = relpath
        by_family[family].append((relpath, task))

    train, heldout_instance, heldout_chain = [], [], []
    train_families, heldout_chain_families = [], []
    for family, items in sorted(by_family.items()):
        items = sorted(items, key=lambda it: it[0])
        if family in UNSEEN_CHAIN_FAMILIES:
            heldout_chain_families.append(family)
            heldout_chain.extend(rp for rp, _ in items)
        else:
            train_families.append(family)
            # hold out the first instance of each train family as unseen-instance
            heldout_instance.append(items[0][0])
            train.extend(rp for rp, _ in items[1:])

    # Signature sets for the integrity audit.
    sig = {}
    for path in bundled_task_paths():
        task = load_task_config(path)
        sig["tasks/" + path.name] = plan_signature(task)
    train_sigs = {sig[p] for p in train}
    chain_sig_overlap = sum(1 for p in heldout_chain if sig[p] in train_sigs)
    instance_sig_overlap = sum(1 for p in heldout_instance if sig[p] in train_sigs)

    return {
        "unseen_chain_families": heldout_chain_families,
        "train_families": train_families,
        "train": sorted(train),
        "heldout_instance": sorted(heldout_instance),
        "heldout_chain": sorted(heldout_chain),
        "heldout_all": sorted(heldout_instance + heldout_chain),
        "audit": {
            "num_train": len(train),
            "num_heldout_instance": len(heldout_instance),
            "num_heldout_chain": len(heldout_chain),
            "chain_signature_overlap_with_train": chain_sig_overlap,
            "instance_signature_overlap_with_train": instance_sig_overlap,
            "note": (
                "chain_signature_overlap MUST be 0 (genuine unseen chains); instance overlap is "
                "EXPECTED non-zero (same family, different tokens = unseen-instance)."
            ),
        },
    }


def get_split() -> dict[str, Any]:
    return compute_split()


def write_split(path: Path | None = None) -> dict[str, Any]:
    split = compute_split()
    out = path or (ROOT / "outputs" / "task_split.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    return split


if __name__ == "__main__":
    s = write_split()
    print(json.dumps({k: v for k, v in s.items() if k not in ("train", "heldout_instance", "heldout_chain", "heldout_all")}, ensure_ascii=False, indent=2))
    print(f"train={len(s['train'])} heldout_instance={len(s['heldout_instance'])} heldout_chain={len(s['heldout_chain'])}")
