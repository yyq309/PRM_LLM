"""Tests for the structural-family split integrity (method §13.1)."""

from task_split import compute_split, plan_signature
from web_attack_sim.tasks import load_task_config


def test_split_partitions_all_tasks():
    s = compute_split()
    total = len(s["train"]) + len(s["heldout_instance"]) + len(s["heldout_chain"])
    assert total >= 30
    # disjoint
    assert not (set(s["train"]) & set(s["heldout_instance"]))
    assert not (set(s["train"]) & set(s["heldout_chain"]))


def test_unseen_chain_has_zero_signature_overlap_with_train():
    s = compute_split()
    assert s["audit"]["chain_signature_overlap_with_train"] == 0


def test_unseen_instance_shares_signature_with_train():
    # unseen-instance is the same family with different tokens -> signatures DO overlap
    s = compute_split()
    assert s["audit"]["instance_signature_overlap_with_train"] > 0


def test_heldout_chain_primitives_all_present_in_train():
    # every primitive action in a held-out chain must appear in some train family
    s = compute_split()
    train_prims = set()
    for p in s["train"]:
        train_prims.update(plan_signature(load_task_config(p.replace("tasks/", "tasks/"))))
    for p in s["heldout_chain"]:
        for prim in plan_signature(load_task_config(p)):
            assert prim in train_prims, f"primitive {prim} in held-out chain not seen in train"
