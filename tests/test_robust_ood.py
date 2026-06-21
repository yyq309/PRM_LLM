"""Tests for the held-out OOD corruption family B in the robust PRM trainer."""

import random

from train_prm_robust import OOD_RELABEL, degrade_context_ood, parse_context_fields

CONTEXT = (
    "Scenario abc123. Known paths: ['/', '/login']. Known forms: ['/login']. "
    "Auth state: anonymous. Shell state: none. Remaining budget: 12. "
    "Failed branches: {}. Recent feedback: path_found."
)


def test_parse_context_fields_splits_labels():
    prefix, fields = parse_context_fields(CONTEXT)
    assert prefix.startswith("Scenario")
    labels = [lab for lab, _ in fields]
    assert "Known paths:" in labels and "Auth state:" in labels


def test_ood_relabels_reorders_and_keeps_no_original_labels():
    rng = random.Random(0)
    out = degrade_context_ood(CONTEXT, rng)
    # original field labels are renamed -> none of the masking-family labels survive
    assert "Known paths:" not in out
    assert "Auth state:" not in out
    # at least one synonym label is present
    assert any(syn in out for syn in OOD_RELABEL.values())


def test_ood_is_distinct_from_masking():
    # OOD family must not introduce the masking-family artifacts ([MASKED]/OOA token)
    rng = random.Random(1)
    out = degrade_context_ood(CONTEXT, rng)
    assert "[MASKED]" not in out
    assert "out_of_abstraction_event" not in out


def test_ood_preserves_field_values():
    # values (e.g. the discovered paths) survive relabeling so the row stays informative
    rng = random.Random(2)
    out = degrade_context_ood(CONTEXT, rng)
    assert "/login" in out
