"""Tests for the training-stage helper logic added to complete method §10/§11.2/§12.1."""

import argparse

from normalizer_benchmark import confusion, per_category_prf
from oracle_label_confidence import merge
from qstar_labels import plan_to_int
from train_prm_joint import build_pairs


def _args():
    return argparse.Namespace(margin_scale=3.0, var_weight=2.0)


def test_merge_rule_sample_gets_full_confidence():
    rule = {"sample_id": "r", "label_source": "precondition_rule", "rank_label": "precondition_missing", "score": 0.0}
    seed_rows = [{"r": dict(rule)} for _ in range(3)]
    merged, report = merge(seed_rows, _args())
    assert merged[0]["multiseed_confidence"] == 1.0
    assert report["num_oracle_labeled"] == 0


def test_merge_oracle_full_agreement_high_confidence():
    def oracle(rank, score):
        return {"sample_id": "o", "label_source": "oracle", "rank_label": rank, "score": score,
                "value_gap": 1.0, "oracle_q_report": {"top_actions": [{"q": 10.0}, {"q": 7.0}]}}
    seed_rows = [{"o": oracle("high", 0.9)}, {"o": oracle("high", 0.88)}, {"o": oracle("high", 0.92)}]
    merged, report = merge(seed_rows, _args())
    assert merged[0]["seed_rank_agreement"] == 1.0
    assert merged[0]["rank_label"] == "high"
    assert merged[0]["multiseed_confidence"] > 0.5
    assert report["num_oracle_labeled"] == 1


def test_merge_oracle_disagreement_lowers_confidence():
    def oracle(rank, score):
        return {"sample_id": "o", "label_source": "oracle", "rank_label": rank, "score": score,
                "value_gap": 1.0, "oracle_q_report": {"top_actions": [{"q": 10.0}, {"q": 9.5}]}}
    seed_rows = [{"o": oracle("high", 0.9)}, {"o": oracle("medium", 0.5)}, {"o": oracle("low", 0.2)}]
    merged, _ = merge(seed_rows, _args())
    assert merged[0]["seed_rank_agreement"] < 0.5  # 1/3 modal
    assert merged[0]["multiseed_confidence"] < 0.5


def test_build_pairs_direction_and_filtering():
    rows = [
        {"label_source": "oracle", "candidate_group": "g", "score": 0.9, "multiseed_confidence": 0.8},
        {"label_source": "oracle", "candidate_group": "g", "score": 0.3, "multiseed_confidence": 0.6},
        {"label_source": "oracle", "candidate_group": "g", "score": 0.9, "multiseed_confidence": 1.0},  # tie with row0
        {"label_source": "precondition_rule", "candidate_group": "g", "score": 0.0},  # not oracle -> excluded
    ]
    pairs = build_pairs(rows, eps=1e-6)
    # row0 (0.9) > row1 (0.3): better first
    assert (0, 1, 0.6) in pairs
    # tie (row0 vs row2) excluded; rule row excluded
    assert all(set(p[:2]) != {0, 2} for p in pairs)


def test_plan_to_int():
    assert plan_to_int("web_path_enumeration") == 2
    assert plan_to_int({"action_type": "credential_use"}) == 7


def test_per_category_prf_perfect():
    pairs = [("valid", "valid"), ("unsafe", "unsafe"), ("valid", "valid")]
    prf = per_category_prf(pairs)
    assert prf["valid"]["f1"] == 1.0
    assert prf["valid"]["support"] == 2


def test_confusion_counts():
    pairs = [("valid", "valid"), ("valid", "unsupported")]
    table = confusion(pairs)
    assert table["valid"]["valid"] == 1
    assert table["valid"]["unsupported"] == 1
