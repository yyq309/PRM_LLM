"""Tests for the error-action identification metric helpers."""

from error_action_eval import detector_metrics, is_error


def test_is_error_buckets():
    assert is_error("high") == 0
    assert is_error("medium") == 0
    assert is_error("low") == 1
    assert is_error("precondition_missing") == 1
    assert is_error("unsafe") == 1
    assert is_error("schema_gap") == 1


def test_detector_metrics_perfect_separation():
    # errors get high err_score and are predicted as error; good get low err_score
    y_true = [1, 1, 0, 0]
    err_score = [0.9, 0.8, 0.1, 0.2]
    err_pred = [1, 1, 0, 0]
    m = detector_metrics(y_true, err_score, err_pred)
    assert m["score_roc_auc"] == 1.0
    assert m["rankhead_precision"] == 1.0
    assert m["rankhead_recall"] == 1.0
    assert m["n_error"] == 2


def test_detector_metrics_degenerate_single_class_auc_none():
    # all errors -> AUC undefined (None), rank-head recall still computable
    m = detector_metrics([1, 1, 1], [0.9, 0.8, 0.7], [1, 1, 0])
    assert m["score_roc_auc"] is None
    assert m["rankhead_recall"] == round(2 / 3, 4)
