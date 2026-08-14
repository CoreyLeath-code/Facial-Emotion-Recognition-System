"""Analytical tests for the research reference classification operations."""

from math import isclose, log

import pytest

from src.research.classification_reference import (
    categorical_cross_entropy,
    classification_summary,
    confusion_matrix,
    stable_softmax,
)


def test_stable_softmax_is_normalized_and_shift_invariant():
    baseline = stable_softmax([-2.0, 0.0, 3.0])
    shifted = stable_softmax([98.0, 100.0, 103.0])

    assert isclose(sum(baseline), 1.0)
    assert baseline[2] > baseline[1] > baseline[0]
    assert all(isclose(left, right) for left, right in zip(baseline, shifted))


def test_cross_entropy_matches_uniform_two_class_case():
    assert isclose(categorical_cross_entropy([0.0, 0.0], target=1), log(2.0))


def test_confusion_matrix_and_summary_match_known_case():
    targets = [0, 1, 2, 2]
    predictions = [0, 0, 2, 1]

    assert confusion_matrix(targets, predictions, num_classes=3) == [
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
    ]
    summary = classification_summary(targets, predictions, num_classes=3)
    assert isclose(summary["accuracy"], 0.5)
    assert isclose(summary["macro_precision"], 0.5)
    assert isclose(summary["macro_recall"], 0.5)
    assert isclose(summary["macro_f1"], 4.0 / 9.0)


@pytest.mark.parametrize(
    ("logits", "target"),
    [([], 0), ([0.0, 1.0], -1), ([0.0, 1.0], 2)],
)
def test_cross_entropy_rejects_invalid_inputs(logits, target):
    with pytest.raises(ValueError):
        categorical_cross_entropy(logits, target)
