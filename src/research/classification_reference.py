"""Numerically stable reference operations for seven-class logit classification.

These functions are educational baselines for the logits emitted by EmotionCNN.
They do not replace PyTorch's optimized kernels in the inference service.
"""

from __future__ import annotations

from math import exp, isfinite, log
from sys import float_info
from typing import Sequence


def stable_softmax(logits: Sequence[float]) -> list[float]:
    """Return a probability distribution using max-shifted exponentiation."""
    if not logits:
        raise ValueError("logits must not be empty")
    if not all(isfinite(value) for value in logits):
        raise ValueError("logits must be finite")

    maximum = max(logits)
    exponentials = [exp(value - maximum) for value in logits]
    normalizer = sum(exponentials)
    return [value / normalizer for value in exponentials]


def categorical_cross_entropy(logits: Sequence[float], target: int) -> float:
    """Return negative log p(target | logits) after stable softmax normalization."""
    if not 0 <= target < len(logits):
        raise ValueError("target must index logits")
    probability = stable_softmax(logits)[target]
    return -log(max(probability, float_info.min))


def confusion_matrix(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> list[list[int]]:
    """Return rows=true labels and columns=predicted labels."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal length")

    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions, strict=True):
        if not 0 <= target < num_classes or not 0 <= prediction < num_classes:
            raise ValueError("labels must be in [0, num_classes)")
        matrix[target][prediction] += 1
    return matrix


def classification_summary(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> dict[str, float]:
    """Compute accuracy and macro precision, recall, and F1 with zero support=0."""
    matrix = confusion_matrix(targets, predictions, num_classes)
    total = sum(sum(row) for row in matrix)
    if total == 0:
        raise ValueError("at least one example is required")

    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    correct = 0
    for label in range(num_classes):
        true_positive = matrix[label][label]
        correct += true_positive
        predicted_positive = sum(matrix[row][label] for row in range(num_classes))
        actual_positive = sum(matrix[label])
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": correct / total,
        "macro_precision": sum(precisions) / num_classes,
        "macro_recall": sum(recalls) / num_classes,
        "macro_f1": sum(f1_scores) / num_classes,
    }
