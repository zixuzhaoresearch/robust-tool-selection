"""Evaluation metrics for tool selection experiments."""

from __future__ import annotations

import numpy as np


def tool_selection_accuracy(selected: list[str], ground_truth: list[str]) -> float:
    """Fraction of queries where selected tool matches ground truth."""
    if not selected:
        return 0.0
    correct = sum(s == g for s, g in zip(selected, ground_truth))
    return correct / len(selected)


def windowed_tsa(
    selected: list[str], ground_truth: list[str], window: int = 50
) -> list[float]:
    """TSA computed over sliding windows."""
    results = []
    for start in range(0, len(selected), window):
        end = min(start + window, len(selected))
        chunk_sel = selected[start:end]
        chunk_gt = ground_truth[start:end]
        results.append(tool_selection_accuracy(chunk_sel, chunk_gt))
    return results


def expected_calibration_error(
    predicted_probs: list[float],
    actual_outcomes: list[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error: weighted average of |confidence - accuracy| per bin."""
    if not predicted_probs:
        return 0.0

    preds = np.array(predicted_probs)
    actuals = np.array(actual_outcomes, dtype=float)
    n = len(preds)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (preds > bin_edges[i]) & (preds <= bin_edges[i + 1])
        # Include 0.0 in the first bin
        if i == 0:
            mask |= preds == 0.0
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = preds[mask].mean()
        avg_acc = actuals[mask].mean()
        ece += (count / n) * abs(avg_conf - avg_acc)

    return float(ece)


def cumulative_regret(selected: list[str], ground_truth: list[str]) -> list[float]:
    """Cumulative regret: running count of suboptimal selections."""
    regret = []
    total = 0.0
    for s, g in zip(selected, ground_truth):
        if s != g:
            total += 1.0
        regret.append(total)
    return regret


def learning_curve(
    selected: list[str], ground_truth: list[str], window: int = 20
) -> tuple[list[int], list[float]]:
    """Rolling accuracy over time."""
    xs = []
    ys = []
    correct_window: list[int] = []

    for i, (s, g) in enumerate(zip(selected, ground_truth)):
        correct_window.append(1 if s == g else 0)
        if len(correct_window) > window:
            correct_window.pop(0)
        if i >= window - 1:
            xs.append(i)
            ys.append(sum(correct_window) / len(correct_window))

    return xs, ys
