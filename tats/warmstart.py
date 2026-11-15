"""Warm-start initialization for BilinTS from BFCL ground-truth labels.

Fits a ridge regression on (feature, label) pairs extracted from BFCL
scenarios, producing a prior mean theta_0 AND precision matrix B_0 that
encode which description-level features correlate with the correct tool.

IMPORTANT: The training scenarios passed here must be from a DIFFERENT
dataset than the evaluation scenarios to avoid data leakage.  For example,
fit on ``v4_multiple`` and evaluate on ``v4_live_multiple``.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .bfcl_adapter import BFCLScenario
from .features import ToolFeatureExtractor


def compute_warm_start(
    train_scenarios: list[BFCLScenario],
    feature_extractor: ToolFeatureExtractor,
    lambda_reg: float = 1.0,
    sigma2: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute warm-start theta_0 AND precision matrix B_0 from training data.

    IMPORTANT: train_scenarios must be from a DIFFERENT dataset than evaluation
    to avoid data leakage. E.g., fit on v4_multiple, evaluate on v4_live_multiple.

    For each scenario, for each candidate tool:
        x = feature_extractor.extract(scenario_idx, tool, n_candidates)
        r = 1.0 if tool is ground truth, 0.0 otherwise

    Then compute:
        B_0 = (1/sigma2) * X^T X + lambda * I
        f_0 = (1/sigma2) * X^T y
        theta_0 = B_0^{-1} f_0

    Parameters
    ----------
    train_scenarios:
        List of BFCL scenarios with candidate tools and ground-truth labels.
        Must NOT overlap with the evaluation scenarios.
    feature_extractor:
        Feature extractor initialized with the query corpus.
    lambda_reg:
        Ridge regularization strength (prior precision).
    sigma2:
        Observation noise variance. Must match the sigma2 used in
        BilinTSSelector for consistent Bayesian updates.

    Returns
    -------
    (theta_0, B_0) : tuple[np.ndarray, np.ndarray]
        theta_0 — shape (d,), warm-start prior mean.
        B_0     — shape (d, d), warm-start precision matrix.
    """
    d = feature_extractor.n_features

    # Accumulate X^T X and X^T y incrementally (avoids storing full matrix X)
    XtX = np.zeros((d, d))
    Xty = np.zeros(d)

    for scenario_idx, scenario in enumerate(train_scenarios):
        n_candidates = len(scenario.candidate_tools)
        for tool in scenario.candidate_tools:
            x = feature_extractor.extract(scenario_idx, tool, n_candidates)
            r = 1.0 if tool.tool_id == scenario.ground_truth_tool_id else 0.0

            XtX += np.outer(x, x)
            Xty += r * x

    # B_0 = (1/sigma2) * X^T X + lambda * I
    B_0 = (1.0 / sigma2) * XtX + lambda_reg * np.eye(d)

    # f_0 = (1/sigma2) * X^T y
    f_0 = (1.0 / sigma2) * Xty

    # theta_0 = B_0^{-1} f_0
    theta_0 = np.linalg.solve(B_0, f_0)

    return theta_0, B_0


def train_offline_logistic(
    train_scenarios: list[BFCLScenario],
    feature_extractor: ToolFeatureExtractor,
    C: float = 1.0,
    seed: int = 0,
) -> LogisticRegression:
    """Train a logistic regression on (feature, label) pairs from scenarios.

    For each scenario, for each candidate tool:
        x = feature_extractor.extract(scenario_idx, tool, n_candidates)
        y = 1 if tool is ground truth, 0 otherwise

    IMPORTANT: train_scenarios must be from a DIFFERENT dataset than evaluation
    to avoid data leakage (same protocol as compute_warm_start).

    Parameters
    ----------
    train_scenarios:
        List of BFCL scenarios with candidate tools and ground-truth labels.
    feature_extractor:
        Feature extractor initialized with the training query corpus.
    C:
        Inverse of regularization strength (larger = less regularization).
    seed:
        Random seed for solver.

    Returns
    -------
    LogisticRegression
        Fitted sklearn model with predict_proba() support.
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for scenario_idx, scenario in enumerate(train_scenarios):
        n_candidates = len(scenario.candidate_tools)
        for tool in scenario.candidate_tools:
            x = feature_extractor.extract(scenario_idx, tool, n_candidates)
            y = 1 if tool.tool_id == scenario.ground_truth_tool_id else 0
            X_list.append(x)
            y_list.append(y)

    X = np.array(X_list)
    y = np.array(y_list)

    model = LogisticRegression(
        C=C, l1_ratio=0, random_state=seed, max_iter=500, solver="lbfgs",
    )
    model.fit(X, y)
    return model
