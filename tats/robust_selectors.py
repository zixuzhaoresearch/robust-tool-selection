"""Robust contextual bandit baselines (reward-corruption defense).

Adapts the spirit of Ding et al. 2022 (Robust linear contextual bandits under
corruption) for empirical comparison in our tool-selection setting. The core
idea is that under adversarial reward corruption, the learner should limit the
influence of any single observation whose realized reward disagrees sharply
with the current predicted reward.

These baselines are designed for the threat model of *reward* corruption --
NOT for the *tool injection* threat model we evaluate. Including them as
and demonstrates that off-the-shelf robust-CB methods do not generalize to
candidate-set injection attacks.
"""
from __future__ import annotations

import numpy as np

from .bilints_selector import LinUCBSelector
from .models import Query


class RobustLinUCBSelector(LinUCBSelector):
    """LinUCB with reward-corruption-robust updates (Ding et al. 2022 style).

    Each observation is clipped so that individual rewards cannot drift the
    posterior mean by more than a bounded amount per step. Concretely, we
    clip the effective reward used in the update to the interval
    ``[r_pred - tau, r_pred + tau]`` where ``r_pred = x^T theta_mean`` and
    ``tau`` is the per-round corruption budget.

    This is NOT the full Ding 2022 algorithm (which tracks a global corruption
    budget ``C`` and uses a variance-adjusted confidence bonus). It captures
    the practical essence: outlier rewards cannot individually capture the
    posterior.

    Parameters
    ----------
    tau:
        Per-round reward clipping radius. Smaller = more conservative
        (robust to corruption, slower learning). Default 0.5 (half the binary
        reward range).
    """

    name = "RobustLinUCB"

    def __init__(
        self,
        feature_extractor,
        seed: int = 0,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        tau: float = 0.5,
    ) -> None:
        super().__init__(
            feature_extractor=feature_extractor,
            seed=seed,
            alpha=alpha,
            lambda_reg=lambda_reg,
            sigma2=sigma2,
        )
        self._tau = tau

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        x = self._cached_features.get(tool_id)
        if x is None:
            return

        r = 1.0 if success else 0.0
        r_pred = float(x @ self._theta_mean)
        r_clipped = float(np.clip(r, r_pred - self._tau, r_pred + self._tau))

        inv_sigma2 = 1.0 / self._sigma2
        self._B += inv_sigma2 * np.outer(x, x)
        self._f += inv_sigma2 * r_clipped * x
        self._theta_mean = np.linalg.solve(self._B, self._f)
        self._B_inv = None
        self.n_updates += 1
