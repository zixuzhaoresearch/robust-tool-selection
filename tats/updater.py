"""Bayesian updater: Beta-Bernoulli conjugate updates."""

from __future__ import annotations

from .models import Observation, ToolMentalModel


class BayesianUpdater:
    """Update tool mental models from observations (TATS-NoReflect variant)."""

    def update(self, model: ToolMentalModel, observation: Observation) -> None:
        belief = model.get_belief(observation.capability_used)
        belief.update(observation.success)
