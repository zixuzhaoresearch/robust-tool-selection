"""Core data structures for TATS framework."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BetaParams:
    """Beta distribution parameters for Bayesian belief tracking."""

    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def sample(self, rng: np.random.Generator) -> float:
        return rng.beta(self.alpha, self.beta)

    def update(self, success: bool, weight: float = 1.0) -> None:
        if success:
            self.alpha += weight
        else:
            self.beta += weight


@dataclass
class DiscountedBetaParams:
    """Beta params with gamma^age discounting over timestamped observations."""

    gamma: float = 0.95
    observations: list[tuple[bool, int]] = field(default_factory=list)

    def record(self, success: bool, timestamp: int) -> None:
        self.observations.append((success, timestamp))

    def weighted_params(self, current_time: int) -> tuple[float, float]:
        """Compute discounted alpha and beta at current_time."""
        alpha = 1.0
        beta = 1.0
        for success, t in self.observations:
            weight = self.gamma ** (current_time - t)
            if success:
                alpha += weight
            else:
                beta += weight
        return alpha, beta

    def sample(self, rng: np.random.Generator, current_time: int) -> float:
        alpha, beta = self.weighted_params(current_time)
        return rng.beta(alpha, beta)

    @property
    def n_obs(self) -> int:
        return len(self.observations)


@dataclass
class ToolMentalModel:
    """Agent's mental model of a single tool's capabilities."""

    tool_id: str
    capability_beliefs: dict[str, BetaParams] = field(default_factory=dict)

    def get_belief(self, capability: str) -> BetaParams:
        if capability not in self.capability_beliefs:
            self.capability_beliefs[capability] = BetaParams()
        return self.capability_beliefs[capability]


@dataclass
class RelationTracker:
    """Track substitutability relationships between tools.

    When tool A fails on capability C, this tracker records a co-failure event.
    Tools that frequently share failure patterns are considered substitutable.
    When tool A fails, substitutable tools get a negative boost while
    non-substitutable alternatives get a positive boost.
    """

    # (tool_a, tool_b, capability) -> co-failure count
    co_failures: dict[tuple[str, str, str], int] = field(default_factory=dict)
    # (tool_id, capability) -> total failure count
    failure_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    # Last failed tool per capability (for co-failure tracking)
    last_failed: dict[str, str | None] = field(default_factory=dict)

    def record_failure(self, tool_id: str, capability: str) -> None:
        """Record a tool failure, updating co-failure counts with previous failure."""
        key = (tool_id, capability)
        self.failure_counts[key] = self.failure_counts.get(key, 0) + 1

        prev = self.last_failed.get(capability)
        if prev is not None and prev != tool_id:
            pair = tuple(sorted([prev, tool_id]))
            co_key = (pair[0], pair[1], capability)
            self.co_failures[co_key] = self.co_failures.get(co_key, 0) + 1

        self.last_failed[capability] = tool_id

    def record_success(self, tool_id: str, capability: str) -> None:
        """Clear last_failed on success (breaks co-failure chain)."""
        self.last_failed[capability] = None

    def substitutability_score(
        self, tool_a: str, tool_b: str, capability: str
    ) -> float:
        """Higher score = more substitutable (both tend to fail together).

        Returns 0 if no co-failure data. Score in [0, 1].
        """
        pair = tuple(sorted([tool_a, tool_b]))
        co_key = (pair[0], pair[1], capability)
        co_count = self.co_failures.get(co_key, 0)
        if co_count == 0:
            return 0.0
        # Normalize by min failures of either tool
        fa = self.failure_counts.get((tool_a, capability), 0)
        fb = self.failure_counts.get((tool_b, capability), 0)
        denom = min(fa, fb)
        if denom == 0:
            return 0.0
        return min(1.0, co_count / denom)

    def get_substitution_bonus(
        self,
        candidate_id: str,
        failed_tool_id: str,
        capability: str,
        bonus_weight: float = 0.1,
    ) -> float:
        """Bonus for candidate when failed_tool just failed.

        Returns positive bonus if candidate is NOT substitutable (good alternative),
        returns negative bonus if candidate IS substitutable (likely to also fail).
        """
        sub_score = self.substitutability_score(candidate_id, failed_tool_id, capability)
        # High substitutability -> negative bonus (avoid); low -> positive (prefer)
        return bonus_weight * (1.0 - 2.0 * sub_score)


@dataclass
class Tool:
    """A tool with visible declarations and hidden ground truth."""

    tool_id: str
    name: str
    declared_capabilities: set[str]
    true_success_rates: dict[str, float]  # hidden from agent
    description_affinity: dict[str, float]  # weights for Vanilla selector

    def declares(self, capability: str) -> bool:
        return capability in self.declared_capabilities


@dataclass
class Query:
    """A query requiring a specific capability."""

    query_id: int
    required_capability: str
    ground_truth_tool_id: str


@dataclass
class Observation:
    """Result of executing a tool for a query."""

    query: Query
    tool_id: str
    capability_used: str
    success: bool
