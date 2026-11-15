"""Tool selection strategies for TATS experimental framework.

Includes: Random, Vanilla, TATS (Thompson Sampling), EpsilonGreedy, UCB1,
DiscountedTS, SlidingWindowUCB, TATSFrequency (-Bayesian), TATSGreedy (-Explore).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from .environment import SyntheticEnvironment
from .models import DiscountedBetaParams, Query, RelationTracker, ToolMentalModel


class Selector(ABC):
    """Base class for tool selection strategies."""

    @abstractmethod
    def select(self, query: Query, env: SyntheticEnvironment) -> str:
        """Select a tool_id for the given query."""


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class RandomSelector(Selector):
    """Uniform random selection from all tools."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, query: Query, env: SyntheticEnvironment) -> str:
        idx = self.rng.integers(len(env.tools))
        return env.tools[idx].tool_id


class VanillaSelector(Selector):
    """Select from declarers using softmax over description_affinity."""

    def __init__(self, seed: int = 0, temperature: float = 1.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature

    def select(self, query: Query, env: SyntheticEnvironment) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            return env.tools[self.rng.integers(len(env.tools))].tool_id

        affinities = np.array([t.description_affinity.get(cap, 0.1) for t in declarers])
        logits = affinities / self.temperature
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        idx = self.rng.choice(len(declarers), p=probs)
        return declarers[idx].tool_id


# ---------------------------------------------------------------------------
# TATS (full model)
# ---------------------------------------------------------------------------


class TATSSelector(Selector):
    """Thompson Sampling over per-capability Beta posteriors."""

    def __init__(
        self,
        seed: int = 0,
        cold_start_threshold: int = 10,
        obs_weight: float = 1.0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.mental_models: dict[str, ToolMentalModel] = {}
        # If total observations for a (tool, cap) < threshold, skip risk penalty.
        # Currently a hook for future risk penalty integration.
        self.cold_start_threshold = cold_start_threshold
        # Observation weight: 1.0 = full tool-call signal, <1.0 simulates
        # noisier step/task-level observations (see hierarchical update Eq. 9)
        self.obs_weight = obs_weight

    def _get_model(self, tool_id: str) -> ToolMentalModel:
        if tool_id not in self.mental_models:
            self.mental_models[tool_id] = ToolMentalModel(tool_id)
        return self.mental_models[tool_id]

    def _obs_count(self, tool_id: str, capability: str) -> int:
        model = self._get_model(tool_id)
        belief = model.get_belief(capability)
        return int(belief.alpha + belief.beta - 2)  # subtract the two priors

    def select(self, query: Query, env: SyntheticEnvironment) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            return env.tools[self.rng.integers(len(env.tools))].tool_id

        best_tool_id = declarers[0].tool_id
        best_sample = -1.0

        for tool in declarers:
            model = self._get_model(tool.tool_id)
            belief = model.get_belief(cap)
            sample = belief.sample(self.rng)

            # Cold start gating: if insufficient observations, no risk penalty
            # (risk penalty integration point -- currently a no-op)
            _in_cold_start = self._obs_count(tool.tool_id, cap) < self.cold_start_threshold

            if sample > best_sample:
                best_sample = sample
                best_tool_id = tool.tool_id

        return best_tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        model = self._get_model(tool_id)
        model.get_belief(capability).update(success, weight=self.obs_weight)

    def get_predicted_prob(self, tool_id: str, capability: str) -> float:
        model = self._get_model(tool_id)
        return model.get_belief(capability).mean


# ---------------------------------------------------------------------------
# Bandit baselines
# ---------------------------------------------------------------------------


class EpsilonGreedySelector(Selector):
    """Epsilon-greedy: exploit best observed success rate, explore with prob epsilon."""

    def __init__(self, epsilon: float = 0.1, seed: int = 0) -> None:
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.successes: dict[tuple[str, str], int] = defaultdict(int)
        self.counts: dict[tuple[str, str], int] = defaultdict(int)

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        key = (tool_id, capability)
        self.counts[key] += 1
        if success:
            self.successes[key] += 1

    def _success_rate(self, tool_id: str, capability: str) -> float:
        key = (tool_id, capability)
        n = self.counts[key]
        if n == 0:
            return 0.0
        return self.successes[key] / n

    def select(self, query: Query, env: SyntheticEnvironment) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            return env.tools[self.rng.integers(len(env.tools))].tool_id

        if self.rng.random() < self.epsilon:
            idx = self.rng.integers(len(declarers))
            return declarers[idx].tool_id

        rates = [self._success_rate(t.tool_id, cap) for t in declarers]
        best_rate = max(rates)
        best_indices = [i for i, r in enumerate(rates) if r == best_rate]
        idx = self.rng.choice(best_indices)
        return declarers[idx].tool_id


class UCBSelector(Selector):
    """UCB1: mean + c * sqrt(log(t) / n). Ties broken randomly."""

    def __init__(self, c: float = 2.0, seed: int = 0) -> None:
        self.c = c
        self.rng = np.random.default_rng(seed)
        self.successes: dict[tuple[str, str], int] = defaultdict(int)
        self.counts: dict[tuple[str, str], int] = defaultdict(int)
        self.total: int = 0

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        key = (tool_id, capability)
        self.counts[key] += 1
        if success:
            self.successes[key] += 1
        self.total += 1

    def _ucb_score(self, tool_id: str, capability: str) -> float:
        key = (tool_id, capability)
        n = self.counts[key]
        if n == 0:
            return float("inf")
        mean = self.successes[key] / n
        exploration = self.c * math.sqrt(math.log(max(self.total, 1)) / n)
        return mean + exploration

    def select(self, query: Query, env: SyntheticEnvironment) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            return env.tools[self.rng.integers(len(env.tools))].tool_id

        scores = [self._ucb_score(t.tool_id, cap) for t in declarers]
        best_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if s == best_score]
        idx = self.rng.choice(best_indices)
        return declarers[idx].tool_id


# ---------------------------------------------------------------------------
# Discounted Thompson Sampling
# ---------------------------------------------------------------------------


class DiscountedTSSelector(Selector):
    """Thompson Sampling with gamma^age discounting on observations."""

    def __init__(self, gamma: float = 0.95, seed: int = 0) -> None:
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.beliefs: dict[tuple[str, str], DiscountedBetaParams] = {}
        self.timestep: int = 0

    def _get_belief(self, tool_id: str, capability: str) -> DiscountedBetaParams:
        key = (tool_id, capability)
        if key not in self.beliefs:
            self.beliefs[key] = DiscountedBetaParams(gamma=self.gamma)
        return self.beliefs[key]

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        belief = self._get_belief(tool_id, capability)
        belief.record(success, self.timestep)
        self.timestep += 1

    def select(self, query: Query, env: SyntheticEnvironment) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            return env.tools[self.rng.integers(len(env.tools))].tool_id

        best_tool_id = declarers[0].tool_id
        best_sample = -1.0

        for tool in declarers:
            belief = self._get_belief(tool.tool_id, cap)
            sample = belief.sample(self.rng, self.timestep)
            if sample > best_sample:
                best_sample = sample
                best_tool_id = tool.tool_id

        return best_tool_id


# ---------------------------------------------------------------------------
# Sliding Window UCB
# ---------------------------------------------------------------------------


class SlidingWindowUCBSelector(Selector):
    """UCB1 using only the most recent W observations per (tool, capability)."""

    def __init__(self, window: int = 50, c: float = 2.0, seed: int = 0) -> None:
        self.window = window
        self.c = c
        self.rng = np.random.default_rng(seed)
        # Store full history per key; slice at query time
        self.history: dict[tuple[str, str], list[bool]] = defaultdict(list)
        self.total: int = 0

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        self.history[(tool_id, capability)].append(success)
        self.total += 1

    def _ucb_score(self, tool_id: str, capability: str) -> float:
        key = (tool_id, capability)
        obs = self.history[key][-self.window:]
        n = len(obs)
        if n == 0:
            return float("inf")
        mean = sum(obs) / n
        exploration = self.c * math.sqrt(math.log(max(self.total, 1)) / n)
        return mean + exploration

    def select(self, query: Query, env: SyntheticEnvironment) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            return env.tools[self.rng.integers(len(env.tools))].tool_id

        scores = [self._ucb_score(t.tool_id, cap) for t in declarers]
        best_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if s == best_score]
        idx = self.rng.choice(best_indices)
        return declarers[idx].tool_id


# ---------------------------------------------------------------------------
# TATS ablations
# ---------------------------------------------------------------------------


class LinUCBSelector(Selector):
    """LinUCB contextual bandit baseline.

    Uses a one-hot encoding of the required capability as context vector.
    Maintains a separate linear model (A, b) per tool, predicting reward
    from context features.

    LinUCB score: θ̂^T x + α * sqrt(x^T A^{-1} x)
    where θ̂ = A^{-1} b.

    Parameters
    ----------
    n_features:
        Dimensionality of the context vector (= number of capabilities).
    alpha:
        Exploration parameter controlling confidence bound width.
    capability_list:
        Ordered list of capability names for one-hot encoding.
    seed:
        Random seed for tie-breaking.
    """

    def __init__(
        self,
        n_features: int = 5,
        alpha: float = 1.0,
        capability_list: list[str] | None = None,
        seed: int = 0,
    ) -> None:
        self.n_features = n_features
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.capability_list = capability_list or []
        self._cap_to_idx: dict[str, int] = {
            c: i for i, c in enumerate(self.capability_list)
        }
        # Per-tool: A (d×d matrix), b (d×1 vector)
        self._A: dict[str, np.ndarray] = {}
        self._b: dict[str, np.ndarray] = {}

    def _ensure_tool(self, tool_id: str) -> None:
        if tool_id not in self._A:
            self._A[tool_id] = np.eye(self.n_features)
            self._b[tool_id] = np.zeros(self.n_features)

    def _context(self, capability: str) -> np.ndarray:
        """One-hot encoding of the required capability."""
        x = np.zeros(self.n_features)
        idx = self._cap_to_idx.get(capability)
        if idx is not None:
            x[idx] = 1.0
        return x

    def _score(self, tool_id: str, x: np.ndarray) -> float:
        self._ensure_tool(tool_id)
        A_inv = np.linalg.inv(self._A[tool_id])
        theta_hat = A_inv @ self._b[tool_id]
        exploit = float(theta_hat @ x)
        explore = self.alpha * float(np.sqrt(x @ A_inv @ x))
        return exploit + explore

    def select(self, query: Query, env: SyntheticEnvironment) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            return env.tools[self.rng.integers(len(env.tools))].tool_id

        x = self._context(cap)
        scores = [self._score(t.tool_id, x) for t in declarers]
        best_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if abs(s - best_score) < 1e-12]
        idx = self.rng.choice(best_indices)
        return declarers[idx].tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        self._ensure_tool(tool_id)
        x = self._context(capability)
        reward = 1.0 if success else 0.0
        self._A[tool_id] += np.outer(x, x)
        self._b[tool_id] += reward * x


class TATSFrequencySelector(Selector):
    """TATS minus Bayesian: raw frequency success rate, no sampling."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.successes: dict[tuple[str, str], int] = defaultdict(int)
        self.counts: dict[tuple[str, str], int] = defaultdict(int)

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        key = (tool_id, capability)
        self.counts[key] += 1
        if success:
            self.successes[key] += 1

    def select(self, query: Query, env: SyntheticEnvironment) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            return env.tools[self.rng.integers(len(env.tools))].tool_id

        # Check if any declarer has observations
        has_obs = any(self.counts[(t.tool_id, cap)] > 0 for t in declarers)
        if not has_obs:
            idx = self.rng.integers(len(declarers))
            return declarers[idx].tool_id

        rates = []
        for t in declarers:
            key = (t.tool_id, cap)
            n = self.counts[key]
            rates.append(self.successes[key] / n if n > 0 else 0.0)

        best_rate = max(rates)
        best_indices = [i for i, r in enumerate(rates) if r == best_rate]
        idx = self.rng.choice(best_indices)
        return declarers[idx].tool_id


class TATSGreedySelector(Selector):
    """TATS minus Explore: argmax of Beta mean instead of Thompson Sampling."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.mental_models: dict[str, ToolMentalModel] = {}

    def _get_model(self, tool_id: str) -> ToolMentalModel:
        if tool_id not in self.mental_models:
            self.mental_models[tool_id] = ToolMentalModel(tool_id)
        return self.mental_models[tool_id]

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        model = self._get_model(tool_id)
        model.get_belief(capability).update(success)

    def select(self, query: Query, env: SyntheticEnvironment) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            return env.tools[self.rng.integers(len(env.tools))].tool_id

        means = []
        for tool in declarers:
            model = self._get_model(tool.tool_id)
            means.append(model.get_belief(cap).mean)

        best_mean = max(means)
        best_indices = [i for i, m in enumerate(means) if m == best_mean]
        idx = self.rng.choice(best_indices)
        return declarers[idx].tool_id


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TATSRelationalSelector(Selector):
    """TATS + substitutability-aware selection.

    Extends TATSSelector with a RelationTracker that boosts alternatives
    when the previously selected tool failed. When tool A just failed on
    capability C, tools that are NOT substitutable with A get a positive
    bonus, while substitutable tools get a penalty.
    """

    def __init__(self, seed: int = 0, bonus_weight: float = 0.1) -> None:
        self.rng = np.random.default_rng(seed)
        self.mental_models: dict[str, ToolMentalModel] = {}
        self.relations = RelationTracker()
        self.bonus_weight = bonus_weight
        self._last_selected: dict[str, str] = {}  # capability -> last tool_id
        self._last_success: dict[str, bool] = {}  # capability -> last success

    def _get_model(self, tool_id: str) -> ToolMentalModel:
        if tool_id not in self.mental_models:
            self.mental_models[tool_id] = ToolMentalModel(tool_id)
        return self.mental_models[tool_id]

    def select(self, query: Query, env: SyntheticEnvironment) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            return env.tools[self.rng.integers(len(env.tools))].tool_id

        best_tool_id = declarers[0].tool_id
        best_score = -1.0

        last_tool = self._last_selected.get(cap)
        last_was_failure = last_tool is not None and not self._last_success.get(cap, True)

        for tool in declarers:
            model = self._get_model(tool.tool_id)
            belief = model.get_belief(cap)
            sample = belief.sample(self.rng)

            # Add substitutability bonus when last selection failed
            bonus = 0.0
            if last_was_failure and last_tool is not None:
                bonus = self.relations.get_substitution_bonus(
                    tool.tool_id, last_tool, cap, self.bonus_weight
                )

            score = sample + bonus
            if score > best_score:
                best_score = score
                best_tool_id = tool.tool_id

        return best_tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        model = self._get_model(tool_id)
        model.get_belief(capability).update(success)

        if success:
            self.relations.record_success(tool_id, capability)
        else:
            self.relations.record_failure(tool_id, capability)

        self._last_selected[capability] = tool_id
        self._last_success[capability] = success

    def get_predicted_prob(self, tool_id: str, capability: str) -> float:
        model = self._get_model(tool_id)
        return model.get_belief(capability).mean
