"""BFCL-specific selectors for tool selection.

In BFCL, each scenario has a unique ``scenario_id`` as its "capability",
which means the standard capability-based learning (Beta posteriors keyed
by capability) cannot generalise across scenarios.  These selectors address
that gap with text-based matching and domain-level posterior sharing.

Selectors
---------
RandomBFCLSelector
    Uniform random among per-scenario candidate tools.
DescriptionMatchSelector
    Static word-overlap between query text and tool descriptions.
ParameterMatchSelector
    Counts how many parameter names from the tool spec appear in the query.
TATSBFCLSelector
    Thompson Sampling with posteriors keyed by *domain* (prefix of the
    function name) instead of scenario_id, enabling cross-scenario learning.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from .models import Query, ToolMentalModel
from .selector import Selector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenise(text: str) -> set[str]:
    """Lowercase word tokens (alphanumeric runs)."""
    return {w.lower() for w in _WORD_RE.findall(text)}


# ---------------------------------------------------------------------------
# 1. RandomBFCLSelector
# ---------------------------------------------------------------------------


class RandomBFCLSelector(Selector):
    """Uniform random selection among per-scenario declarers.

    Identical in spirit to :class:`RandomSelector`, but always draws from the
    candidate tools returned by ``env.get_declarers(capability)`` rather than
    the global tool list.  This is the correct behaviour for BFCL where each
    scenario defines its own candidate set.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, query: Query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            # Fallback: pick from global tool list (should not happen in
            # well-formed BFCL data, but keeps parity with other selectors).
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id
        idx = self.rng.integers(len(declarers))
        return declarers[idx].tool_id


# ---------------------------------------------------------------------------
# 2. DescriptionMatchSelector
# ---------------------------------------------------------------------------


class DescriptionMatchSelector(Selector):
    """Select the tool whose description has the highest word overlap with
    the query text.

    Uses a simple recall-oriented overlap score::

        score = |query_words ∩ desc_words| / |query_words|

    No external NLP dependencies — just lowercased alphanumeric tokens.

    Parameters
    ----------
    query_texts:
        Mapping from ``query_id`` to the raw query string (typically
        extracted from :class:`BFCLScenario.query_text`).
    seed:
        Random seed for tie-breaking.
    """

    def __init__(self, query_texts: dict[int, str], seed: int = 0) -> None:
        self.query_texts = query_texts
        self.rng = np.random.default_rng(seed)

    def _similarity(self, query_text: str, tool_description: str) -> float:
        """Word-level recall of query tokens in the tool description."""
        q_tokens = _tokenise(query_text)
        if not q_tokens:
            return 0.0
        d_tokens = _tokenise(tool_description)
        return len(q_tokens & d_tokens) / len(q_tokens)

    def select(self, query: Query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id

        query_text = self.query_texts.get(query.query_id, "")

        scores = [
            self._similarity(query_text, getattr(t, "description", ""))
            for t in declarers
        ]
        best_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if s == best_score]
        idx = self.rng.choice(best_indices)
        return declarers[idx].tool_id


# ---------------------------------------------------------------------------
# 3. TATSBFCLSelector
# ---------------------------------------------------------------------------


class TATSBFCLSelector(Selector):
    """Thompson Sampling with domain-level posterior sharing for BFCL.

    Standard TATS keys Beta posteriors by ``(tool_id, capability)`` where
    ``capability`` is the scenario_id.  Since each BFCL scenario has a
    unique id, posteriors never transfer across scenarios.

    This selector instead extracts a **domain key** from the tool's function
    name (the prefix before the first ``.``) and uses that as the capability
    key.  Tools from the same domain thereby share posterior evidence,
    enabling cross-scenario learning.

    Domain key examples::

        "triangle_properties.get"       → "triangle_properties"
        "math.solve_equation"           → "math"
        "get_weather"                   → "get_weather"  (no dot → full name)

    Parameters
    ----------
    seed:
        Random seed for Thompson Sampling.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.mental_models: dict[str, ToolMentalModel] = {}

    def _get_model(self, tool_id: str) -> ToolMentalModel:
        if tool_id not in self.mental_models:
            self.mental_models[tool_id] = ToolMentalModel(tool_id)
        return self.mental_models[tool_id]

    @staticmethod
    def _extract_domain(tool_id: str) -> str:
        """Extract domain key: text before the first ``'.'``.

        If no dot is present the full tool_id is returned as its own domain.
        """
        dot_idx = tool_id.find(".")
        if dot_idx > 0:
            return tool_id[:dot_idx]
        return tool_id

    def select(self, query: Query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id

        best_tool_id = declarers[0].tool_id
        best_sample = -1.0

        for tool in declarers:
            domain = self._extract_domain(tool.tool_id)
            model = self._get_model(tool.tool_id)
            belief = model.get_belief(domain)
            sample = belief.sample(self.rng)

            if sample > best_sample:
                best_sample = sample
                best_tool_id = tool.tool_id

        return best_tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        """Update the posterior for *tool_id* using its domain key.

        The ``capability`` argument (typically the scenario_id) is **ignored**;
        the domain key is derived from ``tool_id`` instead.
        """
        domain = self._extract_domain(tool_id)
        model = self._get_model(tool_id)
        model.get_belief(domain).update(success)

    def get_predicted_prob(self, tool_id: str, capability: str) -> float:
        """Return the posterior mean for *tool_id* under its domain key.

        ``capability`` is accepted for API compatibility but unused.
        """
        domain = self._extract_domain(tool_id)
        model = self._get_model(tool_id)
        return model.get_belief(domain).mean


# ---------------------------------------------------------------------------
# 4. RepeatedBFCLSelector (multi-round learning)
# ---------------------------------------------------------------------------


class RepeatedBFCLSelector(Selector):
    """Thompson Sampling with exact tool_id-based posterior for repeated BFCL.

    Unlike TATSBFCLSelector which uses domain-prefix sharing, this selector
    keys posteriors by the exact function name (tool_id). When the same
    function appears across multiple scenarios and rounds, posterior
    accumulates directly.

    Designed for use with RepeatedBFCLEnvironment where the same scenarios
    repeat over K rounds, giving TATS opportunity to learn.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.mental_models: dict[str, ToolMentalModel] = {}

    def _get_model(self, tool_id: str) -> ToolMentalModel:
        if tool_id not in self.mental_models:
            self.mental_models[tool_id] = ToolMentalModel(tool_id)
        return self.mental_models[tool_id]

    def select(self, query: Query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id

        best_tool_id = declarers[0].tool_id
        best_sample = -1.0

        for tool in declarers:
            model = self._get_model(tool.tool_id)
            # Use tool_id itself as the "capability" key for cross-scenario sharing
            belief = model.get_belief(tool.tool_id)
            sample = belief.sample(self.rng)

            if sample > best_sample:
                best_sample = sample
                best_tool_id = tool.tool_id

        return best_tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        """Update posterior keyed by exact tool_id (ignores scenario capability)."""
        model = self._get_model(tool_id)
        model.get_belief(tool_id).update(success)

    def get_predicted_prob(self, tool_id: str, capability: str) -> float:
        model = self._get_model(tool_id)
        return model.get_belief(tool_id).mean


# ---------------------------------------------------------------------------
# 5. ParameterMatchSelector
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. EpsilonGreedyBFCLSelector (learning baseline for repeated BFCL)
# ---------------------------------------------------------------------------


class EpsilonGreedyBFCLSelector(Selector):
    """ε-greedy selector for BFCL with per-tool empirical success tracking.

    Maintains a running count of successes and trials for each tool_id.
    With probability ε, selects uniformly at random among candidates;
    with probability 1-ε, selects the tool with the highest empirical
    success rate. Ties broken randomly.

    Parameters
    ----------
    epsilon:
        Exploration rate (probability of random selection).
    seed:
        Random seed.
    """

    def __init__(self, epsilon: float = 0.1, seed: int = 0) -> None:
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.successes: dict[str, int] = defaultdict(int)
        self.trials: dict[str, int] = defaultdict(int)

    def _success_rate(self, tool_id: str) -> float:
        t = self.trials[tool_id]
        if t == 0:
            return 0.5  # uninformed prior
        return self.successes[tool_id] / t

    def select(self, query: Query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id

        # ε-greedy
        if self.rng.random() < self.epsilon:
            idx = self.rng.integers(len(declarers))
            return declarers[idx].tool_id

        # Exploit: pick best empirical success rate
        rates = [self._success_rate(t.tool_id) for t in declarers]
        best_rate = max(rates)
        best_indices = [i for i, r in enumerate(rates) if r == best_rate]
        idx = self.rng.choice(best_indices)
        return declarers[idx].tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        self.trials[tool_id] += 1
        if success:
            self.successes[tool_id] += 1

    def get_predicted_prob(self, tool_id: str, capability: str) -> float:
        return self._success_rate(tool_id)


# ---------------------------------------------------------------------------
# 6. UCBBFCLSelector (learning baseline for repeated BFCL)
# ---------------------------------------------------------------------------


class UCBBFCLSelector(Selector):
    """UCB1 selector for BFCL with per-tool confidence bounds.

    Uses standard UCB1 formula: score = empirical_mean + c * sqrt(ln(N) / n_i)
    where N is total trials and n_i is trials for tool i.

    Parameters
    ----------
    c:
        Exploration coefficient (default: sqrt(2) ≈ 1.41).
    seed:
        Random seed for tie-breaking and initial exploration.
    """

    def __init__(self, c: float = 1.41, seed: int = 0) -> None:
        self.c = c
        self.rng = np.random.default_rng(seed)
        self.successes: dict[str, int] = defaultdict(int)
        self.trials: dict[str, int] = defaultdict(int)
        self.total_trials: int = 0

    def select(self, query: Query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id

        scores: list[float] = []
        for tool in declarers:
            n_i = self.trials[tool.tool_id]
            if n_i == 0:
                scores.append(float("inf"))  # unexplored: try first
            else:
                mean = self.successes[tool.tool_id] / n_i
                bonus = self.c * math.sqrt(math.log(max(self.total_trials, 1)) / n_i)
                scores.append(mean + bonus)

        best_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if s == best_score]
        idx = self.rng.choice(best_indices)
        return declarers[idx].tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        self.trials[tool_id] += 1
        self.total_trials += 1
        if success:
            self.successes[tool_id] += 1

    def get_predicted_prob(self, tool_id: str, capability: str) -> float:
        n_i = self.trials[tool_id]
        if n_i == 0:
            return 0.5
        return self.successes[tool_id] / n_i


# ---------------------------------------------------------------------------
# 7. ParameterMatchSelector
# ---------------------------------------------------------------------------


class ParameterMatchSelector(Selector):
    """Select the tool whose parameter names overlap most with the query text.

    For each candidate tool, the score is the number of parameter names
    (from ``tool.parameter_spec["properties"]``) that appear as tokens in
    the query text.  Ties are broken randomly.

    Parameters
    ----------
    query_texts:
        Mapping from ``query_id`` to raw query string.
    seed:
        Random seed for tie-breaking.
    """

    def __init__(self, query_texts: dict[int, str], seed: int = 0) -> None:
        self.query_texts = query_texts
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _param_names(tool) -> set[str]:
        """Extract lowercased parameter names from a tool's parameter_spec."""
        spec = getattr(tool, "parameter_spec", {})
        properties = spec.get("properties", {})
        return {name.lower() for name in properties}

    def select(self, query: Query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id

        query_text = self.query_texts.get(query.query_id, "")
        q_tokens = _tokenise(query_text)

        scores: list[int] = []
        for tool in declarers:
            param_names = self._param_names(tool)
            # Count how many parameter names appear as tokens in the query.
            # Parameter names may contain underscores; we also tokenise them
            # so multi-word names like "start_date" match individual tokens
            # ("start", "date") as well as the joined form.
            expanded: set[str] = set()
            for p in param_names:
                expanded.add(p)
                expanded.update(_tokenise(p))

            scores.append(len(q_tokens & expanded))

        best_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if s == best_score]
        idx = self.rng.choice(best_indices)
        return declarers[idx].tool_id
