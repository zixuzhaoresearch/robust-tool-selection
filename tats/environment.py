"""Synthetic environment with 10 tools × 5 capabilities."""

from __future__ import annotations

import numpy as np

from .models import Observation, Query, Tool

# Ground truth: best tool per capability
GROUND_TRUTH = {"A": "t1", "B": "t2", "C": "t4", "D": "t6", "E": "t8"}

CAPABILITIES = list(GROUND_TRUTH.keys())


def _build_tools() -> list[Tool]:
    """Create the 10-tool synthetic environment."""
    return [
        Tool("t1", "Tool-1", {"A", "B"}, {"A": 0.90, "B": 0.30}, {"A": 0.8, "B": 0.7}),
        Tool("t2", "Tool-2", {"A", "B"}, {"A": 0.35, "B": 0.85}, {"A": 0.7, "B": 0.8}),
        Tool("t3", "Tool-3", {"A", "C"}, {"A": 0.50, "C": 0.50}, {"A": 0.4, "C": 0.5}),
        Tool("t4", "Tool-4", {"C"}, {"C": 0.88}, {"C": 0.9}),
        Tool("t5", "Tool-5", {"C", "D"}, {"C": 0.55, "D": 0.20}, {"C": 0.6, "D": 0.7}),
        Tool("t6", "Tool-6", {"D"}, {"D": 0.82}, {"D": 0.8}),
        Tool("t7", "Tool-7", {"D", "E"}, {"D": 0.60, "E": 0.45}, {"D": 0.5, "E": 0.5}),
        Tool("t8", "Tool-8", {"E"}, {"E": 0.95}, {"E": 0.7}),
        Tool("t9", "Tool-9", {"B", "E"}, {"B": 0.50, "E": 0.55}, {"B": 0.4, "E": 0.4}),
        Tool("t10", "Tool-10", {"B", "D"}, {"B": 0.50, "D": 0.40}, {"B": 0.5, "D": 0.4}),
    ]


class SyntheticEnvironment:
    """Deterministic synthetic environment for TATS evaluation."""

    def __init__(self, n_queries: int = 500, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.tools = _build_tools()
        self.tool_map = {t.tool_id: t for t in self.tools}
        self.n_queries = n_queries

        self.queries = self._generate_queries()
        self.outcome_matrix = self._pregenerate_outcomes()

    def _generate_queries(self) -> list[Query]:
        """Generate queries with uniform capability distribution, shuffled."""
        caps = CAPABILITIES * (self.n_queries // len(CAPABILITIES))
        # Handle remainder
        remainder = self.n_queries - len(caps)
        if remainder > 0:
            caps += list(self.rng.choice(CAPABILITIES, size=remainder, replace=False))
        self.rng.shuffle(caps)

        return [
            Query(i, cap, GROUND_TRUTH[cap])
            for i, cap in enumerate(caps)
        ]

    def _pregenerate_outcomes(self) -> dict[tuple[int, str], bool]:
        """Pregenerate outcome matrix so different selectors get same results."""
        outcomes: dict[tuple[int, str], bool] = {}
        for q in self.queries:
            for tool in self.tools:
                rate = tool.true_success_rates.get(q.required_capability, 0.0)
                outcomes[(q.query_id, tool.tool_id)] = bool(
                    self.rng.random() < rate
                )
        return outcomes

    def execute(self, tool_id: str, query: Query) -> Observation:
        """Execute a tool on a query, returning deterministic outcome."""
        success = self.outcome_matrix[(query.query_id, tool_id)]
        return Observation(
            query=query,
            tool_id=tool_id,
            capability_used=query.required_capability,
            success=success,
        )

    def get_declarers(self, capability: str) -> list[Tool]:
        """Get tools that declare a given capability."""
        return [t for t in self.tools if t.declares(capability)]
