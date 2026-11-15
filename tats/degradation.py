"""Documentation degradation engine for BFCL-Opaque experiments.

Transforms BFCL tool descriptions to simulate information loss at three levels:

- **L1 (Name-only)**: Anonymize function names, strip all descriptions.
- **L2 (Partial)**: Keep names, truncate descriptions to 30%, strip param descriptions.
- **L3 (Misleading)**: Cross-swap descriptions for 20% of tools (deterministic subset).

All transformations produce deep copies — originals are never mutated.
Deterministic given seed.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from .bfcl_adapter import BFCLScenario, BFCLTool

logger = logging.getLogger(__name__)

# Valid degradation levels
_VALID_LEVELS = {"L1", "L2", "L3"}


class DegradationEngine:
    """Apply documentation degradation to BFCLScenarios."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def degrade(
        self, scenarios: list[BFCLScenario], level: str
    ) -> list[BFCLScenario]:
        """Return degraded COPIES of scenarios.

        Parameters
        ----------
        scenarios:
            Original BFCL scenarios to degrade.
        level:
            One of ``'L1'``, ``'L2'``, ``'L3'``.

        Returns
        -------
        list[BFCLScenario]
            New scenario objects with degraded tool descriptions.
            Originals are untouched.

        Raises
        ------
        ValueError
            If *level* is not a recognised degradation level.
        """
        level = level.upper()
        if level not in _VALID_LEVELS:
            raise ValueError(
                f"Unknown degradation level {level!r}. "
                f"Expected one of {sorted(_VALID_LEVELS)}."
            )

        # Reset RNG so the same seed + level always produces the same output,
        # regardless of how many times `degrade` has been called before.
        self.rng = np.random.default_rng(self.seed)

        dispatch = {
            "L1": self.degrade_l1_nameonly,
            "L2": self.degrade_l2_partial,
            "L3": self.degrade_l3_misleading,
        }
        degraded = dispatch[level](scenarios)
        logger.info(
            "Degraded %d scenarios at level %s (seed=%d)",
            len(degraded),
            level,
            self.seed,
        )
        return degraded

    # ------------------------------------------------------------------
    # L1: Name-only
    # ------------------------------------------------------------------

    def degrade_l1_nameonly(
        self, scenarios: list[BFCLScenario]
    ) -> list[BFCLScenario]:
        """Anonymize names, remove ALL descriptions.

        Per-scenario, each candidate tool gets a deterministic anonymous name
        ``func_0``, ``func_1``, ... The mapping is shuffled using the RNG so
        that the anonymous index does NOT simply mirror the original ordering.
        Ground-truth tool IDs are updated to the new anonymous names.

        Preserved:
        - parameter names, types, required/optional markers
        Removed:
        - function descriptions (set to ``""``)
        - parameter descriptions (set to ``""``)
        """
        result: list[BFCLScenario] = []

        for scenario in scenarios:
            n_tools = len(scenario.candidate_tools)

            # Deterministic shuffled assignment: indices 0..n-1 assigned to
            # tools in a permuted order.
            indices = self.rng.permutation(n_tools)
            name_map: dict[str, str] = {}  # original_name -> anonymous_name

            new_tools: list[BFCLTool] = []
            for idx, tool in zip(indices, scenario.candidate_tools):
                anon_name = f"func_{idx}"
                name_map[tool.tool_id] = anon_name

                new_tool = _copy_tool(tool)
                new_tool.tool_id = anon_name
                new_tool.name = anon_name
                new_tool.description = ""
                new_tool.parameter_spec = _strip_param_descriptions(
                    new_tool.parameter_spec
                )
                new_tools.append(new_tool)

            # Update ground truth to the anonymized name.
            new_gt = name_map.get(scenario.ground_truth_tool_id)
            if new_gt is None:
                # Defensive: ground truth tool not in candidate list.
                logger.warning(
                    "Scenario %s: ground_truth_tool_id %r not found among "
                    "candidates — leaving unchanged.",
                    scenario.scenario_id,
                    scenario.ground_truth_tool_id,
                )
                new_gt = scenario.ground_truth_tool_id

            result.append(
                BFCLScenario(
                    scenario_id=scenario.scenario_id,
                    query_text=scenario.query_text,
                    candidate_tools=new_tools,
                    ground_truth_tool_id=new_gt,
                    ground_truth_args=copy.deepcopy(scenario.ground_truth_args),
                )
            )

        return result

    # ------------------------------------------------------------------
    # L2: Partial
    # ------------------------------------------------------------------

    def degrade_l2_partial(
        self, scenarios: list[BFCLScenario]
    ) -> list[BFCLScenario]:
        """Keep names, truncate descriptions to first 30%, strip param descriptions.

        Preserved:
        - original function names
        - parameter names, types, required/optional markers
        Modified:
        - function description truncated to first 30% of characters
        Removed:
        - parameter descriptions
        """
        result: list[BFCLScenario] = []

        for scenario in scenarios:
            new_tools: list[BFCLTool] = []
            for tool in scenario.candidate_tools:
                new_tool = _copy_tool(tool)
                new_tool.description = _truncate(tool.description, fraction=0.3)
                new_tool.parameter_spec = _strip_param_descriptions(
                    new_tool.parameter_spec
                )
                new_tools.append(new_tool)

            result.append(
                BFCLScenario(
                    scenario_id=scenario.scenario_id,
                    query_text=scenario.query_text,
                    candidate_tools=new_tools,
                    ground_truth_tool_id=scenario.ground_truth_tool_id,
                    ground_truth_args=copy.deepcopy(scenario.ground_truth_args),
                )
            )

        return result

    # ------------------------------------------------------------------
    # L3: Misleading
    # ------------------------------------------------------------------

    def degrade_l3_misleading(
        self, scenarios: list[BFCLScenario]
    ) -> list[BFCLScenario]:
        """Cross-swap descriptions for 20% of tools per scenario.

        For each scenario independently:
        1. Select 20% of candidate tools (deterministic subset via RNG).
           Must be >= 2 tools to perform any swaps; skip scenario otherwise.
        2. Within the selected subset, cyclically shift descriptions: tool[i]
           gets the description of tool[i+1], and the last gets the first's.
           This guarantees every selected tool receives a *different* tool's
           description.

        Preserved:
        - original function names
        - parameters (names, types, required/optional, descriptions)
        Modified:
        - function-level descriptions are cross-swapped for the selected subset
        """
        result: list[BFCLScenario] = []

        for scenario in scenarios:
            n_tools = len(scenario.candidate_tools)
            n_swap = max(1, int(n_tools * 0.2))

            # Deep copy all tools first.
            new_tools = [_copy_tool(t) for t in scenario.candidate_tools]

            if n_swap >= 2:
                # Pick which tool indices to swap (deterministic).
                swap_indices = self.rng.choice(
                    n_tools, size=n_swap, replace=False
                )
                swap_indices.sort()  # deterministic ordering

                # Collect original descriptions for the selected tools.
                original_descs = [
                    scenario.candidate_tools[i].description
                    for i in swap_indices
                ]

                # Cyclic shift: tool at swap_indices[k] gets description from
                # swap_indices[k-1]'s original. This is equivalent to rotating
                # the descriptions list by +1 position.
                rotated_descs = [original_descs[-1]] + original_descs[:-1]

                for idx, new_desc in zip(swap_indices, rotated_descs):
                    new_tools[idx].description = new_desc

            result.append(
                BFCLScenario(
                    scenario_id=scenario.scenario_id,
                    query_text=scenario.query_text,
                    candidate_tools=new_tools,
                    ground_truth_tool_id=scenario.ground_truth_tool_id,
                    ground_truth_args=copy.deepcopy(scenario.ground_truth_args),
                )
            )

        return result


# ======================================================================
# Helpers (module-private)
# ======================================================================


def _copy_tool(tool: BFCLTool) -> BFCLTool:
    """Deep-copy a BFCLTool."""
    return BFCLTool(
        tool_id=tool.tool_id,
        name=tool.name,
        declared_capabilities=set(tool.declared_capabilities),
        true_success_rates=dict(tool.true_success_rates),
        description_affinity=dict(tool.description_affinity),
        description=tool.description,
        parameter_spec=copy.deepcopy(tool.parameter_spec),
    )


def _strip_param_descriptions(param_spec: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of *param_spec* with all parameter descriptions cleared.

    Handles the standard JSON-Schema style used by BFCL::

        {
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "...",
                    ...
                },
                ...
            },
            "required": [...]
        }

    Only the ``"description"`` key inside each property is set to ``""``.
    Top-level keys (``type``, ``required``, ``properties``) and per-property
    keys other than ``description`` (``type``, ``enum``, ``default``, etc.)
    are preserved.
    """
    spec = copy.deepcopy(param_spec)
    properties = spec.get("properties", {})
    for _param_name, param_def in properties.items():
        if isinstance(param_def, dict) and "description" in param_def:
            param_def["description"] = ""
    return spec


def _truncate(text: str, fraction: float) -> str:
    """Truncate *text* to the first *fraction* of its characters.

    Returns the empty string for empty input. The result length is
    ``max(1, int(len(text) * fraction))`` to avoid producing empty strings
    from non-empty input (at least one character survives).
    """
    if not text:
        return ""
    keep = max(1, int(len(text) * fraction))
    return text[:keep]
