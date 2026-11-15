"""BFCL (Berkeley Function Call Leaderboard) adapter for tool selection experiments.

Loads BFCL v4 "multiple" data and presents it through the same environment
interface as SyntheticEnvironment, so existing selectors and experiment
infrastructure can be reused without modification.

Data format
-----------
Each BFCL entry is a JSONL line with:
- "id": scenario identifier, e.g. "multiple_0"
- "question": [[{"role": "user", "content": "..."}]]
- "function": list of candidate function specs (name, description, parameters)

Answer format (parallel JSONL):
- "id": matching identifier
- "ground_truth": [{"correct_func_name": {...params...}}]
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .models import Observation, Query, Tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BFCLTool(Tool):
    """A Tool enriched with BFCL-specific metadata.

    Inherits the standard Tool fields (tool_id, name, declared_capabilities,
    true_success_rates, description_affinity) and adds the raw function
    specification from the BFCL dataset.
    """

    description: str = ""
    parameter_spec: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_function_spec(cls, func: dict[str, Any]) -> BFCLTool:
        """Create a BFCLTool from a BFCL function specification dict."""
        name = func["name"]
        params = func.get("parameters", {})

        # Extract parameter names as declared capabilities.
        # For BFCL the "capability" concept maps to the function itself;
        # we store individual parameter names as a lightweight proxy.
        properties = params.get("properties", {})
        param_names: set[str] = set(properties.keys())

        return cls(
            tool_id=name,
            name=name,
            declared_capabilities=param_names,
            true_success_rates={},
            description_affinity={},
            description=func.get("description", ""),
            parameter_spec=params,
        )


@dataclass
class BFCLScenario:
    """A single BFCL evaluation scenario (one question with candidate tools)."""

    scenario_id: str
    query_text: str
    candidate_tools: list[BFCLTool]
    ground_truth_tool_id: str

    # Optional: full ground-truth arguments for richer evaluation later.
    ground_truth_args: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------


def find_bfcl_data(
    filename: str = "BFCL_v4_multiple.json",
) -> tuple[Path, Path]:
    """Try to locate the BFCL data and answer files automatically.

    Search order:
    1. ``BFCL_DATA_DIR`` environment variable
    2. Gorilla repo relative path (``../gorilla/...`` from the project root)
    3. Installed ``bfcl_eval`` package data directory

    Returns
    -------
    (data_path, answer_path)

    Raises
    ------
    FileNotFoundError
        If neither location exists.
    """
    candidates: list[tuple[Path, Path]] = []

    # 1. Environment variable
    env_dir = os.environ.get("BFCL_DATA_DIR")
    if env_dir:
        base = Path(env_dir)
        candidates.append((base / filename, base / "possible_answer" / filename))

    # 2. Gorilla repo relative to this file's project root
    project_root = Path(__file__).resolve().parent.parent
    gorilla_base = (
        project_root.parent
        / "gorilla"
        / "berkeley-function-call-leaderboard"
        / "bfcl_eval"
        / "data"
    )
    candidates.append((gorilla_base / filename, gorilla_base / "possible_answer" / filename))

    # 3. Installed bfcl_eval package
    try:
        import bfcl_eval  # type: ignore[import-untyped]

        pkg_dir = Path(bfcl_eval.__file__).resolve().parent / "data"
        candidates.append((pkg_dir / filename, pkg_dir / "possible_answer" / filename))
    except (ImportError, AttributeError):
        pass

    for data_path, answer_path in candidates:
        if data_path.is_file() and answer_path.is_file():
            logger.info("Found BFCL data at %s", data_path.parent)
            return data_path, answer_path

    searched = [str(d) for d, _ in candidates]
    raise FileNotFoundError(
        f"Could not find BFCL data files. Searched:\n"
        + "\n".join(f"  - {p}" for p in searched)
        + "\nSet BFCL_DATA_DIR or place the gorilla repo alongside this project."
    )


# ---------------------------------------------------------------------------
# Benchmark loader
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file (one JSON object per line)."""
    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


class BFCLToolSelectionBenchmark:
    """Loads BFCL v4 'multiple' data for tool-selection evaluation.

    Parameters
    ----------
    data_path, answer_path:
        Explicit paths to the BFCL JSONL files. If *both* are ``None`` the
        loader will call :func:`find_bfcl_data` to auto-discover them.
    seed:
        Random seed (used when the environment wrapper shuffles queries).
    bfcl_eval_package:
        If ``True``, force auto-detection via ``find_bfcl_data()`` even when
        explicit paths are given.
    dataset:
        Which BFCL dataset to load. Supported values:

        - ``"multiple"`` (default) — ``BFCL_v4_multiple.json``
        - ``"live_multiple"`` — ``BFCL_v4_live_multiple.json``

        Only used when paths are auto-detected (i.e. *data_path* / *answer_path*
        are ``None`` or *bfcl_eval_package* is ``True``).
    """

    _DATASET_FILENAMES: dict[str, str] = {
        "multiple": "BFCL_v4_multiple.json",
        "live_multiple": "BFCL_v4_live_multiple.json",
    }

    def __init__(
        self,
        data_path: str | Path | None = None,
        answer_path: str | Path | None = None,
        seed: int = 42,
        bfcl_eval_package: bool = False,
        dataset: str = "multiple",
    ) -> None:
        # Resolve paths -------------------------------------------------
        if bfcl_eval_package or (data_path is None and answer_path is None):
            if dataset not in self._DATASET_FILENAMES:
                raise ValueError(
                    f"Unknown dataset {dataset!r}. "
                    f"Supported: {sorted(self._DATASET_FILENAMES)}"
                )
            filename = self._DATASET_FILENAMES[dataset]
            resolved_data, resolved_answer = find_bfcl_data(filename=filename)
        else:
            if data_path is None or answer_path is None:
                raise ValueError(
                    "Provide both data_path and answer_path, or set both to None "
                    "for auto-detection."
                )
            resolved_data = Path(data_path)
            resolved_answer = Path(answer_path)

        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Load raw data -------------------------------------------------
        raw_data = _load_jsonl(resolved_data)
        raw_answers = _load_jsonl(resolved_answer)

        answer_map: dict[str, dict[str, Any]] = {a["id"]: a for a in raw_answers}

        # Build scenarios -----------------------------------------------
        self._scenarios: list[BFCLScenario] = []
        for entry in raw_data:
            sid = entry["id"]
            answer_entry = answer_map.get(sid)
            if answer_entry is None:
                logger.warning("No answer found for scenario %s — skipping.", sid)
                continue

            # Parse query text
            question_block = entry["question"]
            # question is [[{role, content}, ...]]  — take first turn's content
            query_text = question_block[0][0]["content"]

            # Parse candidate tools
            candidate_tools = [
                BFCLTool.from_function_spec(func) for func in entry["function"]
            ]

            # Parse ground truth: first element of ground_truth list,
            # the single key of that dict is the correct function name.
            gt_entry = answer_entry["ground_truth"][0]
            gt_tool_id = next(iter(gt_entry.keys()))
            gt_args = gt_entry[gt_tool_id]

            self._scenarios.append(
                BFCLScenario(
                    scenario_id=sid,
                    query_text=query_text,
                    candidate_tools=candidate_tools,
                    ground_truth_tool_id=gt_tool_id,
                    ground_truth_args=gt_args,
                )
            )

        logger.info(
            "Loaded %d BFCL scenarios from %s", len(self._scenarios), resolved_data
        )

    # ------------------------------------------------------------------ API
    @property
    def scenarios(self) -> list[BFCLScenario]:
        """All loaded BFCL scenarios."""
        return list(self._scenarios)

    def __len__(self) -> int:
        return len(self._scenarios)

    def __getitem__(self, idx: int) -> BFCLScenario:
        return self._scenarios[idx]

    # --------------------------------------------- Environment-compatible API

    def to_sequential_env(self) -> BFCLEnvironment:
        """Return a :class:`BFCLEnvironment` that mimics :class:`SyntheticEnvironment`.

        The returned object exposes ``.tools``, ``.queries``, ``.execute()``, and
        ``.get_declarers()`` so that existing selectors work unchanged.
        """
        return BFCLEnvironment(self._scenarios, seed=self._seed)


# ---------------------------------------------------------------------------
# Environment wrapper (SyntheticEnvironment-compatible interface)
# ---------------------------------------------------------------------------


class BFCLEnvironment:
    """Presents BFCL scenarios through the same interface as SyntheticEnvironment.

    Key design choices
    ------------------
    * Each scenario becomes a single :class:`Query` whose ``required_capability``
      equals the ``scenario_id``.  This means "capability" in the BFCL context
      is scenario-specific rather than a shared abstract concept.
    * ``get_declarers(capability)`` returns the candidate tools for the scenario
      whose id matches the capability string.
    * ``execute()`` is deterministic: success iff the selected tool matches
      the ground-truth tool for that scenario.
    * ``tools`` is the deduplicated union of all candidate tools across scenarios
      (by ``tool_id``).  Since different scenarios can define different functions
      with the same name but different descriptions, we keep the *first seen*
      version.  ``get_declarers()`` always returns the per-scenario candidates.
    """

    def __init__(self, scenarios: list[BFCLScenario], seed: int = 42) -> None:
        self._scenarios = scenarios
        self._rng = np.random.default_rng(seed)

        # Build scenario lookup
        self._scenario_map: dict[str, BFCLScenario] = {
            s.scenario_id: s for s in scenarios
        }

        # Build deduplicated global tool list (first-seen wins)
        seen: dict[str, BFCLTool] = {}
        for scenario in scenarios:
            for tool in scenario.candidate_tools:
                if tool.tool_id not in seen:
                    seen[tool.tool_id] = tool
        self.tools: list[BFCLTool] = list(seen.values())
        self.tool_map: dict[str, BFCLTool] = seen

        # Build queries — one per scenario, then shuffle presentation order
        self.queries: list[Query] = [
            Query(
                query_id=i,
                required_capability=scenario.scenario_id,
                ground_truth_tool_id=scenario.ground_truth_tool_id,
            )
            for i, scenario in enumerate(scenarios)
        ]
        self._rng.shuffle(self.queries)

        # Pregenerate outcome matrix for compatibility
        self.outcome_matrix: dict[tuple[int, str], bool] = {}
        for q in self.queries:
            scenario = self._scenario_map[q.required_capability]
            for tool in scenario.candidate_tools:
                is_correct = tool.tool_id == scenario.ground_truth_tool_id
                self.outcome_matrix[(q.query_id, tool.tool_id)] = is_correct
            # Any tool NOT in the candidate set is always wrong
            for tool in self.tools:
                key = (q.query_id, tool.tool_id)
                if key not in self.outcome_matrix:
                    self.outcome_matrix[key] = False

    def execute(self, tool_id: str, query: Query) -> Observation:
        """Execute a tool on a query. Success iff tool matches ground truth."""
        key = (query.query_id, tool_id)
        success = self.outcome_matrix.get(key, False)
        return Observation(
            query=query,
            tool_id=tool_id,
            capability_used=query.required_capability,
            success=success,
        )

    def get_declarers(self, capability: str) -> list[BFCLTool]:
        """Return candidate tools for the given capability (= scenario_id)."""
        scenario = self._scenario_map.get(capability)
        if scenario is None:
            return []
        return list(scenario.candidate_tools)

    def get_scenario(self, scenario_id: str) -> BFCLScenario | None:
        """Look up a scenario by id (convenience accessor)."""
        return self._scenario_map.get(scenario_id)

    @property
    def n_scenarios(self) -> int:
        return len(self._scenarios)

    @property
    def n_unique_tools(self) -> int:
        return len(self.tools)


# ---------------------------------------------------------------------------
# Repeated-interaction BFCL environment (multi-round protocol)
# ---------------------------------------------------------------------------


class RepeatedBFCLEnvironment:
    """Multi-round BFCL environment enabling cross-scenario learning.

    Repeats all scenarios K rounds. Within each round, scenarios are shuffled
    (deterministic per seed). Queries span all rounds with globally unique IDs,
    so selectors accumulate posterior across rounds.

    demonstrating learning: with K rounds, the same tool pool is seen multiple
    times, allowing posterior accumulation.

    Parameters
    ----------
    scenarios:
        Base BFCL scenarios to repeat.
    n_rounds:
        Number of times to repeat all scenarios (default: 5).
    seed:
        Random seed for shuffle order per round.
    """

    def __init__(
        self,
        scenarios: list[BFCLScenario],
        n_rounds: int = 5,
        seed: int = 42,
    ) -> None:
        self._base_scenarios = scenarios
        self.n_rounds = n_rounds
        self._rng = np.random.default_rng(seed)

        # Build deduplicated global tool list
        seen: dict[str, "BFCLTool"] = {}
        for scenario in scenarios:
            for tool in scenario.candidate_tools:
                if tool.tool_id not in seen:
                    seen[tool.tool_id] = tool
        self.tools: list["BFCLTool"] = list(seen.values())
        self.tool_map: dict[str, "BFCLTool"] = seen

        # Build scenario lookup
        self._scenario_map: dict[str, BFCLScenario] = {
            s.scenario_id: s for s in scenarios
        }

        # Build multi-round queries
        self.queries: list[Query] = []
        self.round_boundaries: list[int] = []  # start index of each round
        query_id = 0
        for _round in range(n_rounds):
            self.round_boundaries.append(query_id)
            # Shuffle scenario order for this round
            indices = self._rng.permutation(len(scenarios))
            for idx in indices:
                scenario = scenarios[idx]
                self.queries.append(
                    Query(
                        query_id=query_id,
                        required_capability=scenario.scenario_id,
                        ground_truth_tool_id=scenario.ground_truth_tool_id,
                    )
                )
                query_id += 1

        # Pregenerate outcome matrix
        self.outcome_matrix: dict[tuple[int, str], bool] = {}
        for q in self.queries:
            scenario = self._scenario_map[q.required_capability]
            for tool in scenario.candidate_tools:
                is_correct = tool.tool_id == scenario.ground_truth_tool_id
                self.outcome_matrix[(q.query_id, tool.tool_id)] = is_correct
            for tool in self.tools:
                key = (q.query_id, tool.tool_id)
                if key not in self.outcome_matrix:
                    self.outcome_matrix[key] = False

    def execute(self, tool_id: str, query: Query) -> Observation:
        """Execute a tool on a query. Success iff tool matches ground truth."""
        key = (query.query_id, tool_id)
        success = self.outcome_matrix.get(key, False)
        return Observation(
            query=query,
            tool_id=tool_id,
            capability_used=query.required_capability,
            success=success,
        )

    def get_declarers(self, capability: str) -> list["BFCLTool"]:
        """Return candidate tools for the given scenario_id."""
        scenario = self._scenario_map.get(capability)
        if scenario is None:
            return []
        return list(scenario.candidate_tools)

    @property
    def n_scenarios(self) -> int:
        return len(self._base_scenarios)

    @property
    def n_total_queries(self) -> int:
        return len(self.queries)

    def per_round_tsa(
        self, selected_ids: list[str], ground_truth_ids: list[str]
    ) -> list[float]:
        """Compute TSA per round for learning curve analysis."""
        n_per_round = len(self._base_scenarios)
        round_tsas = []
        for r in range(self.n_rounds):
            start = self.round_boundaries[r]
            end = start + n_per_round
            if end > len(selected_ids):
                end = len(selected_ids)
            round_sel = selected_ids[start:end]
            round_gt = ground_truth_ids[start:end]
            if round_sel:
                correct = sum(1 for s, g in zip(round_sel, round_gt) if s == g)
                round_tsas.append(correct / len(round_sel))
        return round_tsas
