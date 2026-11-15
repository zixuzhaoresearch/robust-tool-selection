"""MetaTool (ICLR 2024) adapter for tool selection experiments.

Loads MetaTool ToolE dataset and presents it through the same environment
interface as BFCLEnvironment, so existing selectors and experiment
infrastructure can be reused without modification.

Data format
-----------
- plugin_des.json: {tool_name: description_string} — 199 tools
- all_clean_data.csv: (Query, Tool) pairs — 20,614 single-tool queries
- tool_embedding.pkl: list of {tool: name, embedding: vector} — precomputed

Candidate set construction
--------------------------
For each query, the candidate set consists of:
  1. The ground-truth tool
  2. The (K-1) most similar tools to the ground-truth tool (by embedding cosine)
This creates a challenging selection task where all candidates "look similar."

Key difference from BFCL
-------------------------
MetaTool tools have NO parameter specifications — only name + description.
Features like param_match, n_params_norm, required_ratio will be zero.
This tests whether the bandit framework generalizes beyond rich API schemas.
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .bfcl_adapter import BFCLTool, BFCLScenario
from .models import Observation, Query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

_DEFAULT_METATOOL_DIR = Path(__file__).resolve().parent.parent.parent / "MetaTool" / "dataset"


def find_metatool_data(base_dir: str | Path | None = None) -> Path:
    """Locate the MetaTool dataset directory.

    Search order:
    1. Explicit base_dir argument
    2. METATOOL_DATA_DIR environment variable
    3. ../MetaTool/dataset/ relative to project root
    """
    import os

    candidates: list[Path] = []

    if base_dir is not None:
        candidates.append(Path(base_dir))

    env_dir = os.environ.get("METATOOL_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.append(_DEFAULT_METATOOL_DIR)

    for path in candidates:
        if (path / "plugin_des.json").is_file() and (path / "data" / "all_clean_data.csv").is_file():
            logger.info("Found MetaTool data at %s", path)
            return path

    searched = [str(p) for p in candidates]
    raise FileNotFoundError(
        "Could not find MetaTool dataset. Searched:\n"
        + "\n".join(f"  - {p}" for p in searched)
        + "\nClone https://github.com/HowieHwong/MetaTool alongside this project."
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_tool_descriptions(base: Path) -> dict[str, str]:
    """Load tool_name -> description mapping from plugin_des.json."""
    with open(base / "plugin_des.json", encoding="utf-8") as f:
        return json.load(f)


def _load_queries(base: Path) -> list[tuple[str, str]]:
    """Load (query_text, ground_truth_tool) pairs from all_clean_data.csv."""
    pairs: list[tuple[str, str]] = []
    with open(base / "data" / "all_clean_data.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                pairs.append((row[0].strip(), row[1].strip()))
    return pairs


def _load_embeddings(base: Path) -> dict[str, np.ndarray]:
    """Load tool embeddings from tool_embedding.pkl.

    Returns dict mapping tool_name -> embedding_vector.
    """
    emb_path = base / "tool_embedding.pkl"
    if not emb_path.is_file():
        logger.warning("tool_embedding.pkl not found; will compute similarities from TF-IDF")
        return {}

    with open(emb_path, "rb") as f:
        raw = pickle.load(f)

    result: dict[str, np.ndarray] = {}
    for entry in raw:
        name = entry["tool"]
        vec = np.array(entry["embedding"], dtype=np.float64)
        result[name] = vec
    return result


# ---------------------------------------------------------------------------
# Similarity computation for candidate set construction
# ---------------------------------------------------------------------------


def _compute_similarity_matrix(
    tool_names: list[str],
    embeddings: dict[str, np.ndarray],
    tool_descriptions: dict[str, str],
) -> np.ndarray:
    """Compute pairwise cosine similarity between tools.

    Uses precomputed embeddings if available, falls back to TF-IDF.
    Returns (N, N) similarity matrix.
    """
    n = len(tool_names)

    if embeddings:
        # Use precomputed embeddings
        dim = next(iter(embeddings.values())).shape[0]
        mat = np.zeros((n, dim), dtype=np.float64)
        for i, name in enumerate(tool_names):
            if name in embeddings:
                mat[i] = embeddings[name]
            else:
                logger.warning("No embedding for tool %s; using zero vector", name)
    else:
        # Fallback: TF-IDF on descriptions
        import re
        from collections import Counter

        docs = [tool_descriptions.get(name, "") for name in tool_names]
        # Simple TF-IDF
        all_tokens: list[set[str]] = []
        for doc in docs:
            all_tokens.append(set(re.findall(r"\w+", doc.lower())))

        df: Counter[str] = Counter()
        for tset in all_tokens:
            for tok in tset:
                df[tok] += 1

        # Build vocabulary
        vocab = sorted(df.keys())
        vocab_idx = {tok: i for i, tok in enumerate(vocab)}
        dim = len(vocab)
        mat = np.zeros((n, dim), dtype=np.float64)
        for i, doc in enumerate(docs):
            tokens = re.findall(r"\w+", doc.lower())
            tf = Counter(tokens)
            max_tf = max(tf.values()) if tf else 1
            for tok, count in tf.items():
                if tok in vocab_idx:
                    import math
                    idf = math.log(n / (1 + df[tok])) + 1.0
                    mat[i, vocab_idx[tok]] = (0.5 + 0.5 * count / max_tf) * idf

    # Normalize rows
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat_normed = mat / norms

    # Cosine similarity
    sim = mat_normed @ mat_normed.T
    return sim


# ---------------------------------------------------------------------------
# Benchmark loader
# ---------------------------------------------------------------------------


class MetaToolBenchmark:
    """Loads MetaTool ToolE dataset for tool selection evaluation.

    Parameters
    ----------
    base_dir:
        Path to MetaTool/dataset/ directory. Auto-detected if None.
    n_candidates:
        Number of candidate tools per query (including ground truth).
    seed:
        Random seed for any shuffling.
    max_queries:
        Maximum number of queries to load (None = all ~20k).
        Use a smaller value for faster experiments.
    similarity_mode:
        How to construct candidate sets:
        - "embedding": use precomputed tool embeddings (default)
        - "random": random distractors
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        n_candidates: int = 10,
        seed: int = 42,
        max_queries: int | None = None,
        similarity_mode: str = "embedding",
    ) -> None:
        base = find_metatool_data(base_dir)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._n_candidates = n_candidates

        # Load data
        tool_descriptions = _load_tool_descriptions(base)
        raw_queries = _load_queries(base)
        embeddings = _load_embeddings(base) if similarity_mode == "embedding" else {}

        # Build tool list
        self._tool_names = sorted(tool_descriptions.keys())
        self._tool_name_to_idx = {name: i for i, name in enumerate(self._tool_names)}

        # Build BFCLTool objects (no parameter_spec — MetaTool has no params)
        self._tools: dict[str, BFCLTool] = {}
        for name in self._tool_names:
            desc = tool_descriptions[name]
            self._tools[name] = BFCLTool(
                tool_id=name,
                name=name,
                declared_capabilities=set(),
                true_success_rates={},
                description_affinity={},
                description=desc,
                parameter_spec={},  # MetaTool has no parameter specs
            )

        # Compute tool similarity for candidate set construction
        self._sim_matrix = _compute_similarity_matrix(
            self._tool_names, embeddings, tool_descriptions
        )

        # Precompute: for each tool, the indices of most similar tools
        self._similar_tools: dict[str, list[str]] = {}
        for i, name in enumerate(self._tool_names):
            sims = self._sim_matrix[i].copy()
            sims[i] = -1.0  # exclude self
            top_indices = np.argsort(sims)[::-1]
            self._similar_tools[name] = [self._tool_names[j] for j in top_indices]

        # Filter queries to only those with valid tools
        valid_queries = [(q, t) for q, t in raw_queries if t in self._tools]
        if len(valid_queries) < len(raw_queries):
            logger.warning(
                "Dropped %d queries with unknown tools",
                len(raw_queries) - len(valid_queries),
            )

        # Optional: limit number of queries
        if max_queries is not None and max_queries < len(valid_queries):
            # Stratified sampling: sample proportionally from each tool
            self._rng.shuffle(valid_queries)
            valid_queries = valid_queries[:max_queries]

        # Build scenarios
        self._scenarios: list[BFCLScenario] = []
        for i, (query_text, gt_tool) in enumerate(valid_queries):
            candidates = self._build_candidate_set(gt_tool, n_candidates, similarity_mode)
            candidate_tools = [self._tools[name] for name in candidates]

            self._scenarios.append(
                BFCLScenario(
                    scenario_id=f"metatool_{i}",
                    query_text=query_text,
                    candidate_tools=candidate_tools,
                    ground_truth_tool_id=gt_tool,
                )
            )

        logger.info(
            "Loaded %d MetaTool scenarios (%d candidates each) from %s",
            len(self._scenarios),
            n_candidates,
            base,
        )

    def _build_candidate_set(
        self, gt_tool: str, n_candidates: int, mode: str
    ) -> list[str]:
        """Build a candidate set of n_candidates tools including gt_tool."""
        if n_candidates >= len(self._tool_names):
            return list(self._tool_names)

        candidates = [gt_tool]

        if mode == "embedding":
            # Add most similar tools as distractors
            for similar_name in self._similar_tools.get(gt_tool, []):
                if len(candidates) >= n_candidates:
                    break
                if similar_name != gt_tool:
                    candidates.append(similar_name)
        else:
            # Random distractors
            available = [n for n in self._tool_names if n != gt_tool]
            chosen = self._rng.choice(
                len(available),
                size=min(n_candidates - 1, len(available)),
                replace=False,
            )
            candidates.extend(available[i] for i in chosen)

        return candidates[:n_candidates]

    @property
    def scenarios(self) -> list[BFCLScenario]:
        return list(self._scenarios)

    @property
    def n_candidates(self) -> int:
        return self._n_candidates

    def __len__(self) -> int:
        return len(self._scenarios)

    def __getitem__(self, idx: int) -> BFCLScenario:
        return self._scenarios[idx]

    def to_sequential_env(self) -> "MetaToolEnvironment":
        """Return a MetaToolEnvironment that mimics BFCLEnvironment."""
        return MetaToolEnvironment(self._scenarios, seed=self._seed)


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------


class MetaToolEnvironment:
    """Presents MetaTool scenarios through the BFCLEnvironment interface.

    Identical interface to BFCLEnvironment so that all existing selectors,
    feature extractors, and experiment infrastructure work unchanged.
    """

    def __init__(self, scenarios: list[BFCLScenario], seed: int = 42) -> None:
        self._scenarios = scenarios
        self._rng = np.random.default_rng(seed)

        # Build scenario lookup
        self._scenario_map: dict[str, BFCLScenario] = {
            s.scenario_id: s for s in scenarios
        }

        # Build deduplicated global tool list
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
        key = (query.query_id, tool_id)
        success = self.outcome_matrix.get(key, False)
        return Observation(
            query=query,
            tool_id=tool_id,
            capability_used=query.required_capability,
            success=success,
        )

    def get_declarers(self, capability: str) -> list[BFCLTool]:
        scenario = self._scenario_map.get(capability)
        if scenario is None:
            return []
        return list(scenario.candidate_tools)

    def get_scenario(self, scenario_id: str) -> BFCLScenario | None:
        return self._scenario_map.get(scenario_id)

    @property
    def n_scenarios(self) -> int:
        return len(self._scenarios)

    @property
    def n_unique_tools(self) -> int:
        return len(self.tools)
