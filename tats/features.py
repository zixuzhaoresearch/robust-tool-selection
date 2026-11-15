"""Hand-crafted feature extraction for (query, tool) pairs.

Used by contextual bandit selectors (e.g. BilinTS) that need a fixed-
dimensional feature vector to represent each (query, tool) combination.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from .bfcl_adapter import BFCLScenario, BFCLTool


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization."""
    return re.findall(r"\w+", text.lower())


class ToolFeatureExtractor:
    """Extract (query, tool) feature vectors for contextual bandit tool selection.

    Produces an 8-dimensional feature vector with values roughly in [0, 1]:

    0. desc_sim      — TF-IDF cosine similarity between query and tool description
    1. name_sim      — Token overlap ratio between query and tool name
    2. param_match   — Fraction of tool parameter name tokens found in query
    3. n_params_norm — Number of required parameters, normalized
    4. desc_len_norm — Tool description length, normalized
    5. n_cand_norm   — Number of candidates, log-normalized
    6. required_ratio— Fraction of parameters that are required
    7. has_desc      — Binary: 1.0 if description length > 10, else 0.0
    """

    def __init__(self, query_texts: dict[int, str]) -> None:
        """Build IDF weights from the query corpus.

        Args:
            query_texts: mapping from query_id (int) to query text (str).
        """
        self._query_texts = query_texts

        # Build document-frequency counts across the query corpus
        n_docs = len(query_texts)
        df: Counter[str] = Counter()
        for text in query_texts.values():
            unique_tokens = set(_tokenize(text))
            for tok in unique_tokens:
                df[tok] += 1

        # IDF = log(N / df_t), with smoothing to avoid division by zero
        # and log(0). We use log(N / (1 + df_t)) + 1 style smoothing.
        self._idf: dict[str, float] = {}
        for tok, count in df.items():
            self._idf[tok] = math.log(n_docs / (1 + count)) + 1.0

    @property
    def n_features(self) -> int:
        """Number of features produced by extract(). Always 8."""
        return 8

    def extract(self, query_id: int, tool: BFCLTool, n_candidates: int) -> np.ndarray:
        """Extract feature vector for a (query, tool) pair.

        Args:
            query_id: Index into the query_texts dict passed at init.
            tool: A BFCLTool instance.
            n_candidates: Number of candidate tools for this query.

        Returns:
            np.ndarray of shape (8,), values approximately in [0, 1].
        """
        query_text = self._query_texts[query_id]
        query_tokens = _tokenize(query_text)
        query_token_set = set(query_tokens)

        # --- Feature 0: desc_sim (TF-IDF cosine similarity) ---
        desc_sim = self._tfidf_cosine(query_tokens, tool.description)

        # --- Feature 1: name_sim (token overlap ratio) ---
        name_tokens = set(re.split(r"[._]", tool.name.lower()))
        name_tokens.discard("")
        if name_tokens:
            overlap = len(query_token_set & name_tokens)
            name_sim = overlap / len(name_tokens)
        else:
            name_sim = 0.0

        # --- Feature 2: param_match ---
        properties: dict[str, Any] = tool.parameter_spec.get("properties", {})
        if properties:
            # Split each parameter name on underscores, collect unique tokens
            param_tokens: set[str] = set()
            for param_name in properties:
                for part in param_name.lower().split("_"):
                    if part:
                        param_tokens.add(part)
            if param_tokens:
                param_overlap = len(query_token_set & param_tokens)
                param_match = param_overlap / len(param_tokens)
            else:
                param_match = 0.0
        else:
            param_match = 0.0

        # --- Feature 3: n_params_norm ---
        required: list[str] = tool.parameter_spec.get("required", [])
        n_required = len(required)
        n_params_norm = min(n_required / 10.0, 1.0)

        # --- Feature 4: desc_len_norm ---
        desc_len = len(tool.description)
        desc_len_norm = min(desc_len / 500.0, 1.0)

        # --- Feature 5: n_cand_norm ---
        n_cand_norm = math.log1p(n_candidates) / math.log1p(40)

        # --- Feature 6: required_ratio ---
        n_total_params = len(properties)
        if n_total_params > 0:
            required_ratio = n_required / n_total_params
        else:
            required_ratio = 0.0

        # --- Feature 7: has_desc ---
        has_desc = 1.0 if desc_len > 10 else 0.0

        return np.array(
            [
                desc_sim,
                name_sim,
                param_match,
                n_params_norm,
                desc_len_norm,
                n_cand_norm,
                required_ratio,
                has_desc,
            ],
            dtype=np.float64,
        )

    def _tfidf_cosine(self, query_tokens: list[str], doc_text: str) -> float:
        """Compute TF-IDF cosine similarity between query tokens and a document.

        Uses augmented term frequency: tf(t,d) = 0.5 + 0.5 * f(t,d) / max_f(d)
        multiplied by IDF.
        """
        doc_tokens = _tokenize(doc_text)
        if not query_tokens or not doc_tokens:
            return 0.0

        # Build TF-IDF vectors as dicts (sparse representation)
        q_vec = self._tfidf_vector(query_tokens)
        d_vec = self._tfidf_vector(doc_tokens)

        if not q_vec or not d_vec:
            return 0.0

        # Cosine similarity via dot product / (norm_a * norm_b)
        dot = 0.0
        for tok, w in q_vec.items():
            if tok in d_vec:
                dot += w * d_vec[tok]

        if dot == 0.0:
            return 0.0

        norm_q = math.sqrt(sum(w * w for w in q_vec.values()))
        norm_d = math.sqrt(sum(w * w for w in d_vec.values()))

        if norm_q == 0.0 or norm_d == 0.0:
            return 0.0

        sim = dot / (norm_q * norm_d)
        # Clamp to [0, 1] (cosine can technically be negative with weird IDF)
        return max(0.0, min(1.0, sim))

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        """Build a sparse TF-IDF vector using augmented TF and IDF weights."""
        tf_counts = Counter(tokens)
        if not tf_counts:
            return {}

        max_tf = max(tf_counts.values())
        if max_tf == 0:
            return {}

        vec: dict[str, float] = {}
        for tok, count in tf_counts.items():
            augmented_tf = 0.5 + 0.5 * count / max_tf
            idf = self._idf.get(tok, 1.0)  # unseen tokens get neutral IDF
            vec[tok] = augmented_tf * idf
        return vec


def build_query_texts_from_scenarios(scenarios: list[BFCLScenario]) -> dict[int, str]:
    """Build query_id -> query_text mapping from a list of BFCLScenarios."""
    return {i: s.query_text for i, s in enumerate(scenarios)}
