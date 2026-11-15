"""Dense embedding features for (query, tool) pairs.

Replaces TF-IDF cosine similarity with sentence-transformer embedding
cosine similarity, providing a stronger retrieval baseline.

Requires: sentence-transformers (pip install sentence-transformers)
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from .bfcl_adapter import BFCLTool
from .features import ToolFeatureExtractor, _tokenize


class DenseEmbedder:
    """Pre-computes and caches sentence embeddings for text strings.

    Uses a sentence-transformer model for encoding. Embeddings are
    normalized to unit length for cosine similarity via dot product.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        """Return unit-normalized embedding for a text string (cached)."""
        if text not in self._cache:
            emb = self._model.encode(text, normalize_embeddings=True)
            self._cache[text] = emb
        return self._cache[text]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts, using cache where possible."""
        # Find uncached texts
        uncached_idx = [i for i, t in enumerate(texts) if t not in self._cache]
        if uncached_idx:
            uncached_texts = [texts[i] for i in uncached_idx]
            embeddings = self._model.encode(uncached_texts, normalize_embeddings=True)
            for idx, emb in zip(uncached_idx, embeddings):
                self._cache[texts[idx]] = emb

        return np.array([self._cache[t] for t in texts])

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts (dot product of unit vectors)."""
        emb_a = self.encode(text_a)
        emb_b = self.encode(text_b)
        sim = float(np.dot(emb_a, emb_b))
        return max(0.0, min(1.0, sim))


class DenseFeatureExtractor(ToolFeatureExtractor):
    """Feature extractor with dense embedding desc_sim replacing TF-IDF.

    Produces the same 8-dimensional feature vector as ToolFeatureExtractor,
    but feature 0 (desc_sim) uses sentence-transformer cosine similarity
    instead of TF-IDF cosine similarity.

    Parameters
    ----------
    query_texts:
        Mapping from query_id to query text.
    embedder:
        A DenseEmbedder instance (shared across extractors to reuse cache).
    """

    def __init__(
        self,
        query_texts: dict[int, str],
        embedder: DenseEmbedder,
    ) -> None:
        super().__init__(query_texts)
        self._embedder = embedder

        # Pre-compute query embeddings in batch for efficiency
        texts = list(query_texts.values())
        if texts:
            self._embedder.encode_batch(texts)

    def extract(self, query_id: int, tool: BFCLTool, n_candidates: int) -> np.ndarray:
        """Extract feature vector with dense desc_sim."""
        query_text = self._query_texts[query_id]
        query_tokens = _tokenize(query_text)
        query_token_set = set(query_tokens)

        # Feature 0: dense cosine similarity (replaces TF-IDF)
        desc_sim = self._embedder.cosine_similarity(query_text, tool.description)

        # Features 1-7: same as parent class
        name_tokens = set(re.split(r"[._]", tool.name.lower()))
        name_tokens.discard("")
        if name_tokens:
            overlap = len(query_token_set & name_tokens)
            name_sim = overlap / len(name_tokens)
        else:
            name_sim = 0.0

        properties: dict[str, Any] = tool.parameter_spec.get("properties", {})
        if properties:
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

        required: list[str] = tool.parameter_spec.get("required", [])
        n_required = len(required)
        n_params_norm = min(n_required / 10.0, 1.0)

        desc_len = len(tool.description)
        desc_len_norm = min(desc_len / 500.0, 1.0)

        n_cand_norm = math.log1p(n_candidates) / math.log1p(40)

        n_total_params = len(properties)
        required_ratio = n_required / n_total_params if n_total_params > 0 else 0.0

        has_desc = 1.0 if desc_len > 10 else 0.0

        return np.array(
            [desc_sim, name_sim, param_match, n_params_norm,
             desc_len_norm, n_cand_norm, required_ratio, has_desc],
            dtype=np.float64,
        )


class DenseDescMatchSelector:
    """Dense embedding description matcher — pure retrieval baseline.

    Computes cosine similarity between query and each candidate tool's
    description using a sentence-transformer model, then picks argmax.
    No online learning, no features beyond description similarity.

    This is the dense-embedding analog of DescriptionMatchSelector.
    """

    def __init__(
        self,
        query_texts: dict[int, str],
        embedder: DenseEmbedder,
        seed: int = 0,
    ) -> None:
        self._query_texts = query_texts
        self._embedder = embedder
        self.rng = np.random.default_rng(seed)

        # Pre-compute query embeddings
        texts = list(query_texts.values())
        if texts:
            self._embedder.encode_batch(texts)

        # For RepeatedBFCLEnvironment query_id resolution
        self._cap_to_query_id: dict[str, int] = {}

    def select(self, query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)

        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id

        # Resolve query_id
        qid = query.query_id
        if qid not in self._query_texts:
            qid = self._cap_to_query_id.get(cap, qid)
        else:
            if cap not in self._cap_to_query_id:
                self._cap_to_query_id[cap] = qid

        query_text = self._query_texts.get(qid, "")

        scores = [
            self._embedder.cosine_similarity(query_text, getattr(t, "description", ""))
            for t in declarers
        ]
        best_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if s == best_score]
        idx = self.rng.choice(best_indices)
        return declarers[idx].tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        pass  # No online learning
