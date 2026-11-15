"""BilinTS: Linear Thompson Sampling for contextual tool selection.

Implements Bayesian linear regression with Thompson Sampling over a GLOBAL
parameter vector theta shared across all tools.  Features are per-(query, tool)
pair, produced by :class:`~tats.features.ToolFeatureExtractor`.

Model
-----
r = x^T theta + epsilon,   epsilon ~ N(0, sigma^2)

Prior:      theta ~ N(theta_0, (1/lambda) I)
Precision:  B = lambda I + (1/sigma^2) sum x_i x_i^T
RHS:        f = lambda theta_0 + (1/sigma^2) sum r_i x_i
Posterior:  theta_mean = B^{-1} f

Thompson Sampling: sample theta_tilde ~ N(theta_mean, sigma^2 B^{-1}),
pick argmax_t  x(q,t)^T theta_tilde.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .features import ToolFeatureExtractor
from .models import Query


class BilinTSSelector:
    """Linear Thompson Sampling with description-aware features.

    Parameters
    ----------
    feature_extractor:
        Produces fixed-dim feature vectors for (query, tool) pairs.
    seed:
        Random seed.
    lambda_reg:
        Prior precision (regularization strength).
    noise_var:
        Observation noise variance sigma^2.
    theta_init:
        Prior mean theta_0.  If ``None``, uses the zero vector (cold start).
    feature_mask:
        Binary mask for ablation studies.  If set, element-wise multiplied
        with every feature vector before use.
    """

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        lambda_reg: float = 1.0,
        noise_var: float = 0.25,
        theta_init: np.ndarray | None = None,
        B_init: np.ndarray | None = None,
        feature_mask: np.ndarray | None = None,
    ) -> None:
        self._fe = feature_extractor
        self.rng = np.random.default_rng(seed)
        self._lambda = lambda_reg
        self._sigma2 = noise_var
        self._d = feature_extractor.n_features
        self._mask = feature_mask

        # Prior mean
        if theta_init is not None:
            self._theta0 = theta_init.copy()
        else:
            self._theta0 = np.zeros(self._d)

        # Warm-start precision matrix
        self._B0 = B_init

        # Initialize sufficient statistics from prior
        self._reset()

        # Feature cache: tool_id -> feature vector (set during select, used in update)
        self._cached_features: dict[str, np.ndarray] = {}

        # Map capability (scenario_id) -> first query_id seen for that cap.
        # Needed for RepeatedBFCLEnvironment where query_ids exceed the
        # feature extractor's corpus range but the same scenario repeats.
        self._cap_to_query_id: dict[str, int] = {}

        # Update counter
        self.n_updates: int = 0

    def _reset(self) -> None:
        """Initialize (or re-initialize) sufficient statistics from the prior.

        When ``B_init`` was provided, the prior is N(theta_0, sigma^2 * B_0^{-1})
        so the sufficient statistics are B = B_0, f = B_0 @ theta_0.
        Otherwise, use the default prior B = lambda * I, f = lambda * theta_0.
        """
        if self._B0 is not None:
            self._B = self._B0.copy()
            self._f = self._B0 @ self._theta0  # f = B_0 * theta_0 (Bayesian prior)
        else:
            self._B = self._lambda * np.eye(self._d)
            self._f = self._lambda * self._theta0.copy()
        self._theta_mean = np.linalg.solve(self._B, self._f)
        self._B_inv = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_B_inv(self) -> np.ndarray:
        """Return B^{-1}, computing and caching if needed."""
        if self._B_inv is None:
            self._B_inv = np.linalg.inv(self._B)
        return self._B_inv

    def _apply_mask(self, x: np.ndarray) -> np.ndarray:
        """Apply feature mask if set."""
        if self._mask is not None:
            return x * self._mask
        return x

    # ------------------------------------------------------------------
    # Selector interface
    # ------------------------------------------------------------------

    def select(self, query: Query, env) -> str:
        """Select a tool via Thompson Sampling over linear reward model.

        Steps:
        1. Get candidate tools from environment.
        2. Extract and cache features for each candidate.
        3. Sample theta_tilde from posterior.
        4. Score each candidate: x^T theta_tilde.
        5. Return tool_id with highest score.
        """
        cap = query.required_capability
        declarers = env.get_declarers(cap)

        if not declarers:
            # Fallback: random tool
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id

        n_candidates = len(declarers)

        # Resolve query_id for the feature extractor.
        # In RepeatedBFCLEnvironment, query_ids exceed the feature corpus
        # range but map to the same scenarios.  We cache the first query_id
        # seen per capability and reuse it for subsequent rounds.
        qid = query.query_id
        if qid not in self._fe._query_texts:
            # Fall back to the first query_id seen for this capability
            qid = self._cap_to_query_id.get(cap, qid)
        else:
            # Record mapping from capability to this (valid) query_id
            if cap not in self._cap_to_query_id:
                self._cap_to_query_id[cap] = qid

        # Extract features and cache them
        features: dict[str, np.ndarray] = {}
        for tool in declarers:
            x = self._fe.extract(qid, tool, n_candidates)
            x = self._apply_mask(x)
            features[tool.tool_id] = x

        # Store in instance cache for update()
        self._cached_features = features

        # Sample theta_tilde ~ N(theta_mean, sigma^2 * B^{-1})
        B_inv = self._get_B_inv()
        cov = self._sigma2 * B_inv
        try:
            theta_tilde = self.rng.multivariate_normal(self._theta_mean, cov)
        except np.linalg.LinAlgError:
            cov_safe = cov + 1e-6 * np.eye(self._d)
            theta_tilde = self.rng.multivariate_normal(self._theta_mean, cov_safe)

        # Score candidates
        best_tool_id = declarers[0].tool_id
        best_score = -np.inf
        for tool in declarers:
            score = float(features[tool.tool_id] @ theta_tilde)
            if score > best_score:
                best_score = score
                best_tool_id = tool.tool_id

        return best_tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        """Update posterior with the observed reward.

        Uses cached features from the most recent ``select()`` call.
        """
        x = self._cached_features.get(tool_id)
        if x is None:
            # No cached features (should not happen in normal flow)
            return

        r = 1.0 if success else 0.0
        inv_sigma2 = 1.0 / self._sigma2

        # Rank-1 update to precision: B += (1/sigma^2) x x^T
        self._B += inv_sigma2 * np.outer(x, x)

        # Update RHS: f += (1/sigma^2) r x
        self._f += inv_sigma2 * r * x

        # Recompute posterior mean
        self._theta_mean = np.linalg.solve(self._B, self._f)

        # Invalidate B_inv cache
        self._B_inv = None

        self.n_updates += 1

    def get_predicted_prob(self, tool_id: str, capability: str) -> float:
        """Return clipped predicted reward x^T theta_mean."""
        x = self._cached_features.get(tool_id)
        if x is None:
            return 0.5  # uninformative default
        pred = float(x @ self._theta_mean)
        return max(0.0, min(1.0, pred))

    def get_weights(self) -> dict[str, float]:
        """Return current theta_mean as named feature weights."""
        names = [
            "desc_sim", "name_sim", "param_match", "n_params_norm",
            "desc_len_norm", "n_cand_norm", "required_ratio", "has_desc",
        ]
        return {name: float(self._theta_mean[i]) for i, name in enumerate(names)}


class BilinTSGreedySelector(BilinTSSelector):
    """BilinTS without exploration: uses posterior mean instead of sampling.

    Equivalent to always picking argmax x^T theta_mean (no Thompson sample).
    Useful as an ablation to isolate the value of exploration.
    """

    def select(self, query: Query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)

        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id

        n_candidates = len(declarers)

        # Resolve query_id (same logic as parent; see BilinTSSelector.select)
        qid = query.query_id
        if qid not in self._fe._query_texts:
            qid = self._cap_to_query_id.get(cap, qid)
        else:
            if cap not in self._cap_to_query_id:
                self._cap_to_query_id[cap] = qid

        # Extract features and cache them
        features: dict[str, np.ndarray] = {}
        for tool in declarers:
            x = self._fe.extract(qid, tool, n_candidates)
            x = self._apply_mask(x)
            features[tool.tool_id] = x

        self._cached_features = features

        # Greedy: use posterior mean (no sampling)
        best_tool_id = declarers[0].tool_id
        best_score = -np.inf
        for tool in declarers:
            score = float(features[tool.tool_id] @ self._theta_mean)
            if score > best_score:
                best_score = score
                best_tool_id = tool.tool_id

        return best_tool_id


class ColdStartLinTSSelector(BilinTSSelector):
    """BilinTS with zero prior (cold start, no warm-start initialization).

    Identical to :class:`BilinTSSelector` but always starts with theta_0 = 0.
    """

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        lambda_reg: float = 1.0,
        noise_var: float = 0.25,
        feature_mask: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            feature_extractor=feature_extractor,
            seed=seed,
            lambda_reg=lambda_reg,
            noise_var=noise_var,
            theta_init=None,  # zero prior
            B_init=None,      # no warm-start precision
            feature_mask=feature_mask,
        )


class LinUCBSelector:
    """LinUCB: contextual bandit with Upper Confidence Bound exploration.

    Uses the same Bayesian linear regression model as BilinTS, but replaces
    Thompson Sampling with UCB-style exploration:

        score(t) = x(q,t)^T theta_mean + alpha * sqrt(x(q,t)^T B^{-1} x(q,t))

    The first term is the predicted reward (exploitation), the second is a
    confidence-width bonus (exploration).  Unlike BilinTS, exploration is
    deterministic given the same state — no random posterior sampling.

    Always starts cold (no warm-start initialization).

    Parameters
    ----------
    feature_extractor:
        Produces fixed-dim feature vectors for (query, tool) pairs.
    seed:
        Random seed (used only for tie-breaking / fallback).
    alpha:
        Exploration coefficient controlling the width of the UCB bonus.
    lambda_reg:
        Prior precision (regularization strength).
    sigma2:
        Observation noise variance sigma^2.
    """

    name = "LinUCB"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
    ) -> None:
        self._fe = feature_extractor
        self.rng = np.random.default_rng(seed)
        self._alpha = alpha
        self._lambda = lambda_reg
        self._sigma2 = sigma2
        self._d = feature_extractor.n_features

        # Cold-start prior: theta_0 = 0
        self._B = self._lambda * np.eye(self._d)
        self._f = np.zeros(self._d)
        self._theta_mean = np.zeros(self._d)
        self._B_inv: np.ndarray | None = None

        # Feature cache: tool_id -> feature vector (set during select, used in update)
        self._cached_features: dict[str, np.ndarray] = {}

        # Map capability (scenario_id) -> first query_id seen for that cap.
        # Needed for RepeatedBFCLEnvironment where query_ids exceed the
        # feature extractor's corpus range but the same scenario repeats.
        self._cap_to_query_id: dict[str, int] = {}

        # Update counter
        self.n_updates: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_B_inv(self) -> np.ndarray:
        """Return B^{-1}, computing and caching if needed."""
        if self._B_inv is None:
            self._B_inv = np.linalg.inv(self._B)
        return self._B_inv

    # ------------------------------------------------------------------
    # Selector interface
    # ------------------------------------------------------------------

    def select(self, query: Query, env) -> str:
        """Select a tool via Upper Confidence Bound over linear reward model.

        Steps:
        1. Get candidate tools from environment.
        2. Extract and cache features for each candidate.
        3. Score each candidate: x^T theta_mean + alpha * sqrt(x^T B^{-1} x).
        4. Return tool_id with highest UCB score.
        """
        cap = query.required_capability
        declarers = env.get_declarers(cap)

        if not declarers:
            # Fallback: random tool
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id

        n_candidates = len(declarers)

        # Resolve query_id for the feature extractor.
        # In RepeatedBFCLEnvironment, query_ids exceed the feature corpus
        # range but map to the same scenarios.  We cache the first query_id
        # seen per capability and reuse it for subsequent rounds.
        qid = query.query_id
        if qid not in self._fe._query_texts:
            qid = self._cap_to_query_id.get(cap, qid)
        else:
            if cap not in self._cap_to_query_id:
                self._cap_to_query_id[cap] = qid

        # Extract features and cache them
        features: dict[str, np.ndarray] = {}
        for tool in declarers:
            x = self._fe.extract(qid, tool, n_candidates)
            features[tool.tool_id] = x

        # Store in instance cache for update()
        self._cached_features = features

        # Compute UCB scores
        B_inv = self._get_B_inv()
        best_tool_id = declarers[0].tool_id
        best_score = -np.inf
        for tool in declarers:
            x = features[tool.tool_id]
            exploit = float(x @ self._theta_mean)
            explore = self._alpha * float(np.sqrt(x @ B_inv @ x))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_tool_id = tool.tool_id

        return best_tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        """Update posterior with the observed reward.

        Uses cached features from the most recent ``select()`` call.
        """
        x = self._cached_features.get(tool_id)
        if x is None:
            return

        r = 1.0 if success else 0.0
        inv_sigma2 = 1.0 / self._sigma2

        # Rank-1 update to precision: B += (1/sigma^2) x x^T
        self._B += inv_sigma2 * np.outer(x, x)

        # Update RHS: f += (1/sigma^2) r x
        self._f += inv_sigma2 * r * x

        # Recompute posterior mean
        self._theta_mean = np.linalg.solve(self._B, self._f)

        # Invalidate B_inv cache
        self._B_inv = None

        self.n_updates += 1

    def get_predicted_prob(self, tool_id: str, capability: str = "") -> float:
        """Return clipped predicted reward x^T theta_mean."""
        x = self._cached_features.get(tool_id)
        if x is None:
            return 0.5  # uninformative default
        pred = float(x @ self._theta_mean)
        return max(0.0, min(1.0, pred))

    def get_weights(self) -> dict[str, float]:
        """Return current theta_mean as named feature weights."""
        names = [
            "desc_sim", "name_sim", "param_match", "n_params_norm",
            "desc_len_norm", "n_cand_norm", "required_ratio", "has_desc",
        ]
        return {name: float(self._theta_mean[i]) for i, name in enumerate(names)}


# ======================================================================
# Supervised baselines (for bandit vs supervised comparison)
# ======================================================================


class SupervisedOfflineSelector:
    """Pre-trained logistic regression applied greedily, no online updates.

    Tests whether a simple offline-trained classifier on the same 8 features
    can match bandit methods.  Training uses cross-dataset data (same source
    as warm-start) to avoid data leakage.
    """

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        model: LogisticRegression,
        seed: int = 0,
    ) -> None:
        self._fe = feature_extractor
        self._model = model
        self.rng = np.random.default_rng(seed)
        self._cached_features: dict[str, np.ndarray] = {}
        self._cap_to_query_id: dict[str, int] = {}
        self.n_updates: int = 0

    def select(self, query: Query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)

        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id

        n_candidates = len(declarers)

        # Resolve query_id (same pattern as BilinTS)
        qid = query.query_id
        if qid not in self._fe._query_texts:
            qid = self._cap_to_query_id.get(cap, qid)
        else:
            if cap not in self._cap_to_query_id:
                self._cap_to_query_id[cap] = qid

        # Extract features and cache
        features: dict[str, np.ndarray] = {}
        for tool in declarers:
            x = self._fe.extract(qid, tool, n_candidates)
            features[tool.tool_id] = x
        self._cached_features = features

        # Greedy: select candidate with highest P(success)
        best_tool_id = declarers[0].tool_id
        best_prob = -1.0
        for tool in declarers:
            x = features[tool.tool_id].reshape(1, -1)
            prob = float(self._model.predict_proba(x)[0, 1])
            if prob > best_prob:
                best_prob = prob
                best_tool_id = tool.tool_id

        return best_tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        self.n_updates += 1

    def get_predicted_prob(self, tool_id: str, capability: str = "") -> float:
        x = self._cached_features.get(tool_id)
        if x is None:
            return 0.5
        return float(self._model.predict_proba(x.reshape(1, -1))[0, 1])


class SupervisedOnlineSelector:
    """Online logistic regression with greedy selection.

    Incrementally collects (feature, reward) pairs from interactions and
    periodically retrains a logistic regression, then selects greedily.
    Uses the same information as the bandit (only observes reward for
    the selected arm), but replaces Bayesian updates with MLE retraining.

    Tests whether bandit machinery adds value beyond simple supervised
    learning on the same features.
    """

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        min_samples: int = 20,
        retrain_every: int = 10,
        C: float = 1.0,
    ) -> None:
        self._fe = feature_extractor
        self.rng = np.random.default_rng(seed)
        self._min_samples = min_samples
        self._retrain_every = retrain_every
        self._C = C
        self._seed = seed

        # Training buffer
        self._X: list[np.ndarray] = []
        self._y: list[int] = []

        # Current model
        self._model: LogisticRegression | None = None
        self._fitted = False

        # Caches
        self._cached_features: dict[str, np.ndarray] = {}
        self._cap_to_query_id: dict[str, int] = {}
        self.n_updates: int = 0

    def _retrain(self) -> None:
        X = np.array(self._X)
        y = np.array(self._y)
        if len(np.unique(y)) < 2:
            return
        self._model = LogisticRegression(
            C=self._C, l1_ratio=0, random_state=self._seed,
            max_iter=500, solver="lbfgs",
        )
        self._model.fit(X, y)
        self._fitted = True

    def select(self, query: Query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)

        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id

        n_candidates = len(declarers)

        # Resolve query_id (same pattern as BilinTS)
        qid = query.query_id
        if qid not in self._fe._query_texts:
            qid = self._cap_to_query_id.get(cap, qid)
        else:
            if cap not in self._cap_to_query_id:
                self._cap_to_query_id[cap] = qid

        # Extract features and cache
        features: dict[str, np.ndarray] = {}
        for tool in declarers:
            x = self._fe.extract(qid, tool, n_candidates)
            features[tool.tool_id] = x
        self._cached_features = features

        # Cold start: random selection
        if not self._fitted:
            idx = self.rng.integers(len(declarers))
            return declarers[idx].tool_id

        # Greedy: select candidate with highest P(success)
        best_tool_id = declarers[0].tool_id
        best_prob = -1.0
        for tool in declarers:
            x = features[tool.tool_id].reshape(1, -1)
            prob = float(self._model.predict_proba(x)[0, 1])
            if prob > best_prob:
                best_prob = prob
                best_tool_id = tool.tool_id

        return best_tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        x = self._cached_features.get(tool_id)
        if x is None:
            return

        self._X.append(x.copy())
        self._y.append(1 if success else 0)
        self.n_updates += 1

        if self.n_updates >= self._min_samples and self.n_updates % self._retrain_every == 0:
            self._retrain()

    def get_predicted_prob(self, tool_id: str, capability: str = "") -> float:
        x = self._cached_features.get(tool_id)
        if x is None or not self._fitted:
            return 0.5
        return float(self._model.predict_proba(x.reshape(1, -1))[0, 1])


# ======================================================================
# GLM Bandit: Logistic Thompson Sampling (Laplace approximation)
# ======================================================================


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


class LogisticTSSelector:
    """Logistic Thompson Sampling with Laplace approximation.

    Uses a logistic link function p(r=1|x) = sigmoid(x^T theta) instead of
    the Gaussian linear model used by BilinTS/LinUCB.  The posterior is
    approximated via the Laplace approximation:

        theta_MAP = argmax log p(data|theta) + log p(theta)
        H = -nabla^2 log posterior evaluated at theta_MAP
        posterior ≈ N(theta_MAP, H^{-1})

    Thompson Sampling: sample theta_tilde ~ N(theta_MAP, H^{-1}),
    pick argmax_t sigmoid(x(q,t)^T theta_tilde).

    rather than Gaussian models.  If linear ≈ logistic, proves linear is fine.

    Parameters
    ----------
    feature_extractor:
        Produces fixed-dim feature vectors for (query, tool) pairs.
    seed:
        Random seed.
    lambda_reg:
        L2 regularization on theta (prior precision).
    refit_every:
        Recompute MAP/Hessian every N updates (amortized cost).
    """

    name = "GLM-TS"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        lambda_reg: float = 1.0,
        refit_every: int = 10,
    ) -> None:
        self._fe = feature_extractor
        self.rng = np.random.default_rng(seed)
        self._lambda = lambda_reg
        self._refit_every = refit_every
        self._d = feature_extractor.n_features

        # Prior: theta ~ N(0, (1/lambda) I)
        self._theta_map = np.zeros(self._d)
        self._H = self._lambda * np.eye(self._d)  # precision (Hessian)
        self._H_inv: np.ndarray | None = None

        # Data buffer for MAP computation
        self._X: list[np.ndarray] = []
        self._y: list[float] = []

        # Caches
        self._cached_features: dict[str, np.ndarray] = {}
        self._cap_to_query_id: dict[str, int] = {}
        self.n_updates: int = 0
        self._dirty = False  # needs refit

    def _compute_map_and_hessian(self) -> None:
        """Compute MAP estimate and Hessian via Newton-Raphson."""
        if not self._X:
            return

        X = np.array(self._X)  # (n, d)
        y = np.array(self._y)  # (n,)
        n = len(y)
        theta = self._theta_map.copy()

        # Newton-Raphson iterations
        for _ in range(20):
            z = X @ theta                    # (n,)
            p = _sigmoid(z)                  # (n,)

            # Gradient: X^T (p - y) + lambda * theta
            grad = X.T @ (p - y) + self._lambda * theta

            # Hessian: X^T diag(p*(1-p)) X + lambda I
            w = p * (1.0 - p)               # (n,)
            w = np.clip(w, 1e-8, None)       # numerical stability
            H = (X.T * w) @ X + self._lambda * np.eye(self._d)

            # Newton step
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break

            theta = theta - delta

            if np.linalg.norm(delta) < 1e-6:
                break

        self._theta_map = theta
        # Final Hessian at MAP
        z = X @ theta
        p = _sigmoid(z)
        w = np.clip(p * (1.0 - p), 1e-8, None)
        self._H = (X.T * w) @ X + self._lambda * np.eye(self._d)
        self._H_inv = None
        self._dirty = False

    def _get_H_inv(self) -> np.ndarray:
        if self._H_inv is None:
            try:
                self._H_inv = np.linalg.inv(self._H)
            except np.linalg.LinAlgError:
                self._H_inv = np.linalg.inv(self._H + 1e-6 * np.eye(self._d))
        return self._H_inv

    def select(self, query: Query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)

        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id

        n_candidates = len(declarers)

        # Resolve query_id
        qid = query.query_id
        if qid not in self._fe._query_texts:
            qid = self._cap_to_query_id.get(cap, qid)
        else:
            if cap not in self._cap_to_query_id:
                self._cap_to_query_id[cap] = qid

        # Extract features
        features: dict[str, np.ndarray] = {}
        for tool in declarers:
            x = self._fe.extract(qid, tool, n_candidates)
            features[tool.tool_id] = x
        self._cached_features = features

        # Refit if needed
        if self._dirty and self.n_updates >= self._refit_every:
            self._compute_map_and_hessian()

        # Sample theta_tilde ~ N(theta_MAP, H^{-1})
        H_inv = self._get_H_inv()
        try:
            theta_tilde = self.rng.multivariate_normal(self._theta_map, H_inv)
        except np.linalg.LinAlgError:
            H_inv_safe = H_inv + 1e-6 * np.eye(self._d)
            theta_tilde = self.rng.multivariate_normal(self._theta_map, H_inv_safe)

        # Score candidates: sigmoid(x^T theta_tilde)
        best_tool_id = declarers[0].tool_id
        best_score = -np.inf
        for tool in declarers:
            score = float(features[tool.tool_id] @ theta_tilde)
            if score > best_score:
                best_score = score
                best_tool_id = tool.tool_id

        return best_tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        x = self._cached_features.get(tool_id)
        if x is None:
            return

        self._X.append(x.copy())
        self._y.append(1.0 if success else 0.0)
        self.n_updates += 1
        self._dirty = True

        # Amortized refit
        if self.n_updates % self._refit_every == 0:
            self._compute_map_and_hessian()

    def get_predicted_prob(self, tool_id: str, capability: str = "") -> float:
        x = self._cached_features.get(tool_id)
        if x is None:
            return 0.5
        return float(_sigmoid(x @ self._theta_map))
