"""Novel algorithmic variants for tool selection under adversarial conditions.

Responds to reviewer critique that LinUCB/BilinTS are standard algorithms and
the paper lacks algorithmic novelty. These variants encode the asymmetric
trust structure between description features (attackable) and schema features
(structurally harder to forge) directly into the learning algorithm.

Feature index convention (from tats.features.ToolFeatureExtractor):
  0. desc_sim       (description)
  1. name_sim       (description)
  2. param_match    (SCHEMA)
  3. n_params_norm  (SCHEMA)
  4. desc_len_norm  (description)
  5. n_cand_norm    (context — neutral)
  6. required_ratio (SCHEMA)
  7. has_desc       (description)

Variants
--------
- TWLinUCBSelector: Trust-Weighted UCB. Splits the UCB exploration bonus
  into per-group terms with different exploration coefficients. Schema
  features get aggressive exploration (alpha_s), description features
  get conservative exploration (alpha_d < alpha_s). Rationale: under
  adversarial documentation, description signals are untrusted, so we
  should not "explore" as eagerly based on them.

- ConsistencyGatedSelector (CGB): Maintains two parallel posteriors — one
  fit only on description features, one only on schema features — and at
  selection time computes the disagreement between their predicted scores
  for each candidate. If the schema-only head predicts a candidate much
  worse than the description-only head does, that candidate is penalized
  (likely an adversarial tool crafted to maximize description similarity).
"""

from __future__ import annotations

import numpy as np

from .features import ToolFeatureExtractor
from .models import Query


# Feature groups by index
SCHEMA_IDX = np.array([2, 3, 6])          # param_match, n_params_norm, required_ratio
DESC_IDX = np.array([0, 1, 4, 7])         # desc_sim, name_sim, desc_len_norm, has_desc
CONTEXT_IDX = np.array([5])                # n_cand_norm


def _mask(d: int, idx: np.ndarray) -> np.ndarray:
    """Return a d-dim binary mask with 1s at positions in idx."""
    m = np.zeros(d, dtype=np.float64)
    m[idx] = 1.0
    return m


class TWLinUCBSelector:
    """Trust-Weighted LinUCB: differential exploration for trusted vs untrusted features.

    Same shared posterior over theta as LinUCB, but the UCB bonus is decomposed:

        score(t) = x^T theta_mean
                   + alpha_s * sqrt(x_s^T B^{-1} x_s)      # schema exploration
                   + alpha_d * sqrt(x_d^T B^{-1} x_d)      # description exploration

    where x_s (resp. x_d) is x masked to schema (resp. description) feature
    indices. With alpha_s > alpha_d, the algorithm prefers to explore along
    trusted schema dimensions and avoids being "confidently-hijacked" along
    description dimensions an attacker can manipulate.

    Setting alpha_s = alpha_d recovers standard LinUCB.
    """

    name = "TW-LinUCB"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        alpha_schema: float = 2.0,
        alpha_desc: float = 0.5,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
    ) -> None:
        self._fe = feature_extractor
        self.rng = np.random.default_rng(seed)
        self._alpha_s = alpha_schema
        self._alpha_d = alpha_desc
        self._lambda = lambda_reg
        self._sigma2 = sigma2
        self._d = feature_extractor.n_features

        self._m_s = _mask(self._d, SCHEMA_IDX)
        self._m_d = _mask(self._d, DESC_IDX)

        # Shared posterior (same as LinUCB)
        self._B = self._lambda * np.eye(self._d)
        self._f = np.zeros(self._d)
        self._theta_mean = np.zeros(self._d)
        self._B_inv: np.ndarray | None = None

        self._cached_features: dict[str, np.ndarray] = {}
        self._cap_to_query_id: dict[str, int] = {}
        self.n_updates: int = 0

    def _get_B_inv(self) -> np.ndarray:
        if self._B_inv is None:
            self._B_inv = np.linalg.inv(self._B)
        return self._B_inv

    def select(self, query: Query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id
        n_candidates = len(declarers)

        qid = query.query_id
        if qid not in self._fe._query_texts:
            qid = self._cap_to_query_id.get(cap, qid)
        else:
            if cap not in self._cap_to_query_id:
                self._cap_to_query_id[cap] = qid

        features: dict[str, np.ndarray] = {}
        for tool in declarers:
            features[tool.tool_id] = self._fe.extract(qid, tool, n_candidates)
        self._cached_features = features

        B_inv = self._get_B_inv()
        best_tool_id = declarers[0].tool_id
        best_score = -np.inf
        for tool in declarers:
            x = features[tool.tool_id]
            exploit = float(x @ self._theta_mean)
            x_s = x * self._m_s
            x_d = x * self._m_d
            bonus_s = self._alpha_s * float(np.sqrt(max(0.0, x_s @ B_inv @ x_s)))
            bonus_d = self._alpha_d * float(np.sqrt(max(0.0, x_d @ B_inv @ x_d)))
            score = exploit + bonus_s + bonus_d
            if score > best_score:
                best_score = score
                best_tool_id = tool.tool_id
        return best_tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        x = self._cached_features.get(tool_id)
        if x is None:
            return
        r = 1.0 if success else 0.0
        inv_s2 = 1.0 / self._sigma2
        self._B += inv_s2 * np.outer(x, x)
        self._f += inv_s2 * r * x
        self._theta_mean = np.linalg.solve(self._B, self._f)
        self._B_inv = None
        self.n_updates += 1

    def get_predicted_prob(self, tool_id: str, capability: str = "") -> float:
        x = self._cached_features.get(tool_id)
        if x is None:
            return 0.5
        return max(0.0, min(1.0, float(x @ self._theta_mean)))


class ConsistencyGatedSelector:
    """Consistency-Gated Bandit: runtime adversarial detection via two-head disagreement.

    Maintains two independent Bayesian linear posteriors:
      - theta_d trained on description-only features
      - theta_s trained on schema-only features

    At selection time, for each candidate we compute:
      pred_d(t) = x_d^T theta_d_mean
      pred_s(t) = x_s^T theta_s_mean
      disagree(t) = max(0, pred_d(t) - pred_s(t))   # desc-high but schema-low

    This flags candidates whose description features predict high reward but
    schema features predict low reward — precisely the profile of a maliciously
    injected tool whose description mimics the query but whose schema is
    unrelated. We subtract a disagreement penalty from the UCB score:

      score(t) = x^T theta_mean + alpha * bonus - beta * disagree(t)

    An attacker faces a bind: to fool the combined model they must
    simultaneously drive up both heads, but they cannot obtain the
    ground-truth schema without knowing what parameters the query requires.
    """

    name = "CGB"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        alpha: float = 1.0,
        beta: float = 2.0,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
    ) -> None:
        self._fe = feature_extractor
        self.rng = np.random.default_rng(seed)
        self._alpha = alpha
        self._beta = beta
        self._lambda = lambda_reg
        self._sigma2 = sigma2
        self._d = feature_extractor.n_features
        self._m_s = _mask(self._d, SCHEMA_IDX)
        self._m_d = _mask(self._d, DESC_IDX)

        # Three posteriors: combined (for main score), desc-only, schema-only
        self._B = self._lambda * np.eye(self._d)
        self._f = np.zeros(self._d)
        self._theta = np.zeros(self._d)

        self._B_d = self._lambda * np.eye(self._d)
        self._f_d = np.zeros(self._d)
        self._theta_d = np.zeros(self._d)

        self._B_s = self._lambda * np.eye(self._d)
        self._f_s = np.zeros(self._d)
        self._theta_s = np.zeros(self._d)

        self._B_inv: np.ndarray | None = None

        self._cached_features: dict[str, np.ndarray] = {}
        self._cap_to_query_id: dict[str, int] = {}
        self.n_updates: int = 0

    def _get_B_inv(self) -> np.ndarray:
        if self._B_inv is None:
            self._B_inv = np.linalg.inv(self._B)
        return self._B_inv

    def select(self, query: Query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id
        n_candidates = len(declarers)

        qid = query.query_id
        if qid not in self._fe._query_texts:
            qid = self._cap_to_query_id.get(cap, qid)
        else:
            if cap not in self._cap_to_query_id:
                self._cap_to_query_id[cap] = qid

        features: dict[str, np.ndarray] = {}
        for tool in declarers:
            features[tool.tool_id] = self._fe.extract(qid, tool, n_candidates)
        self._cached_features = features

        B_inv = self._get_B_inv()
        best_tool_id = declarers[0].tool_id
        best_score = -np.inf
        for tool in declarers:
            x = features[tool.tool_id]
            x_d = x * self._m_d
            x_s = x * self._m_s
            exploit = float(x @ self._theta)
            bonus = self._alpha * float(np.sqrt(max(0.0, x @ B_inv @ x)))
            pred_d = float(x_d @ self._theta_d)
            pred_s = float(x_s @ self._theta_s)
            disagree = max(0.0, pred_d - pred_s)
            score = exploit + bonus - self._beta * disagree
            if score > best_score:
                best_score = score
                best_tool_id = tool.tool_id
        return best_tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        x = self._cached_features.get(tool_id)
        if x is None:
            return
        r = 1.0 if success else 0.0
        inv_s2 = 1.0 / self._sigma2

        # Main (combined)
        self._B += inv_s2 * np.outer(x, x)
        self._f += inv_s2 * r * x
        self._theta = np.linalg.solve(self._B, self._f)

        # Description head (only desc features active)
        x_d = x * self._m_d
        self._B_d += inv_s2 * np.outer(x_d, x_d)
        self._f_d += inv_s2 * r * x_d
        # regularized solve with slight jitter to stay PD
        self._theta_d = np.linalg.solve(
            self._B_d + 1e-6 * np.eye(self._d), self._f_d
        )

        # Schema head
        x_s = x * self._m_s
        self._B_s += inv_s2 * np.outer(x_s, x_s)
        self._f_s += inv_s2 * r * x_s
        self._theta_s = np.linalg.solve(
            self._B_s + 1e-6 * np.eye(self._d), self._f_s
        )

        self._B_inv = None
        self.n_updates += 1

    def get_predicted_prob(self, tool_id: str, capability: str = "") -> float:
        x = self._cached_features.get(tool_id)
        if x is None:
            return 0.5
        return max(0.0, min(1.0, float(x @ self._theta)))


class ADABSelector:
    """Anomaly-Detection-Augmented Bandit.

    Addresses the following observation: under adaptive attacks, an adversary
    drives ALL features of the malicious tool to extreme values (param_match=1,
    desc_sim=1, etc.). Benign tools have MODERATE feature profiles because a
    real tool covers some but not all of a query's concerns. We exploit this
    distributional asymmetry explicitly.

    Algorithm:
      1. LinUCB-style posterior over theta.
      2. Running mean mu and covariance Sigma of feature vectors for
         *rewarded* tools (the "benign distribution").
      3. For each candidate, Mahalanobis distance d(x) = (x-mu)^T Sigma^{-1} (x-mu).
      4. Score(t) = x^T theta + alpha * bonus - gamma * d(x).

    A malicious tool with x = [1,1,1,1,1,1,1,1] is far from the benign mu,
    so its score is heavily penalized. Benign tools, near mu, are not penalized.

    The attacker cannot both (a) match individual features highly to dominate
    LinUCB and (b) stay close to the benign distribution mean — the two
    constraints conflict.
    """

    name = "ADAB"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma: float = 1.0,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        # Warm up anomaly detector on this many successes before using it.
        min_samples: int = 20,
    ) -> None:
        self._fe = feature_extractor
        self.rng = np.random.default_rng(seed)
        self._alpha = alpha
        self._gamma = gamma
        self._lambda = lambda_reg
        self._sigma2 = sigma2
        self._min_samples = min_samples
        self._d = feature_extractor.n_features

        # LinUCB posterior
        self._B = self._lambda * np.eye(self._d)
        self._f = np.zeros(self._d)
        self._theta_mean = np.zeros(self._d)
        self._B_inv: np.ndarray | None = None

        # Benign feature distribution (only from successful selections)
        self._benign_sum = np.zeros(self._d)
        self._benign_sq = np.zeros((self._d, self._d))
        self._benign_n = 0

        self._cached_features: dict[str, np.ndarray] = {}
        self._cap_to_query_id: dict[str, int] = {}
        self.n_updates: int = 0

    def _get_B_inv(self) -> np.ndarray:
        if self._B_inv is None:
            self._B_inv = np.linalg.inv(self._B)
        return self._B_inv

    def _mahalanobis(self, x: np.ndarray) -> float:
        """Return Mahalanobis distance of x from benign distribution."""
        if self._benign_n < self._min_samples:
            return 0.0
        mu = self._benign_sum / self._benign_n
        # Sample covariance (with ridge regularization for numerical stability)
        sigma = self._benign_sq / self._benign_n - np.outer(mu, mu)
        sigma += 1e-3 * np.eye(self._d)
        diff = x - mu
        try:
            return float(np.sqrt(max(0.0, diff @ np.linalg.solve(sigma, diff))))
        except np.linalg.LinAlgError:
            return 0.0

    def select(self, query: Query, env) -> str:
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id
        n_candidates = len(declarers)

        qid = query.query_id
        if qid not in self._fe._query_texts:
            qid = self._cap_to_query_id.get(cap, qid)
        else:
            if cap not in self._cap_to_query_id:
                self._cap_to_query_id[cap] = qid

        features: dict[str, np.ndarray] = {}
        for tool in declarers:
            features[tool.tool_id] = self._fe.extract(qid, tool, n_candidates)
        self._cached_features = features

        B_inv = self._get_B_inv()
        best_tool_id = declarers[0].tool_id
        best_score = -np.inf
        for tool in declarers:
            x = features[tool.tool_id]
            exploit = float(x @ self._theta_mean)
            bonus = self._alpha * float(np.sqrt(max(0.0, x @ B_inv @ x)))
            anomaly = self._mahalanobis(x)
            score = exploit + bonus - self._gamma * anomaly
            if score > best_score:
                best_score = score
                best_tool_id = tool.tool_id
        return best_tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        x = self._cached_features.get(tool_id)
        if x is None:
            return
        r = 1.0 if success else 0.0
        inv_s2 = 1.0 / self._sigma2

        # LinUCB update
        self._B += inv_s2 * np.outer(x, x)
        self._f += inv_s2 * r * x
        self._theta_mean = np.linalg.solve(self._B, self._f)
        self._B_inv = None

        # Update benign distribution ONLY on success
        # This is the key mechanism: benign distribution is the set of
        # feature vectors that led to correct selections.
        if success:
            self._benign_sum += x
            self._benign_sq += np.outer(x, x)
            self._benign_n += 1

        self.n_updates += 1

    def get_predicted_prob(self, tool_id: str, capability: str = "") -> float:
        x = self._cached_features.get(tool_id)
        if x is None:
            return 0.5
        return max(0.0, min(1.0, float(x @ self._theta_mean)))


class ADABWarmSelector(ADABSelector):
    """ADAB with cold-start warmup.

    Hypothesis: ADAB's benign TSA drops because the Mahalanobis detector is
    unreliable before the benign distribution is well-estimated. During the
    first `warmup_rounds` interactions, suppress the anomaly penalty entirely
    (falling back to LinUCB), then activate ADAB once a solid benign sample
    is available.
    """

    name = "ADAB-Warm"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma: float = 0.3,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        warmup_rounds: int = 100,
    ) -> None:
        super().__init__(
            feature_extractor, seed=seed, alpha=alpha, gamma=gamma,
            lambda_reg=lambda_reg, sigma2=sigma2,
            min_samples=20,  # still require 20 successes before any anomaly penalty
        )
        self._warmup_rounds = warmup_rounds

    def select(self, query, env):
        # During warmup, zero out gamma temporarily; after warmup restore it.
        if self.n_updates < self._warmup_rounds:
            saved = self._gamma
            self._gamma = 0.0
            try:
                return super().select(query, env)
            finally:
                self._gamma = saved
        return super().select(query, env)


class ADABQuantileSelector:
    """Per-feature quantile anomaly — simpler alternative to Mahalanobis.

    Maintains per-feature empirical quantile q95 over benign (successful)
    observations. For a candidate, count how many feature dims exceed
    their q95. Use this count as a penalty. Simpler and more interpretable
    than joint covariance.
    """

    name = "ADAB-Quantile"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma: float = 0.5,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        quantile: float = 0.95,
        min_samples: int = 20,
    ) -> None:
        self._fe = feature_extractor
        self.rng = np.random.default_rng(seed)
        self._alpha = alpha
        self._gamma = gamma
        self._lambda = lambda_reg
        self._sigma2 = sigma2
        self._q = quantile
        self._min_samples = min_samples
        self._d = feature_extractor.n_features

        self._B = self._lambda * np.eye(self._d)
        self._f = np.zeros(self._d)
        self._theta_mean = np.zeros(self._d)
        self._B_inv: np.ndarray | None = None

        # Streaming benign feature observations (simple list for now)
        self._benign_X: list[np.ndarray] = []

        self._cached_features: dict[str, np.ndarray] = {}
        self._cap_to_query_id: dict[str, int] = {}
        self.n_updates: int = 0

    def _get_B_inv(self):
        if self._B_inv is None:
            self._B_inv = np.linalg.inv(self._B)
        return self._B_inv

    def _anomaly(self, x):
        if len(self._benign_X) < self._min_samples:
            return 0.0
        X = np.array(self._benign_X)
        q = np.quantile(X, self._q, axis=0)
        count = int(np.sum(x > q))
        return count / self._d

    def select(self, query, env):
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id
        n_candidates = len(declarers)

        qid = query.query_id
        if qid not in self._fe._query_texts:
            qid = self._cap_to_query_id.get(cap, qid)
        else:
            if cap not in self._cap_to_query_id:
                self._cap_to_query_id[cap] = qid

        features = {}
        for tool in declarers:
            features[tool.tool_id] = self._fe.extract(qid, tool, n_candidates)
        self._cached_features = features

        B_inv = self._get_B_inv()
        best_id = declarers[0].tool_id
        best_s = -np.inf
        for tool in declarers:
            x = features[tool.tool_id]
            exploit = float(x @ self._theta_mean)
            bonus = self._alpha * float(np.sqrt(max(0.0, x @ B_inv @ x)))
            anomaly = self._anomaly(x)
            score = exploit + bonus - self._gamma * anomaly
            if score > best_s:
                best_s = score
                best_id = tool.tool_id
        return best_id

    def update(self, tool_id, capability, success):
        x = self._cached_features.get(tool_id)
        if x is None:
            return
        r = 1.0 if success else 0.0
        inv_s2 = 1.0 / self._sigma2
        self._B += inv_s2 * np.outer(x, x)
        self._f += inv_s2 * r * x
        self._theta_mean = np.linalg.solve(self._B, self._f)
        self._B_inv = None
        if success:
            self._benign_X.append(x.copy())
            if len(self._benign_X) > 500:
                # cap memory
                self._benign_X = self._benign_X[-500:]
        self.n_updates += 1

    def get_predicted_prob(self, tool_id, capability=""):
        x = self._cached_features.get(tool_id)
        if x is None:
            return 0.5
        return max(0.0, min(1.0, float(x @ self._theta_mean)))


class CommitteeLinUCBSelector:
    """Committee of K LinUCBs each trained on a random subset of features.

    Hypothesis: adaptive attackers must corrupt features that ALL committee
    members see. With K heads each using a random 4/8 feature subset, an
    attacker cannot simultaneously fool all members unless they corrupt
    every feature dimension.

    Selection: sum of the K UCB scores (majority-vote analog).
    """

    name = "Committee"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        n_committees: int = 5,
        subset_size: int = 4,
    ) -> None:
        self._fe = feature_extractor
        self.rng = np.random.default_rng(seed)
        self._alpha = alpha
        self._lambda = lambda_reg
        self._sigma2 = sigma2
        self._K = n_committees
        self._d = feature_extractor.n_features

        # Random feature masks for each committee member
        rng_mask = np.random.default_rng(seed + 42)
        self._masks = []
        for _ in range(n_committees):
            idx = rng_mask.choice(self._d, size=subset_size, replace=False)
            self._masks.append(_mask(self._d, idx))

        # Per-committee posterior
        self._Bs = [self._lambda * np.eye(self._d) for _ in range(n_committees)]
        self._fs = [np.zeros(self._d) for _ in range(n_committees)]
        self._thetas = [np.zeros(self._d) for _ in range(n_committees)]
        self._Binvs = [None] * n_committees

        self._cached_features = {}
        self._cap_to_query_id = {}
        self.n_updates = 0

    def _ucb(self, x, i):
        if self._Binvs[i] is None:
            self._Binvs[i] = np.linalg.inv(self._Bs[i])
        xm = x * self._masks[i]
        exploit = float(xm @ self._thetas[i])
        bonus = self._alpha * float(np.sqrt(max(0.0, xm @ self._Binvs[i] @ xm)))
        return exploit + bonus

    def select(self, query, env):
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id
        n_candidates = len(declarers)

        qid = query.query_id
        if qid not in self._fe._query_texts:
            qid = self._cap_to_query_id.get(cap, qid)
        else:
            if cap not in self._cap_to_query_id:
                self._cap_to_query_id[cap] = qid

        features = {}
        for tool in declarers:
            features[tool.tool_id] = self._fe.extract(qid, tool, n_candidates)
        self._cached_features = features

        best_id = declarers[0].tool_id
        best_s = -np.inf
        for tool in declarers:
            x = features[tool.tool_id]
            score = sum(self._ucb(x, i) for i in range(self._K))
            if score > best_s:
                best_s = score
                best_id = tool.tool_id
        return best_id

    def update(self, tool_id, capability, success):
        x = self._cached_features.get(tool_id)
        if x is None:
            return
        r = 1.0 if success else 0.0
        inv_s2 = 1.0 / self._sigma2
        for i in range(self._K):
            xm = x * self._masks[i]
            self._Bs[i] += inv_s2 * np.outer(xm, xm)
            self._fs[i] += inv_s2 * r * xm
            self._thetas[i] = np.linalg.solve(self._Bs[i], self._fs[i])
            self._Binvs[i] = None
        self.n_updates += 1

    def get_predicted_prob(self, tool_id, capability=""):
        x = self._cached_features.get(tool_id)
        if x is None:
            return 0.5
        preds = [float((x * self._masks[i]) @ self._thetas[i]) for i in range(self._K)]
        return max(0.0, min(1.0, float(np.mean(preds))))


class HybridADABSelector(ADABQuantileSelector):
    """Combine per-feature quantile anomaly AND Mahalanobis distance.

    Hypothesis: the two signals catch different kinds of attacks.
    Quantile catches extreme single-feature values, Mahalanobis catches
    unusual feature combinations. Combining both should strictly improve.

    penalty = gamma_q * quantile_anomaly + gamma_m * mahalanobis_anomaly
    """

    name = "Hybrid-ADAB"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma_quant: float = 2.0,
        gamma_maha: float = 0.3,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        quantile: float = 0.99,
        min_samples: int = 20,
    ) -> None:
        super().__init__(
            feature_extractor, seed=seed, alpha=alpha, gamma=gamma_quant,
            lambda_reg=lambda_reg, sigma2=sigma2,
            quantile=quantile, min_samples=min_samples,
        )
        self._gamma_q = gamma_quant
        self._gamma_m = gamma_maha

        # Covariance stats for Mahalanobis (parallel to Quantile)
        self._benign_sum = np.zeros(self._d)
        self._benign_sq = np.zeros((self._d, self._d))
        self._benign_n = 0

    def _mahalanobis(self, x):
        if self._benign_n < self._min_samples:
            return 0.0
        mu = self._benign_sum / self._benign_n
        sigma = self._benign_sq / self._benign_n - np.outer(mu, mu)
        sigma += 1e-3 * np.eye(self._d)
        diff = x - mu
        try:
            return float(np.sqrt(max(0.0, diff @ np.linalg.solve(sigma, diff))))
        except np.linalg.LinAlgError:
            return 0.0

    def _anomaly(self, x):
        q = super()._anomaly(x)
        m = self._mahalanobis(x)
        return self._gamma_q * q + self._gamma_m * m

    def select(self, query, env):
        # parent select uses self._gamma * _anomaly — we override _anomaly to
        # return the COMBINED penalty already scaled. So set effective gamma=1.
        saved = self._gamma
        self._gamma = 1.0
        try:
            return super().select(query, env)
        finally:
            self._gamma = saved

    def update(self, tool_id, capability, success):
        x = self._cached_features.get(tool_id)
        super().update(tool_id, capability, success)
        # Also update Mahalanobis stats on success
        if x is not None and success:
            self._benign_sum += x
            self._benign_sq += np.outer(x, x)
            self._benign_n += 1


class OutlierAbstainLinUCB(ADABQuantileSelector):
    """Hard-reject extreme-outlier candidates, select best non-outlier.

    Rule: if a candidate's quantile-anomaly count > threshold, MARK it
    as outlier. Prefer non-outlier candidates; only pick outliers if all
    candidates are outliers (fall back to best UCB score).

    This is more aggressive than soft-penalty: suspicious tools are flatly
    excluded unless forced.
    """

    name = "Outlier-Abstain"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        alpha: float = 1.0,
        threshold: int = 4,  # count of features above q99 to flag as outlier
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        quantile: float = 0.99,
        min_samples: int = 20,
    ) -> None:
        super().__init__(
            feature_extractor, seed=seed, alpha=alpha, gamma=0.0,  # no soft penalty
            lambda_reg=lambda_reg, sigma2=sigma2,
            quantile=quantile, min_samples=min_samples,
        )
        self._threshold = threshold  # threshold is in "count of outlier features"

    def select(self, query, env):
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id
        n_candidates = len(declarers)

        qid = query.query_id
        if qid not in self._fe._query_texts:
            qid = self._cap_to_query_id.get(cap, qid)
        else:
            if cap not in self._cap_to_query_id:
                self._cap_to_query_id[cap] = qid

        features = {}
        for tool in declarers:
            features[tool.tool_id] = self._fe.extract(qid, tool, n_candidates)
        self._cached_features = features

        # Compute quantile q per-dim (if enough benign samples)
        if len(self._benign_X) >= self._min_samples:
            X = np.array(self._benign_X)
            q = np.quantile(X, self._q, axis=0)
        else:
            q = None

        B_inv = self._get_B_inv()
        non_outlier = []
        outlier = []
        for tool in declarers:
            x = features[tool.tool_id]
            if q is None:
                non_outlier.append(tool)
                continue
            count = int(np.sum(x > q))
            (outlier if count >= self._threshold else non_outlier).append(tool)

        candidates = non_outlier if non_outlier else outlier

        best_id = candidates[0].tool_id
        best_s = -np.inf
        for tool in candidates:
            x = features[tool.tool_id]
            exploit = float(x @ self._theta_mean)
            bonus = self._alpha * float(np.sqrt(max(0.0, x @ B_inv @ x)))
            score = exploit + bonus
            if score > best_s:
                best_s = score
                best_id = tool.tool_id
        return best_id


class ADABEmaSelector(ADABQuantileSelector):
    """ADAB-Quantile with EMA-tracked quantile (adaptive to distribution drift).

    Instead of a flat sample over all successful observations, maintain an
    exponential moving average. More responsive to distribution shift, and
    bounded memory.
    """

    name = "ADAB-EMA"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma: float = 3.0,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        quantile: float = 0.99,
        min_samples: int = 20,
        decay: float = 0.99,
    ) -> None:
        super().__init__(
            feature_extractor, seed=seed, alpha=alpha, gamma=gamma,
            lambda_reg=lambda_reg, sigma2=sigma2,
            quantile=quantile, min_samples=min_samples,
        )
        self._decay = decay
        self._ema_quantile = None  # will be initialized lazily from first K benign samples

    def _anomaly(self, x):
        if len(self._benign_X) < self._min_samples:
            return 0.0
        # Initialize EMA from the first samples if not yet set
        if self._ema_quantile is None:
            X = np.array(self._benign_X)
            self._ema_quantile = np.quantile(X, self._q, axis=0)
        # Compare x to current EMA quantile
        count = int(np.sum(x > self._ema_quantile))
        return count / self._d

    def update(self, tool_id, capability, success):
        x = self._cached_features.get(tool_id)
        super().update(tool_id, capability, success)
        # After an update, refresh EMA on the parent's benign buffer
        if success and x is not None and len(self._benign_X) >= self._min_samples:
            if self._ema_quantile is None:
                X = np.array(self._benign_X)
                self._ema_quantile = np.quantile(X, self._q, axis=0)
            else:
                # Update: new_quant = decay * old + (1-decay) * max(x, old)
                # But for a quantile, a simpler approach: EMA of x itself,
                # then use a multiple of std (conservative analog).
                # Simpler: recompute quantile from sliding window (last 200 successes)
                window = self._benign_X[-200:]
                X = np.array(window)
                target = np.quantile(X, self._q, axis=0)
                self._ema_quantile = self._decay * self._ema_quantile + (1 - self._decay) * target


class ADABPerFeatureGamma(ADABQuantileSelector):
    """Per-feature gamma weighting.

    Hypothesis: features with a tight benign distribution (low variance)
    are more informative anomaly signals than features with wide distribution.
    Set gamma_i = gamma_base * (1 / std_i_over_benign) for each feature.
    """

    name = "ADAB-PFG"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma_base: float = 3.0,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        quantile: float = 0.99,
        min_samples: int = 20,
    ) -> None:
        super().__init__(
            feature_extractor, seed=seed, alpha=alpha, gamma=gamma_base,
            lambda_reg=lambda_reg, sigma2=sigma2,
            quantile=quantile, min_samples=min_samples,
        )

    def _anomaly(self, x):
        if len(self._benign_X) < self._min_samples:
            return 0.0
        X = np.array(self._benign_X)
        q = np.quantile(X, self._q, axis=0)
        std = np.std(X, axis=0) + 1e-3
        # per-feature weight inversely proportional to std
        w = 1.0 / std
        w = w / (w.sum() + 1e-9)  # normalize
        exceed = (x > q).astype(np.float64)
        return float(np.sum(w * exceed))


class AdaptiveGammaADABSelector(ADABQuantileSelector):
    """ADAB with gamma growing with experience.

    Rationale: early rounds have noisy quantile estimates, so aggressive
    gamma risks penalizing benign tools. Start gentle, ramp up as confidence
    grows: gamma(t) = gamma_target * min(1, n_updates / warmup_K).
    """

    name = "Adaptive-γ"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma_target: float = 3.0,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        quantile: float = 0.99,
        min_samples: int = 20,
        ramp_rounds: int = 200,
    ) -> None:
        super().__init__(
            feature_extractor, seed=seed, alpha=alpha, gamma=gamma_target,
            lambda_reg=lambda_reg, sigma2=sigma2,
            quantile=quantile, min_samples=min_samples,
        )
        self._gamma_target = gamma_target
        self._ramp = ramp_rounds

    def select(self, query, env):
        # Dynamically compute effective gamma based on n_updates
        saved = self._gamma
        t = self.n_updates
        ramp_factor = min(1.0, t / self._ramp)
        self._gamma = self._gamma_target * ramp_factor
        try:
            return super().select(query, env)
        finally:
            self._gamma = saved


class SupervisedOODSelector:
    """Supervised OOD detection via online logistic classifier on feature profiles.

    Train a binary classifier online:
      - Label +1 (benign): features from successful selections
      - Label -1 (adversarial): synthetic malicious profiles sampled at
        high feature extremes (all features > 0.9)

    The classifier's predicted "benign probability" is used as an
    anomaly-free indicator: high probability = trusted, low = suspicious.

    This is an alternative formulation to unsupervised quantile /
    Mahalanobis detection. It can incorporate ANY labeled data, including
    past known-attack traces.
    """

    name = "SupOOD"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma: float = 2.0,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        min_samples: int = 30,
        retrain_every: int = 20,
    ) -> None:
        self._fe = feature_extractor
        self.rng = np.random.default_rng(seed)
        self._alpha = alpha
        self._gamma = gamma
        self._lambda = lambda_reg
        self._sigma2 = sigma2
        self._min_samples = min_samples
        self._retrain_every = retrain_every
        self._d = feature_extractor.n_features

        # LinUCB posterior
        self._B = self._lambda * np.eye(self._d)
        self._f = np.zeros(self._d)
        self._theta_mean = np.zeros(self._d)
        self._B_inv: np.ndarray | None = None

        # OOD classifier training data
        self._benign_X: list[np.ndarray] = []
        self._ood_clf = None
        self._last_train_size = 0

        self._cached_features: dict[str, np.ndarray] = {}
        self._cap_to_query_id: dict[str, int] = {}
        self.n_updates: int = 0

    def _get_B_inv(self):
        if self._B_inv is None:
            self._B_inv = np.linalg.inv(self._B)
        return self._B_inv

    def _sample_synthetic_ood(self, n: int) -> np.ndarray:
        """Sample synthetic adversarial profiles (all features extreme)."""
        rng = self.rng
        # Generate n samples where most features are extreme (>0.8)
        X = rng.uniform(0.8, 1.0, size=(n, self._d))
        # Add some noise dims for realism
        mask = rng.random((n, self._d)) > 0.5
        X = X * mask + rng.uniform(0.0, 0.8, size=(n, self._d)) * (1 - mask)
        return X

    def _retrain(self):
        if len(self._benign_X) < self._min_samples:
            return
        from sklearn.linear_model import LogisticRegression
        X_pos = np.array(self._benign_X)
        n_pos = len(X_pos)
        # Generate OOD synthetic samples (2x benign for class balance)
        X_neg = self._sample_synthetic_ood(2 * n_pos)
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(n_pos), np.zeros(2 * n_pos)])
        try:
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=self.rng.integers(1 << 30))
            clf.fit(X, y)
            self._ood_clf = clf
        except Exception:
            self._ood_clf = None

    def _benign_proba(self, x):
        """Return P(benign | x). Returns 1.0 if classifier not trained."""
        if self._ood_clf is None:
            return 1.0
        try:
            return float(self._ood_clf.predict_proba(x.reshape(1, -1))[0, 1])
        except Exception:
            return 1.0

    def select(self, query, env):
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if not declarers:
            idx = self.rng.integers(len(env.tools))
            return env.tools[idx].tool_id
        n_candidates = len(declarers)

        qid = query.query_id
        if qid not in self._fe._query_texts:
            qid = self._cap_to_query_id.get(cap, qid)
        else:
            if cap not in self._cap_to_query_id:
                self._cap_to_query_id[cap] = qid

        features = {}
        for tool in declarers:
            features[tool.tool_id] = self._fe.extract(qid, tool, n_candidates)
        self._cached_features = features

        B_inv = self._get_B_inv()
        best_id = declarers[0].tool_id
        best_s = -np.inf
        for tool in declarers:
            x = features[tool.tool_id]
            exploit = float(x @ self._theta_mean)
            bonus = self._alpha * float(np.sqrt(max(0.0, x @ B_inv @ x)))
            p_benign = self._benign_proba(x)
            # Penalty proportional to P(adversarial) = 1 - P(benign)
            penalty = self._gamma * (1.0 - p_benign)
            score = exploit + bonus - penalty
            if score > best_s:
                best_s = score
                best_id = tool.tool_id
        return best_id

    def update(self, tool_id, capability, success):
        x = self._cached_features.get(tool_id)
        if x is None:
            return
        r = 1.0 if success else 0.0
        inv_s2 = 1.0 / self._sigma2
        self._B += inv_s2 * np.outer(x, x)
        self._f += inv_s2 * r * x
        self._theta_mean = np.linalg.solve(self._B, self._f)
        self._B_inv = None
        if success:
            self._benign_X.append(x.copy())
            if len(self._benign_X) > 500:
                self._benign_X = self._benign_X[-500:]
        self.n_updates += 1
        if self.n_updates % self._retrain_every == 0:
            self._retrain()

    def get_predicted_prob(self, tool_id, capability=""):
        x = self._cached_features.get(tool_id)
        if x is None:
            return 0.5
        return max(0.0, min(1.0, float(x @ self._theta_mean)))


class ADABMaxExceedance(ADABEmaSelector):
    """Anomaly = MAX of (x_i - τ_i)+ over features.

    Alternative to counting how many features exceed the quantile:
    penalize the single most extreme feature instead.
    """

    name = "ADAB-MaxExc"

    def _anomaly(self, x):
        if len(self._benign_X) < self._min_samples:
            return 0.0
        if self._ema_quantile is None:
            X = np.array(self._benign_X)
            self._ema_quantile = np.quantile(X, self._q, axis=0)
        exceedance = np.maximum(0.0, x - self._ema_quantile)
        return float(np.max(exceedance))


class ADABSumExceedance(ADABEmaSelector):
    """Anomaly = SUM of (x_i - τ_i)+ — continuous version of the count.

    Rather than counting features above quantile, sum the magnitude of
    the exceedance. Smoother signal.
    """

    name = "ADAB-SumExc"

    def _anomaly(self, x):
        if len(self._benign_X) < self._min_samples:
            return 0.0
        if self._ema_quantile is None:
            X = np.array(self._benign_X)
            self._ema_quantile = np.quantile(X, self._q, axis=0)
        exceedance = np.maximum(0.0, x - self._ema_quantile)
        return float(np.sum(exceedance)) / self._d


class ADABAllSelections(ADABEmaSelector):
    """ADAB but build benign distribution from ALL selections (not just success).

    Ablation: does restricting to successful selections matter? Or is any
    selected tool's feature profile a good proxy for 'benign tool profile'?
    """

    name = "ADAB-All"

    def update(self, tool_id, capability, success):
        x = self._cached_features.get(tool_id)
        if x is None:
            return
        r = 1.0 if success else 0.0
        inv_s2 = 1.0 / self._sigma2
        self._B += inv_s2 * np.outer(x, x)
        self._f += inv_s2 * r * x
        self._theta_mean = np.linalg.solve(self._B, self._f)
        self._B_inv = None
        # Key difference: append regardless of success
        self._benign_X.append(x.copy())
        if len(self._benign_X) > 500:
            self._benign_X = self._benign_X[-500:]
        # Also update EMA (copied from parent)
        if len(self._benign_X) >= self._min_samples:
            if self._ema_quantile is None:
                X = np.array(self._benign_X)
                self._ema_quantile = np.quantile(X, self._q, axis=0)
            else:
                window = self._benign_X[-200:]
                X = np.array(window)
                target = np.quantile(X, self._q, axis=0)
                self._ema_quantile = self._decay * self._ema_quantile + (1 - self._decay) * target
        self.n_updates += 1


# =============================================================================
# Path C ideas: beyond per-feature marginal quantile
# =============================================================================

class ADABClusterSelector(ADABQuantileSelector):
    """ADAB with k-means clustering of benign successful profiles.

    Instead of per-feature marginal quantiles, maintain K cluster centroids
    over benign successful feature vectors. Anomaly = min distance to any
    centroid, normalized by median within-cluster distance. This catches
    attackers whose features stay within marginal q99 but whose *combination*
    doesn't match any benign mode.
    """

    name = "ADAB-Cluster"

    def __init__(
        self,
        feature_extractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma: float = 2.0,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        n_clusters: int = 4,
        min_samples: int = 40,
        refit_every: int = 50,
    ) -> None:
        super().__init__(
            feature_extractor, seed=seed, alpha=alpha, gamma=gamma,
            lambda_reg=lambda_reg, sigma2=sigma2,
            quantile=0.99, min_samples=min_samples,
        )
        self._K = n_clusters
        self._refit_every = refit_every
        self._centroids: np.ndarray | None = None
        self._scale: float = 1.0
        self._last_fit: int = 0

    def _maybe_refit(self) -> None:
        n = len(self._benign_X)
        if n < self._min_samples:
            return
        if self._centroids is not None and (n - self._last_fit) < self._refit_every:
            return
        from sklearn.cluster import KMeans
        X = np.array(self._benign_X)
        k = min(self._K, max(1, n // 10))
        km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(X)
        self._centroids = km.cluster_centers_
        # Scale = median min-distance from each benign point to its centroid
        dists = np.linalg.norm(X - self._centroids[km.labels_], axis=1)
        self._scale = float(np.median(dists)) + 1e-6
        self._last_fit = n

    def _anomaly(self, x):
        if len(self._benign_X) < self._min_samples:
            return 0.0
        self._maybe_refit()
        if self._centroids is None:
            return 0.0
        d = np.min(np.linalg.norm(self._centroids - x, axis=1))
        return float(d / self._scale)


class ADABSubspaceSelector(ADABQuantileSelector):
    """ADAB with PCA reconstruction-error anomaly.

    Maintains top-r PCA directions over benign successful profiles. Anomaly =
    reconstruction error when projecting x onto the benign subspace. Attackers
    who maximize score along theta_mean direction but whose direction is OFF
    the benign subspace incur a large penalty, even if their features stay
    within marginal q99.
    """

    name = "ADAB-Subspace"

    def __init__(
        self,
        feature_extractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma: float = 2.0,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        rank: int = 4,
        min_samples: int = 40,
        refit_every: int = 50,
    ) -> None:
        super().__init__(
            feature_extractor, seed=seed, alpha=alpha, gamma=gamma,
            lambda_reg=lambda_reg, sigma2=sigma2,
            quantile=0.99, min_samples=min_samples,
        )
        self._r = rank
        self._refit_every = refit_every
        self._U: np.ndarray | None = None  # orthonormal basis, shape (d, r)
        self._mean: np.ndarray | None = None
        self._scale: float = 1.0
        self._last_fit: int = 0

    def _maybe_refit(self) -> None:
        n = len(self._benign_X)
        if n < self._min_samples:
            return
        if self._U is not None and (n - self._last_fit) < self._refit_every:
            return
        X = np.array(self._benign_X)
        self._mean = X.mean(axis=0)
        Xc = X - self._mean
        # SVD of centered data; keep top-r right singular vectors
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        r = min(self._r, Vt.shape[0])
        self._U = Vt[:r].T  # shape (d, r)
        # Scale = median reconstruction error on benign
        recon = Xc @ (self._U @ self._U.T)
        err = np.linalg.norm(Xc - recon, axis=1)
        self._scale = float(np.median(err)) + 1e-6
        self._last_fit = n

    def _anomaly(self, x):
        if len(self._benign_X) < self._min_samples:
            return 0.0
        self._maybe_refit()
        if self._U is None:
            return 0.0
        xc = x - self._mean
        recon = self._U @ (self._U.T @ xc)
        err = float(np.linalg.norm(xc - recon))
        return err / self._scale


class ADABRandMaskSelector(ADABQuantileSelector):
    """ADAB with per-query random feature-subset anomaly (breaks info asymmetry).

    Each query, sample a random binary mask m of size K < d. Anomaly counts
    exceedances only over masked-in features. The attacker commits x* without
    knowing m, so their expected anomaly over all possible masks stays
    positive, even if x* matches a benign mode exactly in some features.

    This is a Stackelberg-style randomized defense: the defender's move (mask)
    is drawn after the attacker's commit (x*).
    """

    name = "ADAB-RandMask"

    def __init__(
        self,
        feature_extractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma: float = 0.5,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        K: int = 5,
        min_samples: int = 20,
        n_mask_samples: int = 3,
    ) -> None:
        super().__init__(
            feature_extractor, seed=seed, alpha=alpha, gamma=gamma,
            lambda_reg=lambda_reg, sigma2=sigma2,
            quantile=0.99, min_samples=min_samples,
        )
        self._K = K
        self._n_mask = n_mask_samples
        self._rng_mask = np.random.default_rng(seed * 100003 + 7919)
        self._cached_q: np.ndarray | None = None
        self._last_q_refit: int = 0

    def _get_quantiles(self) -> np.ndarray | None:
        n = len(self._benign_X)
        if n < self._min_samples:
            return None
        if self._cached_q is None or (n - self._last_q_refit) >= 50:
            X = np.array(self._benign_X)
            self._cached_q = np.quantile(X, self._q, axis=0)
            self._last_q_refit = n
        return self._cached_q

    def _anomaly(self, x):
        q = self._get_quantiles()
        if q is None:
            return 0.0
        d = len(x)
        K = min(self._K, d)
        # Average anomaly over n_mask_samples random feature subsets
        total = 0.0
        for _ in range(self._n_mask):
            S = self._rng_mask.choice(d, K, replace=False)
            exceed = int(np.sum(x[S] > q[S]))
            total += exceed / K
        return total / self._n_mask


class ADABNoisyQuantileSelector(ADABQuantileSelector):
    """ADAB where tau_q is per-query bootstrap-sampled.

    Each query, resample ceil(n * frac) benign samples with replacement from
    the full benign history and recompute tau_q on the bootstrap. The attacker
    observing committed state sees the empirical tau_q (computed from full
    benign_X), but the DEFENDER uses a fresh bootstrap tau_q' that is tighter
    or looser by chance. E[bootstrap tau_q] = tau_q asymptotically but with
    nontrivial variance, so an attacker who parks x* right at tau_q triggers
    the anomaly roughly half the time.
    """

    name = "ADAB-NoisyQuantile"

    def __init__(
        self,
        feature_extractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma: float = 1.5,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        quantile: float = 0.95,
        bootstrap_frac: float = 0.5,
        min_samples: int = 20,
    ) -> None:
        super().__init__(
            feature_extractor, seed=seed, alpha=alpha, gamma=gamma,
            lambda_reg=lambda_reg, sigma2=sigma2,
            quantile=quantile, min_samples=min_samples,
        )
        self._frac = bootstrap_frac
        self._rng_boot = np.random.default_rng(seed * 93739 + 11)

    def _anomaly(self, x):
        n = len(self._benign_X)
        if n < self._min_samples:
            return 0.0
        m = max(int(self._frac * n), self._min_samples)
        X = np.array(self._benign_X)
        idx = self._rng_boot.integers(0, n, size=m)
        X_boot = X[idx]
        q = np.quantile(X_boot, self._q, axis=0)
        count = int(np.sum(x > q))
        return count / self._d


class ADABInconsistencySelector(ADABQuantileSelector):
    """ADAB where anomaly = cross-channel inconsistency.

    The 8 features split into groups (desc, name, schema, context). A benign
    tool is jointly plausible across all groups: when desc_sim is high,
    name_sim and param_match are also correlated. A malicious tool that only
    spoofs description has HIGH desc_sim but LOW name_sim / param_match (it
    doesn't actually do anything useful for the query).

    Anomaly = gap between actual and predicted cross-channel score. We learn a
    linear map from one channel to another on benign successful data, then
    penalize high residual for a candidate.

    This is ORTHOGONAL to feature-space attacker optimization: even if x*
    satisfies marginal quantiles and within-cluster, the cross-channel
    consistency constraint is a separate axis.
    """

    name = "ADAB-Inconsistency"

    # Indices: 0=desc_sim, 1=name_sim, 2=param_match, 3=n_params_norm,
    # 4=desc_len_norm, 5=n_cand_norm, 6=required_ratio, 7=has_desc
    DESC = [0, 4, 7]     # description channel
    STRUCT = [1, 2, 3, 6]  # structural (name, schema) channel
    CONTEXT = [5]         # candidate context

    def __init__(
        self,
        feature_extractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma: float = 1.0,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        min_samples: int = 30,
        refit_every: int = 50,
    ) -> None:
        super().__init__(
            feature_extractor, seed=seed, alpha=alpha, gamma=gamma,
            lambda_reg=lambda_reg, sigma2=sigma2,
            quantile=0.99, min_samples=min_samples,
        )
        self._refit_every = refit_every
        self._W: np.ndarray | None = None  # STRUCT from DESC map
        self._scale: float = 1.0
        self._last_fit: int = 0

    def _maybe_refit(self) -> None:
        n = len(self._benign_X)
        if n < self._min_samples:
            return
        if self._W is not None and (n - self._last_fit) < self._refit_every:
            return
        X = np.array(self._benign_X)
        D = X[:, self.DESC]              # (n, d_desc)
        S = X[:, self.STRUCT]            # (n, d_struct)
        # Ridge regression: S ≈ D @ W
        A = D.T @ D + 0.1 * np.eye(D.shape[1])
        self._W = np.linalg.solve(A, D.T @ S)  # (d_desc, d_struct)
        residuals = np.linalg.norm(S - D @ self._W, axis=1)
        self._scale = float(np.median(residuals)) + 1e-6
        self._last_fit = n

    def _anomaly(self, x):
        if len(self._benign_X) < self._min_samples:
            return 0.0
        self._maybe_refit()
        if self._W is None:
            return 0.0
        d = x[self.DESC]
        s = x[self.STRUCT]
        predicted_s = d @ self._W
        residual = float(np.linalg.norm(s - predicted_s))
        return residual / self._scale


class ADABEnsembleClusterSelector(ADABQuantileSelector):
    """Ensemble of K cluster-anomaly selectors with different seeds.

    Anomaly = max over the ensemble's per-cluster anomalies, not average.
    Attacker must satisfy ALL cluster constraints simultaneously — the
    intersection of benign-centroid neighborhoods is a smaller region,
    harder for adaptive attacker to exploit.
    """

    name = "ADAB-EnsembleCluster"

    def __init__(
        self,
        feature_extractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma: float = 0.3,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
        n_ensembles: int = 3,
        n_clusters_per: int = 4,
        min_samples: int = 40,
        refit_every: int = 50,
    ) -> None:
        super().__init__(
            feature_extractor, seed=seed, alpha=alpha, gamma=gamma,
            lambda_reg=lambda_reg, sigma2=sigma2,
            quantile=0.99, min_samples=min_samples,
        )
        self._K_ens = n_ensembles
        self._K_clust = n_clusters_per
        self._refit_every = refit_every
        self._centroid_sets: list[np.ndarray] = []
        self._scales: list[float] = []
        self._last_fit: int = 0

    def _maybe_refit(self) -> None:
        n = len(self._benign_X)
        if n < self._min_samples:
            return
        if self._centroid_sets and (n - self._last_fit) < self._refit_every:
            return
        from sklearn.cluster import KMeans
        X = np.array(self._benign_X)
        k = min(self._K_clust, max(1, n // 10))
        self._centroid_sets = []
        self._scales = []
        for e in range(self._K_ens):
            km = KMeans(n_clusters=k, n_init=3, random_state=e * 7919 + 1).fit(X)
            self._centroid_sets.append(km.cluster_centers_)
            dists = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
            self._scales.append(float(np.median(dists)) + 1e-6)
        self._last_fit = n

    def _anomaly(self, x):
        if len(self._benign_X) < self._min_samples:
            return 0.0
        self._maybe_refit()
        if not self._centroid_sets:
            return 0.0
        anomalies = []
        for cents, scale in zip(self._centroid_sets, self._scales):
            d = np.min(np.linalg.norm(cents - x, axis=1))
            anomalies.append(d / scale)
        # MAX over ensemble — must satisfy all
        return float(max(anomalies))


class ADABCandidatePoolSelector(ADABQuantileSelector):
    """Anomaly = candidate-pool-conditional Z-score (not global benign dist).

    Per query, compute mean/std of each feature across the candidate pool.
    Anomaly of a candidate = summed positive Z-scores over features.
    A malicious tool that maximizes features becomes a Z-score outlier
    WITHIN THE CURRENT POOL, regardless of global benign distribution.

    Key insight: attackers who aim to WIN LinUCB's score necessarily push
    features above the pool's other candidates. This signal is query-local
    and cannot be spoofed by sitting at benign centroids or marginal quantiles.
    """

    name = "ADAB-CandPool"

    def __init__(
        self,
        feature_extractor,
        seed: int = 0,
        alpha: float = 1.0,
        gamma: float = 1.0,
        lambda_reg: float = 1.0,
        sigma2: float = 0.25,
    ) -> None:
        super().__init__(
            feature_extractor, seed=seed, alpha=alpha, gamma=gamma,
            lambda_reg=lambda_reg, sigma2=sigma2,
            quantile=0.99, min_samples=5,
        )
        self._pool_stats: dict | None = None  # populated per query

    def select(self, query, env):
        # Pre-compute pool stats before scoring
        cap = query.required_capability
        declarers = env.get_declarers(cap)
        if declarers and len(declarers) >= 2:
            qid = query.query_id
            if qid not in self._fe._query_texts:
                qid = self._cap_to_query_id.get(cap, qid)
            pool_X = np.stack([self._fe.extract(qid, t, len(declarers)) for t in declarers])
            mu = pool_X.mean(axis=0)
            sigma = pool_X.std(axis=0) + 1e-3
            self._pool_stats = {"mu": mu, "sigma": sigma}
        return super().select(query, env)

    def _anomaly(self, x):
        if self._pool_stats is None:
            return 0.0
        mu = self._pool_stats["mu"]
        sigma = self._pool_stats["sigma"]
        z = (x - mu) / sigma
        # Sum of positive Z (how much does this candidate exceed pool average)
        return float(np.sum(np.clip(z, 0, None)) / len(x))
