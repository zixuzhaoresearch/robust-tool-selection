"""Neural contextual bandit baselines for tool selection.

Implements NeuralUCB (Zhou et al. 2020) as a non-linear extension of LinUCB.
Used to test whether the 8-dimensional linear model in BilinTS/LinUCB is

Model
-----
Reward predictor: f(x; theta) = 2-layer MLP with ReLU activation.
UCB score: f(x; theta) + alpha * sqrt(g(x; theta)^T Z^{-1} g(x; theta) / m)
where g is the gradient of f wrt theta, Z is the gradient-covariance matrix
(diagonal approximation for tractability), m is the network width scaling.

Implementation follows the practical NeuralUCB variant: diagonal covariance,
periodic SGD retraining instead of per-step updates.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import ToolFeatureExtractor
from .models import Query


class _RewardMLP(nn.Module):
    """Small 2-layer MLP reward predictor."""

    def __init__(self, d_in: int, d_hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, 1)
        # Glorot init to match NeuralUCB paper
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


class NeuralUCBSelector:
    """NeuralUCB: UCB exploration over a small MLP reward predictor.

    Parameters
    ----------
    feature_extractor:
        Produces 8-dim feature vectors for (query, tool) pairs.
    seed:
        Random seed for reproducibility.
    alpha:
        UCB exploration coefficient (confidence radius scale).
    hidden_dim:
        MLP hidden layer width.
    lambda_reg:
        Regularization for gradient-covariance matrix.
    lr:
        SGD learning rate.
    train_steps:
        SGD steps per retraining.
    retrain_every:
        Retrain the MLP every N observations.
    """

    name = "NeuralUCB"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        alpha: float = 1.0,
        hidden_dim: int = 32,
        lambda_reg: float = 1.0,
        lr: float = 1e-3,
        train_steps: int = 50,
        retrain_every: int = 10,
    ) -> None:
        self._fe = feature_extractor
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self._d = feature_extractor.n_features
        self._alpha = alpha
        self._lambda = lambda_reg
        self._lr = lr
        self._train_steps = train_steps
        self._retrain_every = retrain_every

        self._net = _RewardMLP(self._d, hidden_dim)
        self._n_params = sum(p.numel() for p in self._net.parameters())

        # Gradient-covariance matrix Z (diagonal, for tractability)
        # Z = lambda * I + sum_i g(x_i; theta) g(x_i; theta)^T  (diagonal)
        self._Z_diag = self._lambda * np.ones(self._n_params, dtype=np.float64)

        # Observation buffer for periodic retraining
        self._X_buf: list[np.ndarray] = []
        self._r_buf: list[float] = []

        # Feature cache
        self._cached_features: dict[str, np.ndarray] = {}
        self._cap_to_query_id: dict[str, int] = {}
        self.n_updates: int = 0

    def _flat_grad(self, x_tensor: torch.Tensor) -> np.ndarray:
        """Gradient of f(x; theta) wrt theta, flattened."""
        self._net.zero_grad()
        y = self._net(x_tensor)
        y.backward()
        grads = []
        for p in self._net.parameters():
            grads.append(p.grad.detach().cpu().numpy().ravel())
        return np.concatenate(grads)

    def _ucb_score(self, x: np.ndarray) -> float:
        """Compute NeuralUCB score = f(x) + alpha * sigma(x)."""
        x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).requires_grad_(False)
        with torch.no_grad():
            exploit = float(self._net(x_tensor).item())

        # Gradient for exploration bonus
        x_tensor_grad = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        g = self._flat_grad(x_tensor_grad)
        # Diagonal UCB: sigma = sqrt(g^T Z_diag^{-1} g / m)
        sigma = float(np.sqrt(np.sum(g * g / self._Z_diag) / self._n_params))
        return exploit + self._alpha * sigma

    def _retrain(self) -> None:
        """SGD retraining on accumulated observations."""
        if not self._X_buf:
            return
        X = torch.from_numpy(np.array(self._X_buf, dtype=np.float32))
        r = torch.from_numpy(np.array(self._r_buf, dtype=np.float32))
        opt = torch.optim.SGD(self._net.parameters(), lr=self._lr)
        for _ in range(self._train_steps):
            opt.zero_grad()
            pred = self._net(X)
            loss = F.mse_loss(pred, r) + self._lambda * sum(
                (p * p).sum() for p in self._net.parameters()
            ) / self._n_params
            loss.backward()
            opt.step()

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

        best_tool_id = declarers[0].tool_id
        best_score = -np.inf
        for tool in declarers:
            score = self._ucb_score(features[tool.tool_id])
            if score > best_score:
                best_score = score
                best_tool_id = tool.tool_id
        return best_tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        x = self._cached_features.get(tool_id)
        if x is None:
            return
        r = 1.0 if success else 0.0

        # Update gradient-covariance diagonal
        x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        g = self._flat_grad(x_tensor)
        self._Z_diag += g * g

        # Buffer observation
        self._X_buf.append(x.astype(np.float32))
        self._r_buf.append(r)

        self.n_updates += 1

        # Periodic retraining
        if self.n_updates % self._retrain_every == 0:
            self._retrain()

    def get_predicted_prob(self, tool_id: str, capability: str = "") -> float:
        x = self._cached_features.get(tool_id)
        if x is None:
            return 0.5
        x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            p = float(self._net(x_tensor).item())
        return max(0.0, min(1.0, p))


class XGBoostBanditSelector:
    """Gradient-boosted trees with epsilon-greedy exploration.

    Non-linear supervised baseline that uses the same 8-dimensional feature
    extractor. Fits an XGBoost regressor to observed (feature, reward) pairs,
    then selects greedily with epsilon-probability exploration.

    Requires ``xgboost`` package.
    """

    name = "XGBoost-eGreedy"

    def __init__(
        self,
        feature_extractor: ToolFeatureExtractor,
        seed: int = 0,
        epsilon: float = 0.1,
        n_estimators: int = 50,
        max_depth: int = 4,
        retrain_every: int = 20,
    ) -> None:
        try:
            import xgboost as xgb  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "XGBoostBanditSelector requires xgboost. Install: uv add xgboost"
            ) from e
        import xgboost as xgb

        self._fe = feature_extractor
        self.rng = np.random.default_rng(seed)
        self._epsilon = epsilon
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._retrain_every = retrain_every
        self._seed = seed

        self._xgb = xgb
        self._model = None

        self._X_buf: list[np.ndarray] = []
        self._r_buf: list[float] = []
        self._cached_features: dict[str, np.ndarray] = {}
        self._cap_to_query_id: dict[str, int] = {}
        self.n_updates: int = 0

    def _predict(self, x: np.ndarray) -> float:
        if self._model is None:
            return 0.5  # uninformative prior
        return float(self._model.predict(x.reshape(1, -1))[0])

    def _retrain(self) -> None:
        if len(self._X_buf) < 5:
            return
        X = np.array(self._X_buf, dtype=np.float32)
        r = np.array(self._r_buf, dtype=np.float32)
        self._model = self._xgb.XGBRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=self._seed,
            verbosity=0,
            n_jobs=1,
        )
        self._model.fit(X, r)

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

        # Epsilon-greedy
        if self.rng.random() < self._epsilon:
            idx = self.rng.integers(n_candidates)
            return declarers[idx].tool_id

        best_tool_id = declarers[0].tool_id
        best_score = -np.inf
        for tool in declarers:
            score = self._predict(features[tool.tool_id])
            if score > best_score:
                best_score = score
                best_tool_id = tool.tool_id
        return best_tool_id

    def update(self, tool_id: str, capability: str, success: bool) -> None:
        x = self._cached_features.get(tool_id)
        if x is None:
            return
        r = 1.0 if success else 0.0
        self._X_buf.append(x.astype(np.float32))
        self._r_buf.append(r)
        self.n_updates += 1
        if self.n_updates % self._retrain_every == 0:
            self._retrain()

    def get_predicted_prob(self, tool_id: str, capability: str = "") -> float:
        x = self._cached_features.get(tool_id)
        if x is None:
            return 0.5
        return max(0.0, min(1.0, self._predict(x)))
