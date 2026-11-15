"""Score-optimized attacker (Path B, R7->R8 revision).

Answers the reviewer critique "the attacker should optimize the full ADAB score,
not use handcrafted strategies". We give the attacker white-box knowledge of
the selector's state and let them solve

    x* = argmax_{x in [0,1]^d}  x^T theta_mean + alpha * sqrt(x^T B^-1 x) - gamma * a(x; tau)

where a(x; tau) is the ADAB anomaly count (or zero for LinUCB / RobustLinUCB).
This is an upper bound on attacker power: unlimited feature-realization budget,
full white-box knowledge.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from scipy.optimize import minimize

from .adversarial import compute_asr
from .bfcl_adapter import BFCLEnvironment, BFCLScenario, BFCLTool
from .models import Observation, Query


def normalize_budget(upper, d=None):
    """Shared finite [0,1] oracle box policy; preserve exact zero coordinates."""
    out = np.asarray(upper, dtype=float)
    if out.ndim != 1 or (d is not None and out.shape != (d,)):
        raise ValueError("Budget shape mismatch")
    if not np.isfinite(out).all() or np.any(out < 0):
        raise ValueError("Budget must be finite and nonnegative")
    return np.minimum(out, 1.0).copy()


def count_scores(x, theta, B_inv, alpha, gamma=0.0, tau=None):
    """Defender objective: strict threshold, no exploration/threshold epsilon."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        value = float(x @ theta) + alpha * float(np.sqrt(max(0.0, x @ B_inv @ x)))
    else:
        value = x @ theta + alpha * np.sqrt(np.maximum(
            np.einsum("...i,ij,...j->...", x, B_inv, x), 0.0))
    if tau is not None and gamma > 0:
        value = value - gamma * np.mean(x > np.asarray(tau), axis=-1)
    return value


class WhiteboxAttacker:
    """Computes x* maximizing a selector's defended score.

    Parameters
    ----------
    d: feature dimension
    n_restarts: random L-BFGS-B restarts (more = better optimum, slower)
    beta_smooth: sigmoid sharpness for the anomaly indicator (large = closer to step)
    """

    def __init__(self, d: int = 8, n_restarts: int = 4, beta_smooth: float = 40.0,
                 seed: int = 0) -> None:
        self._d = d
        self._n_restarts = n_restarts
        self._beta = beta_smooth
        self._rng = np.random.default_rng(seed)

    def optimize(
        self,
        theta_mean: np.ndarray,
        B_inv: np.ndarray,
        alpha: float,
        gamma: float = 0.0,
        anomaly_kind: str | None = None,
        anomaly_params: dict | None = None,
        upper_bounds: np.ndarray | None = None,
        hard_indicator: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Return (x*, score*) maximizing defended score in [0, upper_bounds]^d.

        anomaly_kind in {None, "quantile", "cluster", "subspace"} selects the
        anomaly form the attacker is aware of. For "cluster", we do a restart
        from near each centroid (defender's modes) and pick the best.
        """
        d = self._d
        beta = self._beta
        if upper_bounds is None:
            upper_bounds = np.ones(d)
        upper_bounds = normalize_budget(upper_bounds, d)
        anomaly_params = anomaly_params or {}
        if hard_indicator:
            if gamma > 0 and anomaly_kind not in (None, "quantile"):
                raise ValueError("Exact mode only supports count-quantile or no anomaly")
            tau = (np.asarray(anomaly_params["tau_q"], dtype=float)
                   if anomaly_kind == "quantile" and gamma > 0 else upper_bounds.copy())
            if tau.shape != (d,) or not np.isfinite(tau).all() or alpha < 0:
                raise ValueError("Invalid thresholds or negative exploration scale")
            choices = [sorted({0.0, float(np.clip(tau[i], 0, upper_bounds[i])),
                               float(upper_bounds[i])}) for i in range(d)]
            candidates = np.asarray(list(product(*choices)), dtype=float)
            threshold = tau if anomaly_kind == "quantile" else None
            values = count_scores(candidates, theta_mean, B_inv, alpha, gamma, threshold)
            idx = int(np.argmax(values))
            witness = candidates[idx].copy()
            return witness, float(count_scores(witness, theta_mean, B_inv, alpha, gamma, threshold))

        def anomaly_value(x):
            if gamma <= 0 or anomaly_kind is None:
                return 0.0
            if anomaly_kind == "quantile":
                tau_q = anomaly_params["tau_q"]
                z = np.clip(beta * (x - tau_q), -50.0, 50.0)
                return float(np.mean(1.0 / (1.0 + np.exp(-z))))
            if anomaly_kind == "cluster":
                cents = anomaly_params["centroids"]
                scale = anomaly_params["scale"]
                dists = np.linalg.norm(cents - x, axis=1)
                return float(np.min(dists) / scale)
            if anomaly_kind == "ensemble_cluster":
                # Max over ensembles (matches defender's _anomaly)
                worst = 0.0
                for cents, scale in zip(anomaly_params["centroid_sets"], anomaly_params["scales"]):
                    d = float(np.min(np.linalg.norm(cents - x, axis=1)) / scale)
                    if d > worst:
                        worst = d
                return worst
            if anomaly_kind == "subspace":
                U = anomaly_params["U"]
                mean = anomaly_params["mean"]
                scale = anomaly_params["scale"]
                xc = x - mean
                recon = U @ (U.T @ xc)
                return float(np.linalg.norm(xc - recon) / scale)
            return 0.0

        def neg_score(x):
            exploit = float(x @ theta_mean)
            q = float(max(x @ B_inv @ x, 0.0))
            explore = alpha * np.sqrt(q)
            return -(exploit + explore - gamma * anomaly_value(x))

        starts: list[np.ndarray] = []
        # Random restarts within realistic budget
        for _ in range(self._n_restarts):
            starts.append(self._rng.uniform(0.0, 1.0, d) * upper_bounds)
        # Cluster-aware: add restarts AT each defender's centroid (best local min)
        if anomaly_kind == "cluster":
            for c in anomaly_params["centroids"]:
                starts.append(np.minimum(c, upper_bounds))
        if anomaly_kind == "ensemble_cluster":
            # Start from each ensemble centroid, plus mean-of-all (intersection center)
            all_cents = []
            for cents in anomaly_params["centroid_sets"]:
                for c in cents:
                    all_cents.append(c)
            if all_cents:
                mean_pt = np.mean(all_cents, axis=0)
                starts.append(np.clip(mean_pt, 0, upper_bounds))
            for cents in anomaly_params["centroid_sets"]:
                for c in cents:
                    starts.append(np.clip(c, 0, upper_bounds))
        # Subspace-aware: project a random theta-aligned direction into benign subspace
        if anomaly_kind == "subspace":
            U = anomaly_params["U"]; mean = anomaly_params["mean"]
            # Start from mean + projection of theta into benign subspace
            proj = U @ (U.T @ theta_mean)
            x0 = np.clip(mean + proj, 0, upper_bounds)
            starts.append(x0)

        best_x: np.ndarray | None = None
        best_val = np.inf
        for x0 in starts:
            res = minimize(
                neg_score, x0, method="L-BFGS-B",
                bounds=[(0.0, float(upper_bounds[i])) for i in range(d)],
                options={"maxiter": 150, "ftol": 1e-7},
            )
            if res.fun < best_val:
                best_val = float(res.fun)
                best_x = np.clip(res.x, 0.0, upper_bounds)

        assert best_x is not None
        # Return the true defender score, not the smoothed objective.
        if anomaly_kind in (None, "quantile"):
            return best_x, float(count_scores(best_x, theta_mean, B_inv, alpha, gamma,
                                             anomaly_params.get("tau_q")))
        return best_x, -best_val


def read_selector_state(selector):
    """Return (theta_mean, B_inv, alpha, gamma, anomaly_kind, anomaly_params).

    anomaly_kind in {None, "quantile", "cluster", "subspace"}. anomaly_params
    is a dict with the relevant internal state.
    """
    theta_mean = selector._theta_mean.copy()
    B_inv = selector._get_B_inv().copy() if hasattr(selector, "_get_B_inv") else np.linalg.inv(selector._B)
    alpha = float(getattr(selector, "_alpha", 1.0))
    gamma = float(getattr(selector, "_gamma", 0.0))

    kind: str | None = None
    params: dict = {}
    if gamma > 0:
        # Ensemble cluster (list of centroid sets) — highest priority
        cs = getattr(selector, "_centroid_sets", None)
        if cs:
            kind = "ensemble_cluster"
            params = {
                "centroid_sets": [c.copy() for c in cs],
                "scales": list(selector._scales),
            }
        # Single cluster
        elif getattr(selector, "_centroids", None) is not None:
            kind = "cluster"
            params = {"centroids": selector._centroids.copy(), "scale": float(selector._scale)}
        # Subspace (U basis)
        elif getattr(selector, "_U", None) is not None:
            kind = "subspace"
            params = {
                "U": selector._U.copy(),
                "mean": selector._mean.copy(),
                "scale": float(selector._scale),
            }
        else:
            ema = getattr(selector, "_ema_quantile", None)
            benign_X = getattr(selector, "_benign_X", None)
            min_samples = getattr(selector, "_min_samples", 20)
            q = getattr(selector, "_q", 0.99)
            if ema is not None:
                kind = "quantile"; params = {"tau_q": ema.copy()}
            elif benign_X and len(benign_X) >= min_samples:
                kind = "quantile"; params = {"tau_q": np.quantile(np.array(benign_X), q, axis=0)}

    return theta_mean, B_inv, alpha, gamma, kind, params


class OptimizedAttackEnvironment:
    """BFCL-compatible env where attacker injects one optimizer-crafted tool per scenario.

    Unlike AdversarialBFCLEnvironment (text-crafted malicious tools), here the
    malicious tool is a feature-space stub whose 8-dim vector is computed at
    select-time by solving the optimization against the selector's *current*
    state.
    """

    MAL_PREFIX = "__OPT_ATTACK__"

    def __init__(self, scenarios: list[BFCLScenario], seed: int = 42) -> None:
        self._scenarios_by_id: dict[str, BFCLScenario] = {sc.scenario_id: sc for sc in scenarios}
        self._mal_tool_ids: dict[str, str] = {
            sc.scenario_id: f"{self.MAL_PREFIX}{sc.scenario_id}" for sc in scenarios
        }

        seen: dict[str, BFCLTool] = {}
        for sc in scenarios:
            for tool in sc.candidate_tools:
                seen.setdefault(tool.tool_id, tool)
        for sc_id, mal_id in self._mal_tool_ids.items():
            seen[mal_id] = BFCLTool(
                tool_id=mal_id,
                name=f"opt_attack_{sc_id}",
                declared_capabilities=[],
                true_success_rates={},
                description_affinity={},
                description="",
                parameter_spec={},
            )
        self.tools: list[BFCLTool] = list(seen.values())
        self.tool_map: dict[str, BFCLTool] = seen

        rng = np.random.default_rng(seed)
        self.queries: list[Query] = [
            Query(
                query_id=i,
                required_capability=sc.scenario_id,
                ground_truth_tool_id=sc.ground_truth_tool_id,
            )
            for i, sc in enumerate(scenarios)
        ]
        rng.shuffle(self.queries)

        self.outcome_matrix: dict[tuple[int, str], bool] = {}
        for q in self.queries:
            sc = self._scenarios_by_id[q.required_capability]
            # Benign tools: success iff == ground truth
            for tool in sc.candidate_tools:
                self.outcome_matrix[(q.query_id, tool.tool_id)] = (
                    tool.tool_id == sc.ground_truth_tool_id
                )
            # Malicious tool: always failure (attacker's goal)
            self.outcome_matrix[(q.query_id, self._mal_tool_ids[sc.scenario_id])] = False

    def get_declarers(self, capability: str) -> list[BFCLTool]:
        sc = self._scenarios_by_id[capability]
        mal_tool = self.tool_map[self._mal_tool_ids[capability]]
        return list(sc.candidate_tools) + [mal_tool]

    def execute(self, tool_id: str, query: Query) -> Observation:
        key = (query.query_id, tool_id)
        success = self.outcome_matrix.get(key, False)
        return Observation(
            query=query,
            tool_id=tool_id,
            capability_used=query.required_capability,
            success=success,
        )


class OptAttackFeatureExtractor:
    """Wraps a base FE; returns attacker-optimized x* for malicious tools.

    Non-malicious tools are delegated to the base FE. For malicious tools, we
    read the selector state and solve for x*. The selector is passed in via
    set_selector(); state is read fresh on every extract() call.

    Parameters
    ----------
    constraint: "full" (x* in [0,1]^d, unconstrained oracle) or "realistic"
        (x* <= benign_q99 elementwise, a coordinatewise oracle relaxation).
    benign_q99: per-feature 99th percentile of benign candidate features,
        used when constraint="realistic". Required if constraint="realistic".
    """

    def __init__(
        self,
        base_fe,
        mal_tool_ids: set[str],
        constraint: str = "full",
        benign_q99: np.ndarray | None = None,
        commit_after: int | None = None,
        hard_indicator: bool = False,
    ) -> None:
        """
        commit_after: if set, attacker reads selector state once after this many
            queries have been processed, commits a single x*, and reuses it for
            the rest of the run. Realistic attacker model (no live state access).
            If None, attacker recomputes x* on every query (per-query oracle).
        hard_indicator: if True, the attacker uses the exact hard-indicator
            anomaly (Prop. 2 in the paper) via 3^d corner enumeration instead
            of the sigmoid-relaxed L-BFGS-B path. Only affects count-quantile
            ADAB; other anomaly kinds ignore the flag.
        """
        self._base = base_fe
        self._mal_ids = mal_tool_ids
        self._attacker = WhiteboxAttacker(d=base_fe.n_features)
        self._selector = None
        self._constraint = constraint
        if constraint == "realistic" and benign_q99 is None:
            raise ValueError("benign_q99 required for constraint='realistic'")
        self._upper = normalize_budget(benign_q99, base_fe.n_features) if benign_q99 is not None else None
        self._commit_after = commit_after
        self._committed_x: np.ndarray | None = None
        self._query_count: int = 0
        self._hard_indicator = hard_indicator
        self._round_cache = {}
        self._cache_generation = None

    def set_selector(self, selector) -> None:
        self._selector = selector
        self._round_cache.clear()
        self._cache_generation = None

    @property
    def n_features(self) -> int:
        return self._base.n_features

    @property
    def _query_texts(self):
        return self._base._query_texts

    def extract(self, query_id, tool, n_candidates: int) -> np.ndarray:
        if tool.tool_id not in self._mal_ids or self._selector is None:
            return self._base.extract(query_id, tool, n_candidates)
        # Commit-once attacker: compute x* once after warm-up, reuse thereafter.
        if self._commit_after is not None:
            # Use selector's update counter as "queries processed" signal
            n_seen = int(getattr(self._selector, "n_updates", self._query_count))
            if self._committed_x is not None:
                return self._committed_x
            if n_seen < self._commit_after:
                # Before commit time: attacker has no x*, return a zero-vector stub
                # (benign tools will outscore it, malicious rarely selected pre-commit).
                return np.zeros(self._base.n_features)
            # Commit now
            theta_mean, B_inv, alpha, gamma, kind, params = read_selector_state(self._selector)
            upper = self._upper if self._constraint == "realistic" else None
            x_star, _ = self._attacker.optimize(
                theta_mean, B_inv, alpha, gamma,
                anomaly_kind=kind, anomaly_params=params,
                upper_bounds=upper,
                hard_indicator=self._hard_indicator,
            )
            self._committed_x = x_star
            return x_star
        generation = int(getattr(self._selector, "n_updates", 0))
        if self._cache_generation != generation:
            self._round_cache.clear()
            self._cache_generation = generation
        key = (query_id, tool.tool_id, n_candidates)
        if key in self._round_cache:
            return self._round_cache[key].copy()
        # Cache the vector across instrumentation and selection at this state.
        theta_mean, B_inv, alpha, gamma, kind, params = read_selector_state(self._selector)
        upper = self._upper if self._constraint == "realistic" else None
        x_star, _ = self._attacker.optimize(
            theta_mean, B_inv, alpha, gamma,
            anomaly_kind=kind, anomaly_params=params,
            upper_bounds=upper,
            hard_indicator=self._hard_indicator,
        )
        self._round_cache[key] = x_star.copy()
        return x_star.copy()


def compute_benign_q99(fe, scenarios, n_samples: int = 500,
                       query_indices=None, injected_candidates: int = 0) -> np.ndarray:
    """Correct query-index pairing, optional disjoint calibration split.

    The marginal box is an oracle relaxation, not joint metadata realizability
    or containment of every benign vector.
    """
    pool = np.arange(len(scenarios)) if query_indices is None else np.asarray(query_indices, dtype=int)
    if not len(pool) or np.any(pool < 0) or np.any(pool >= len(scenarios)):
        raise ValueError("Invalid or empty calibration split")
    sampled = np.random.default_rng(0).choice(pool, min(n_samples, len(pool)), replace=False)
    feats = []
    for i in sampled:
        qid = int(i)
        if qid not in fe._query_texts:
            raise KeyError(f"Missing query index {qid}; cannot substitute another query")
        sc = scenarios[qid]
        for tool in sc.candidate_tools:
            feats.append(fe.extract(qid, tool, len(sc.candidate_tools) + injected_candidates))
    if not feats:
        raise ValueError("No calibration candidates")
    return normalize_budget(np.quantile(np.stack(feats), 0.99, axis=0), fe.n_features)


def run_optimized_attack(
    selector,
    env: OptimizedAttackEnvironment,
    fe_wrapper: OptAttackFeatureExtractor,
) -> tuple[float, float]:
    """Roll out `selector` on `env`; return (TSA, ASR)."""
    fe_wrapper.set_selector(selector)
    mal_ids = {sc: mid for sc, mid in env._mal_tool_ids.items()}
    selected_ids: list[str] = []
    scenario_ids: list[str] = []
    correct = 0
    for q in env.queries:
        tid = selector.select(q, env)
        obs = env.execute(tid, q)
        selected_ids.append(tid)
        scenario_ids.append(q.required_capability)
        if obs.success:
            correct += 1
        if hasattr(selector, "update"):
            selector.update(tid, q.required_capability, obs.success)
    tsa = correct / len(env.queries)
    asr = compute_asr(selected_ids, scenario_ids, mal_ids)
    return tsa, asr
