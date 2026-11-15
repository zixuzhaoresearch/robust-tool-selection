"""Generalized experiment runner: orchestrate arbitrary selectors on any environment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import (
    cumulative_regret,
    expected_calibration_error,
    learning_curve,
    tool_selection_accuracy,
    windowed_tsa,
)
from .selector import Selector, TATSSelector
from .updater import BayesianUpdater


@dataclass
class ExperimentResult:
    """Results from running one selector through the environment."""

    name: str
    selected_ids: list[str] = field(default_factory=list)
    ground_truth_ids: list[str] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    predicted_probs: list[float] = field(default_factory=list)
    cumulative_tsa: list[float] | None = None  # TSA after each interaction

    @property
    def tsa(self) -> float:
        return tool_selection_accuracy(self.selected_ids, self.ground_truth_ids)

    @property
    def success_rate(self) -> float:
        if not self.successes:
            return 0.0
        return sum(self.successes) / len(self.successes)

    @property
    def ece(self) -> float | None:
        if not self.predicted_probs:
            return None
        return expected_calibration_error(self.predicted_probs, self.successes)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict (metrics only, not raw lists)."""
        regret = cumulative_regret(self.selected_ids, self.ground_truth_ids)
        d = {
            "name": self.name,
            "tsa": self.tsa,
            "success_rate": self.success_rate,
            "ece": self.ece,
            "final_regret": regret[-1] if regret else 0.0,
            "n_queries": len(self.selected_ids),
        }
        if self.cumulative_tsa is not None:
            d["cumulative_tsa"] = self.cumulative_tsa
        return d


def run_selector(
    name: str,
    selector: Selector,
    env: Any,
    updater: BayesianUpdater | None = None,
) -> ExperimentResult:
    """Run a selector through all queries, optionally updating beliefs.

    Works with SyntheticEnvironment, DriftEnvironment, ScalableEnvironment —
    anything with .queries, .execute(), .get_declarers().
    """
    result = ExperimentResult(name=name)
    has_update = hasattr(selector, "update") and callable(selector.update)

    for query in env.queries:
        tool_id = selector.select(query, env)

        # Record predicted probability (TATS variants with get_predicted_prob)
        if hasattr(selector, "get_predicted_prob"):
            pred_prob = selector.get_predicted_prob(
                tool_id, query.required_capability
            )
            result.predicted_probs.append(pred_prob)

        obs = env.execute(tool_id, query)

        result.selected_ids.append(tool_id)
        result.ground_truth_ids.append(query.ground_truth_tool_id)
        result.successes.append(obs.success)

        # Update beliefs: use external updater for legacy TATSSelector,
        # or call selector.update() for new selectors
        if updater is not None and isinstance(selector, TATSSelector):
            model = selector._get_model(tool_id)
            updater.update(model, obs)
        elif has_update:
            selector.update(tool_id, obs.capability_used, obs.success)

    # Compute per-step cumulative TSA (fraction of correct selections so far)
    cumulative_tsa = []
    correct_so_far = 0
    for i, (sel, gt) in enumerate(
        zip(result.selected_ids, result.ground_truth_ids)
    ):
        correct_so_far += int(sel == gt)
        cumulative_tsa.append(correct_so_far / (i + 1))
    result.cumulative_tsa = cumulative_tsa

    return result


def run_experiment(
    selectors: list[tuple[str, Selector]],
    env: Any,
    seed: int = 42,
) -> list[ExperimentResult]:
    """Run multiple selectors on a shared environment.

    Each selector is a (name, selector_instance) tuple. TATS-family selectors
    automatically get a BayesianUpdater attached.
    """
    results = []
    for name, selector in selectors:
        updater = BayesianUpdater() if isinstance(selector, TATSSelector) else None
        result = run_selector(name, selector, env, updater)
        results.append(result)
    return results


def run_multi_seed(
    selector_factory: dict[str, Any],
    env_factory: Any,
    seeds: list[int],
) -> dict[str, list[ExperimentResult]]:
    """Run experiments across multiple seeds for statistical aggregation.

    Parameters
    ----------
    selector_factory:
        ``{name: callable(seed) -> Selector}`` — each callable creates a fresh
        selector for a given seed.
    env_factory:
        ``callable(seed) -> env`` — creates a fresh environment for a given seed.
    seeds:
        List of random seeds to iterate over.

    Returns
    -------
    ``{selector_name: [result_seed_0, result_seed_1, ...]}``
    """
    all_results: dict[str, list[ExperimentResult]] = {
        name: [] for name in selector_factory
    }

    for seed in seeds:
        env = env_factory(seed)
        selectors = [
            (name, factory(seed)) for name, factory in selector_factory.items()
        ]
        results = run_experiment(selectors, env, seed)
        for r in results:
            all_results[r.name].append(r)

    return all_results


def aggregate_results(
    multi_seed_results: dict[str, list[ExperimentResult]],
) -> dict[str, dict[str, Any]]:
    """Compute mean ± std of TSA across seeds.

    Returns ``{name: {"tsa_mean": ..., "tsa_std": ..., "ece_mean": ...,
    "ece_std": ..., "cumulative_tsa_curves": [[...], ...]}}``.

    ``cumulative_tsa_curves`` contains one list per seed (only included when
    at least one result carries cumulative_tsa data).
    """
    agg: dict[str, dict[str, Any]] = {}
    for name, results in multi_seed_results.items():
        tsas = [r.tsa for r in results]
        eces = [r.ece for r in results if r.ece is not None]
        entry: dict[str, Any] = {
            "tsa_mean": float(np.mean(tsas)),
            "tsa_std": float(np.std(tsas)),
            "ece_mean": float(np.mean(eces)) if eces else 0.0,
            "ece_std": float(np.std(eces)) if eces else 0.0,
        }
        # Collect per-seed cumulative TSA curves for learning-curve plots
        curves = [r.cumulative_tsa for r in results if r.cumulative_tsa is not None]
        if curves:
            entry["cumulative_tsa_curves"] = curves
        agg[name] = entry
    return agg


def save_results_json(
    results: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save results dict to JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(results, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Legacy compat: ExperimentRunner for run_mvp.py
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Run the original 3 selectors (Random/Vanilla/TATS) for MVP verification."""

    def __init__(self, n_queries: int = 500, seed: int = 42) -> None:
        self.n_queries = n_queries
        self.seed = seed

    def run(self) -> list[ExperimentResult]:
        from .environment import SyntheticEnvironment
        from .selector import RandomSelector, VanillaSelector

        env = SyntheticEnvironment(n_queries=self.n_queries, seed=self.seed)
        updater = BayesianUpdater()

        configs: list[tuple[str, Selector, BayesianUpdater | None]] = [
            ("Random", RandomSelector(seed=self.seed + 1), None),
            ("Vanilla", VanillaSelector(seed=self.seed + 2), None),
            ("TATS-Thompson", TATSSelector(seed=self.seed + 3), updater),
        ]

        results = []
        for name, selector, upd in configs:
            result = run_selector(name, selector, env, upd)
            results.append(result)

        return results

    @staticmethod
    def print_results(results: list[ExperimentResult]) -> None:
        print("\n" + "=" * 62)
        print("TATS MVP Experiment Results")
        print("=" * 62)

        print(f"\n{'Selector':<18} {'TSA':>7} {'Success Rate':>13} {'ECE':>7}")
        print("-" * 48)
        for r in results:
            ece_str = f"{r.ece:.4f}" if r.ece is not None else "N/A"
            print(f"{r.name:<18} {r.tsa:>7.4f} {r.success_rate:>13.4f} {ece_str:>7}")

        tats_result = next(r for r in results if r.name == "TATS-Thompson")
        windows = windowed_tsa(
            tats_result.selected_ids, tats_result.ground_truth_ids, window=50
        )
        print(f"\nTATS Learning Curve (TSA per 50-query window):")
        for i, w in enumerate(windows):
            bar = "\u2588" * int(w * 40)
            print(f"  Window {i+1:>2}: {w:.3f} {bar}")

        print(f"\nCumulative Regret (final):")
        for r in results:
            regret = cumulative_regret(r.selected_ids, r.ground_truth_ids)
            print(f"  {r.name:<18}: {regret[-1]:.0f}")

        _, tats_lc = learning_curve(
            tats_result.selected_ids, tats_result.ground_truth_ids, window=20
        )
        if len(tats_lc) >= 20:
            first_20 = sum(tats_lc[:20]) / 20
            last_20 = sum(tats_lc[-20:]) / 20
            print(f"\nTATS Rolling Accuracy (window=20):")
            print(f"  First 20 windows avg: {first_20:.3f}")
            print(f"  Last 20 windows avg:  {last_20:.3f}")
            print(f"  Improvement:          {last_20 - first_20:+.3f}")

        print()
