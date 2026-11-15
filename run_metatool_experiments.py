"""MetaTool (ICLR 2024) experiment suite for cross-benchmark validation.

Demonstrates that the contextual bandit framework generalizes beyond BFCL
to a different tool selection benchmark with different characteristics:
  - OpenAI plugin domain (vs BFCL's API function specs)
  - No parameter specifications (description-only tools)
  - Configurable candidate set sizes (5-50 vs BFCL's ~4)
  - 199 tools, ~20k queries

Experiments:
  1. Main Results (K=10) — all selectors incl. Sup-Offline & BilinTS-Warm, 10 seeds
  2. Candidate Set Size Ablation — K=5,10,15,20,50, key selectors, 10 seeds
  3. Dense vs TF-IDF Feature Comparison — K=10, 10 seeds

Sup-Offline and warm-start use a 25/75 train/eval split of MetaTool queries
(MetaTool is a single dataset, so we split to avoid data leakage).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

from tats.metatool_adapter import MetaToolBenchmark, MetaToolEnvironment
from tats.bfcl_selectors import (
    DescriptionMatchSelector,
    EpsilonGreedyBFCLSelector,
    ParameterMatchSelector,
    RandomBFCLSelector,
    UCBBFCLSelector,
)
from tats.bilints_selector import (
    BilinTSSelector,
    ColdStartLinTSSelector,
    LinUCBSelector,
    SupervisedOfflineSelector,
    SupervisedOnlineSelector,
)
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.experiment import run_experiment, run_selector, save_results_json
from tats.warmstart import compute_warm_start, train_offline_logistic

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("paper/figures")
SEEDS = list(range(10))
MAX_QUERIES = 2000  # Use 2000 queries (from 20k total) for reasonable runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_significance(results: dict[str, list[float]], reference: str = "DescMatch"):
    """Print paired t-test p-values vs reference method."""
    if reference not in results:
        return
    ref = results[reference]
    print(f"\n  Statistical significance vs {reference}:")
    for name, scores in sorted(results.items()):
        if name == reference or len(scores) != len(ref):
            continue
        t_stat, p_val = stats.ttest_rel(scores, ref)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"    {name} vs {reference}: t={t_stat:.2f}, p={p_val:.4f} {sig}")


def _print_table(agg: dict[str, dict[str, float]], title: str):
    """Print formatted results table."""
    print(f"\n{'=' * 50}")
    print(title)
    print(f"{'=' * 50}")
    # Sort by TSA descending
    sorted_methods = sorted(agg.items(), key=lambda x: x[1]["tsa_mean"], reverse=True)
    for name, st in sorted_methods:
        print(f"  {name:<20} TSA={st['tsa_mean']:.3f} ± {st['tsa_std']:.3f}")


# ---------------------------------------------------------------------------
# Experiment 1: Main Results (K=10 candidates)
# ---------------------------------------------------------------------------


def run_main_experiment(max_queries: int = MAX_QUERIES, train_frac: float = 0.25) -> dict:
    """All selectors on MetaTool with K=10 candidates.

    Uses a train/eval split (default 25/75) for Sup-Offline and warm-start.
    """
    print("=" * 60)
    print("Experiment 1: Main Results on MetaTool (K=10)")
    print("=" * 60)

    bm = MetaToolBenchmark(n_candidates=10, seed=0, max_queries=max_queries)
    all_scenarios = bm.scenarios

    # Train/eval split (deterministic, seed=0)
    rng = np.random.default_rng(0)
    indices = rng.permutation(len(all_scenarios))
    n_train = int(len(all_scenarios) * train_frac)
    train_scenarios = [all_scenarios[i] for i in indices[:n_train]]
    eval_scenarios = [all_scenarios[i] for i in indices[n_train:]]
    print(f"  Split: {len(train_scenarios)} train / {len(eval_scenarios)} eval")

    # Train offline model and warm-start on train split
    qt_train = build_query_texts_from_scenarios(train_scenarios)
    fe_train = ToolFeatureExtractor(qt_train)
    logistic_model = train_offline_logistic(train_scenarios, fe_train)
    theta0, B0 = compute_warm_start(train_scenarios, fe_train)

    # Eval feature extractor
    qt_eval = build_query_texts_from_scenarios(eval_scenarios)
    fe_eval = ToolFeatureExtractor(qt_eval)

    tsa_per_selector: dict[str, list[float]] = {}

    for seed in SEEDS:
        env = MetaToolEnvironment(eval_scenarios, seed=seed)

        selectors: list[tuple[str, object]] = [
            ("Random", RandomBFCLSelector(seed=seed)),
            ("DescMatch", DescriptionMatchSelector(qt_eval, seed=seed)),
            ("ParamMatch", ParameterMatchSelector(qt_eval, seed=seed)),
            ("EpsGreedy", EpsilonGreedyBFCLSelector(epsilon=0.1, seed=seed)),
            ("UCB", UCBBFCLSelector(seed=seed)),
            ("LinUCB", LinUCBSelector(fe_eval, seed=seed)),
            ("BilinTS-Cold", ColdStartLinTSSelector(feature_extractor=fe_eval, seed=seed)),
            ("BilinTS-Warm", BilinTSSelector(
                feature_extractor=fe_eval, seed=seed,
                theta_init=theta0, B_init=B0,
            )),
            ("Sup-Offline", SupervisedOfflineSelector(
                feature_extractor=fe_eval, model=logistic_model, seed=seed,
            )),
            ("Sup-Online", SupervisedOnlineSelector(feature_extractor=fe_eval, seed=seed)),
        ]

        results = run_experiment(selectors, env, seed)

        for r in results:
            tsa_per_selector.setdefault(r.name, []).append(r.tsa)
            print(f"  seed={seed} {r.name:<15} TSA={r.tsa:.4f}")

    # Aggregate
    agg: dict[str, dict[str, float]] = {}
    for name, tsas in tsa_per_selector.items():
        agg[name] = {
            "tsa_mean": float(np.mean(tsas)),
            "tsa_std": float(np.std(tsas)),
        }

    _print_table(agg, "Experiment 1: Main Results (MetaTool, K=10)")
    _print_significance(tsa_per_selector)
    print("\n  --- vs Sup-Offline ---")
    _print_significance(tsa_per_selector, "Sup-Offline")

    return agg


# ---------------------------------------------------------------------------
# Experiment 2: Candidate Set Size Ablation
# ---------------------------------------------------------------------------


def run_candidate_size_experiment(max_queries: int = MAX_QUERIES) -> dict:
    """Vary K in {5, 10, 15, 20, 50} to show how methods scale with more candidates.

    """
    print("\n" + "=" * 60)
    print("Experiment 2: Candidate Set Size Ablation")
    print("=" * 60)

    k_values = [5, 10, 15, 20, 50]
    all_results: dict[int, dict[str, dict[str, float]]] = {}

    for k in k_values:
        print(f"\n--- K={k} ---")
        bm = MetaToolBenchmark(n_candidates=k, seed=0, max_queries=max_queries)
        scenarios = bm.scenarios
        qt = build_query_texts_from_scenarios(scenarios)
        fe = ToolFeatureExtractor(qt)

        tsa_per_selector: dict[str, list[float]] = {}

        for seed in SEEDS:
            env = MetaToolEnvironment(scenarios, seed=seed)

            # Use subset of methods for efficiency
            selectors: list[tuple[str, object]] = [
                ("Random", RandomBFCLSelector(seed=seed)),
                ("DescMatch", DescriptionMatchSelector(qt, seed=seed)),
                ("LinUCB", LinUCBSelector(fe, seed=seed)),
                ("BilinTS-Cold", ColdStartLinTSSelector(feature_extractor=fe, seed=seed)),
            ]

            results = run_experiment(selectors, env, seed)
            for r in results:
                tsa_per_selector.setdefault(r.name, []).append(r.tsa)

        agg: dict[str, dict[str, float]] = {}
        for name, tsas in tsa_per_selector.items():
            agg[name] = {
                "tsa_mean": float(np.mean(tsas)),
                "tsa_std": float(np.std(tsas)),
            }

        _print_table(agg, f"K={k} Results")
        all_results[k] = agg

    # Print combined table
    print("\n" + "=" * 60)
    print("Combined: Method TSA by Candidate Set Size")
    print("=" * 60)
    methods = ["Random", "DescMatch", "LinUCB", "BilinTS-Cold"]
    header = f"{'Method':<15}" + "".join(f"{'K='+str(k):<12}" for k in k_values)
    print(header)
    print("-" * len(header))
    for method in methods:
        row = f"{method:<15}"
        for k in k_values:
            if method in all_results[k]:
                m = all_results[k][method]["tsa_mean"]
                s = all_results[k][method]["tsa_std"]
                row += f"{m:.3f}±{s:.3f}  "
            else:
                row += f"{'N/A':<12}"
        print(row)

    return all_results


# ---------------------------------------------------------------------------
# Experiment 3: Dense vs TF-IDF (K=10)
# ---------------------------------------------------------------------------


def run_dense_comparison(max_queries: int = MAX_QUERIES) -> dict:
    """Compare TF-IDF vs dense embedding features on MetaTool.

    Tests whether dense embeddings also help on a different benchmark.
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Dense vs TF-IDF on MetaTool (K=10)")
    print("=" * 60)

    try:
        from tats.dense_features import DenseEmbedder, DenseFeatureExtractor, DenseDescMatchSelector
    except ImportError:
        print("sentence-transformers not available, skipping dense comparison")
        return {}

    bm = MetaToolBenchmark(n_candidates=10, seed=0, max_queries=max_queries)
    scenarios = bm.scenarios
    qt = build_query_texts_from_scenarios(scenarios)

    # TF-IDF features
    fe_tfidf = ToolFeatureExtractor(qt)

    # Dense features
    embedder = DenseEmbedder()
    fe_dense = DenseFeatureExtractor(qt, embedder)

    tsa_per_selector: dict[str, list[float]] = {}

    for seed in SEEDS:
        env = MetaToolEnvironment(scenarios, seed=seed)

        selectors: list[tuple[str, object]] = [
            ("DescMatch", DescriptionMatchSelector(qt, seed=seed)),
            ("DenseDescMatch", DenseDescMatchSelector(qt, embedder, seed=seed)),
            ("LinUCB-TfIdf", LinUCBSelector(fe_tfidf, seed=seed)),
            ("LinUCB-Dense", LinUCBSelector(fe_dense, seed=seed)),
            ("BilinTS-TfIdf", ColdStartLinTSSelector(feature_extractor=fe_tfidf, seed=seed)),
            ("BilinTS-Dense", ColdStartLinTSSelector(feature_extractor=fe_dense, seed=seed)),
        ]

        results = run_experiment(selectors, env, seed)
        for r in results:
            tsa_per_selector.setdefault(r.name, []).append(r.tsa)
            print(f"  seed={seed} {r.name:<18} TSA={r.tsa:.4f}")

    agg: dict[str, dict[str, float]] = {}
    for name, tsas in tsa_per_selector.items():
        agg[name] = {
            "tsa_mean": float(np.mean(tsas)),
            "tsa_std": float(np.std(tsas)),
        }

    _print_table(agg, "Dense vs TF-IDF on MetaTool (K=10)")
    _print_significance(tsa_per_selector)

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    all_results = {}

    # Experiment 1: Main results
    all_results["main"] = run_main_experiment()

    # Experiment 2: Candidate set size ablation
    all_results["candidate_size"] = run_candidate_size_experiment()

    # Experiment 3: Dense vs TF-IDF
    all_results["dense_vs_tfidf"] = run_dense_comparison()

    # Save all results
    save_results_json(all_results, str(RESULTS_DIR / "metatool_results.json"))
    print("\nAll results saved to results/metatool_results.json")


if __name__ == "__main__":
    main()
