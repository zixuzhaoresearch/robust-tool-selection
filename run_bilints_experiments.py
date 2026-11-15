"""BilinTS experiment suite: main results, degradation, repeated-BFCL, and ablation.

Produces:
  - results/bilints_results.json    (all experiment results)
  - figures/bilints_main.pdf        (main comparison bar chart)
  - figures/bilints_degradation.pdf (degradation robustness)
  - figures/bilints_repeated.pdf    (per-round learning curves)
  - figures/bilints_ablation.pdf    (ablation study)
  - figures/bilints_learning_curves.pdf  (cumulative TSA over interactions)
  - figures/bilints_feature_weights.pdf  (feature weight heatmap)

Four experiments:
  1. Main Results on live_multiple — 8 selectors, 10 seeds
  2. Degradation Robustness — Full/L1/L2/L3 x 6 selectors, 10 seeds
  3. Repeated-BFCL on v4_multiple — 8 selectors, 5 rounds x 10 seeds
  4. Ablation Study on live_multiple — 5 variants, 10 seeds

IMPORTANT — Cross-dataset warm-start protocol:
  Experiments 1, 2, 4 (eval on live_multiple): warm-start from v4_multiple
  Experiment 3 (eval on v4_multiple): warm-start from live_multiple
  This prevents data leakage from ground-truth labels into the prior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

from tats.bfcl_adapter import (
    BFCLEnvironment,
    BFCLToolSelectionBenchmark,
    RepeatedBFCLEnvironment,
)
from tats.bfcl_selectors import (
    DescriptionMatchSelector,
    EpsilonGreedyBFCLSelector,
    ParameterMatchSelector,
    RandomBFCLSelector,
    RepeatedBFCLSelector,
    UCBBFCLSelector,
)
from tats.bilints_selector import (
    BilinTSGreedySelector,
    BilinTSSelector,
    ColdStartLinTSSelector,
    LinUCBSelector,
    SupervisedOfflineSelector,
    SupervisedOnlineSelector,
)
from tats.bilints_plotting import plot_learning_curves, plot_feature_weight_heatmap
from tats.degradation import DegradationEngine
from tats.experiment import run_experiment, run_selector, save_results_json
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.warmstart import compute_warm_start, train_offline_logistic

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("paper/figures")
SEEDS = list(range(10))
DEGRADATION_LEVELS = ["Full", "L1", "L2", "L3"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_significance(results: dict[str, list[float]], reference: str = "DescMatch"):
    """Print paired t-test p-values vs reference method."""
    if reference not in results:
        return
    ref = results[reference]
    for name, scores in sorted(results.items()):
        if name == reference or len(scores) != len(ref):
            continue
        t_stat, p_val = stats.ttest_rel(scores, ref)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"    {name} vs {reference}: t={t_stat:.2f}, p={p_val:.4f} {sig}")


# ---------------------------------------------------------------------------
# Experiment 1: Main Results on live_multiple
# ---------------------------------------------------------------------------


def run_main_experiment() -> dict:
    """8 selectors on live_multiple, 10 seeds. Returns aggregated TSA stats.

    Cross-dataset warm-start: train on v4_multiple, evaluate on live_multiple.
    """
    print("=" * 60)
    print("Experiment 1: Main Results on live_multiple")
    print("=" * 60)

    # Load BOTH datasets (cross-dataset warm-start)
    bm_live = BFCLToolSelectionBenchmark(
        bfcl_eval_package=True, dataset="live_multiple"
    )
    bm_multi = BFCLToolSelectionBenchmark(
        bfcl_eval_package=True, dataset="multiple"
    )
    scenarios_eval = bm_live.scenarios   # 1053 scenarios for evaluation
    scenarios_ws = bm_multi.scenarios     # 200 scenarios for warm-start

    print(f"Eval: {len(scenarios_eval)} live_multiple scenarios")
    print(f"Warm-start: {len(scenarios_ws)} v4_multiple scenarios")

    # Build feature extractors for warm-start and evaluation separately
    qt_ws = build_query_texts_from_scenarios(scenarios_ws)
    fe_ws = ToolFeatureExtractor(qt_ws)
    theta0, B0 = compute_warm_start(scenarios_ws, fe_ws)

    # Train offline logistic for supervised baseline (same cross-dataset source)
    logistic_model = train_offline_logistic(scenarios_ws, fe_ws)
    print(f"Offline logistic trained on {len(scenarios_ws)} scenarios")

    qt_eval = build_query_texts_from_scenarios(scenarios_eval)
    fe_eval = ToolFeatureExtractor(qt_eval)

    tsa_per_selector: dict[str, list[float]] = {}
    learning_curves: dict[str, list[list[float]]] = {}

    for seed in SEEDS:
        env = BFCLEnvironment(scenarios_eval, seed=seed)

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
            ("Sup-Online", SupervisedOnlineSelector(
                feature_extractor=fe_eval, seed=seed,
            )),
        ]

        results = run_experiment(selectors, env, seed)

        for r in results:
            tsa_per_selector.setdefault(r.name, []).append(r.tsa)
            if r.cumulative_tsa is not None:
                learning_curves.setdefault(r.name, []).append(r.cumulative_tsa)
            print(f"  seed={seed} {r.name:<15} TSA={r.tsa:.4f}")

    # Aggregate
    agg: dict[str, dict[str, float]] = {}
    for name, tsas in tsa_per_selector.items():
        agg[name] = {
            "tsa_mean": float(np.mean(tsas)),
            "tsa_std": float(np.std(tsas)),
        }

    print("\n--- Main Results Summary ---")
    for name, st in agg.items():
        print(f"  {name:<15} TSA={st['tsa_mean']:.3f} +/- {st['tsa_std']:.3f}")

    _print_significance(tsa_per_selector)

    # Plot learning curves
    if learning_curves:
        plot_learning_curves(
            learning_curves,
            str(FIGURES_DIR / "bilints_learning_curves.pdf"),
        )

    return agg


# ---------------------------------------------------------------------------
# Experiment 2: Degradation Robustness
# ---------------------------------------------------------------------------


def run_degradation_experiment() -> dict:
    """BilinTS under Full/L1/L2/L3 degradation on live_multiple, 10 seeds.

    Cross-dataset warm-start: train on v4_multiple, evaluate on live_multiple.

    Key design:
    - query_texts come from EVALUATION scenarios (queries don't change).
    - warm-start theta0/B0 is computed from v4_multiple (separate dataset).
    - At selection time, the environment uses degraded scenarios.
    - The feature extractor uses the eval query corpus throughout.
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Degradation Robustness on live_multiple")
    print("=" * 60)

    # Load BOTH datasets
    bm_live = BFCLToolSelectionBenchmark(
        bfcl_eval_package=True, dataset="live_multiple"
    )
    bm_multi = BFCLToolSelectionBenchmark(
        bfcl_eval_package=True, dataset="multiple"
    )
    scenarios_eval = bm_live.scenarios
    scenarios_ws = bm_multi.scenarios

    print(f"Eval: {len(scenarios_eval)} live_multiple scenarios")
    print(f"Warm-start: {len(scenarios_ws)} v4_multiple scenarios")

    # Build warm-start from v4_multiple
    qt_ws = build_query_texts_from_scenarios(scenarios_ws)
    fe_ws = ToolFeatureExtractor(qt_ws)
    theta0, B0 = compute_warm_start(scenarios_ws, fe_ws)

    # Build eval FE from live_multiple query corpus
    qt_eval = build_query_texts_from_scenarios(scenarios_eval)
    fe_eval = ToolFeatureExtractor(qt_eval)

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    feature_weights: dict[str, dict[str, float]] = {}

    for level in DEGRADATION_LEVELS:
        print(f"\n--- Degradation Level: {level} ---")
        tsa_per_selector: dict[str, list[float]] = {}
        last_bilints_warm_selector = None

        for seed in SEEDS:
            # Build environment from (possibly degraded) scenarios
            if level == "Full":
                level_scenarios = scenarios_eval
            else:
                engine = DegradationEngine(seed=seed)
                level_scenarios = engine.degrade(scenarios_eval, level)

            env = BFCLEnvironment(level_scenarios, seed=seed)

            bilints_warm = BilinTSSelector(
                feature_extractor=fe_eval, seed=seed,
                theta_init=theta0, B_init=B0,
            )

            selectors: list[tuple[str, object]] = [
                ("Random", RandomBFCLSelector(seed=seed)),
                ("DescMatch", DescriptionMatchSelector(qt_eval, seed=seed)),
                ("ParamMatch", ParameterMatchSelector(qt_eval, seed=seed)),
                ("LinUCB", LinUCBSelector(fe_eval, seed=seed)),
                ("BilinTS-Cold", ColdStartLinTSSelector(
                    feature_extractor=fe_eval, seed=seed,
                )),
                ("BilinTS-Warm", bilints_warm),
            ]

            results = run_experiment(selectors, env, seed)

            for r in results:
                tsa_per_selector.setdefault(r.name, []).append(r.tsa)

            # Keep the last seed's BilinTS-Warm for feature weight extraction
            last_bilints_warm_selector = bilints_warm

        # Extract feature weights from the last BilinTS-Warm
        if last_bilints_warm_selector is not None:
            feature_weights[level] = last_bilints_warm_selector.get_weights()

        # Aggregate this level
        level_agg: dict[str, dict[str, float]] = {}
        for name, tsas in tsa_per_selector.items():
            level_agg[name] = {
                "tsa_mean": float(np.mean(tsas)),
                "tsa_std": float(np.std(tsas)),
            }
            print(f"  {name:<15} TSA={np.mean(tsas):.3f} +/- {np.std(tsas):.3f}")

        _print_significance(tsa_per_selector)
        all_results[level] = level_agg

    # Plot feature weight heatmap
    if feature_weights:
        plot_feature_weight_heatmap(
            feature_weights,
            str(FIGURES_DIR / "bilints_feature_weights.pdf"),
        )

    return all_results


# ---------------------------------------------------------------------------
# Experiment 3: Repeated-BFCL on v4_multiple
# ---------------------------------------------------------------------------


def run_repeated_experiment() -> dict:
    """8 selectors on v4_multiple with 5 rounds, 10 seeds.

    Cross-dataset warm-start: train on live_multiple, evaluate on v4_multiple.
    Tracks per-round TSA to demonstrate learning over rounds.
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Repeated-BFCL on v4_multiple (5 rounds)")
    print("=" * 60)

    # Load BOTH datasets (cross-dataset warm-start — reversed direction)
    bm_multi = BFCLToolSelectionBenchmark(
        bfcl_eval_package=True, dataset="multiple"
    )
    bm_live = BFCLToolSelectionBenchmark(
        bfcl_eval_package=True, dataset="live_multiple"
    )
    scenarios_eval = bm_multi.scenarios   # 200 scenarios for evaluation
    scenarios_ws = bm_live.scenarios       # 1053 scenarios for warm-start

    n_rounds = 5
    print(f"Eval: {len(scenarios_eval)} v4_multiple scenarios, {n_rounds} rounds")
    print(f"Warm-start: {len(scenarios_ws)} live_multiple scenarios")

    # Build warm-start from live_multiple
    qt_ws = build_query_texts_from_scenarios(scenarios_ws)
    fe_ws = ToolFeatureExtractor(qt_ws)
    theta0, B0 = compute_warm_start(scenarios_ws, fe_ws)

    # Train offline logistic for supervised baseline (from live_multiple)
    logistic_model = train_offline_logistic(scenarios_ws, fe_ws)
    print(f"Offline logistic trained on {len(scenarios_ws)} scenarios")

    # Build base query_texts from eval scenarios
    base_query_texts = build_query_texts_from_scenarios(scenarios_eval)

    per_round_tsas: dict[str, list[list[float]]] = {}
    overall_tsas: dict[str, list[float]] = {}

    for seed in SEEDS:
        env = RepeatedBFCLEnvironment(scenarios_eval, n_rounds=n_rounds, seed=seed)

        # Build extended query_texts covering ALL repeated query IDs.
        # Each query's required_capability is the original scenario_id,
        # so we can look up the original scenario's query_text.
        scenario_text_map = {s.scenario_id: s.query_text for s in scenarios_eval}
        extended_qtexts: dict[int, str] = {}
        for q in env.queries:
            extended_qtexts[q.query_id] = scenario_text_map[q.required_capability]

        # Build per-seed feature extractor with full query coverage
        fe_repeated = ToolFeatureExtractor(extended_qtexts)

        selectors: list[tuple[str, object]] = [
            ("Random", RandomBFCLSelector(seed=seed)),
            ("DescMatch", DescriptionMatchSelector(extended_qtexts, seed=seed)),
            ("ParamMatch", ParameterMatchSelector(extended_qtexts, seed=seed)),
            ("EpsGreedy", EpsilonGreedyBFCLSelector(epsilon=0.1, seed=seed)),
            ("RepeatedTS", RepeatedBFCLSelector(seed=seed)),
            ("LinUCB", LinUCBSelector(fe_repeated, seed=seed)),
            ("BilinTS-Cold", ColdStartLinTSSelector(
                feature_extractor=fe_repeated, seed=seed,
            )),
            ("BilinTS-Warm", BilinTSSelector(
                feature_extractor=fe_repeated, seed=seed,
                theta_init=theta0, B_init=B0,
            )),
            ("Sup-Offline", SupervisedOfflineSelector(
                feature_extractor=fe_repeated, model=logistic_model, seed=seed,
            )),
            ("Sup-Online", SupervisedOnlineSelector(
                feature_extractor=fe_repeated, seed=seed,
            )),
        ]

        results = run_experiment(selectors, env, seed)

        for r in results:
            overall_tsas.setdefault(r.name, []).append(r.tsa)
            round_tsa = env.per_round_tsa(r.selected_ids, r.ground_truth_ids)
            per_round_tsas.setdefault(r.name, []).append(round_tsa)

    # Aggregate
    agg: dict[str, dict] = {}
    for name in overall_tsas:
        tsas = overall_tsas[name]
        round_lists = per_round_tsas[name]
        n_r = len(round_lists[0])

        avg_per_round = []
        for r_idx in range(n_r):
            vals = [rl[r_idx] for rl in round_lists if r_idx < len(rl)]
            avg_per_round.append(float(np.mean(vals)))

        agg[name] = {
            "tsa_mean": float(np.mean(tsas)),
            "tsa_std": float(np.std(tsas)),
            "per_round_tsa": avg_per_round,
        }
        print(
            f"  {name:<15} Overall={np.mean(tsas):.3f} +/- {np.std(tsas):.3f}  "
            f"Per-round={[f'{t:.3f}' for t in avg_per_round]}"
        )

    _print_significance(overall_tsas)

    # Plot per-round learning curves
    _plot_repeated_learning(agg)

    return agg


# ---------------------------------------------------------------------------
# Experiment 4: Ablation Study
# ---------------------------------------------------------------------------


def run_ablation_experiment() -> dict:
    """Ablation study on live_multiple (1053 scenarios, 10 seeds).

    Cross-dataset warm-start: train on v4_multiple, evaluate on live_multiple.

    Tests 5 variants to isolate contributions of feature groups and design
    choices:
      Full           -- all 8 features, warm start, Thompson Sampling
      -DescFeatures  -- mask out description-derived features (0,1,4,7)
      -SchemaFeatures-- mask out schema-derived features (2,3,6)
      -WarmStart     -- all features, cold start (theta_init = 0)
      -Exploration   -- all features, warm start, greedy (no TS sampling)

    Returns dict[ablation_name -> {tsa_mean, tsa_std}].
    """
    print("\n" + "=" * 60)
    print("Experiment 4: Ablation Study on live_multiple")
    print("=" * 60)

    # Load BOTH datasets (cross-dataset warm-start)
    bm_live = BFCLToolSelectionBenchmark(
        bfcl_eval_package=True, dataset="live_multiple"
    )
    bm_multi = BFCLToolSelectionBenchmark(
        bfcl_eval_package=True, dataset="multiple"
    )
    scenarios_eval = bm_live.scenarios
    scenarios_ws = bm_multi.scenarios

    print(f"Eval: {len(scenarios_eval)} live_multiple scenarios")
    print(f"Warm-start: {len(scenarios_ws)} v4_multiple scenarios")

    # Build warm-start from v4_multiple
    qt_ws = build_query_texts_from_scenarios(scenarios_ws)
    fe_ws = ToolFeatureExtractor(qt_ws)
    theta0, B0 = compute_warm_start(scenarios_ws, fe_ws)

    # Build eval FE from live_multiple
    qt_eval = build_query_texts_from_scenarios(scenarios_eval)
    fe_eval = ToolFeatureExtractor(qt_eval)

    # Feature indices:
    # [0] desc_sim  [1] name_sim  [2] param_match  [3] n_params_norm
    # [4] desc_len_norm  [5] n_cand_norm  [6] required_ratio  [7] has_desc
    ablations: dict[str, dict] = {
        "Full": {"mask": np.ones(8), "warm": True, "greedy": False},
        "-DescFeatures": {
            "mask": np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=float),
            "warm": True,
            "greedy": False,
        },
        "-SchemaFeatures": {
            "mask": np.array([1, 1, 0, 0, 1, 1, 0, 1], dtype=float),
            "warm": True,
            "greedy": False,
        },
        "-WarmStart": {"mask": np.ones(8), "warm": False, "greedy": False},
        "-Exploration": {"mask": np.ones(8), "warm": True, "greedy": True},
    }

    tsa_per_ablation: dict[str, list[float]] = {}

    for seed in SEEDS:
        env = BFCLEnvironment(scenarios_eval, seed=seed)

        for abl_name, cfg in ablations.items():
            mask = cfg["mask"]

            if cfg["greedy"]:
                # Greedy: warm start, no exploration
                sel = BilinTSGreedySelector(
                    feature_extractor=fe_eval,
                    seed=seed,
                    theta_init=theta0,
                    B_init=B0,
                    feature_mask=mask,
                )
            elif not cfg["warm"]:
                # Cold start (no warm-start theta)
                sel = ColdStartLinTSSelector(
                    feature_extractor=fe_eval,
                    seed=seed,
                    feature_mask=mask,
                )
            else:
                # Standard: warm start + Thompson Sampling
                sel = BilinTSSelector(
                    feature_extractor=fe_eval,
                    seed=seed,
                    theta_init=theta0,
                    B_init=B0,
                    feature_mask=mask,
                )

            selectors = [(abl_name, sel)]
            results = run_experiment(selectors, env, seed)

            for r in results:
                tsa_per_ablation.setdefault(r.name, []).append(r.tsa)
                print(f"  seed={seed} {r.name:<20} TSA={r.tsa:.4f}")

    # Aggregate
    agg: dict[str, dict[str, float]] = {}
    for name, tsas in tsa_per_ablation.items():
        agg[name] = {
            "tsa_mean": float(np.mean(tsas)),
            "tsa_std": float(np.std(tsas)),
        }

    print("\n--- Ablation Study Summary ---")
    for name, st in agg.items():
        print(f"  {name:<20} TSA={st['tsa_mean']:.3f} +/- {st['tsa_std']:.3f}")

    _print_significance(tsa_per_ablation, reference="Full")

    # Plot
    _plot_ablation(agg)

    return agg


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_main_comparison(agg: dict) -> None:
    """Bar chart comparing all selectors on live_multiple."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(agg.keys())
    means = [agg[n]["tsa_mean"] for n in names]
    stds = [agg[n]["tsa_std"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.get_cmap("tab10")
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=[colors(i) for i in range(len(names))])

    ax.set_xlabel("Selector")
    ax.set_ylabel("Tool Selection Accuracy (TSA)")
    ax.set_title("BilinTS: Main Results on BFCL live_multiple")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bilints_main.pdf", dpi=300)
    plt.close(fig)
    print(f"  Figure saved to {FIGURES_DIR}/bilints_main.pdf")


def _plot_degradation(results: dict) -> None:
    """Grouped bar chart: selector performance across degradation levels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selectors = list(next(iter(results.values())).keys())
    levels = list(results.keys())
    n_selectors = len(selectors)
    n_levels = len(levels)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_levels)
    width = 0.8 / n_selectors
    colors = plt.get_cmap("tab10")

    for i, sel in enumerate(selectors):
        means = [results[lv][sel]["tsa_mean"] for lv in levels]
        stds = [results[lv][sel]["tsa_std"] for lv in levels]
        ax.bar(
            x + i * width - 0.4 + width / 2,
            means,
            width,
            yerr=stds,
            label=sel,
            color=colors(i),
            capsize=3,
        )

    ax.set_xlabel("Degradation Level")
    ax.set_ylabel("Tool Selection Accuracy (TSA)")
    ax.set_title("BilinTS: Degradation Robustness on live_multiple")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bilints_degradation.pdf", dpi=300)
    plt.close(fig)
    print(f"  Figure saved to {FIGURES_DIR}/bilints_degradation.pdf")


def _plot_repeated_learning(agg: dict) -> None:
    """Per-round TSA learning curves for repeated-BFCL experiment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    colors = [
        "#377eb8", "#e41a1c", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#17becf",
    ]

    for i, (name, data) in enumerate(agg.items()):
        rounds = list(range(1, len(data["per_round_tsa"]) + 1))
        ax.plot(
            rounds,
            data["per_round_tsa"],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=f"{name} (avg={data['tsa_mean']:.3f})",
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Tool Selection Accuracy (TSA)", fontsize=12)
    ax.set_title("Repeated-BFCL: BilinTS Learning Across Rounds", fontsize=13)
    ax.set_xticks(list(range(1, 6)))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bilints_repeated.pdf", dpi=300)
    plt.close(fig)
    print(f"  Figure saved to {FIGURES_DIR}/bilints_repeated.pdf")


def _plot_ablation(agg: dict) -> None:
    """Bar chart for ablation study results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(agg.keys())
    means = [agg[n]["tsa_mean"] for n in names]
    stds = [agg[n]["tsa_std"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4", "#9467bd"]
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors[: len(names)], edgecolor="black", linewidth=0.5)

    # Draw horizontal line at Full model performance for reference
    ax.axhline(y=means[0], color="grey", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Ablation Variant", fontsize=12)
    ax.set_ylabel("Tool Selection Accuracy (TSA)", fontsize=12)
    ax.set_title("BilinTS Ablation Study (live_multiple)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bilints_ablation.pdf", dpi=300)
    plt.close(fig)
    print(f"  Figure saved to {FIGURES_DIR}/bilints_ablation.pdf")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(results: dict) -> None:
    """Print final summary table across all experiments."""
    print("\n" + "=" * 70)
    print("BILINTS EXPERIMENT SUMMARY")
    print("=" * 70)

    # Main results
    if "main" in results:
        print("\n--- Experiment 1: Main Results (live_multiple) ---")
        main = results["main"]
        print(f"  {'Selector':<15} {'TSA Mean':>10} {'TSA Std':>10}")
        print("  " + "-" * 37)
        for name, st in main.items():
            print(f"  {name:<15} {st['tsa_mean']:>10.4f} {st['tsa_std']:>10.4f}")

    # Degradation
    if "degradation" in results:
        print("\n--- Experiment 2: Degradation Robustness ---")
        deg = results["degradation"]
        selectors = list(next(iter(deg.values())).keys())
        header = f"  {'Level':<8}" + "".join(f"{s:<16}" for s in selectors)
        print(header)
        print("  " + "-" * (8 + 16 * len(selectors)))
        for level in DEGRADATION_LEVELS:
            if level in deg:
                row = f"  {level:<8}"
                for sel in selectors:
                    m = deg[level][sel]["tsa_mean"]
                    s = deg[level][sel]["tsa_std"]
                    row += f"{m:.3f}+/-{s:.3f}   "
                print(row)

    # Repeated
    if "repeated" in results:
        print("\n--- Experiment 3: Repeated-BFCL (v4_multiple, 5 rounds) ---")
        rep = results["repeated"]
        print(f"  {'Selector':<15} {'Overall TSA':>12} {'Round 1':>8} {'Round 5':>8}")
        print("  " + "-" * 45)
        for name, st in rep.items():
            pr = st.get("per_round_tsa", [])
            r1 = f"{pr[0]:.3f}" if pr else "N/A"
            r5 = f"{pr[-1]:.3f}" if pr else "N/A"
            print(
                f"  {name:<15} {st['tsa_mean']:>10.4f}+/-{st['tsa_std']:.4f} "
                f"{r1:>8} {r5:>8}"
            )

    # Ablation
    if "ablation" in results:
        print("\n--- Experiment 4: Ablation Study (live_multiple) ---")
        abl = results["ablation"]
        print(f"  {'Variant':<20} {'TSA Mean':>10} {'TSA Std':>10}")
        print("  " + "-" * 42)
        for name, st in abl.items():
            print(f"  {name:<20} {st['tsa_mean']:>10.4f} {st['tsa_std']:>10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    # Experiment 1
    results["main"] = run_main_experiment()
    _plot_main_comparison(results["main"])

    # Experiment 2
    results["degradation"] = run_degradation_experiment()
    _plot_degradation(results["degradation"])

    # Experiment 3
    results["repeated"] = run_repeated_experiment()
    # (plotting done inside run_repeated_experiment)

    # Experiment 4
    results["ablation"] = run_ablation_experiment()
    # (plotting done inside run_ablation_experiment)

    # Save all
    save_results_json(results, RESULTS_DIR / "bilints_results.json")

    # Print summary
    _print_summary(results)

    print(f"\nResults saved to {RESULTS_DIR}/bilints_results.json")
    print(f"Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
