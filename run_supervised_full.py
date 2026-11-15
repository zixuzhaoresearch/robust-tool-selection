"""Full 10-seed experiment: supervised baselines vs bandits.

Runs Experiment 1 (main, live_multiple) and Experiment 3 (repeated, v4_multiple)
with supervised baselines added.
"""
from __future__ import annotations

import json
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
    BilinTSSelector,
    ColdStartLinTSSelector,
    LinUCBSelector,
    SupervisedOfflineSelector,
    SupervisedOnlineSelector,
)
from tats.experiment import run_experiment
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.warmstart import compute_warm_start, train_offline_logistic

SEEDS = list(range(10))


def _print_significance(results: dict[str, list[float]], reference: str):
    if reference not in results:
        return
    ref = results[reference]
    for name, scores in sorted(results.items()):
        if name == reference or len(scores) != len(ref):
            continue
        t_stat, p_val = stats.ttest_rel(scores, ref)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"    {name} vs {reference}: t={t_stat:.2f}, p={p_val:.4f} {sig}")


def run_exp1_main():
    """Experiment 1: Main results on live_multiple, 10 seeds."""
    print("=" * 60)
    print("Experiment 1: Main Results on live_multiple (10 seeds)")
    print("=" * 60)

    bm_live = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple")
    bm_multi = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="multiple")
    scenarios_eval = bm_live.scenarios
    scenarios_ws = bm_multi.scenarios

    qt_ws = build_query_texts_from_scenarios(scenarios_ws)
    fe_ws = ToolFeatureExtractor(qt_ws)
    theta0, B0 = compute_warm_start(scenarios_ws, fe_ws)
    logistic_model = train_offline_logistic(scenarios_ws, fe_ws)

    qt_eval = build_query_texts_from_scenarios(scenarios_eval)
    fe_eval = ToolFeatureExtractor(qt_eval)

    tsa_per: dict[str, list[float]] = {}

    for seed in SEEDS:
        env = BFCLEnvironment(scenarios_eval, seed=seed)
        selectors = [
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
            tsa_per.setdefault(r.name, []).append(r.tsa)
        print(f"  seed={seed} done")

    print("\n--- Experiment 1 Summary ---")
    print(f"  {'Selector':<15} {'TSA Mean':>10} {'TSA Std':>10}")
    print("  " + "-" * 37)
    for name in sorted(tsa_per, key=lambda n: -np.mean(tsa_per[n])):
        tsas = tsa_per[name]
        print(f"  {name:<15} {np.mean(tsas):>10.4f} {np.std(tsas):>10.4f}")

    print("\n  --- vs DescMatch ---")
    _print_significance(tsa_per, "DescMatch")
    print("\n  --- vs Sup-Offline ---")
    _print_significance(tsa_per, "Sup-Offline")
    print("\n  --- vs LinUCB ---")
    _print_significance(tsa_per, "LinUCB")

    return tsa_per


def run_exp3_repeated():
    """Experiment 3: Repeated-BFCL on v4_multiple, 5 rounds, 10 seeds."""
    print("\n" + "=" * 60)
    print("Experiment 3: Repeated-BFCL on v4_multiple (5 rounds, 10 seeds)")
    print("=" * 60)

    bm_multi = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="multiple")
    bm_live = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple")
    scenarios_eval = bm_multi.scenarios
    scenarios_ws = bm_live.scenarios
    n_rounds = 5

    qt_ws = build_query_texts_from_scenarios(scenarios_ws)
    fe_ws = ToolFeatureExtractor(qt_ws)
    theta0, B0 = compute_warm_start(scenarios_ws, fe_ws)
    logistic_model = train_offline_logistic(scenarios_ws, fe_ws)

    per_round_tsas: dict[str, list[list[float]]] = {}
    overall_tsas: dict[str, list[float]] = {}

    for seed in SEEDS:
        env = RepeatedBFCLEnvironment(scenarios_eval, n_rounds=n_rounds, seed=seed)

        scenario_text_map = {s.scenario_id: s.query_text for s in scenarios_eval}
        extended_qtexts: dict[int, str] = {}
        for q in env.queries:
            extended_qtexts[q.query_id] = scenario_text_map[q.required_capability]

        fe_repeated = ToolFeatureExtractor(extended_qtexts)

        selectors = [
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

        print(f"  seed={seed} done")

    print("\n--- Experiment 3 Summary ---")
    print(f"  {'Selector':<15} {'Overall':>10} {'R1':>8} {'R5':>8}")
    print("  " + "-" * 42)
    for name in sorted(overall_tsas, key=lambda n: -np.mean(overall_tsas[n])):
        tsas = overall_tsas[name]
        rounds = per_round_tsas[name]
        n_r = len(rounds[0])
        r1 = np.mean([rl[0] for rl in rounds])
        r5 = np.mean([rl[-1] for rl in rounds])
        print(f"  {name:<15} {np.mean(tsas):>10.4f} {r1:>8.4f} {r5:>8.4f}")

    print("\n  --- vs Sup-Offline ---")
    _print_significance(overall_tsas, "Sup-Offline")
    print("\n  --- vs LinUCB ---")
    _print_significance(overall_tsas, "LinUCB")

    return overall_tsas, per_round_tsas


if __name__ == "__main__":
    exp1 = run_exp1_main()
    exp3_overall, exp3_rounds = run_exp3_repeated()

    # Save results to JSON for audit trail
    results = {
        "main": {
            name: {"tsa_mean": float(np.mean(tsas)), "tsa_std": float(np.std(tsas))}
            for name, tsas in exp1.items()
        },
        "repeated": {
            name: {
                "tsa_mean": float(np.mean(tsas)),
                "tsa_std": float(np.std(tsas)),
                "per_round_tsa": [
                    float(np.mean([rl[r] for rl in exp3_rounds[name]]))
                    for r in range(len(exp3_rounds[name][0]))
                ],
            }
            for name, tsas in exp3_overall.items()
        },
    }
    out_path = Path("results/supervised_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")
