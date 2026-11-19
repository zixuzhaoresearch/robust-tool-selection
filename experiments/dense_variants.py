"""Full 10-seed experiments with dense embedding baselines.

Runs:
  Experiment 1: Main results (live_multiple) — TF-IDF vs Dense × DescMatch/LinUCB
  Experiment 2: Degradation robustness — Full/L1/L2/L3 with dense baselines
  Experiment 3: Repeated-BFCL (v4_multiple)
"""
from __future__ import annotations

import json
import time
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
    ParameterMatchSelector,
    RandomBFCLSelector,
)
from tats.bilints_selector import (
    BilinTSSelector,
    ColdStartLinTSSelector,
    LinUCBSelector,
    SupervisedOfflineSelector,
)
from tats.degradation import DegradationEngine
from tats.dense_features import DenseDescMatchSelector, DenseEmbedder, DenseFeatureExtractor
from tats.experiment import run_experiment
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.warmstart import compute_warm_start, train_offline_logistic

SEEDS = list(range(10))
DEGRADATION_LEVELS = ["Full", "L1", "L2", "L3"]


def _print_sig(results: dict[str, list[float]], ref: str):
    if ref not in results:
        return
    r = results[ref]
    for name, scores in sorted(results.items()):
        if name == ref or len(scores) != len(r):
            continue
        t_stat, p_val = stats.ttest_rel(scores, r)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"    {name} vs {ref}: t={t_stat:.2f}, p={p_val:.4f} {sig}")


def _precompute_descs(scenarios, embedder: DenseEmbedder):
    """Pre-compute tool description embeddings for efficiency."""
    all_descs = set()
    for s in scenarios:
        for t in s.candidate_tools:
            all_descs.add(t.description)
    if all_descs:
        embedder.encode_batch(list(all_descs))


def run_exp1_main(embedder: DenseEmbedder):
    print("=" * 60)
    print("Experiment 1: Main Results on live_multiple (10 seeds)")
    print("=" * 60)

    bm_live = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple")
    bm_multi = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="multiple")
    scenarios_eval = bm_live.scenarios
    scenarios_ws = bm_multi.scenarios

    # TF-IDF features
    qt_ws = build_query_texts_from_scenarios(scenarios_ws)
    fe_ws = ToolFeatureExtractor(qt_ws)
    theta0, B0 = compute_warm_start(scenarios_ws, fe_ws)
    logistic_model = train_offline_logistic(scenarios_ws, fe_ws)

    qt_eval = build_query_texts_from_scenarios(scenarios_eval)
    fe_tfidf = ToolFeatureExtractor(qt_eval)

    # Dense features
    _precompute_descs(scenarios_eval, embedder)
    fe_dense = DenseFeatureExtractor(qt_eval, embedder)

    tsa_per: dict[str, list[float]] = {}

    for seed in SEEDS:
        env = BFCLEnvironment(scenarios_eval, seed=seed)
        selectors = [
            ("Random", RandomBFCLSelector(seed=seed)),
            ("DescMatch", DescriptionMatchSelector(qt_eval, seed=seed)),
            ("DenseDescMatch", DenseDescMatchSelector(qt_eval, embedder, seed=seed)),
            ("ParamMatch", ParameterMatchSelector(qt_eval, seed=seed)),
            ("LinUCB", LinUCBSelector(fe_tfidf, seed=seed)),
            ("LinUCB-Dense", LinUCBSelector(fe_dense, seed=seed)),
            ("BilinTS-Cold", ColdStartLinTSSelector(feature_extractor=fe_tfidf, seed=seed)),
            ("BilinTS-Warm", BilinTSSelector(
                feature_extractor=fe_tfidf, seed=seed,
                theta_init=theta0, B_init=B0,
            )),
            ("Sup-Offline", SupervisedOfflineSelector(
                feature_extractor=fe_tfidf, model=logistic_model, seed=seed,
            )),
        ]
        results = run_experiment(selectors, env, seed)
        for r in results:
            tsa_per.setdefault(r.name, []).append(r.tsa)
        print(f"  seed={seed} done")

    print("\n--- Experiment 1 Summary ---")
    print(f"  {'Selector':<20} {'TSA Mean':>10} {'TSA Std':>10}")
    print("  " + "-" * 42)
    for name in sorted(tsa_per, key=lambda n: -np.mean(tsa_per[n])):
        tsas = tsa_per[name]
        print(f"  {name:<20} {np.mean(tsas):>10.4f} {np.std(tsas):>10.4f}")

    print("\n  --- vs DenseDescMatch ---")
    _print_sig(tsa_per, "DenseDescMatch")
    print("\n  --- vs LinUCB ---")
    _print_sig(tsa_per, "LinUCB")

    return tsa_per


def run_exp2_degradation(embedder: DenseEmbedder):
    print("\n" + "=" * 60)
    print("Experiment 2: Degradation Robustness (10 seeds)")
    print("=" * 60)

    bm_live = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple")
    scenarios_eval = bm_live.scenarios

    qt_eval = build_query_texts_from_scenarios(scenarios_eval)
    fe_tfidf = ToolFeatureExtractor(qt_eval)

    all_results: dict[str, dict[str, dict[str, float]]] = {}

    for level in DEGRADATION_LEVELS:
        print(f"\n--- Degradation Level: {level} ---")
        tsa_per: dict[str, list[float]] = {}

        for seed in SEEDS:
            if level == "Full":
                level_scenarios = scenarios_eval
            else:
                engine = DegradationEngine(seed=seed)
                level_scenarios = engine.degrade(scenarios_eval, level)

            # Pre-compute dense embeddings for (possibly degraded) descriptions
            _precompute_descs(level_scenarios, embedder)
            fe_dense = DenseFeatureExtractor(qt_eval, embedder)

            env = BFCLEnvironment(level_scenarios, seed=seed)

            selectors = [
                ("Random", RandomBFCLSelector(seed=seed)),
                ("DescMatch", DescriptionMatchSelector(qt_eval, seed=seed)),
                ("DenseDescMatch", DenseDescMatchSelector(qt_eval, embedder, seed=seed)),
                ("ParamMatch", ParameterMatchSelector(qt_eval, seed=seed)),
                ("LinUCB", LinUCBSelector(fe_tfidf, seed=seed)),
                ("LinUCB-Dense", LinUCBSelector(fe_dense, seed=seed)),
            ]

            results = run_experiment(selectors, env, seed)
            for r in results:
                tsa_per.setdefault(r.name, []).append(r.tsa)

        level_agg: dict[str, dict[str, float]] = {}
        for name, tsas in tsa_per.items():
            level_agg[name] = {
                "tsa_mean": float(np.mean(tsas)),
                "tsa_std": float(np.std(tsas)),
            }
            print(f"  {name:<20} TSA={np.mean(tsas):.3f} +/- {np.std(tsas):.3f}")

        all_results[level] = level_agg

    # Print degradation table
    print("\n--- Degradation Summary Table ---")
    selectors_order = ["Random", "DescMatch", "DenseDescMatch", "ParamMatch", "LinUCB", "LinUCB-Dense"]
    header = f"  {'Level':<8}" + "".join(f"{s:<18}" for s in selectors_order)
    print(header)
    for level in DEGRADATION_LEVELS:
        row = f"  {level:<8}"
        for sel in selectors_order:
            if sel in all_results[level]:
                m = all_results[level][sel]["tsa_mean"]
                s = all_results[level][sel]["tsa_std"]
                row += f"{m:.3f}+/-{s:.3f}    "
            else:
                row += f"{'N/A':<18}"
        print(row)

    return all_results


def run_exp3_repeated(embedder: DenseEmbedder):
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

    _precompute_descs(scenarios_eval, embedder)

    overall_tsas: dict[str, list[float]] = {}
    per_round_tsas: dict[str, list[list[float]]] = {}

    for seed in SEEDS:
        env = RepeatedBFCLEnvironment(scenarios_eval, n_rounds=n_rounds, seed=seed)

        scenario_text_map = {s.scenario_id: s.query_text for s in scenarios_eval}
        extended_qtexts: dict[int, str] = {}
        for q in env.queries:
            extended_qtexts[q.query_id] = scenario_text_map[q.required_capability]

        fe_tfidf = ToolFeatureExtractor(extended_qtexts)
        fe_dense = DenseFeatureExtractor(extended_qtexts, embedder)

        selectors = [
            ("DescMatch", DescriptionMatchSelector(extended_qtexts, seed=seed)),
            ("DenseDescMatch", DenseDescMatchSelector(extended_qtexts, embedder, seed=seed)),
            ("LinUCB", LinUCBSelector(fe_tfidf, seed=seed)),
            ("LinUCB-Dense", LinUCBSelector(fe_dense, seed=seed)),
            ("BilinTS-Cold", ColdStartLinTSSelector(feature_extractor=fe_tfidf, seed=seed)),
            ("BilinTS-Warm", BilinTSSelector(
                feature_extractor=fe_tfidf, seed=seed,
                theta_init=theta0, B_init=B0,
            )),
        ]

        results = run_experiment(selectors, env, seed)
        for r in results:
            overall_tsas.setdefault(r.name, []).append(r.tsa)
            round_tsa = env.per_round_tsa(r.selected_ids, r.ground_truth_ids)
            per_round_tsas.setdefault(r.name, []).append(round_tsa)

        print(f"  seed={seed} done")

    print("\n--- Experiment 3 Summary ---")
    print(f"  {'Selector':<20} {'Overall':>10} {'R1':>8} {'R5':>8}")
    print("  " + "-" * 50)
    for name in sorted(overall_tsas, key=lambda n: -np.mean(overall_tsas[n])):
        tsas = overall_tsas[name]
        rounds = per_round_tsas[name]
        r1 = np.mean([rl[0] for rl in rounds])
        r5 = np.mean([rl[-1] for rl in rounds])
        print(f"  {name:<20} {np.mean(tsas):>10.4f} {r1:>8.4f} {r5:>8.4f}")

    return overall_tsas, per_round_tsas


if __name__ == "__main__":
    t_start = time.time()

    print("Initializing dense embedder...")
    embedder = DenseEmbedder("all-MiniLM-L6-v2")
    print(f"  Model loaded in {time.time() - t_start:.1f}s\n")

    exp1 = run_exp1_main(embedder)
    exp2 = run_exp2_degradation(embedder)
    exp3_overall, exp3_rounds = run_exp3_repeated(embedder)

    # Save results to JSON for audit trail
    results = {
        "main": {
            name: {"tsa_mean": float(np.mean(tsas)), "tsa_std": float(np.std(tsas))}
            for name, tsas in exp1.items()
        },
        "degradation": exp2,
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
    out_path = Path("results/dense_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    print(f"\nTotal time: {time.time() - t_start:.0f}s")
