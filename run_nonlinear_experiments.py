"""Experiment 2: Non-linear baseline comparison (Exp 2 in improvement_plan).

is capacity-limited. We test NeuralUCB (Zhou 2020) and XGBoost+eGreedy on the
same feature pipeline.

Outcomes:
  - If NeuralUCB wins significantly: rewrite paper to use NeuralUCB-Dense as
    main method.
  - If NeuralUCB ties/loses: evidence that 8-dim linear is sufficient.

Usage:
  uv run python run_nonlinear_experiments.py --mode smoke  # 1 seed, fast sanity
  uv run python run_nonlinear_experiments.py --mode full   # 10 seeds, full run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from tats.bfcl_adapter import BFCLEnvironment, BFCLToolSelectionBenchmark
from tats.bfcl_selectors import DescriptionMatchSelector
from tats.bilints_selector import LinUCBSelector
from tats.experiment import run_experiment
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.neural_bandits import NeuralUCBSelector, XGBoostBanditSelector

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_nonlinear_baselines(seeds: list[int]) -> dict:
    """Run LinUCB vs NeuralUCB vs XGBoost on BFCL live_multiple."""
    print("=" * 60)
    print("Experiment 2: Non-linear Baselines on live_multiple")
    print("=" * 60)

    bm = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple")
    scenarios = bm.scenarios
    print(f"Loaded {len(scenarios)} scenarios from live_multiple")

    qt = build_query_texts_from_scenarios(scenarios)
    fe = ToolFeatureExtractor(qt)

    tsa_per_selector: dict[str, list[float]] = {}
    time_per_selector: dict[str, list[float]] = {}

    for seed in seeds:
        env = BFCLEnvironment(scenarios, seed=seed)

        selectors = [
            ("DescMatch", DescriptionMatchSelector(qt, seed=seed)),
            ("LinUCB", LinUCBSelector(fe, seed=seed)),
            ("NeuralUCB", NeuralUCBSelector(fe, seed=seed, alpha=1.0)),
            ("XGBoost-eGreedy", XGBoostBanditSelector(fe, seed=seed, epsilon=0.1)),
        ]

        # Time each selector individually — wrap run_experiment per (name, sel)
        seed_results = []
        for name, sel in selectors:
            t_start = time.time()
            r_list = run_experiment([(name, sel)], env, seed)
            elapsed = time.time() - t_start
            for r in r_list:
                r_dict_name = r.name
                tsa_per_selector.setdefault(r_dict_name, []).append(r.tsa)
                time_per_selector.setdefault(r_dict_name, []).append(elapsed)
                print(f"  seed={seed} {r_dict_name:<18} TSA={r.tsa:.4f} time={elapsed:.1f}s")
                seed_results.append(r)

    # Aggregate
    agg: dict[str, dict[str, float]] = {}
    for name, tsas in tsa_per_selector.items():
        agg[name] = {
            "tsa_mean": float(np.mean(tsas)),
            "tsa_std": float(np.std(tsas)),
            "time_mean_sec": float(np.mean(time_per_selector[name])),
            "n_seeds": len(tsas),
        }

    print("\nAggregated results:")
    for name, s in sorted(agg.items()):
        print(f"  {name:<18}  TSA: {s['tsa_mean']:.4f} ± {s['tsa_std']:.4f}"
              f"  ({s['time_mean_sec']:.1f}s/run)")

    # Paired t-test: NeuralUCB vs LinUCB, XGBoost vs LinUCB
    print("\nSignificance vs LinUCB:")
    ref = tsa_per_selector.get("LinUCB", [])
    for name in ["NeuralUCB", "XGBoost-eGreedy"]:
        scores = tsa_per_selector.get(name, [])
        if len(scores) != len(ref) or not ref:
            continue
        t_stat, p_val = stats.ttest_rel(scores, ref)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        delta = np.mean(scores) - np.mean(ref)
        print(f"  {name} vs LinUCB:  delta={delta:+.4f} pp  t={t_stat:.2f}  p={p_val:.4f} {sig}")

    return {
        "per_seed": tsa_per_selector,
        "aggregate": agg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", default="results/nonlinear_results.json")
    args = parser.parse_args()

    seeds = [0] if args.mode == "smoke" else list(range(10))
    print(f"Mode: {args.mode}  Seeds: {seeds}")

    t0 = time.time()
    results = run_nonlinear_baselines(seeds)
    t1 = time.time()
    results["wall_time_sec"] = t1 - t0
    results["mode"] = args.mode

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}  (total {t1-t0:.1f}s)")


if __name__ == "__main__":
    main()
