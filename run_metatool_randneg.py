"""Experiment 3: MetaTool with random negatives (debias test).

similarity to the ground-truth tool, which favors dense retrieval methods.
This experiment re-runs key selectors with RANDOM distractors to test
whether our method's gains persist when candidate construction doesn't
advantage embedding-based baselines.

Usage:
  uv run python run_metatool_randneg.py --mode smoke   # 1 seed
  uv run python run_metatool_randneg.py --mode full    # 10 seeds
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from tats.bfcl_selectors import (
    DescriptionMatchSelector,
    RandomBFCLSelector,
    ParameterMatchSelector,
)
from tats.bilints_selector import LinUCBSelector
from tats.experiment import run_experiment
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.metatool_adapter import MetaToolBenchmark

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
MAX_QUERIES = 2000


def run_comparison(seeds: list[int], n_candidates: int = 10) -> dict:
    """Compare embedding vs random candidate construction on MetaTool."""
    print("=" * 70)
    print(f"Exp 3: MetaTool candidate-construction comparison (K={n_candidates})")
    print("=" * 70)

    per_mode: dict[str, dict[str, list[float]]] = {}

    for mode in ["embedding", "random"]:
        print(f"\n--- Candidate construction: {mode} ---")
        bm = MetaToolBenchmark(
            n_candidates=n_candidates,
            max_queries=MAX_QUERIES,
            similarity_mode=mode,
            seed=42,
        )
        scenarios = bm.scenarios
        qt = build_query_texts_from_scenarios(scenarios)
        fe = ToolFeatureExtractor(qt)

        mode_results: dict[str, list[float]] = {}

        for seed in seeds:
            env = bm.to_sequential_env()
            # reshuffle queries per seed
            rng = np.random.default_rng(seed)
            rng.shuffle(env.queries)

            selectors = [
                ("Random", RandomBFCLSelector(seed=seed)),
                ("DescMatch", DescriptionMatchSelector(qt, seed=seed)),
                ("LinUCB", LinUCBSelector(fe, seed=seed)),
            ]
            results = run_experiment(selectors, env, seed)
            for r in results:
                mode_results.setdefault(r.name, []).append(r.tsa)
                print(f"  seed={seed} mode={mode:<10} {r.name:<12} TSA={r.tsa:.4f}")

        per_mode[mode] = mode_results

    # Aggregated comparison
    print("\n=== MetaTool K=10: embedding-vs-random candidate construction ===")
    print(f"{'Method':<12} {'Embedding':>16} {'Random':>16} {'Δ':>10}")
    for name in ["Random", "DescMatch", "LinUCB"]:
        emb = per_mode["embedding"].get(name, [])
        rnd = per_mode["random"].get(name, [])
        if not emb or not rnd:
            continue
        emb_m, emb_s = np.mean(emb), np.std(emb)
        rnd_m, rnd_s = np.mean(rnd), np.std(rnd)
        delta = rnd_m - emb_m
        print(
            f"{name:<12}  {emb_m*100:6.2f}±{emb_s*100:4.2f}  "
            f"{rnd_m*100:6.2f}±{rnd_s*100:4.2f}  {delta*100:+6.2f}"
        )

    # Paired t-test for LinUCB's relative advantage over DescMatch
    print("\nLinUCB-vs-DescMatch gap (LinUCB − DescMatch):")
    for mode in ["embedding", "random"]:
        lin = per_mode[mode].get("LinUCB", [])
        dm = per_mode[mode].get("DescMatch", [])
        if len(lin) != len(dm) or not lin:
            continue
        gaps = np.array(lin) - np.array(dm)
        t_stat, p_val = stats.ttest_1samp(gaps, 0)
        sig = (
            "***" if p_val < 0.001
            else "**" if p_val < 0.01
            else "*" if p_val < 0.05
            else "ns"
        )
        print(
            f"  {mode:<10}  gap={gaps.mean()*100:+6.2f}pp  "
            f"std={gaps.std()*100:.2f}  t={t_stat:.2f}  p={p_val:.4g} {sig}"
        )

    return {"per_mode": per_mode, "n_candidates": n_candidates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", default="results/metatool_randneg.json")
    args = parser.parse_args()

    seeds = [0] if args.mode == "smoke" else list(range(10))

    t0 = time.time()
    result = run_comparison(seeds)
    result["wall_time_sec"] = time.time() - t0

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}  ({result['wall_time_sec']:.1f}s)")


if __name__ == "__main__":
    main()
