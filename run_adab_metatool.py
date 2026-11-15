"""Validate ADAB (Quantile) on MetaTool — cross-benchmark generalization test.

If ADAB only works on BFCL (where we designed the attack), the contribution
is narrow. If it also Pareto-dominates LinUCB on MetaTool (different tool
ecosystem, no parameter schemas), the algorithmic contribution generalizes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from tats.adversarial import AdversarialBFCLEnvironment, compute_asr
from tats.bfcl_selectors import DescriptionMatchSelector, RandomBFCLSelector
from tats.bilints_selector import LinUCBSelector
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.metatool_adapter import MetaToolBenchmark
from tats.novel_selectors import ADABSelector, ADABQuantileSelector, ADABEmaSelector

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
MAX_QUERIES = 2000


def _build_selectors(fe, qt, seed):
    return [
        ("DescMatch", DescriptionMatchSelector(qt, seed=seed)),
        ("LinUCB", LinUCBSelector(fe, seed=seed)),
        ("ADAB-EMA-2.0", ADABEmaSelector(fe, seed=seed, gamma=2.0, quantile=0.99, decay=0.99)),
        ("ADAB-EMA-3.0", ADABEmaSelector(fe, seed=seed, gamma=3.0, quantile=0.99, decay=0.99)),
    ]


def run_benign(seeds):
    print("=" * 70)
    print("MetaTool Benign")
    print("=" * 70)
    bm = MetaToolBenchmark(n_candidates=10, max_queries=MAX_QUERIES,
                            similarity_mode="embedding", seed=42)
    scenarios = bm.scenarios
    qt = build_query_texts_from_scenarios(scenarios)
    fe = ToolFeatureExtractor(qt)
    per_sel = {}
    for seed in seeds:
        env = bm.to_sequential_env()
        rng = np.random.default_rng(seed)
        rng.shuffle(env.queries)
        selectors = _build_selectors(fe, qt, seed)
        for name, sel in selectors:
            selected, correct = [], 0
            for q in env.queries:
                tid = sel.select(q, env)
                obs = env.execute(tid, q)
                selected.append(tid)
                if obs.success:
                    correct += 1
                if hasattr(sel, "update"):
                    sel.update(tid, q.required_capability, obs.success)
            tsa = correct / len(env.queries)
            per_sel.setdefault(name, []).append(tsa)
            print(f"  seed={seed} {name:<14} TSA={tsa:.4f}")
    return per_sel


def run_attack(seeds, strategy):
    print("=" * 70)
    print(f"MetaTool Adversarial ({strategy})")
    print("=" * 70)
    bm = MetaToolBenchmark(n_candidates=10, max_queries=MAX_QUERIES,
                            similarity_mode="embedding", seed=42)
    scenarios = bm.scenarios
    qt = build_query_texts_from_scenarios(scenarios)
    fe = ToolFeatureExtractor(qt)
    per_sel = {}
    for seed in seeds:
        env = AdversarialBFCLEnvironment(scenarios, strategy=strategy, seed=seed)
        mal_ids = {sc: mid for sc, mid in env._mal_tool_ids.items()}
        selectors = _build_selectors(fe, qt, seed)
        for name, sel in selectors:
            selected_ids, scenario_ids, correct = [], [], 0
            for q in env.queries:
                tid = sel.select(q, env)
                obs = env.execute(tid, q)
                selected_ids.append(tid)
                scenario_ids.append(q.required_capability)
                if obs.success:
                    correct += 1
                if hasattr(sel, "update"):
                    sel.update(tid, q.required_capability, obs.success)
            tsa = correct / len(env.queries)
            asr = compute_asr(selected_ids, scenario_ids, mal_ids)
            per_sel.setdefault(name, {"tsa": [], "asr": []})
            per_sel[name]["tsa"].append(tsa)
            per_sel[name]["asr"].append(asr)
            print(f"  seed={seed} {name:<14} TSA={tsa:.4f} ASR={asr:.4f}")
    return per_sel


def summarize(per_sel, is_attack=False, title=""):
    print(f"\n--- {title} ---")
    if is_attack:
        for name, v in sorted(per_sel.items()):
            tsa_m = np.mean(v["tsa"]) * 100
            tsa_s = np.std(v["tsa"]) * 100
            asr_m = np.mean(v["asr"]) * 100
            asr_s = np.std(v["asr"]) * 100
            print(f"  {name:<14} TSA={tsa_m:5.2f}±{tsa_s:.2f}  ASR={asr_m:5.2f}±{asr_s:.2f}")
    else:
        for name, vals in sorted(per_sel.items()):
            m, s = np.mean(vals) * 100, np.std(vals) * 100
            print(f"  {name:<14} TSA={m:5.2f}±{s:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", default="results/adab_metatool.json")
    args = parser.parse_args()
    seeds = [0] if args.mode == "smoke" else list(range(10))

    t0 = time.time()
    results = {}

    benign = run_benign(seeds)
    summarize(benign, is_attack=False, title="MetaTool Benign")
    results["benign"] = benign

    for strategy in ["desc_copy", "full_adaptive"]:
        adv = run_attack(seeds, strategy)
        summarize(adv, is_attack=True, title=f"MetaTool Adv ({strategy})")
        results[f"adv_{strategy}"] = adv

    results["wall_time_sec"] = time.time() - t0
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ({results['wall_time_sec']:.1f}s)")


if __name__ == "__main__":
    main()
