"""Experiment: Novel algorithmic variants (TW-LinUCB, CGB) vs LinUCB.

Evaluates two novel algorithmic contributions:
  - TW-LinUCB: Trust-Weighted UCB (asymmetric exploration per feature group)
  - CGB: Consistency-Gated Bandit (runtime adversarial detection)

Against standard LinUCB across three regimes:
  - Benign: BFCL live_multiple (no attack)
  - Adversarial: 4 attack strategies (desc_copy, param_generic, param_steal, full_adaptive)
  - Cross-benchmark: MetaTool (schema-absent)

Usage:
  uv run python adab_design_ablation.py --mode smoke
  uv run python adab_design_ablation.py --mode full
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from tats.adversarial import AdversarialBFCLEnvironment, compute_asr
from tats.bfcl_adapter import BFCLEnvironment, BFCLToolSelectionBenchmark
from tats.bfcl_selectors import DescriptionMatchSelector
from tats.bilints_selector import LinUCBSelector
from tats.experiment import run_experiment
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.novel_selectors import (
    TWLinUCBSelector, ConsistencyGatedSelector, ADABSelector,
    ADABWarmSelector, ADABQuantileSelector, CommitteeLinUCBSelector,
    HybridADABSelector, OutlierAbstainLinUCB,
    ADABEmaSelector, ADABPerFeatureGamma, AdaptiveGammaADABSelector,
    SupervisedOODSelector,
    ADABMaxExceedance, ADABSumExceedance, ADABAllSelections,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _build_selectors(fe: ToolFeatureExtractor, qt: dict, seed: int):
    """Four selectors: DescMatch + LinUCB + 2 TW variants + CGB.

    TW-LinUCB-HS: 'High schema' — aggressive schema exploration (fails under adaptive).
    TW-LinUCB-LS: 'Low schema'  — conservative schema, standard desc (hypothesis).
    """
    return [
        ("LinUCB", LinUCBSelector(fe, seed=seed)),
        ("EMA-count-2.0", ADABEmaSelector(fe, seed=seed, gamma=2.0, quantile=0.99, decay=0.99)),
        ("EMA-max-2.0", ADABMaxExceedance(fe, seed=seed, gamma=2.0, quantile=0.99, decay=0.99)),
        ("EMA-sum-2.0", ADABSumExceedance(fe, seed=seed, gamma=2.0, quantile=0.99, decay=0.99)),
        ("EMA-all-2.0", ADABAllSelections(fe, seed=seed, gamma=2.0, quantile=0.99, decay=0.99)),
    ]


def eval_benign(seeds: list[int]) -> dict:
    print("\n" + "=" * 70)
    print("Benign evaluation on BFCL live_multiple")
    print("=" * 70)
    bm = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple")
    scenarios = bm.scenarios
    qt = build_query_texts_from_scenarios(scenarios)
    fe = ToolFeatureExtractor(qt)

    per_sel: dict[str, list[float]] = {}
    for seed in seeds:
        env = BFCLEnvironment(scenarios, seed=seed)
        selectors = _build_selectors(fe, qt, seed)
        results = run_experiment(selectors, env, seed)
        for r in results:
            per_sel.setdefault(r.name, []).append(r.tsa)
            print(f"  seed={seed}  {r.name:<12}  TSA={r.tsa:.4f}")
    return per_sel


def eval_attack(seeds: list[int], strategy: str) -> dict:
    print("\n" + "=" * 70)
    print(f"Adversarial ({strategy}) on BFCL live_multiple")
    print("=" * 70)
    bm = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple")
    scenarios = bm.scenarios
    qt = build_query_texts_from_scenarios(scenarios)
    fe = ToolFeatureExtractor(qt)

    per_sel: dict[str, dict[str, list[float]]] = {}
    for seed in seeds:
        env = AdversarialBFCLEnvironment(scenarios, strategy=strategy, seed=seed)
        mal_ids = {sc_id: mid for sc_id, mid in env._mal_tool_ids.items()}
        selectors = _build_selectors(fe, qt, seed)
        for name, selector in selectors:
            selected_ids: list[str] = []
            scenario_ids: list[str] = []
            correct = 0
            for query in env.queries:
                tool_id = selector.select(query, env)
                obs = env.execute(tool_id, query)
                selected_ids.append(tool_id)
                scenario_ids.append(query.required_capability)
                if obs.success:
                    correct += 1
                if hasattr(selector, "update"):
                    selector.update(tool_id, query.required_capability, obs.success)
            tsa = correct / len(env.queries)
            asr = compute_asr(selected_ids, scenario_ids, mal_ids)
            per_sel.setdefault(name, {"tsa": [], "asr": []})
            per_sel[name]["tsa"].append(tsa)
            per_sel[name]["asr"].append(asr)
            print(f"  seed={seed} {name:<12} TSA={tsa:.4f}  ASR={asr:.4f}")
    return per_sel


def summarize_benign(per_sel: dict, title: str) -> dict:
    print(f"\n--- {title} ---")
    print(f"{'Method':<12} {'TSA':>18}")
    agg: dict[str, dict[str, float]] = {}
    for name, vals in sorted(per_sel.items()):
        m, s = float(np.mean(vals)), float(np.std(vals))
        agg[name] = {"tsa_mean": m, "tsa_std": s, "n_seeds": len(vals)}
        print(f"{name:<12}  {m*100:6.2f}±{s*100:4.2f}")
    ref = per_sel.get("LinUCB", [])
    if ref and len(ref) > 1:
        print("\nvs LinUCB (paired t):")
        for name in ["TW-LinUCB", "CGB"]:
            if name not in per_sel or len(per_sel[name]) != len(ref):
                continue
            t, p = stats.ttest_rel(per_sel[name], ref)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            dmean = np.mean(per_sel[name]) - np.mean(ref)
            print(f"  {name:<12}  delta={dmean*100:+.2f}pp  t={t:.2f}  p={p:.4g} {sig}")
    return agg


def summarize_attack(per_sel: dict, title: str) -> dict:
    print(f"\n--- {title} ---")
    print(f"{'Method':<12} {'TSA':>14} {'ASR':>14}")
    agg: dict[str, dict[str, float]] = {}
    for name, vals in sorted(per_sel.items()):
        tsa_m, tsa_s = float(np.mean(vals["tsa"])), float(np.std(vals["tsa"]))
        asr_m, asr_s = float(np.mean(vals["asr"])), float(np.std(vals["asr"]))
        agg[name] = {
            "tsa_mean": tsa_m, "tsa_std": tsa_s,
            "asr_mean": asr_m, "asr_std": asr_s,
            "n_seeds": len(vals["tsa"]),
        }
        print(f"{name:<12}  {tsa_m*100:6.2f}±{tsa_s*100:4.2f}  "
              f"{asr_m*100:6.2f}±{asr_s*100:4.2f}")
    ref = per_sel.get("LinUCB", {}).get("asr", [])
    if ref and len(ref) > 1:
        print("\nASR vs LinUCB (paired t):")
        for name in ["TW-LinUCB", "CGB"]:
            if name not in per_sel:
                continue
            vals = per_sel[name]["asr"]
            if len(vals) != len(ref):
                continue
            t, p = stats.ttest_rel(vals, ref)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            d = np.mean(vals) - np.mean(ref)
            print(f"  {name:<12}  ΔASR={d*100:+.2f}pp  t={t:.2f}  p={p:.4g} {sig}")
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", default="results/novel_selectors.json")
    args = parser.parse_args()

    seeds = [0] if args.mode == "smoke" else list(range(10))
    t0 = time.time()

    results = {"config": {"seeds": seeds}, "benign": {}, "adversarial": {}}

    benign = eval_benign(seeds)
    results["benign"] = {
        "per_seed": benign,
        "aggregate": summarize_benign(benign, "Benign BFCL"),
    }

    for strategy in ["desc_copy", "param_steal", "full_adaptive"]:
        adv = eval_attack(seeds, strategy)
        results["adversarial"][strategy] = {
            "per_seed": adv,
            "aggregate": summarize_attack(adv, f"Adversarial ({strategy})"),
        }

    results["wall_time_sec"] = time.time() - t0
    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}  (total {results['wall_time_sec']:.1f}s)")


if __name__ == "__main__":
    main()
