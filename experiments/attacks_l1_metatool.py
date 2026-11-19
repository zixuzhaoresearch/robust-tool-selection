"""Adversarial robustness on MetaTool (cross-benchmark attack eval).

MetaTool tools have NO parameter schema — only name + description. This tests
whether the bandit defense generalizes when schema features are unavailable.
Expected: partial defense via name_sim; weaker than BFCL but still positive.
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

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
MAX_QUERIES = 2000


def evaluate(selectors, env, seed):
    mal_ids = {sc_id: mid for sc_id, mid in env._mal_tool_ids.items()}
    rows = []
    for name, selector in selectors:
        selected_ids, scenario_ids, correct = [], [], 0
        for query in env.queries:
            tool_id = selector.select(query, env)
            obs = env.execute(tool_id, query)
            selected_ids.append(tool_id)
            scenario_ids.append(query.required_capability)
            if obs.success:
                correct += 1
            if hasattr(selector, "update"):
                selector.update(tool_id, query.required_capability, obs.success)
        rows.append({
            "name": name,
            "tsa": correct / len(env.queries),
            "asr": compute_asr(selected_ids, scenario_ids, mal_ids),
        })
        print(f"  seed={seed} {rows[-1]['name']:<12} "
              f"TSA={rows[-1]['tsa']:.4f}  ASR={rows[-1]['asr']:.4f}")
    return rows


def run(seeds, strategy="desc_copy"):
    print("=" * 70)
    print(f"Adversarial MetaTool: strategy={strategy}")
    print("=" * 70)

    # Load MetaTool benchmark (using embedding mode — harder for attacker since
    # candidates are already similar to GT)
    bm = MetaToolBenchmark(
        n_candidates=10, max_queries=MAX_QUERIES,
        similarity_mode="embedding", seed=42,
    )
    scenarios = bm.scenarios
    print(f"Loaded {len(scenarios)} MetaTool scenarios")

    qt = build_query_texts_from_scenarios(scenarios)
    fe = ToolFeatureExtractor(qt)

    per_seed: dict[str, dict[str, list[float]]] = {}

    for seed in seeds:
        # Wrap MetaTool scenarios with adversarial injection
        env = AdversarialBFCLEnvironment(
            scenarios, strategy=strategy, inject_schema=False, seed=seed
        )

        selectors = [
            ("Random", RandomBFCLSelector(seed=seed)),
            ("DescMatch", DescriptionMatchSelector(qt, seed=seed)),
            ("LinUCB", LinUCBSelector(fe, seed=seed)),
        ]
        rows = evaluate(selectors, env, seed)
        for r in rows:
            per_seed.setdefault(r["name"], {"tsa": [], "asr": []})
            per_seed[r["name"]]["tsa"].append(r["tsa"])
            per_seed[r["name"]]["asr"].append(r["asr"])

    print(f"\n=== MetaTool Adversarial Aggregated ({strategy}) ===")
    print(f"{'Method':<12} {'TSA':>14} {'ASR':>14}")
    for name, vals in per_seed.items():
        tsa_m = np.mean(vals["tsa"]) * 100
        tsa_s = np.std(vals["tsa"]) * 100
        asr_m = np.mean(vals["asr"]) * 100
        asr_s = np.std(vals["asr"]) * 100
        print(f"{name:<12}  {tsa_m:6.2f}±{tsa_s:4.2f}  {asr_m:6.2f}±{asr_s:4.2f}")

    print("\nSignificance (ASR reduction vs DescMatch):")
    ref = per_seed.get("DescMatch", {}).get("asr", [])
    for name in ["LinUCB"]:
        scores = per_seed.get(name, {}).get("asr", [])
        if len(scores) != len(ref) or not ref:
            continue
        t_stat, p_val = stats.ttest_rel(scores, ref)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        delta = np.mean(scores) - np.mean(ref)
        print(f"  {name}:  ΔASR={delta*100:+6.2f}pp  t={t_stat:.2f}  p={p_val:.4g} {sig}")

    return per_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", default="results/adversarial_metatool.json")
    args = parser.parse_args()

    seeds = [0] if args.mode == "smoke" else list(range(10))
    t0 = time.time()
    result = {"per_seed": run(seeds), "wall_time_sec": 0.0}
    result["wall_time_sec"] = time.time() - t0
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}  ({result['wall_time_sec']:.1f}s)")


if __name__ == "__main__":
    main()
