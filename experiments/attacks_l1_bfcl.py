"""Experiment 1: Adversarial robustness (ToolHijacker-simplified attack).

Evaluates how well each selector resists ToolHijacker-style description-based
attacks, where the attacker injects a malicious tool with crafted description
into the candidate set.

Usage:
  uv run python attacks_l1_bfcl.py --mode smoke
  uv run python attacks_l1_bfcl.py --mode full
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from tats.adversarial import AdversarialBFCLEnvironment, compute_asr
from tats.bfcl_adapter import BFCLToolSelectionBenchmark
from tats.bfcl_selectors import (
    DescriptionMatchSelector,
    ParameterMatchSelector,
    RandomBFCLSelector,
)
from tats.bilints_selector import LinUCBSelector
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate_under_attack(
    selectors: list[tuple[str, object]],
    env: AdversarialBFCLEnvironment,
    seed: int,
) -> list[dict]:
    """Run each selector through the adversarial env; return per-selector metrics."""
    results = []
    mal_ids = {sc_id: mid for sc_id, mid in env._mal_tool_ids.items()}

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
        results.append({"name": name, "tsa": tsa, "asr": asr})
        print(f"  seed={seed} {name:<18}  TSA={tsa:.4f}  ASR={asr:.4f}")

    return results


def run_attack_experiment(
    seeds: list[int],
    strategy: str = "desc_copy",
    inject_schema: bool = False,
) -> dict:
    """Run all methods under the specified attack strategy."""
    print("=" * 70)
    print(
        f"Adversarial Experiment: strategy={strategy}, "
        f"schema_injected={inject_schema}"
    )
    print("=" * 70)

    bm = BFCLToolSelectionBenchmark(
        bfcl_eval_package=True, dataset="live_multiple"
    )
    scenarios = bm.scenarios
    print(f"Loaded {len(scenarios)} scenarios from live_multiple")

    qt = build_query_texts_from_scenarios(scenarios)

    # Feature extractor must see the augmented tool universe (including malicious
    # tools) to compute features for them. Build fe after seeing one env.
    # Since malicious tools have synthetic descriptions from query_text, the fe
    # built from original scenarios is fine (it only indexes query corpus, not
    # tool descriptions, for IDF).
    fe = ToolFeatureExtractor(qt)

    all_rows: list[dict] = []

    for seed in seeds:
        env = AdversarialBFCLEnvironment(
            scenarios, strategy=strategy, inject_schema=inject_schema, seed=seed
        )

        selectors = [
            ("Random", RandomBFCLSelector(seed=seed)),
            ("DescMatch", DescriptionMatchSelector(qt, seed=seed)),
            ("ParamMatch", ParameterMatchSelector(qt, seed=seed)),
            ("LinUCB", LinUCBSelector(fe, seed=seed)),
        ]

        seed_results = evaluate_under_attack(selectors, env, seed)
        for r in seed_results:
            r["seed"] = seed
            r["strategy"] = strategy
            r["inject_schema"] = inject_schema
            all_rows.append(r)

    # Aggregate
    agg: dict[str, dict[str, float]] = {}
    per_seed: dict[str, dict[str, list[float]]] = {}
    for row in all_rows:
        name = row["name"]
        per_seed.setdefault(name, {"tsa": [], "asr": []})
        per_seed[name]["tsa"].append(row["tsa"])
        per_seed[name]["asr"].append(row["asr"])

    for name, vals in per_seed.items():
        agg[name] = {
            "tsa_mean": float(np.mean(vals["tsa"])),
            "tsa_std": float(np.std(vals["tsa"])),
            "asr_mean": float(np.mean(vals["asr"])),
            "asr_std": float(np.std(vals["asr"])),
            "n_seeds": len(vals["tsa"]),
        }

    print(f"\n=== Aggregated ({strategy}, schema={inject_schema}) ===")
    print(f"{'Method':<18} {'TSA':>12} {'ASR':>12}")
    for name, s in sorted(agg.items()):
        print(
            f"{name:<18}  {s['tsa_mean']*100:6.2f}±{s['tsa_std']*100:4.2f}  "
            f"{s['asr_mean']*100:6.2f}±{s['asr_std']*100:4.2f}"
        )

    # Paired t-test ASR vs DescMatch (baseline being attacked)
    print("\nSignificance of ASR reduction vs DescMatch:")
    ref = per_seed.get("DescMatch", {}).get("asr", [])
    for name in ["ParamMatch", "LinUCB"]:
        scores = per_seed.get(name, {}).get("asr", [])
        if len(scores) != len(ref) or not ref:
            continue
        t_stat, p_val = stats.ttest_rel(scores, ref)
        sig = (
            "***" if p_val < 0.001
            else "**" if p_val < 0.01
            else "*" if p_val < 0.05
            else "ns"
        )
        delta = np.mean(scores) - np.mean(ref)
        print(
            f"  {name:<18} delta_ASR={delta*100:+6.2f} pp  "
            f"t={t_stat:5.2f}  p={p_val:.4g} {sig}"
        )

    return {
        "config": {"strategy": strategy, "inject_schema": inject_schema},
        "per_seed": per_seed,
        "aggregate": agg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--strategy",
        choices=[
            "desc_copy", "desc_stuff", "name_stuff",
            "param_generic", "param_steal", "full_adaptive",
        ],
        default="desc_copy",
    )
    parser.add_argument(
        "--inject-schema",
        action="store_true",
        help="Also plant a minimal fake schema in the malicious tool.",
    )
    parser.add_argument("--output", default="results/adversarial_results.json")
    args = parser.parse_args()

    seeds = [0] if args.mode == "smoke" else list(range(10))

    t0 = time.time()
    result = run_attack_experiment(seeds, args.strategy, args.inject_schema)
    result["wall_time_sec"] = time.time() - t0
    result["mode"] = args.mode

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}  ({result['wall_time_sec']:.1f}s)")


if __name__ == "__main__":
    main()
