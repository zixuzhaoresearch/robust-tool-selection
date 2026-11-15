"""ADAB-Dense adversarial experiment.

Question: Does ADAB's adversarial robustness transfer to the dense-embedding
feature regime, or is it coupled to TF-IDF?

adversarial results use TF-IDF. If ADAB-Dense maintains Pareto improvement over
LinUCB-Dense under attack, it directly kills the "only helps in TF-IDF" limitation.

Experimental design:
  Features:  TF-IDF (reference) and Dense (MiniLM-L6-v2)
  Datasets:  BFCL live_multiple and MetaTool
  Attacks:   benign, desc_copy, full_adaptive
  γ sweep:   {1.0, 2.0, 3.0, 4.0}
  Seeds:     10 (full mode) or [0] (smoke)

Output: results/adab_dense_adv.json with per-(dataset, attack, method, γ) TSA + ASR.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tats.adversarial import AdversarialBFCLEnvironment, compute_asr
from tats.bfcl_adapter import BFCLEnvironment, BFCLToolSelectionBenchmark
from tats.bilints_selector import LinUCBSelector
from tats.dense_features import DenseEmbedder, DenseFeatureExtractor
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.metatool_adapter import MetaToolBenchmark
from tats.novel_selectors import ADABEmaSelector

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
GAMMAS = [1.0, 2.0, 3.0, 4.0]
MT_MAX_QUERIES = 2000


def _precompute_descs(scenarios, embedder: DenseEmbedder) -> None:
    all_descs: set[str] = set()
    for s in scenarios:
        for t in s.candidate_tools:
            all_descs.add(t.description)
    if all_descs:
        embedder.encode_batch(list(all_descs))


def _build_selectors(fe_tfidf, fe_dense, seed: int):
    sels: list[tuple[str, object]] = []
    sels.append(("LinUCB", LinUCBSelector(fe_tfidf, seed=seed)))
    sels.append(("LinUCB-Dense", LinUCBSelector(fe_dense, seed=seed)))
    for g in GAMMAS:
        sels.append((f"ADAB-EMA-g{g}", ADABEmaSelector(
            fe_tfidf, seed=seed, gamma=g, quantile=0.99, decay=0.99,
        )))
        sels.append((f"ADAB-Dense-g{g}", ADABEmaSelector(
            fe_dense, seed=seed, gamma=g, quantile=0.99, decay=0.99,
        )))
    return sels


def _run_env(env, selectors, mal_ids_by_scenario: dict | None):
    """Run every selector through env.queries; return per-selector TSA + ASR."""
    results: dict[str, dict[str, float]] = {}
    for name, sel in selectors:
        selected_ids: list[str] = []
        scenario_ids: list[str] = []
        correct = 0
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
        asr = compute_asr(selected_ids, scenario_ids, mal_ids_by_scenario) if mal_ids_by_scenario else 0.0
        results[name] = {"tsa": tsa, "asr": asr}
    return results


def run_bfcl(seeds: list[int], embedder: DenseEmbedder) -> dict:
    print("=" * 70)
    print("BFCL (live_multiple): ADAB-Dense adversarial sweep")
    print("=" * 70)
    bm = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple")
    scenarios = bm.scenarios
    qt = build_query_texts_from_scenarios(scenarios)
    fe_tfidf = ToolFeatureExtractor(qt)
    _precompute_descs(scenarios, embedder)
    fe_dense = DenseFeatureExtractor(qt, embedder)

    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    attacks = {
        "benign": None,
        "desc_copy": "desc_copy",
        "full_adaptive": "full_adaptive",
    }
    for attack_name, strategy in attacks.items():
        print(f"\n--- BFCL attack: {attack_name} ---")
        per_sel: dict[str, dict[str, list[float]]] = {}
        for seed in seeds:
            if strategy is None:
                env = BFCLEnvironment(scenarios, seed=seed)
                mal_ids = None
            else:
                env = AdversarialBFCLEnvironment(
                    scenarios, strategy=strategy, seed=seed,
                )
                mal_ids = {sc: mid for sc, mid in env._mal_tool_ids.items()}
                # Pre-compute dense embeddings for malicious tool descs.
                mal_descs = {t.description for t in env.tools}
                if mal_descs:
                    embedder.encode_batch(list(mal_descs))
            sels = _build_selectors(fe_tfidf, fe_dense, seed)
            r = _run_env(env, sels, mal_ids)
            for name, v in r.items():
                per_sel.setdefault(name, {"tsa": [], "asr": []})
                per_sel[name]["tsa"].append(v["tsa"])
                per_sel[name]["asr"].append(v["asr"])
            print(f"  seed={seed} done (bfcl/{attack_name})")
        _summarize(per_sel)
        out[attack_name] = per_sel
    return out


def run_metatool(seeds: list[int], embedder: DenseEmbedder) -> dict:
    print("=" * 70)
    print("MetaTool: ADAB-Dense adversarial sweep")
    print("=" * 70)
    bm = MetaToolBenchmark(
        n_candidates=10, max_queries=MT_MAX_QUERIES,
        similarity_mode="embedding", seed=42,
    )
    scenarios = bm.scenarios
    qt = build_query_texts_from_scenarios(scenarios)
    fe_tfidf = ToolFeatureExtractor(qt)
    _precompute_descs(scenarios, embedder)
    fe_dense = DenseFeatureExtractor(qt, embedder)

    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    attacks = {
        "benign": None,
        "desc_copy": "desc_copy",
        "full_adaptive": "full_adaptive",
    }
    for attack_name, strategy in attacks.items():
        print(f"\n--- MetaTool attack: {attack_name} ---")
        per_sel: dict[str, dict[str, list[float]]] = {}
        for seed in seeds:
            if strategy is None:
                env = bm.to_sequential_env()
                rng = np.random.default_rng(seed)
                rng.shuffle(env.queries)
                mal_ids = None
            else:
                env = AdversarialBFCLEnvironment(
                    scenarios, strategy=strategy, seed=seed,
                )
                mal_ids = {sc: mid for sc, mid in env._mal_tool_ids.items()}
                mal_descs = {t.description for t in env.tools}
                if mal_descs:
                    embedder.encode_batch(list(mal_descs))
            sels = _build_selectors(fe_tfidf, fe_dense, seed)
            r = _run_env(env, sels, mal_ids)
            for name, v in r.items():
                per_sel.setdefault(name, {"tsa": [], "asr": []})
                per_sel[name]["tsa"].append(v["tsa"])
                per_sel[name]["asr"].append(v["asr"])
            print(f"  seed={seed} done")
        _summarize(per_sel)
        out[attack_name] = per_sel
    return out


def _summarize(per_sel: dict[str, dict[str, list[float]]]) -> None:
    print(f"    {'Method':<22} {'TSA':>12} {'ASR':>12}")
    for name in sorted(per_sel):
        v = per_sel[name]
        tsa_m = np.mean(v["tsa"]) * 100
        tsa_s = np.std(v["tsa"]) * 100
        asr_m = np.mean(v["asr"]) * 100
        asr_s = np.std(v["asr"]) * 100
        print(f"    {name:<22}  {tsa_m:6.2f}\u00b1{tsa_s:4.2f}  {asr_m:6.2f}\u00b1{asr_s:4.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", default="results/adab_dense_adv.json")
    parser.add_argument("--datasets", default="bfcl,metatool",
                        help="comma-separated: bfcl, metatool")
    args = parser.parse_args()

    seeds = [0] if args.mode == "smoke" else list(range(10))
    t0 = time.time()

    print(f"Initializing dense embedder (seeds={seeds})...")
    embedder = DenseEmbedder("all-MiniLM-L6-v2")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    datasets = set(args.datasets.split(","))
    results: dict = {"mode": args.mode, "gammas": GAMMAS}
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)

    def _save():
        results["wall_time_sec"] = time.time() - t0
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [checkpoint] saved {out_path} ({results['wall_time_sec']:.1f}s)")

    if "bfcl" in datasets:
        results["bfcl"] = run_bfcl(seeds, embedder)
        _save()
    if "metatool" in datasets:
        results["metatool"] = run_metatool(seeds, embedder)
        _save()
    print(f"\nDone ({results['wall_time_sec']:.1f}s)")


if __name__ == "__main__":
    main()
