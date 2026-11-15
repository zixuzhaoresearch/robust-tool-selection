"""Experiments: noisy labels + RobustLinUCB baseline.


  success labels? We flip labels at rate p in {0, 0.05, 0.10, 0.20} and compare
  LinUCB and ADAB-EMA performance degradation on BFCL benign.

  tool-injection attack? We add RobustLinUCB (reward-clipping, Ding 2022 style)
  to the benign and full_adaptive comparisons on BFCL.

Both experiments use TF-IDF features (same as main-paper adversarial results).
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
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.novel_selectors import ADABEmaSelector
from tats.robust_selectors import RobustLinUCBSelector

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _run_env_with_noisy_update(env, selectors, flip_rate: float, seed: int):
    """Run selectors but flip success labels fed to update() at given rate."""
    rng = np.random.default_rng(seed * 10_007 + 7919)
    out: dict[str, float] = {}
    for name, sel in selectors:
        correct = 0
        for q in env.queries:
            tid = sel.select(q, env)
            obs = env.execute(tid, q)
            if obs.success:
                correct += 1
            # Feed potentially-flipped label to the selector's update().
            # Note: the scored TSA still reflects the true success.
            label = obs.success
            if flip_rate > 0.0 and rng.random() < flip_rate:
                label = not label
            if hasattr(sel, "update"):
                sel.update(tid, q.required_capability, label)
        tsa = correct / len(env.queries)
        out[name] = tsa
    return out


def _run_env_plain(env, selectors, mal_ids):
    out: dict[str, dict[str, float]] = {}
    for name, sel in selectors:
        selected, scenarios, correct = [], [], 0
        for q in env.queries:
            tid = sel.select(q, env)
            obs = env.execute(tid, q)
            selected.append(tid)
            scenarios.append(q.required_capability)
            if obs.success:
                correct += 1
            if hasattr(sel, "update"):
                sel.update(tid, q.required_capability, obs.success)
        tsa = correct / len(env.queries)
        asr = compute_asr(selected, scenarios, mal_ids) if mal_ids else 0.0
        out[name] = {"tsa": tsa, "asr": asr}
    return out


def run_noisy_label(seeds: list[int]) -> dict:
    print("=" * 70)
    print("=" * 70)
    bm = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple")
    scenarios = bm.scenarios
    qt = build_query_texts_from_scenarios(scenarios)
    fe = ToolFeatureExtractor(qt)

    flip_rates = [0.0, 0.05, 0.10, 0.20]
    out: dict[str, dict[str, list[float]]] = {}
    for flip in flip_rates:
        print(f"\n--- flip_rate = {flip:.2f} ---")
        per_sel: dict[str, list[float]] = {}
        for seed in seeds:
            env = BFCLEnvironment(scenarios, seed=seed)
            sels = [
                ("LinUCB", LinUCBSelector(fe, seed=seed)),
                ("ADAB-EMA-g2", ADABEmaSelector(
                    fe, seed=seed, gamma=2.0, quantile=0.99, decay=0.99,
                )),
            ]
            r = _run_env_with_noisy_update(env, sels, flip, seed)
            for name, tsa in r.items():
                per_sel.setdefault(name, []).append(tsa)
            print(f"  seed={seed}  LinUCB={r['LinUCB']:.4f}  ADAB={r['ADAB-EMA-g2']:.4f}")
        out[f"flip_{flip:.2f}"] = per_sel
        for name, tsas in sorted(per_sel.items()):
            print(f"  {name}: {np.mean(tsas)*100:.2f} +/- {np.std(tsas)*100:.2f}")
    return out


def run_robust_linucb_baseline(seeds: list[int]) -> dict:
    print("=" * 70)
    print("=" * 70)
    bm = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple")
    scenarios = bm.scenarios
    qt = build_query_texts_from_scenarios(scenarios)
    fe = ToolFeatureExtractor(qt)

    attacks = {"benign": None, "full_adaptive": "full_adaptive"}
    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for atk_name, strategy in attacks.items():
        print(f"\n--- BFCL attack: {atk_name} ---")
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
            sels = [
                ("LinUCB", LinUCBSelector(fe, seed=seed)),
                ("RobustLinUCB-tau0.5", RobustLinUCBSelector(
                    fe, seed=seed, tau=0.5,
                )),
                ("RobustLinUCB-tau0.3", RobustLinUCBSelector(
                    fe, seed=seed, tau=0.3,
                )),
                ("ADAB-EMA-g2", ADABEmaSelector(
                    fe, seed=seed, gamma=2.0, quantile=0.99, decay=0.99,
                )),
            ]
            r = _run_env_plain(env, sels, mal_ids)
            for name, v in r.items():
                per_sel.setdefault(name, {"tsa": [], "asr": []})
                per_sel[name]["tsa"].append(v["tsa"])
                per_sel[name]["asr"].append(v["asr"])
            print(f"  seed={seed} done")
        for name, v in sorted(per_sel.items()):
            tsa_m = np.mean(v["tsa"]) * 100
            asr_m = np.mean(v["asr"]) * 100
            print(f"  {name:<22}  TSA={tsa_m:6.2f}  ASR={asr_m:6.2f}")
        out[atk_name] = per_sel
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", default="results/day3_experiments.json")
    parser.add_argument("--task", choices=["noisy", "robust", "all"], default="all")
    args = parser.parse_args()
    seeds = [0] if args.mode == "smoke" else list(range(10))

    t0 = time.time()
    out: dict = {"mode": args.mode}
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)

    def _save():
        out["wall_time_sec"] = time.time() - t0
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  [checkpoint] saved {out_path}")

    if args.task in ("noisy", "all"):
        out["noisy_label"] = run_noisy_label(seeds)
        _save()
    if args.task in ("robust", "all"):
        out["robust_baseline"] = run_robust_linucb_baseline(seeds)
        _save()
    print(f"\nDone ({out['wall_time_sec']:.1f}s)")


if __name__ == "__main__":
    main()
