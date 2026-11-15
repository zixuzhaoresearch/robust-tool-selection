"""10-seed source for Table 8 of the paper: ADAB-EMA gamma sweep on BFCL live_multiple.

Selectors: LinUCB, ADAB-EMA gamma in {0.5, 1, 2, 3, 5}, ADAB-Mahalanobis gamma_M = 0.3.
Attacks:   benign, param_steal, full_adaptive (L1, query-word knowledge only).
Output:    results/tab8_gamma_sweep_10seed.json with per-seed TSA/ASR and mean +- sample SD.
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
from tats.novel_selectors import ADABEmaSelector, ADABSelector

GAMMAS = [0.5, 1.0, 2.0, 3.0, 5.0]
ATTACKS = {"benign": None, "param_steal": "param_steal", "full_adaptive": "full_adaptive"}


def build_selectors(fe, seed):
    sels = [("LinUCB", LinUCBSelector(fe, seed=seed))]
    for g in GAMMAS:
        sels.append((f"ADAB-EMA-g{g:g}", ADABEmaSelector(fe, seed=seed, gamma=g, quantile=0.99, decay=0.99)))
    sels.append(("ADAB-Mahalanobis-g0.3", ADABSelector(fe, seed=seed, gamma=0.3)))
    return sels


def run_env(env, selectors, mal_ids):
    out = {}
    for name, sel in selectors:
        selected, scen, correct = [], [], 0
        for q in env.queries:
            tid = sel.select(q, env)
            obs = env.execute(tid, q)
            selected.append(tid)
            scen.append(q.required_capability)
            correct += obs.success
            if hasattr(sel, "update"):
                sel.update(tid, q.required_capability, obs.success)
        out[name] = {"tsa": correct / len(env.queries),
                     "asr": compute_asr(selected, scen, mal_ids) if mal_ids else 0.0}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "full"], default="full")
    p.add_argument("--output", default="results/tab8_gamma_sweep_10seed.json")
    args = p.parse_args()
    seeds = [0] if args.mode == "smoke" else list(range(10))
    t0 = time.time()
    bm = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple")
    scenarios = bm.scenarios
    fe = ToolFeatureExtractor(build_query_texts_from_scenarios(scenarios))
    results = {"mode": args.mode, "gammas": GAMMAS, "seeds": seeds, "dataset": "bfcl_live_multiple", "per_seed": {}, "aggregate": {}}
    for attack, strategy in ATTACKS.items():
        per_sel = {}
        for seed in seeds:
            if strategy is None:
                env, mal = BFCLEnvironment(scenarios, seed=seed), None
            else:
                env = AdversarialBFCLEnvironment(scenarios, strategy=strategy, seed=seed)
                mal = dict(env._mal_tool_ids)
            for name, v in run_env(env, build_selectors(fe, seed), mal).items():
                per_sel.setdefault(name, {"tsa": [], "asr": []})
                per_sel[name]["tsa"].append(v["tsa"])
                per_sel[name]["asr"].append(v["asr"])
            print(f"  {attack} seed={seed} done ({time.time()-t0:.0f}s)", flush=True)
        results["per_seed"][attack] = per_sel
        results["aggregate"][attack] = {
            n: {k: {"mean": float(np.mean(v[k]) * 100), "std": float(np.std(v[k], ddof=1) * 100) if len(v[k]) > 1 else 0.0}
                for k in ("tsa", "asr")} for n, v in per_sel.items()}
        for n, a in results["aggregate"][attack].items():
            print(f"{attack:14s} {n:24s} TSA {a['tsa']['mean']:.1f}±{a['tsa']['std']:.1f}  ASR {a['asr']['mean']:.2f}±{a['asr']['std']:.2f}", flush=True)
    results["wall_time_sec"] = time.time() - t0
    Path(args.output).parent.mkdir(exist_ok=True)
    json.dump(results, open(args.output, "w"), indent=2)
    print(f"saved {args.output} ({results['wall_time_sec']:.0f}s)")


if __name__ == "__main__":
    main()
