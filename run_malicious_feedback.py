"""Feedback-manipulation stress test for ADAB (Table A6 of the paper).

The injected tool reports success with probability p (a malicious tool that
sometimes completes the task, or an attacker who manipulates the observed
outcome), so its feature vector can enter ADAB's success-filtered buffer.
TSA counts ground-truth selections; ASR counts injected-tool selections.
BFCL live_multiple, TF-IDF features, 10 seeds.
Output: results/malicious_feedback_10seed.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tats.adversarial import AdversarialBFCLEnvironment
from tats.bfcl_adapter import BFCLToolSelectionBenchmark
from tats.bilints_selector import LinUCBSelector
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.models import Observation
from tats.novel_selectors import ADABEmaSelector

PS = [0.0, 0.1, 0.3, 0.5, 1.0]
ATTACKS = ["full_adaptive", "desc_copy"]


class ManipulatedFeedbackEnv:
    """Delegates to an adversarial env; the injected tool reports success w.p. p."""

    def __init__(self, base, p, seed):
        self._base, self._p = base, p
        self._rng = np.random.default_rng(seed * 7919 + 101)
        self.n_fake_success = 0

    def __getattr__(self, name):
        return getattr(self._base, name)

    def execute(self, tool_id, query):
        obs = self._base.execute(tool_id, query)
        if tool_id == self._base._mal_tool_ids.get(query.required_capability, "") and self._rng.random() < self._p:
            self.n_fake_success += 1
            return Observation(query=obs.query, tool_id=obs.tool_id,
                               capability_used=obs.capability_used, success=True)
        return obs


def build_selectors(fe, seed):
    return [("LinUCB", LinUCBSelector(fe, seed=seed)),
            ("ADAB-EMA-g2", ADABEmaSelector(fe, seed=seed, gamma=2.0, quantile=0.99, decay=0.99)),
            ("ADAB-EMA-g5", ADABEmaSelector(fe, seed=seed, gamma=5.0, quantile=0.99, decay=0.99))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--output", default="results/malicious_feedback_10seed.json")
    args = ap.parse_args()
    seeds = [0] if args.mode == "smoke" else list(range(10))
    t0 = time.time()
    scenarios = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios
    fe = ToolFeatureExtractor(build_query_texts_from_scenarios(scenarios))
    res = {"mode": args.mode, "ps": PS, "seeds": seeds, "per_seed": {}, "aggregate": {}}
    for attack in ATTACKS:
        for p in PS:
            per = {}
            for seed in seeds:
                env = ManipulatedFeedbackEnv(AdversarialBFCLEnvironment(scenarios, strategy=attack, seed=seed), p, seed)
                for name, sel in build_selectors(fe, seed):
                    env.n_fake_success = 0
                    correct = attacked = 0
                    for q in env.queries:
                        tid = sel.select(q, env)
                        obs = env.execute(tid, q)
                        mal = env._mal_tool_ids.get(q.required_capability, "")
                        correct += tid == q.ground_truth_tool_id
                        attacked += tid == mal
                        sel.update(tid, q.required_capability, obs.success)
                    n = len(env.queries)
                    d = per.setdefault(name, {"tsa": [], "asr": [], "fake_success": []})
                    d["tsa"].append(correct / n); d["asr"].append(attacked / n); d["fake_success"].append(env.n_fake_success / n)
            key = f"{attack}/p={p:g}"
            res["per_seed"][key] = per
            res["aggregate"][key] = {m: {k: {"mean": float(np.mean(v[k]) * 100), "std": float(np.std(v[k], ddof=1) * 100) if len(v[k]) > 1 else 0.0}
                                          for k in ("tsa", "asr", "fake_success")} for m, v in per.items()}
            for m, a in res["aggregate"][key].items():
                print(f"{key:22s} {m:12s} TSA {a['tsa']['mean']:5.1f}±{a['tsa']['std']:4.1f}  ASR {a['asr']['mean']:5.2f}±{a['asr']['std']:4.2f}  fake-success/q {a['fake_success']['mean']:5.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    res["wall_time_sec"] = time.time() - t0
    Path(args.output).parent.mkdir(exist_ok=True)
    json.dump(res, open(args.output, "w"), indent=2)
    print(f"saved {args.output} ({res['wall_time_sec']:.0f}s)")


if __name__ == "__main__":
    main()
