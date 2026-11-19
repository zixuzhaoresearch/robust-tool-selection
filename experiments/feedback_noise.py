"""Revision 1, experiment E3: reward is a noisy, partial, or delayed function of the correct choice.

Reviewer 2 (comment 3) / Reviewer 3. The selector never sees the ground truth; it sees
a reward produced by a feedback model:
  sym(p)      - reward flipped with probability p (the submitted Table A5 text)
  fn(p)       - a correct selection is reported as a failure with probability p
  fp(p)       - a wrong benign selection is reported as a success with probability p
  mix(a, b)   - fn(a) and fp(b) together (execution success != correct tool)
  miss(p)     - feedback is missing with probability p (no update)
  delay(d)    - the reward arrives d decisions later
TSA / ASR are always scored against the ground truth. In these runs the injected tool
never reports success (Table "malicious feedback" covers that separately).

Output: results/revision_r1/feedback.json
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

from tats.adversarial import AdversarialBFCLEnvironment
from tats.bfcl_adapter import BFCLEnvironment, BFCLToolSelectionBenchmark
from tats.bilints_selector import LinUCBSelector
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.novel_selectors import ADABEmaSelector
from tats.rev1_common import save, summarize

MODELS = [("clean", {}), ("sym", {"p": 0.2}),
          ("fn", {"p": 0.1}), ("fn", {"p": 0.3}), ("fn", {"p": 0.5}),
          ("fp", {"p": 0.1}), ("fp", {"p": 0.3}), ("fp", {"p": 0.5}),
          ("mix", {"fn": 0.3, "fp": 0.1}), ("mix", {"fn": 0.1, "fp": 0.3}),
          ("miss", {"p": 0.5}), ("miss", {"p": 0.9}),
          ("delay", {"d": 10}), ("delay", {"d": 50}), ("delay", {"d": 200})]


def run(sel, env, n, mal, kind, kw, rng):
    correct, attacked = np.zeros(n), np.zeros(n)
    pending = deque()
    d = kw.get("d", 0)
    for q in env.queries:
        tid = sel.select(q, env)
        ok = tid == q.ground_truth_tool_id
        is_mal = bool(mal) and tid == mal.get(q.required_capability, "")
        correct[q.query_id], attacked[q.query_id] = ok, is_mal
        r = ok
        u = rng.random()
        if kind == "sym" and u < kw["p"]:
            r = not r
        elif kind == "fn" and ok and u < kw["p"]:
            r = False
        elif kind == "fp" and not ok and not is_mal and u < kw["p"]:
            r = True
        elif kind == "mix":
            if ok and u < kw["fn"]:
                r = False
            elif not ok and not is_mal and u < kw["fp"]:
                r = True
        elif kind == "miss" and u < kw["p"]:
            continue
        pending.append((dict(sel._cached_features), tid, q.required_capability, r))
        while len(pending) > d:  # the reward for a decision arrives d decisions later
            feats, t, cap, rew = pending.popleft()
            cur = sel._cached_features
            sel._cached_features = feats
            sel.update(t, cap, bool(rew))
            sel._cached_features = cur
    return correct, attacked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    t0 = time.time()
    scen = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios
    fe = ToolFeatureExtractor(build_query_texts_from_scenarios(scen))
    res = {}
    for kind, kw in MODELS:
        key = kind + "".join(f"_{k}{v:g}" for k, v in kw.items())
        for strategy in [None, "full_adaptive"]:
            per = {}
            for seed in range(args.seeds):
                for name, mk in [("LinUCB", lambda: LinUCBSelector(fe, seed=seed)),
                                 ("ADAB-g2", lambda: ADABEmaSelector(fe, seed=seed, gamma=2.0, quantile=0.99, decay=0.99))]:
                    if strategy is None:
                        env, mal = BFCLEnvironment(scen, seed=seed), None
                    else:
                        env = AdversarialBFCLEnvironment(scen, strategy=strategy, seed=seed)
                        mal = dict(env._mal_tool_ids)
                    c, a = run(mk(), env, len(scen), mal, kind, kw, np.random.default_rng(seed * 7919 + 13))
                    d = per.setdefault(name, {"tsa": [], "asr": []})
                    d["tsa"].append(c)
                    d["asr"].append(a)
            cond = strategy or "benign"
            res.setdefault(key, {})[cond] = {n: {"tsa": summarize(v["tsa"]), "asr": summarize(v["asr"])} for n, v in per.items()}
            line = "  ".join(f"{n}: TSA {v['tsa']['mean']:.1f}±{v['tsa']['sd']:.1f} ASR {v['asr']['mean']:.2f}±{v['asr']['sd']:.2f}"
                             for n, v in res[key][cond].items())
            print(f"[{time.time()-t0:4.0f}s] {key:16s} {cond:14s} {line}", flush=True)
    save("feedback", res)


if __name__ == "__main__":
    main()
