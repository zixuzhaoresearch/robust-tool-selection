"""Revision 1, experiment E5: per-scenario outcomes for the headline comparisons,
paired bootstrap intervals over scenarios, and held-out selection of gamma.

Reviewer 2 (comment 5). Three parts:
  core     - reruns the headline BFCL / MetaTool comparisons and stores per-scenario
             outcomes (results/revision_r1/core_{bfcl,metatool}.json)
  holdout  - per seed, splits BFCL live_multiple 50/50 into a development stream and a
             test stream; gamma is chosen on the development stream (highest
             under-attack TSA under full_adaptive, ties to the smaller gamma) and
             reported on the test stream (results/revision_r1/gamma_holdout.json)
  boot     - paired bootstrap CIs over scenarios for a fixed list of comparisons
             (results/revision_r1/bootstrap.json)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from tats.adversarial import AdversarialBFCLEnvironment
from tats.bfcl_adapter import BFCLEnvironment, BFCLToolSelectionBenchmark
from tats.bfcl_selectors import DescriptionMatchSelector, ParameterMatchSelector
from tats.bilints_selector import LinUCBSelector, SupervisedOfflineSelector, SupervisedOnlineSelector
from tats.dense_features import DenseDescMatchSelector, DenseEmbedder, DenseFeatureExtractor
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.metatool_adapter import MetaToolBenchmark
from tats.novel_selectors import ADABEmaSelector
from tats.rev1_common import OUT, ArgmaxFeatureSelector, CachedExtractor, paired_bootstrap, rollout, save, summarize
from tats.warmstart import train_offline_logistic

GAMMAS = [0.5, 1.0, 2.0, 3.0, 5.0]


def adab(fe, seed, g):
    return ADABEmaSelector(fe, seed=seed, gamma=g, quantile=0.99, decay=0.99)


def run_condition(scen, strategy, seed, selectors, benign_env=None):
    """selectors: dict name -> selector. Returns dict name -> (correct, attacked)."""
    out = {}
    for name, sel in selectors.items():
        if strategy is None:
            env = benign_env() if benign_env else BFCLEnvironment(scen, seed=seed)
            mal = None
        else:
            env = AdversarialBFCLEnvironment(scen, strategy=strategy, seed=seed)
            mal = dict(env._mal_tool_ids)
        out[name] = rollout(sel, env, len(scen), mal)
    return out


def collect(res, cond, out):
    for name, (c, a) in out.items():
        d = res.setdefault(cond, {}).setdefault(name, {"tsa": [], "asr": []})
        d["tsa"].append(c)
        d["asr"].append(a)


def finalize(res):
    return {cond: {n: {"tsa": summarize(v["tsa"]), "asr": summarize(v["asr"])} for n, v in sels.items()}
            for cond, sels in res.items()}


def report(tag, fin):
    for cond, sels in fin.items():
        for n, v in sels.items():
            print(f"{tag} {cond:14s} {n:16s} TSA {v['tsa']['mean']:5.1f}±{v['tsa']['sd']:4.1f}  "
                  f"ASR {v['asr']['mean']:5.2f}±{v['asr']['sd']:5.2f}", flush=True)


def core_bfcl(seeds, emb):
    scen = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios
    train = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="multiple").scenarios
    qt = build_query_texts_from_scenarios(scen)
    fe = ToolFeatureExtractor(qt)
    logit = train_offline_logistic(train, ToolFeatureExtractor(build_query_texts_from_scenarios(train)))
    res = {}
    for strategy in [None, "desc_copy", "param_generic", "param_steal", "full_adaptive"]:
        for seed in seeds:
            # the dense extractor caches description embeddings; attacks add new descriptions
            fed = DenseFeatureExtractor(qt, emb)
            sels = {"DescMatch": DescriptionMatchSelector(qt, seed=seed),
                    "TFIDF-Desc": ArgmaxFeatureSelector(fe, 0, seed),
                    "ParamMatch": ParameterMatchSelector(qt, seed=seed),
                    "LinUCB": LinUCBSelector(fe, seed=seed),
                    "LinUCB-Dense": LinUCBSelector(fed, seed=seed),
                    "ADAB-Dense-g2": adab(fed, seed, 2.0)}
            sels.update({f"ADAB-g{g:g}": adab(fe, seed, g) for g in GAMMAS})
            if strategy is None:
                sels.update({"DenseDescMatch": DenseDescMatchSelector(qt, emb, seed=seed),
                             "Sup-Offline": SupervisedOfflineSelector(fe, logit, seed=seed),
                             "Sup-Online": SupervisedOnlineSelector(fe, seed=seed)})
            collect(res, strategy or "benign", run_condition(scen, strategy, seed, sels))
    fin = finalize(res)
    report("bfcl", fin)
    save("core_bfcl", fin)


def core_metatool(seeds, emb):
    bm = MetaToolBenchmark(n_candidates=10, max_queries=2000, similarity_mode="embedding", seed=42)
    scen = bm.scenarios
    qt = build_query_texts_from_scenarios(scen)
    fe = CachedExtractor(ToolFeatureExtractor(qt))
    emb.encode_batch(list(qt.values()) + list({t.description for s in scen for t in s.candidate_tools}))
    res = {}
    for strategy in [None, "desc_copy"]:
        for seed in seeds:
            fed = DenseFeatureExtractor(qt, emb)
            sels = {"DescMatch": DescriptionMatchSelector(qt, seed=seed),
                    "TFIDF-Desc": ArgmaxFeatureSelector(fe, 0, seed),
                    "DenseDescMatch": DenseDescMatchSelector(qt, emb, seed=seed),
                    "LinUCB": LinUCBSelector(fe, seed=seed),
                    "ADAB-g2": adab(fe, seed, 2.0), "ADAB-g3": adab(fe, seed, 3.0),
                    "LinUCB-Dense": LinUCBSelector(fed, seed=seed),
                    "ADAB-Dense-g2": adab(fed, seed, 2.0), "ADAB-Dense-g4": adab(fed, seed, 4.0)}

            def benign_env(seed=seed):  # same construction as the submitted Table 9 run
                env = bm.to_sequential_env()
                np.random.default_rng(seed).shuffle(env.queries)
                return env
            collect(res, strategy or "benign", run_condition(scen, strategy, seed, sels, benign_env))
    fin = finalize(res)
    report("metatool", fin)
    save("core_metatool", fin)


def holdout(seeds):
    scen = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios
    qt = build_query_texts_from_scenarios(scen)
    rows = []
    for seed in seeds:
        perm = np.random.default_rng(10_000 + seed).permutation(len(scen))
        half = len(scen) // 2
        parts = {"dev": [scen[i] for i in perm[:half]], "test": [scen[i] for i in perm[half:]]}
        m = {}
        for part, sub in parts.items():
            sub_qt = {i: s.query_text for i, s in enumerate(sub)}
            fe = ToolFeatureExtractor(sub_qt)  # IDF from this stream only
            for g in [0.0] + GAMMAS:
                for strategy in [None, "full_adaptive", "param_steal"]:
                    sel = LinUCBSelector(fe, seed=seed) if g == 0 else adab(fe, seed, g)
                    c, a = run_condition(sub, strategy, seed, {"s": sel})["s"]
                    m[(part, g, strategy or "benign")] = (100 * c.mean(), 100 * a.mean())
        # rule fixed before looking at the test stream
        best = max(GAMMAS, key=lambda g: (round(m[("dev", g, "full_adaptive")][0], 6), -g))
        row = {"seed": seed, "chosen_gamma": best}
        for g, tag in [(0.0, "linucb"), (best, "chosen"), (2.0, "g2")]:
            row[tag] = {k: m[("test", g, k)] for k in ("benign", "full_adaptive", "param_steal")}
        row["dev"] = {f"g{g:g}": {k: m[("dev", g, k)] for k in ("benign", "full_adaptive")} for g in [0.0] + GAMMAS}
        rows.append(row)
        print(f"holdout seed={seed} chosen gamma={best:g} test benign TSA {row['chosen']['benign'][0]:.1f} "
              f"(LinUCB {row['linucb']['benign'][0]:.1f}) full_adaptive TSA/ASR "
              f"{row['chosen']['full_adaptive'][0]:.1f}/{row['chosen']['full_adaptive'][1]:.2f} "
              f"(LinUCB {row['linucb']['full_adaptive'][0]:.1f}/{row['linucb']['full_adaptive'][1]:.2f})", flush=True)
    agg = {}
    for tag in ("linucb", "chosen", "g2"):
        for k in ("benign", "full_adaptive", "param_steal"):
            v = np.array([r[tag][k] for r in rows])
            agg[f"{tag}/{k}"] = {"tsa_mean": float(v[:, 0].mean()), "tsa_sd": float(v[:, 0].std(ddof=1)),
                                 "asr_mean": float(v[:, 1].mean()), "asr_sd": float(v[:, 1].std(ddof=1))}
    counts = {f"{g:g}": int(sum(r["chosen_gamma"] == g for r in rows)) for g in GAMMAS}
    print("chosen-gamma counts", counts)
    for k, v in agg.items():
        print(f"  {k:24s} TSA {v['tsa_mean']:.1f}±{v['tsa_sd']:.1f}  ASR {v['asr_mean']:.2f}±{v['asr_sd']:.2f}")
    save("gamma_holdout", {"rule": "argmax dev full_adaptive TSA, ties to smaller gamma",
                           "gammas": GAMMAS, "chosen_counts": counts, "aggregate": agg, "per_seed": rows})


PAIRS = [  # (file, condition, metric, A, B)
    ("core_bfcl", "benign", "tsa", "LinUCB", "DescMatch"),
    ("core_bfcl", "benign", "tsa", "LinUCB", "TFIDF-Desc"),
    ("core_bfcl", "benign", "tsa", "LinUCB", "Sup-Offline"),
    ("core_bfcl", "benign", "tsa", "LinUCB", "Sup-Online"),
    ("core_bfcl", "benign", "tsa", "LinUCB", "DenseDescMatch"),
    ("core_bfcl", "benign", "tsa", "LinUCB-Dense", "DenseDescMatch"),
    ("core_bfcl", "benign", "tsa", "ADAB-g2", "LinUCB"),
    ("core_bfcl", "benign", "tsa", "ADAB-Dense-g2", "LinUCB-Dense"),
    ("core_bfcl", "full_adaptive", "tsa", "ADAB-g2", "LinUCB"),
    ("core_bfcl", "full_adaptive", "asr", "ADAB-g2", "LinUCB"),
    ("core_bfcl", "full_adaptive", "asr", "ADAB-g5", "ADAB-g2"),
    ("core_bfcl", "full_adaptive", "asr", "ADAB-Dense-g2", "LinUCB-Dense"),
    ("core_bfcl", "param_steal", "asr", "ADAB-g2", "LinUCB"),
    ("core_metatool", "benign", "tsa", "LinUCB", "DescMatch"),
    ("core_metatool", "benign", "tsa", "LinUCB", "TFIDF-Desc"),
    ("core_metatool", "benign", "tsa", "LinUCB-Dense", "DenseDescMatch"),
    ("core_metatool", "benign", "tsa", "ADAB-g2", "LinUCB"),
    ("core_metatool", "benign", "tsa", "ADAB-Dense-g4", "LinUCB-Dense"),
    ("core_metatool", "desc_copy", "asr", "ADAB-g2", "LinUCB"),
    ("core_metatool", "desc_copy", "asr", "ADAB-g3", "LinUCB"),
    ("core_metatool", "desc_copy", "asr", "ADAB-g3", "ADAB-g2"),
    ("core_metatool", "desc_copy", "asr", "ADAB-Dense-g2", "LinUCB-Dense"),
    ("core_metatool", "desc_copy", "asr", "ADAB-Dense-g4", "LinUCB-Dense"),
    ("core_metatool", "desc_copy", "asr", "ADAB-Dense-g4", "ADAB-Dense-g2"),
    ("core_metatool", "desc_copy", "tsa", "ADAB-g3", "LinUCB"),
    ("core_metatool", "desc_copy", "tsa", "ADAB-Dense-g4", "LinUCB-Dense"),
]


def boot():
    cache, rows = {}, []
    for f, cond, metric, a, b in PAIRS:
        d = cache.setdefault(f, json.load(open(OUT / f"{f}.json")))
        diff, lo, hi = paired_bootstrap(d[cond][a][metric]["per_scenario"], d[cond][b][metric]["per_scenario"])
        rows.append({"file": f, "condition": cond, "metric": metric, "a": a, "b": b,
                     "a_mean": d[cond][a][metric]["mean"], "b_mean": d[cond][b][metric]["mean"],
                     "diff_pp": diff, "ci95": [lo, hi]})
        print(f"{f:14s} {cond:14s} {metric} {a:14s} - {b:14s} = {diff:+6.2f} pp  [{lo:+6.2f}, {hi:+6.2f}]")
    save("bootstrap", {"method": "paired percentile bootstrap over scenarios, 10,000 resamples; "
                                 "per-scenario outcomes are averaged over seeds first", "rows": rows})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["core_bfcl", "core_metatool", "holdout", "boot"])
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    t0 = time.time()
    if args.part == "core_bfcl":
        core_bfcl(seeds, DenseEmbedder("all-MiniLM-L6-v2"))
    elif args.part == "core_metatool":
        core_metatool(seeds, DenseEmbedder("all-MiniLM-L6-v2"))
    elif args.part == "holdout":
        holdout(seeds)
    else:
        boot()
    print(f"done in {time.time()-t0:.0f}s")
