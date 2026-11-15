"""Revision 1, experiment E1: documentation AND schema degradation on BFCL live_multiple.

Reviewer 2 (comment 1) / Reviewer 1 (comment 2). Adds schema-only and name-only
baselines to the degradation table, a description level that keeps names (D1n),
a misleading level that touches every scenario (D3s), and a second axis in which
the schema itself is degraded (abbreviated / renamed / dropped / added parameters,
missing required flags, no schema), alone and combined with D1.

Output: results/revision_r1/degradation.json
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np

from tats.bfcl_adapter import BFCLEnvironment, BFCLScenario, BFCLToolSelectionBenchmark
from tats.bfcl_selectors import DescriptionMatchSelector, ParameterMatchSelector, RandomBFCLSelector
from tats.bilints_selector import LinUCBSelector, SupervisedOfflineSelector
from tats.degradation import DegradationEngine, _copy_tool
from tats.dense_features import DenseDescMatchSelector, DenseEmbedder, DenseFeatureExtractor
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.rev1_common import ArgmaxFeatureSelector, rollout, save, summarize
from tats.warmstart import train_offline_logistic

DESC_LEVELS = ["Full", "D3", "D3s", "D2", "D1n", "D1"]
SCHEMA_LEVELS = ["intact", "S-abbrev", "S-rename", "S-noreq", "S-drop50", "S-add3", "S-none"]
_FILLER = ["verbose", "trace_id", "locale", "retries", "format", "timeout_ms"]


def _rebuild(sc, tools, gt=None):
    return BFCLScenario(scenario_id=sc.scenario_id, query_text=sc.query_text, candidate_tools=tools,
                        ground_truth_tool_id=gt or sc.ground_truth_tool_id,
                        ground_truth_args=copy.deepcopy(sc.ground_truth_args))


def degrade_desc(scenarios, level, seed):
    if level == "Full":
        return scenarios
    if level in ("D1", "D2", "D3"):
        return DegradationEngine(seed=seed).degrade(scenarios, {"D1": "L1", "D2": "L2", "D3": "L3"}[level])
    rng = np.random.default_rng(seed)
    out = []
    for sc in scenarios:
        tools = [_copy_tool(t) for t in sc.candidate_tools]
        if level == "D1n":  # descriptions removed, real names kept
            for t in tools:
                t.description = ""
                for p in t.parameter_spec.get("properties", {}).values():
                    if isinstance(p, dict) and "description" in p:
                        p["description"] = ""
        elif level == "D3s":  # two candidates exchange descriptions in every scenario
            i, j = rng.choice(len(tools), size=2, replace=False)
            tools[i].description, tools[j].description = tools[j].description, tools[i].description
        else:
            raise ValueError(level)
        out.append(_rebuild(sc, tools))
    return out


def degrade_schema(scenarios, level, seed):
    if level == "intact":
        return scenarios
    rng = np.random.default_rng(seed + 1000)
    out = []
    for sc in scenarios:
        tools = []
        for t in sc.candidate_tools:
            t = _copy_tool(t)
            spec = t.parameter_spec if isinstance(t.parameter_spec, dict) else {}
            props = dict(spec.get("properties", {}) or {})
            req = list(spec.get("required", []) or [])
            if level == "S-abbrev":  # start_date -> sta_dat
                m = {k: "_".join(p[:3] for p in k.split("_")) for k in props}
            elif level == "S-rename":  # opaque names
                m = {k: f"arg{i}" for i, k in enumerate(props)}
            else:
                m = {k: k for k in props}
            # dict comprehension keeps the last value on abbreviation collisions
            props = {m[k]: v for k, v in props.items()}
            req = [m[k] for k in req if k in m]
            if level == "S-noreq":
                req = []
            elif level == "S-drop50" and props:
                keep = [k for k in props if rng.random() >= 0.5]
                props = {k: props[k] for k in keep}
                req = [k for k in req if k in props]
            elif level == "S-add3":
                for k in rng.choice(_FILLER, size=3, replace=False):
                    props.setdefault(str(k), {"type": "string", "description": ""})
            elif level == "S-none":
                props, req = {}, []
            t.parameter_spec = {"type": "object", "properties": props, "required": req}
            tools.append(t)
        out.append(_rebuild(sc, tools))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    t0 = time.time()
    scen = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios
    train = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="multiple").scenarios
    qt = build_query_texts_from_scenarios(scen)
    fe = ToolFeatureExtractor(qt)
    logit = train_offline_logistic(train, ToolFeatureExtractor(build_query_texts_from_scenarios(train)))
    emb = DenseEmbedder("all-MiniLM-L6-v2")
    emb.encode_batch(list(qt.values()))
    conds = [(d, "intact") for d in DESC_LEVELS] + [(d, s) for d in ("Full", "D1") for s in SCHEMA_LEVELS[1:]]
    res = {"conditions": [f"{d}|{s}" for d, s in conds], "n": len(scen), "results": {}}
    for d, s in conds:
        per = {}
        for seed in range(args.seeds):
            cur = degrade_schema(degrade_desc(scen, d, seed), s, seed)
            emb.encode_batch(list({t.description for sc in cur for t in sc.candidate_tools}))
            fed = DenseFeatureExtractor(qt, emb)
            sels = {
                "Random": RandomBFCLSelector(seed=seed),
                "DescMatch": DescriptionMatchSelector(qt, seed=seed),
                "TFIDF-Desc": ArgmaxFeatureSelector(fe, 0, seed),
                "DenseDescMatch": DenseDescMatchSelector(qt, emb, seed=seed),
                "NameMatch": ArgmaxFeatureSelector(fe, 1, seed),
                "ParamMatch": ParameterMatchSelector(qt, seed=seed),
                "Sup-Offline": SupervisedOfflineSelector(fe, logit, seed=seed),
                "LinUCB": LinUCBSelector(fe, seed=seed),
                "LinUCB-Dense": LinUCBSelector(fed, seed=seed),
            }
            for name, sel in sels.items():
                c, _ = rollout(sel, BFCLEnvironment(cur, seed=seed), len(scen))
                per.setdefault(name, []).append(c)
        res["results"][f"{d}|{s}"] = {k: summarize(v) for k, v in per.items()}
        line = "  ".join(f"{k}={v['mean']:.1f}±{v['sd']:.1f}" for k, v in res["results"][f"{d}|{s}"].items())
        print(f"[{time.time()-t0:5.0f}s] {d:5s}|{s:9s} {line}", flush=True)
    save("degradation", res)


if __name__ == "__main__":
    main()
