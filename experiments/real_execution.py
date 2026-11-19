"""Real execution: the selected tool is actually run and the outcome is the reward.

BFCL's executable categories (exec_simple, exec_multiple, exec_parallel) ship Python implementations of
their functions: 32 local computations, 5 public HTTP APIs that need no key, and 14 APIs that need keys
we do not have (these stay in the catalog and fail when called). BFCL retired the categories in commit
4ed459c of github.com/ShishirPatil/gorilla; the implementations are recovered from that commit's parent:
  git show 4ed459c^:berkeley-function-call-leaderboard/bfcl/eval_checker/executable_eval/executable_python_function.py \
      > data/bfcl_exec/executable_python_function.py

Pipeline per decision: selector picks one function out of the whole catalog -> a small LLM (Qwen2.5-3B)
fills that function's arguments from the query -> the function is executed -> the result is compared with
the result of the benchmark's reference call (BFCL's exact / real-time / structural rules). No label is
shown to the learner in the "exec" runs; the reward is the execution outcome.

  prepare - reference results, LLM arguments and real outcomes for every (query, function) pair
  run     - selectors on the stream, rewards from real outcomes or (for comparison) from labels

Output: data/bfcl_exec/outcomes.json, results/revision_r1/real_exec.json
"""
import argparse
import ast
import json
import math
import re
import signal
import time
import types
from pathlib import Path

import numpy as np

D = Path("data/bfcl_exec")
CATS = ["exec_simple", "exec_multiple", "exec_parallel"]
TYPES = {"dict": "object", "float": "number", "tuple": "array", "list": "array", "bool": "boolean", "any": "string", "str": "string", "int": "integer"}


def norm(x):
    if isinstance(x, dict):
        return {k: (TYPES.get(v, v) if k == "type" and isinstance(v, str) else norm(v)) for k, v in x.items()}
    return [norm(v) for v in x] if isinstance(x, list) else x


def load_functions():
    src = (D / "executable_python_function.py").read_text()
    src = src.replace("from bfcl.eval_checker.executable_eval.custom_exception import NoAPIKeyError", "class NoAPIKeyError(Exception):\n    pass")
    import os
    for var in ("GEOCODE_API_KEY", "RAPID_API_KEY", "OMDB_API_KEY", "EXCHANGERATE_API_KEY"):
        os.environ.setdefault(var, "MISSING")  # we hold no keys: these 14 functions stay in the catalog and fail when called
    mod = types.ModuleType("bfcl_exec_functions")
    exec(compile(src, "executable_python_function.py", "exec"), mod.__dict__)
    # functions that read an API key are not called at all: without the key they cannot work
    mod.NEEDS_KEY = {n.name for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and "api_key[" in ast.get_source_segment(src, n)}
    return mod


def load_data():
    import bfcl_eval
    base = Path(bfcl_eval.__file__).parent / "data/unused_datasets"
    specs, items = {}, []
    for c in CATS + ["exec_parallel_multiple"]:
        for line in open(base / f"question/BFCL_v4_{c}.json"):
            for f in json.loads(line)["function"]:
                specs.setdefault(f["name"], f)
    for c in CATS:
        qs = [json.loads(l) for l in open(base / f"question/BFCL_v4_{c}.json")]
        ans = {a["id"]: a for a in map(json.loads, open(base / f"possible_answer/BFCL_v4_{c}.json"))}
        for q in qs:
            a = ans[q["id"]]
            call = ast.parse(a["ground_truth"][0], mode="eval").body
            name = call.func.id if isinstance(call.func, ast.Name) else ast.unparse(call.func)
            kwargs = {k.arg: eval(compile(ast.Expression(k.value), "<gt>", "eval"), {"__builtins__": {}}, {}) for k in call.keywords}
            items.append({"id": q["id"], "query": q["question"][0][0]["content"], "gold": name, "gold_args": kwargs,
                          "rtype": a["execution_result_type"][0]})
    return specs, items


class Timeout(Exception):
    pass


def call(mod, name, args, limit=10):
    fn = getattr(mod, name.replace(".", "_"), None)
    if fn is None:
        return False, "no implementation"
    if fn.__name__ in mod.NEEDS_KEY:
        return False, "needs an API key"

    def handler(*_):
        raise Timeout()
    signal.signal(signal.SIGALRM, handler); signal.alarm(limit)
    try:
        r = fn(**args)
        # the recovered functions report missing keys and API failures as strings / dicts with an error
        if isinstance(r, str) and re.search(r"error|invalid|not found|api key", r, re.I):
            return False, r[:120]
        return True, r
    except BaseException as e:  # noqa: BLE001 - any failure of the tool is an outcome
        return False, f"{type(e).__name__}: {str(e)[:100]}"
    finally:
        signal.alarm(0)


def match(out, ref, rtype):
    """BFCL executable_checker rules: exact_match, real_time_match (20%), structural_match."""
    if rtype == "exact_match":
        if isinstance(out, float) or isinstance(ref, float):
            try:
                return math.isclose(float(out), float(ref), rel_tol=1e-6, abs_tol=1e-9)
            except (TypeError, ValueError, OverflowError):
                return out == ref
        return out == ref
    if rtype == "real_time_match":
        if isinstance(ref, (int, float)) and isinstance(out, (int, float)) and not isinstance(out, bool):
            return ref * 0.8 <= out <= ref * 1.2 if ref >= 0 else ref * 1.2 <= out <= ref * 0.8
        return out == ref
    if type(out) is not type(ref):
        return False
    if isinstance(ref, dict):
        return set(out) == set(ref)
    if isinstance(ref, list):
        return len(out) == len(ref)
    return True


def prepare(limit=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mod = load_functions()
    specs, items = load_data()
    names = sorted(specs)
    print(f"{len(items)} queries, catalog of {len(names)} functions", flush=True)
    for it in items:  # reference result of the benchmark's own call, executed now
        it["ref_ok"], it["ref"] = call(mod, it["gold"], it["gold_args"])
    ok = [it for it in items if it["ref_ok"]]
    print(f"reference call executes for {len(ok)} of {len(items)} queries; failing gold functions:",
          sorted({it['gold'] for it in items if not it['ref_ok']}), flush=True)
    if limit:
        ok = ok[:limit]
    name = "Qwen/Qwen2.5-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    pairs = [(i, n) for i in range(len(ok)) for n in names]
    prompts = [tok.apply_chat_template([{"role": "user", "content": ok[i]["query"]}],
                                       tools=[{"type": "function", "function": {"name": n.replace(".", "_"), "description": specs[n]["description"],
                                                                                 "parameters": norm(specs[n]["parameters"])}}],
                                       tokenize=False, add_generation_prompt=True) + "<tool_call>\n{\"name\": \"" + n.replace(".", "_") + "\", \"arguments\": "
               for i, n in pairs]
    args = [None] * len(pairs)
    cache_file = D / ("arg_cache.json" if not limit else "arg_cache_smoke.json")  # LLM arguments are saved, so a crash later does not lose them
    done = json.load(open(cache_file)) if cache_file.exists() else {}
    keyof = lambda j: f"{ok[pairs[j][0]]['id']}|{pairs[j][1]}"
    for j in range(len(pairs)):
        if keyof(j) in done:
            args[j] = done[keyof(j)]
    order = np.array([j for j in np.argsort([len(p) for p in prompts]) if keyof(j) not in done]); t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(order), 16):
            idx = order[s:s + 16]
            enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=3000).to("cuda")
            gen = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, g in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
                txt = tok.decode(g, skip_special_tokens=True)
                try:
                    obj, _ = json.JSONDecoder().raw_decode(txt.strip())
                    args[j] = obj if isinstance(obj, dict) else None
                except ValueError:
                    args[j] = None
            for j in idx:
                done[keyof(j)] = args[j]
            if (s // 16) % 50 == 0:
                json.dump(done, open(cache_file, "w"))
                print(f"  arguments {s}/{len(order)} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(done, open(cache_file, "w"))
    del model; torch.cuda.empty_cache()
    outcome, detail, cache = {}, {}, {}
    for (i, n), a in zip(pairs, args):
        it = ok[i]
        if a is None:
            res = (False, "arguments not parsable")
        else:
            key = n + json.dumps(a, sort_keys=True, default=str)
            if key not in cache:
                cache[key] = call(mod, n, a)
            res = cache[key]
        success = bool(res[0] and match(res[1], it["ref"], it["rtype"]))
        outcome[f"{it['id']}|{n}"] = success
        if n == it["gold"] or success:
            detail[f"{it['id']}|{n}"] = {"args": a, "executed": bool(res[0]), "result": str(res[1])[:80], "ref": str(it["ref"])[:80], "success": success}
    json.dump({"queries": [{k: (str(v) if k == "ref" else v) for k, v in it.items()} for it in ok], "functions": names,
               "specs": {n: specs[n] for n in names}, "outcome": outcome, "detail": detail,
               "n_all_queries": len(items), "arg_model": name}, open(D / "outcomes.json", "w"), default=str)
    g = np.mean([outcome[f"{it['id']}|{it['gold']}"] for it in ok])
    off = sum(v for k, v in outcome.items() if k.split("|")[1] != next(it["gold"] for it in ok if it["id"] == k.split("|")[0]))
    print(f"gold function + LLM arguments succeeds on {100*g:.1f}% of queries; {off} (query, non-gold function) pairs also succeed", flush=True)


def run(seeds, passes):
    from tats.adversarial import craft_malicious_tool
    from tats.bfcl_adapter import BFCLScenario, BFCLTool, BFCLToolSelectionBenchmark
    from tats.bilints_selector import LinUCBSelector, SupervisedOfflineSelector
    from tats.dense_features import DenseEmbedder, DenseFeatureExtractor
    from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
    from tats.models import Observation, Query
    from tats.novel_selectors import ADABEmaSelector
    from tats.rev1_common import ArgmaxFeatureSelector, CachedExtractor, save
    from tats.warmstart import train_offline_logistic
    O = json.load(open(D / "outcomes.json"))
    tools = [BFCLTool.from_function_spec(O["specs"][n]) for n in O["functions"]]
    scen = [BFCLScenario(scenario_id=q["id"], query_text=q["query"], candidate_tools=tools, ground_truth_tool_id=q["gold"]) for q in O["queries"]]
    qt = build_query_texts_from_scenarios(scen)
    emb = DenseEmbedder("all-MiniLM-L6-v2"); emb.encode_batch(list(qt.values()) + [t.description for t in tools])
    fe, fed = CachedExtractor(ToolFeatureExtractor(qt)), CachedExtractor(DenseFeatureExtractor(qt, emb))
    train = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="multiple").scenarios
    logit = train_offline_logistic(train, ToolFeatureExtractor(build_query_texts_from_scenarios(train)))
    n = len(scen)

    class Env:
        def __init__(self, seed, attack=None):
            rng = np.random.default_rng(seed)
            self.queries = []
            for p in range(passes):
                order = rng.permutation(n)
                self.queries += [Query(query_id=int(i), required_capability=scen[i].scenario_id, ground_truth_tool_id=scen[i].ground_truth_tool_id) for i in order]
            self.sc = {s.scenario_id: s for s in scen}
            self.mal = {s.scenario_id: craft_malicious_tool(s.scenario_id, s.query_text, attack) for s in scen} if attack else {}
            self.tools = tools

        def get_declarers(self, cap):
            return tools + ([self.mal[cap]] if self.mal else [])

        def real(self, tid, q):  # outcome of actually executing the selected function with LLM-filled arguments
            return bool(O["outcome"].get(f"{q.required_capability}|{tid}", False))

    def stream(sel, env, reward):
        lab, ex, atk = [], [], []
        for q in env.queries:
            tid = sel.select(q, env)
            l, e = tid == q.ground_truth_tool_id, env.real(tid, q)
            lab.append(l); ex.append(e); atk.append(bool(env.mal) and tid == env.mal[q.required_capability].tool_id)
            if hasattr(sel, "update"):
                sel.update(tid, q.required_capability, e if reward == "exec" else l)
        return np.array(lab, float), np.array(ex, float), np.array(atk, float)

    def make(seed):
        return {"TF-IDF-Desc": (ArgmaxFeatureSelector(fe, 0, seed), "exec"), "DenseDescMatch": (ArgmaxFeatureSelector(fed, 0, seed), "exec"),
                "ParamMatch": (ArgmaxFeatureSelector(fe, 2, seed), "exec"), "Sup-Offline": (SupervisedOfflineSelector(fe, logit, seed=seed), "exec"),
                "LinUCB (execution reward)": (LinUCBSelector(fe, seed=seed), "exec"), "LinUCB (label reward)": (LinUCBSelector(fe, seed=seed), "label"),
                "LinUCB-Dense (execution reward)": (LinUCBSelector(fed, seed=seed), "exec"), "LinUCB-Dense (label reward)": (LinUCBSelector(fed, seed=seed), "label"),
                "ADAB-Dense g=2 (execution reward)": (ADABEmaSelector(fed, seed=seed, gamma=2.0, quantile=0.99, decay=0.99), "exec")}

    res = {"n_queries": n, "n_functions": len(tools), "passes": passes, "n_all_queries": O["n_all_queries"], "arg_model": O["arg_model"],
           "gold_exec_success": float(100 * np.mean([O["outcome"][f"{q['id']}|{q['gold']}"] for q in O["queries"]])), "runs": {}}
    for attack in [None, "full_adaptive"]:
        per = {}
        for seed in seeds:
            for name, (sel, reward) in make(seed).items():
                if attack and "Dense" not in name:
                    continue
                l, e, a = stream(sel, Env(seed, attack), reward)
                d = per.setdefault(name, {"label": [], "exec": [], "asr": []})
                d["label"].append(l); d["exec"].append(e); d["asr"].append(a)
        out = {}
        for name, d in per.items():
            L, E, A = (np.array(d[k]) for k in ("label", "exec", "asr"))
            out[name] = {"label_tsa": float(100 * L.mean()), "exec_success": float(100 * E.mean()), "exec_sd": float(100 * E.mean(1).std(ddof=1)),
                         "exec_first_pass": float(100 * E[:, :n].mean()), "exec_last_pass": float(100 * E[:, -n:].mean()),
                         "exec_last_pass_sd": float(100 * E[:, -n:].mean(1).std(ddof=1)), "label_last_pass": float(100 * L[:, -n:].mean()),
                         "asr": float(100 * A.mean()), "asr_sd": float(100 * A.mean(1).std(ddof=1)),
                         "label_right_exec_fail": float(100 * ((L == 1) & (E == 0)).mean()), "label_wrong_exec_ok": float(100 * ((L == 0) & (E == 1)).mean())}
            print(f"{attack or 'benign':13s} {name:36s} label TSA {out[name]['label_tsa']:5.1f}  real success {out[name]['exec_success']:5.1f}±{out[name]['exec_sd']:.1f} "
                  f"(first pass {out[name]['exec_first_pass']:5.1f}, last {out[name]['exec_last_pass']:5.1f})  ASR {out[name]['asr']:.1f}", flush=True)
        res["runs"][attack or "benign"] = out
    save("real_exec", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["prepare", "run"])
    ap.add_argument("--limit", type=int); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--passes", type=int, default=5)
    a = ap.parse_args()
    prepare(a.limit) if a.part == "prepare" else run(list(range(a.seeds)), a.passes)
