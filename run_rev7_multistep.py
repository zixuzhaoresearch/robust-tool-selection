"""Multi-step agent with real stateful tools and a registry of several hundred tools.

BFCL multi_turn_base: 200 tasks, 734 user turns, 1,142 reference calls on eight stateful Python backends
(file system, messaging, posting, ticketing, trading, travel, vehicle control, math; 128 functions).
The registry adds the 457 function specifications of BFCL live_multiple as tools of other servers that
are not connected in the session and fail when called.

At every user turn a shortlister picks k tools from the registry; the agent may call only these.

  coverage - no LLM: is every tool of the reference solution in the shortlist? (an ideal agent calls
             exactly the reference tools it is offered)
  agent    - a local LLM agent calls the shortlisted tools on the real backends, turn by turn; BFCL's
             state/response checker decides task success. The online shortlister learns from execution
             outcomes only (call returned without error / with error / tool offered but not used).

Output: results/revision_r1/multistep_{coverage,agent}.json
"""
import argparse
import ast
import json
import math
import pickle
import re
import time
from pathlib import Path

import numpy as np

from tats.rev1_common import OUT, save, tokens

SERVERS = ["GorillaFileSystem", "MathAPI", "MessageAPI", "TwitterAPI", "TicketAPI", "TradingBot", "TravelAPI", "VehicleControlAPI"]


TYPES = {"dict": "object", "float": "number", "tuple": "array", "list": "array", "bool": "boolean", "any": "string", "str": "string", "int": "integer"}


def norm(x):
    if isinstance(x, dict):
        return {k: (TYPES.get(v, v) if k == "type" and isinstance(v, str) else norm(v)) for k, v in x.items()}
    return [norm(v) for v in x] if isinstance(x, list) else x


def load(registry="full"):
    import bfcl_eval
    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING as DOC
    base = Path(bfcl_eval.__file__).parent / "data"
    tools = []
    for s in SERVERS:
        for line in open(base / "multi_turn_func_doc" / DOC[s]):
            f = json.loads(line)
            tools.append({"name": f["name"], "description": f["description"], "parameters": f["parameters"], "server": s})
    assert len({t["name"] for t in tools}) == len(tools)
    seen = {t["name"] for t in tools}
    for line in open(base / "BFCL_v4_live_multiple.json") if registry == "full" else []:
        for f in json.loads(line)["function"]:
            if f["name"] not in seen:
                seen.add(f["name"])
                tools.append({"name": f["name"], "description": f["description"], "parameters": f["parameters"],
                              "server": "ext:" + re.split(r"[._]", f["name"])[0]})
    tasks = [json.loads(l) for l in open(base / "BFCL_v4_multi_turn_base.json")]
    gt = {a["id"]: a["ground_truth"] for a in map(json.loads, open(base / "possible_answer/BFCL_v4_multi_turn_base.json"))}
    for t in tasks:
        t["gt"] = gt[t["id"]]
        t["gold"] = [sorted({ast.parse(c, mode="eval").body.func.id for c in turn}) for turn in t["gt"]]
        t["query"] = [" ".join(m["content"] for m in turn if m["role"] == "user") for turn in t["question"]]
    return tools, tasks


class Registry:
    """Static text features of every (query, tool) pair, computed once."""

    def __init__(self, tools, tasks):
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tools, self.n = tools, len(tools)
        self.idx = {t["name"]: i for i, t in enumerate(tools)}
        self.server = np.array([t["server"] for t in tools])
        docs = [t["name"].replace("_", " ").replace(".", " ") + ". " + t["description"] for t in tools]
        queries = sorted({q for t in tasks for q in t["query"]})
        self.qid = {q: i for i, q in enumerate(queries)}
        st = SentenceTransformer("all-MiniLM-L6-v2")
        self.D = st.encode(docs, normalize_embeddings=True, batch_size=128)
        self.Q = st.encode(queries, normalize_embeddings=True, batch_size=128)
        tv = TfidfVectorizer().fit(docs)
        self.dense = self.Q @ self.D.T
        self.tfidf = (tv.transform(queries) @ tv.transform(docs).T).toarray()
        name_tok = [set(tokens(t["name"].replace("_", " ").replace(".", " "))) for t in tools]
        par_tok = [set(tokens(" ".join(t["parameters"].get("properties", {})).replace("_", " "))) for t in tools]
        self.name = np.zeros((len(queries), self.n)); self.param = np.zeros((len(queries), self.n))
        for q, i in self.qid.items():
            qt = set(tokens(q))
            self.name[i] = [len(qt & s) / max(len(s), 1) for s in name_tok]
            self.param[i] = [len(qt & s) / max(len(s), 1) for s in par_tok]


class Static:
    def __init__(self, reg, which):
        self.reg, self.which = reg, which

    def start_task(self):
        pass

    def shortlist(self, q, k):
        return np.argsort(-getattr(self.reg, self.which)[self.reg.qid[q]], kind="stable")[:k]

    def update(self, q, called, offered):
        pass


class Online:
    """LinUCB over text features and, optionally, feedback channels:
    usage memory (queries the tool has served without error), windowed reliability and novelty,
    and two session features (the tool / a tool of the same server already worked in this task)."""

    def __init__(self, reg, feedback=True, session=True, alpha=0.5, seed=0, win=8):
        self.reg, self.fb, self.sess, self.alpha, self.win = reg, feedback, session, alpha, win
        self.rng = np.random.default_rng(seed)
        self.d = 5 + (4 if feedback else 0) + (2 if session else 0)
        self.A = np.eye(self.d); self.b = np.zeros(self.d)
        self.mem = [[] for _ in range(reg.n)]; self.out = [[] for _ in range(reg.n)]
        self.start_task()

    def start_task(self):
        self.used = np.zeros(self.reg.n); self.srv = set()

    def feats(self, q):
        r, i = self.reg, self.reg.qid[q]
        cols = [np.ones(r.n), r.dense[i], r.tfidf[i], r.name[i], r.param[i]]
        if self.fb:
            qv = r.Q[i]
            us = np.zeros(r.n); seen = np.zeros(r.n); rel = np.full(r.n, 0.5); nov = np.ones(r.n)
            for t in range(r.n):
                if self.mem[t]:
                    us[t] = max(0.0, float(np.sort(np.asarray(self.mem[t]) @ qv)[-3:].mean()))
                    seen[t] = min(math.log1p(len(self.mem[t])) / math.log(21), 1.0)
                if self.out[t]:
                    rel[t] = float(np.mean(self.out[t][-self.win:])); nov[t] = 1 / math.sqrt(len(self.out[t]) + 1)
            cols += [us, seen, rel, nov]
        if self.sess:
            cols += [self.used, np.isin(r.server, list(self.srv)).astype(float)]
        return np.stack(cols, axis=1)

    def shortlist(self, q, k):
        X = self.feats(q)
        Ainv = np.linalg.inv(self.A)
        s = X @ (Ainv @ self.b) + self.alpha * np.sqrt(np.einsum("ij,jk,ik->i", X, Ainv, X))
        self._X = X
        return np.argsort(-s, kind="stable")[:k]

    def update(self, q, called, offered):
        """called: {tool index: bool success}; offered: the shortlist."""
        X, qv = self._X, self.reg.Q[self.reg.qid[q]]
        idle = [t for t in offered if t not in called]
        neg = self.rng.choice(idle, size=min(len(idle), max(2, len(called))), replace=False) if idle else []
        for t, r in list(called.items()) + [(int(t), False) for t in neg]:
            self.A += np.outer(X[t], X[t]); self.b += float(r) * X[t]
        for t, ok in called.items():
            self.out[t].append(float(ok))
            if ok:
                self.mem[t] = (self.mem[t] + [qv])[-8:]
                self.used[t] = 1.0; self.srv.add(self.reg.server[t])


def selectors(reg, seed):
    return {"TF-IDF": Static(reg, "tfidf"), "Dense": Static(reg, "dense"),
            "LinUCB-Dense (text features)": Online(reg, False, False, seed=seed),
            "LinUCB-Dense + feedback": Online(reg, True, False, seed=seed),
            "LinUCB-Dense + feedback + session": Online(reg, True, True, seed=seed)}


def coverage(seeds, registry, ks=(5, 10, 20)):
    tools, tasks = load(registry)
    reg = Registry(tools, tasks)
    nturn = sum(1 for t in tasks for g in t["gold"] if g)
    print(f"registry {reg.n} tools ({sum(1 for t in tools if not t['server'].startswith('ext:'))} connected), "
          f"{len(tasks)} tasks, {nturn} turns with calls", flush=True)
    res = {}
    for k in ks:
        for name in selectors(reg, 0):
            full, rec, late = [], [], []
            for seed in range(seeds):
                sel = selectors(reg, seed)[name]
                order = np.random.default_rng(seed).permutation(len(tasks))
                f, r = [], []
                for ti in order:
                    task = tasks[ti]; sel.start_task()
                    for q, gold in zip(task["query"], task["gold"]):
                        if not gold:
                            continue
                        sl = [int(x) for x in sel.shortlist(q, k)]
                        g = [reg.idx[n] for n in gold]
                        hit = [t for t in g if t in sl]
                        f.append(len(hit) == len(g)); r.append(len(hit) / len(g))
                        sel.update(q, {t: True for t in hit}, sl)
                full.append(np.mean(f) * 100); rec.append(np.mean(r) * 100); late.append(np.mean(f[len(f) // 2:]) * 100)
                if isinstance(sel, Static):
                    break
            res[f"{name}|k={k}"] = {"full": float(np.mean(full)), "full_sd": float(np.std(full, ddof=1)) if len(full) > 1 else 0.0,
                                     "recall": float(np.mean(rec)), "full_second_half": float(np.mean(late))}
            v = res[f"{name}|k={k}"]
            print(f"k={k:2d} {name:36s} all tools offered {v['full']:5.1f} ±{v['full_sd']:.1f}  recall {v['recall']:5.1f}  second half {v['full_second_half']:5.1f}", flush=True)
    save(f"multistep_coverage_{registry}", {"n_tools": reg.n, "n_turns": nturn, "seeds": seeds, "results": res})


SYSTEM = ("You are an assistant that completes the user's request by calling the available tools. "
          "Call tools one step at a time and use their results. When the request is done, or cannot be done "
          "with the available tools, reply in plain text without a tool call.")


class LLM:
    def __init__(self, name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to("cuda").eval()

    def __call__(self, messages, tools):
        ids = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=384, do_sample=False)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_calls(text):
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        try:
            d = json.loads(m.group(1))
            if isinstance(d.get("name"), str) and isinstance(d.get("arguments", {}), dict):
                calls.append((d["name"], d.get("arguments", {})))
        except json.JSONDecodeError:
            pass
    return calls


def failed(result):
    if result.startswith("Error"):
        return True
    try:
        d = json.loads(result)
        return isinstance(d, dict) and "error" in d
    except (json.JSONDecodeError, TypeError):
        return False


def run_task(llm, task, sel, reg, k, run_id, max_steps=8):
    """One task on the real backends. Returns the calls per turn and the per-turn offered/called tools."""
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
    live = {t["name"] for t in reg.tools if t["server"] in task["involved_classes"]}
    messages = [{"role": "system", "content": SYSTEM}]
    result, log = [], []
    if sel != "oracle":
        sel.start_task()
    for turn, q in zip(task["question"], task["query"]):
        messages += turn
        if sel == "oracle":
            sl = [i for i, t in enumerate(reg.tools) if t["server"] in task["involved_classes"]]
        else:
            sl = [int(x) for x in sel.shortlist(q, k)]
        offered = {reg.tools[i]["name"]: i for i in sl}
        specs = [{"type": "function", "function": {"name": reg.tools[i]["name"], "description": reg.tools[i]["description"],
                                                   "parameters": norm(reg.tools[i]["parameters"])}} for i in sl]
        steps, called = [], {}
        for _ in range(max_steps):
            text = llm(messages, specs)
            calls = parse_calls(text)
            if not calls:
                messages.append({"role": "assistant", "content": text})
                break
            messages.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]})
            step = []
            for n, a in calls:
                if n not in offered:
                    res = f"Error: tool '{n}' is not available."
                elif n not in live:
                    res = f"Error: the server of tool '{n}' is not connected in this session."
                    called[offered[n]] = called.get(offered[n], False)
                else:
                    cs = f"{n}({', '.join(f'{key}={val!r}' for key, val in a.items())})"
                    res = execute_multi_turn_func_call([cs], task["initial_config"], task["involved_classes"], run_id, task["id"])[0][0]
                    step.append(cs)
                    called[offered[n]] = called.get(offered[n], False) or not failed(res)
                messages.append({"role": "tool", "name": n, "content": res[:1500]})
            if step:
                steps.append(step)
        result.append(steps)
        if sel != "oracle":
            sel.update(q, called, sl)
        log.append({"offered": sl, "called": {int(a): bool(b) for a, b in called.items()}})
    return result, log


def check(task, result, run_id):
    """Number of leading turns that pass BFCL's state/response checker, and whether the task passes."""
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
    passed = 0
    for n in range(1, len(task["gt"]) + 1):
        try:
            ok = multi_turn_checker(result[:n], task["gt"][:n], task, "multi_turn_base", f"{run_id}_p{n}")["valid"]
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            break
        passed = n
    return passed, passed == len(task["gt"])


def agent(model, registry, k, passes, limit, which, seed=0):
    tools, tasks = load(registry)
    reg = Registry(tools, tasks)
    if limit:
        tasks = tasks[:limit]
    llm = LLM(model)
    conds = {"Oracle servers": "oracle", "TF-IDF": Static(reg, "tfidf"), "Dense": Static(reg, "dense"),
             "LinUCB-Dense + feedback + session": Online(reg, True, True, seed=seed)}
    if which and "LinUCB-Dense (text features)" in which:
        conds["LinUCB-Dense (text features)"] = Online(reg, False, False, seed=seed)
    out_path = f"multistep_agent_{registry}_{model.split('/')[-1]}"
    try:
        out = json.load(open(f"results/revision_r1/{out_path}.json"))
    except FileNotFoundError:
        out = {"model": model, "registry": registry, "n_tools": reg.n, "k": k, "n_tasks": len(tasks), "runs": {}}
    for name, sel in conds.items():
        if which and name not in which:
            continue
        n_pass = passes if isinstance(sel, Online) else 1
        tag = (f"|seed{seed}" if seed else "")
        # checkpoint after every 10 tasks: rows of the current pass and the selector state, so that a run
        # stopped by the walltime continues from the last saved task instead of starting again
        ck_path = OUT / f"ckpt_{out_path}_{re.sub(r'[^A-Za-z0-9]+', '_', name)}{tag.replace('|', '_')}.pkl"
        ck = pickle.load(open(ck_path, "rb")) if ck_path.exists() else {"pass": 0, "j": 0, "rows": [], "sel": None, "seconds": 0.0}
        if ck["sel"] is not None:
            sel.__dict__.update(ck["sel"])
            print(f"  resuming {name}{tag} at pass {ck['pass']}, task {ck['j']}", flush=True)

        def checkpoint(pp, jj, rows, secs):
            state = {k_: v for k_, v in sel.__dict__.items() if k_ not in ("reg", "_X")} if isinstance(sel, Online) else None
            tmp = ck_path.with_suffix(".tmp")
            pickle.dump({"pass": pp, "j": jj, "rows": rows, "sel": state, "seconds": secs}, open(tmp, "wb"))
            tmp.replace(ck_path)

        for p in range(n_pass):
            key = f"{name}|pass{p}{tag}"
            if p < ck["pass"] or (key in out["runs"] and ck["pass"] == 0 and ck["j"] == 0 and not isinstance(sel, Online)):
                continue
            order = np.random.default_rng(seed * 100 + p).permutation(len(tasks))
            start = ck["j"] if p == ck["pass"] else 0
            rows = ck["rows"] if p == ck["pass"] else []
            t0 = time.time() - (ck["seconds"] if p == ck["pass"] else 0.0)
            for j, ti in enumerate(order):
                if j < start:
                    continue
                task = tasks[int(ti)]
                run_id = re.sub(r"\W", "_", f"{name}_{registry}_{p}_{time.time():.0f}")
                result, log = run_task(llm, task, sel, reg, k, run_id)
                turns, ok = check(task, result, run_id)
                gold_cov = [all(reg.idx[g] in l["offered"] for g in gold) for l, gold in zip(log, task["gold"]) if gold]
                rows.append({"id": task["id"], "ok": ok, "turns_passed": turns, "n_turns": len(task["gt"]),
                             "offered_all": gold_cov, "n_calls": sum(len(s) for t in result for s in t),
                             "n_err": sum(1 for l in log for v in l["called"].values() if not v)})
                if (j + 1) % 10 == 0:
                    checkpoint(p, j + 1, rows, time.time() - t0)
                if (j + 1) % 20 == 0:
                    print(f"  {name} pass {p}: {j + 1}/{len(order)} tasks, success {np.mean([r['ok'] for r in rows]) * 100:.1f}%, "
                          f"turns {sum(r['turns_passed'] for r in rows) / sum(r['n_turns'] for r in rows) * 100:.1f}%, {time.time() - t0:.0f}s", flush=True)
            summ = {"task_success": float(np.mean([r["ok"] for r in rows]) * 100),
                    "turn_success": float(sum(r["turns_passed"] for r in rows) / sum(r["n_turns"] for r in rows) * 100),
                    "offered_all": float(np.mean([x for r in rows for x in r["offered_all"]]) * 100),
                    "calls": int(sum(r["n_calls"] for r in rows)), "seconds": time.time() - t0, "rows": rows}
            # another process may have added runs to the same file since this one started
            try:
                out["runs"] = {**json.load(open(OUT / f"{out_path}.json"))["runs"], **out["runs"]}
            except FileNotFoundError:
                pass
            out["runs"][key] = summ
            print(f"{name} pass {p}: task success {summ['task_success']:.1f}%, turn success {summ['turn_success']:.1f}%, "
                  f"all reference tools offered {summ['offered_all']:.1f}%", flush=True)
            save(out_path, out)
            checkpoint(p + 1, 0, [], 0.0)
            ck = {"pass": p + 1, "j": 0, "rows": [], "sel": None, "seconds": 0.0}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["coverage", "agent"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--registry", choices=["full", "connected"], default="full")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    if a.mode == "coverage":
        coverage(a.seeds, a.registry)
    else:
        agent(a.model, a.registry, a.k, a.passes, a.limit, a.only, a.seed)
    print(f"done in {time.time() - t0:.0f}s")
