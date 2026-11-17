"""Adaptation to execution feedback: what a feedback-driven selector can do that static matching cannot.

  twins   - hidden reliability: every MetaTool candidate has a twin with identical name and description;
            one twin of each pair executes reliably (success prob. p_good when it is the right tool), the
            other does not (p_bad). Documentation cannot tell them apart; outcomes can.
  outage  - the same, but at mid-stream the reliable and unreliable twins swap roles.
  drift   - BFCL documentation drift: intact descriptions in the first half of the stream, two
            descriptions exchanged in every scenario (D3s) in the second half.

Per-tool feedback features (success-filtered usage memory; windowed reliability with a novelty term)
are added to the linear bandit. Output: results/revision_r1/adaptation.json and
adaptation.pdf
"""
import argparse
import copy
import math
import time

import numpy as np

from run_rev1_baselines import STEmbedder, finetune, metatool_split
from run_rev1_degradation import degrade_desc
from tats.bfcl_adapter import BFCLTool, BFCLToolSelectionBenchmark
from tats.bilints_selector import LinUCBSelector, SupervisedOfflineSelector
from tats.dense_features import DenseEmbedder, DenseFeatureExtractor
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.models import Observation, Query
from tats.rev1_common import ArgmaxFeatureSelector, CachedExtractor, save
from tats.warmstart import train_offline_logistic

P_GOOD, P_BAD, WINDOW = 0.9, 0.2, 100


class FeedbackExtractor:
    """base features + usage memory (usage_sim, usage_seen) and/or reliability (rate, novelty)."""

    def __init__(self, base, emb, memory=True, reliability=True, k=3, rel_window=30, mem_cap=50, forget=False):
        """forget=True: a failure of a tool evicts its oldest stored success (the memory tracks current behavior)."""
        self._base, self._emb, self._mem_on, self._rel_on, self._k, self._w = base, emb, memory, reliability, k, rel_window
        self._cap, self._forget = mem_cap, forget
        self._query_texts = base._query_texts
        self._mem, self._hist, self._q = {}, {}, {}

    @property
    def n_features(self):
        return self._base.n_features + 2 * self._mem_on + 2 * self._rel_on

    def _qvec(self, qid):
        v = self._q.get(qid)
        if v is None:
            v = self._emb.encode(self._query_texts[qid])
            v = self._q[qid] = v / (np.linalg.norm(v) + 1e-9)
        return v

    def extract(self, qid, tool, n):
        x = list(self._base.extract(qid, tool, n))
        if self._mem_on:
            m = self._mem.get(tool.tool_id)
            x += [max(0.0, float(np.sort(np.asarray(m) @ self._qvec(qid))[-self._k:].mean())),
                  min(math.log1p(len(m)) / math.log(21), 1.0)] if m else [0.0, 0.0]
        if self._rel_on:
            h = self._hist.get(tool.tool_id, [])[-self._w:]
            x += [(sum(h) + 1) / (len(h) + 2), 1 / math.sqrt(len(h) + 1)]
        return np.array(x)

    def observe(self, tool_id, qid, success):
        self._hist.setdefault(tool_id, []).append(float(success))
        if success:
            m = self._mem.setdefault(tool_id, [])
            m.append(self._qvec(qid))
            del m[:-self._cap]
        elif self._forget and self._mem.get(tool_id):
            self._mem[tool_id].pop(0)


class Online:
    def __init__(self, sel, fe=None):
        self.sel, self.fe, self.qid = sel, fe, None

    def select(self, q, env):
        self.qid = q.query_id
        return self.sel.select(q, env)

    def update(self, tid, cap, success):
        if hasattr(self.sel, "update"):
            self.sel.update(tid, cap, success)
        if self.fe is not None:
            self.fe.observe(tid, self.qid, success)


class TwinEnv:
    """Each candidate appears twice (ids '<id>#a', '<id>#b', identical documentation)."""

    def __init__(self, scen, seed, flip_at=None):
        rng = np.random.default_rng(seed)
        self.sc = {s.scenario_id: s for s in scen}
        base = sorted({t.tool_id for s in scen for t in s.candidate_tools})
        self.good = {b: ("a" if rng.random() < 0.5 else "b") for b in base}
        self.queries = [Query(query_id=i, required_capability=s.scenario_id, ground_truth_tool_id=s.ground_truth_tool_id) for i, s in enumerate(scen)]
        rng.shuffle(self.queries)
        self.rng, self.t, self.flip_at, self._tw, self.tools = rng, 0, flip_at, {}, []

    def _twin(self, t, s):
        k = (t.tool_id, s)
        if k not in self._tw:
            c = BFCLTool(tool_id=f"{t.tool_id}#{s}", name=t.name, declared_capabilities=[], true_success_rates={}, description_affinity={},
                         description=t.description, parameter_spec=copy.deepcopy(t.parameter_spec))
            self._tw[k] = c
        return self._tw[k]

    def get_declarers(self, cap):
        c = [self._twin(t, s) for t in self.sc[cap].candidate_tools for s in "ab"]
        order = np.random.default_rng(hash(cap) % (2**32)).permutation(len(c))
        return [c[i] for i in order]

    def good_now(self, base):
        g = self.good[base]
        return g if (self.flip_at is None or self.t < self.flip_at) else ("b" if g == "a" else "a")

    def execute(self, tid, q):
        base, s = tid.rsplit("#", 1)
        p = 0.0 if base != q.ground_truth_tool_id else (P_GOOD if s == self.good_now(base) else P_BAD)
        ok = self.rng.random() < p
        self.t += 1
        return Observation(query=q, tool_id=tid, capability_used=q.required_capability, success=ok)


def run_stream(sel, env, twins=True):
    succ, right, rel = [], [], []
    for q in env.queries:
        tid = sel.select(q, env)
        base = tid.rsplit("#", 1)[0] if twins else tid
        if twins:
            rel.append(float(base == q.ground_truth_tool_id and tid.endswith("#" + env.good_now(base))))
        right.append(float(base == q.ground_truth_tool_id))
        obs = env.execute(tid, q)
        succ.append(float(obs.success))
        sel.update(tid, q.required_capability, obs.success)
    return np.array(succ), np.array(right), np.array(rel) if twins else None


def curve(x):
    c = np.cumsum(np.insert(x, 0, 0.0))
    return ((c[WINDOW:] - c[:-WINDOW]) / WINDOW)[::10]


def agg(runs):
    a = np.array(runs)
    h = a.shape[1] // 2
    return {"mean": float(100 * a.mean()), "sd": float(100 * a.mean(1).std(ddof=1)),
            "first_half": float(100 * a[:, :h].mean()), "second_half": float(100 * a[:, h:].mean()),
            "last_quarter": float(100 * a[:, -a.shape[1] // 4:].mean()), "last_quarter_sd": float(100 * a[:, -a.shape[1] // 4:].mean(1).std(ddof=1)),
            "curve": (100 * np.mean([curve(r) for r in a], 0)).round(2).tolist()}


def part_twins(seeds, res, t0):
    train, ev = metatool_split(10, max_queries=8000)  # 2,000 training / 6,000 distinct evaluation queries, about 30 per tool
    qt = build_query_texts_from_scenarios(ev)
    emb = DenseEmbedder("all-MiniLM-L6-v2")
    ft = STEmbedder(finetune(train, 3))
    docs = list({t.description for s in ev for t in s.candidate_tools})
    for e in (emb, ft):
        e.encode_batch(list(qt.values()) + docs)
    fed = CachedExtractor(DenseFeatureExtractor(qt, emb))
    fft = CachedExtractor(DenseFeatureExtractor(qt, ft))

    def make(seed):
        d = {"DenseDescMatch": Online(ArgmaxFeatureSelector(fed, 0, seed)),
             "DenseDescMatch-FT": Online(ArgmaxFeatureSelector(fft, 0, seed)),
             "LinUCB-Dense": Online(LinUCBSelector(fed, seed=seed))}
        fg = dict(rel_window=8, mem_cap=8, forget=True)  # short windows; failures evict stored successes
        for name, kw, alpha in [("+Mem (no forgetting)", dict(memory=True, reliability=False), 1.0), ("+Rel (window 30)", dict(memory=False, reliability=True), 1.0),
                                ("+Mem+Rel (no forgetting)", dict(memory=True, reliability=True), 1.0),
                                ("+Mem", dict(memory=True, reliability=False, **fg), 1.0), ("+Rel", dict(memory=False, reliability=True, **fg), 1.0),
                                ("+Mem+Rel", dict(memory=True, reliability=True, **fg), 1.0), ("+Mem+Rel (greedy)", dict(memory=True, reliability=True, **fg), 0.0)]:
            f = FeedbackExtractor(fed, emb, **kw)
            d["LinUCB-Dense" + name] = Online(LinUCBSelector(f, seed=seed, alpha=alpha), f)
        return d

    for tag, flip in [("twins", None), ("outage", len(ev) // 2)]:
        per = {}
        for seed in seeds:
            for name, sel in make(seed).items():
                s, r, g = run_stream(sel, TwinEnv(ev, seed, flip))
                d = per.setdefault(name, {"success": [], "right_tool": [], "reliable_twin": []})
                d["success"].append(s); d["right_tool"].append(r); d["reliable_twin"].append(g)
        res[tag] = {n: {k: agg(v) for k, v in d.items()} for n, d in per.items()}
        res[tag]["_oracle_success"] = 100 * P_GOOD
        for n, d in res[tag].items():
            if n[0] != "_":
                print(f"[{time.time()-t0:4.0f}s] {tag:7s} {n:30s} success {d['success']['mean']:5.1f}±{d['success']['sd']:.1f}  "
                      f"last quarter {d['success']['last_quarter']:5.1f}  reliable twin chosen {d['reliable_twin']['mean']:5.1f} (last q. {d['reliable_twin']['last_quarter']:.1f})", flush=True)


class DriftEnv:
    """BFCL stream whose documentation changes at mid-stream (variant A before, variant B after)."""

    def __init__(self, a, b, seed):
        self.a, self.b = {s.scenario_id: s for s in a}, {s.scenario_id: s for s in b}
        self.queries = [Query(query_id=i, required_capability=s.scenario_id, ground_truth_tool_id=s.ground_truth_tool_id) for i, s in enumerate(a)]
        np.random.default_rng(seed).shuffle(self.queries)
        self.t, self.half, self.tools = 0, len(a) // 2, []

    def get_declarers(self, cap):
        return list((self.a if self.t < self.half else self.b)[cap].candidate_tools)

    def execute(self, tid, q):
        self.t += 1
        return Observation(query=q, tool_id=tid, capability_used=q.required_capability, success=tid == q.ground_truth_tool_id)


def part_drift(seeds, res, t0):
    scen = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios
    train = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="multiple").scenarios
    qt = build_query_texts_from_scenarios(scen)
    fe = ToolFeatureExtractor(qt)
    emb = DenseEmbedder("all-MiniLM-L6-v2")
    emb.encode_batch(list(qt.values()) + list({t.description for s in scen for t in s.candidate_tools}))
    logit = train_offline_logistic(train, ToolFeatureExtractor(build_query_texts_from_scenarios(train)))
    per = {}
    for seed in seeds:
        fed = DenseFeatureExtractor(qt, emb)
        sels = {"TF-IDF-Desc": Online(ArgmaxFeatureSelector(fe, 0, seed)), "DenseDescMatch": Online(ArgmaxFeatureSelector(fed, 0, seed)),
                "Sup-Offline": Online(SupervisedOfflineSelector(fe, logit, seed=seed)),
                "LinUCB": Online(LinUCBSelector(fe, seed=seed)), "LinUCB-Dense": Online(LinUCBSelector(fed, seed=seed))}
        for name, sel in sels.items():
            _, r, _ = run_stream(sel, DriftEnv(scen, degrade_desc(scen, "D3s", seed), seed), twins=False)
            per.setdefault(name, []).append(r)
    res["drift"] = {n: {"tsa": agg(v)} for n, v in per.items()}
    for n, d in res["drift"].items():
        print(f"[{time.time()-t0:4.0f}s] drift   {n:16s} TSA first half {d['tsa']['first_half']:5.1f}  second half {d['tsa']['second_half']:5.1f}  last quarter {d['tsa']['last_quarter']:5.1f}", flush=True)


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.1))
    styles = {"DenseDescMatch": ("#888888", "--"), "DenseDescMatch-FT": ("#444444", ":"), "LinUCB-Dense": ("#1f77b4", "-."),
              "LinUCB-Dense+Mem+Rel": ("#d62728", "-"), "TF-IDF-Desc": ("#888888", ":"), "Sup-Offline": ("#2ca02c", "-."), "LinUCB": ("#d62728", "-")}
    for a, key, metric, title, names in [
            (ax[0], "twins", "success", "(a) Hidden reliability", ["DenseDescMatch", "DenseDescMatch-FT", "LinUCB-Dense", "LinUCB-Dense+Mem+Rel"]),
            (ax[1], "outage", "success", "(b) Reliability swaps at mid-stream", ["DenseDescMatch", "DenseDescMatch-FT", "LinUCB-Dense", "LinUCB-Dense+Mem+Rel"]),
            (ax[2], "drift", "tsa", "(c) Descriptions swap at mid-stream", ["DenseDescMatch", "Sup-Offline", "LinUCB", "LinUCB-Dense"])]:
        for n in names:
            c = res[key][n][metric]["curve"]
            col, ls = styles[n] if not (key == "drift" and n == "LinUCB-Dense") else ("#1f77b4", "-")
            a.plot(WINDOW + 10 * np.arange(len(c)), c, color=col, ls=ls, lw=1.3, label=n.replace("LinUCB-Dense+Mem+Rel", "LinUCB-Dense + feedback features"))
        if key != "twins":
            a.axvline((WINDOW + 10 * len(c)) / 2, color="k", lw=0.5, alpha=0.5)
        if key != "drift":
            a.axhline(100 * P_GOOD, color="k", lw=0.5, ls=":", alpha=0.6)
        a.set_title(title, fontsize=8); a.set_xlabel("decision"); a.set_ylim(0, 100)
        a.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=2.6)
    ax[0].set_ylabel("task success (%), window 100"); ax[2].set_ylabel("TSA (%), window 100")
    fig.tight_layout(); fig.savefig("adaptation.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--figure-only", action="store_true"); a = ap.parse_args()
    if a.figure_only:
        import json
        figure(json.load(open("results/revision_r1/adaptation.json"))); raise SystemExit
    res, t0 = {}, time.time()
    part_twins(list(range(a.seeds)), res, t0)
    part_drift(list(range(a.seeds)), res, t0)
    save("adaptation", res)
    figure(res)
