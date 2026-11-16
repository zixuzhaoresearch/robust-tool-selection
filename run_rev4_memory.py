"""Revision 1, third batch: ideas taken from recent work.

  memory     - a usage-memory channel (in the spirit of Tool2Vec and online-optimized tool retrieval):
               each tool is also represented by the queries on which it was selected AND succeeded;
               two extra features, usage_sim (mean of the top-3 cosines between the query and those
               stored queries) and usage_seen (log-count of stored successes). Success-filtered like ADAB.
  ppl        - perplexity filtering of tool descriptions (GPT-2), the standard detection baseline
  toolbench  - the ToolBench / ToolLLM API retriever as a description retriever

Output: results/revision_r1/{memory,ppl,toolbench_ir}.json
"""
import argparse
import math
import time

import numpy as np

from run_rev1_baselines import metatool_split
from tats.adversarial import AdversarialBFCLEnvironment, craft_malicious_tool
from tats.bfcl_adapter import BFCLEnvironment, BFCLToolSelectionBenchmark
from tats.bilints_selector import LinUCBSelector
from tats.dense_features import DenseEmbedder, DenseFeatureExtractor
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.metatool_adapter import MetaToolEnvironment
from tats.novel_selectors import ADABEmaSelector
from tats.rev1_common import ArgmaxFeatureSelector, CachedExtractor, rollout, save, summarize


class MemoryExtractor:
    """base features + [usage_sim, usage_seen]; memory is keyed by tool_id and holds query embeddings."""

    def __init__(self, base, emb, k=3):
        self._base, self._emb, self._k = base, emb, k
        self._query_texts = base._query_texts
        self._mem, self._q = {}, {}

    @property
    def n_features(self):
        return self._base.n_features + 2

    def _qvec(self, qid):
        v = self._q.get(qid)
        if v is None:
            v = self._emb.encode(self._query_texts[qid])
            v = self._q[qid] = v / (np.linalg.norm(v) + 1e-9)
        return v

    def extract(self, qid, tool, n):
        m = self._mem.get(tool.tool_id)
        if m:
            s = np.sort(np.asarray(m) @ self._qvec(qid))[-self._k:]
            extra = [max(0.0, float(s.mean())), min(math.log1p(len(m)) / math.log(21), 1.0)]
        else:
            extra = [0.0, 0.0]
        return np.append(self._base.extract(qid, tool, n), extra)

    def remember(self, tool_id, qid):
        self._mem.setdefault(tool_id, []).append(self._qvec(qid))


class WithMemory:
    """Feeds successful (query, tool) pairs to the memory; everything else is the wrapped selector."""

    def __init__(self, sel, fe):
        self._sel, self._fe, self._qid = sel, fe, None

    def select(self, query, env):
        self._qid = query.query_id
        return self._sel.select(query, env)

    def update(self, tid, cap, success):
        self._sel.update(tid, cap, success)
        if success:
            self._fe.remember(tid, self._qid)


def mem_selectors(fe, fed, emb, seed, adab=False):
    out = {}
    for tag, f in [("LinUCB", fe), ("LinUCB-Dense", fed)]:
        out[tag] = LinUCBSelector(f, seed=seed)
        m = MemoryExtractor(f, emb)
        out[tag + "+Mem"] = WithMemory(LinUCBSelector(m, seed=seed), m)
        if adab:
            out[tag.replace("LinUCB", "ADAB") + "-g2"] = ADABEmaSelector(f, seed=seed, gamma=2.0, quantile=0.99, decay=0.99)
            m2 = MemoryExtractor(f, emb)
            out[tag.replace("LinUCB", "ADAB") + "+Mem-g2"] = WithMemory(ADABEmaSelector(m2, seed=seed, gamma=2.0, quantile=0.99, decay=0.99), m2)
    return out


def run_block(scen, env_cls, seeds, emb, strategies, tag, res, t0):
    qt = build_query_texts_from_scenarios(scen)
    emb.encode_batch(list(qt.values()) + list({t.description for s in scen for t in s.candidate_tools}))
    fe, fed = CachedExtractor(ToolFeatureExtractor(qt)), CachedExtractor(DenseFeatureExtractor(qt, emb))
    for strategy in strategies:
        per = {}
        for seed in seeds:
            sels = mem_selectors(fe, fed, emb, seed, adab=strategy is not None or tag.endswith("K10") or tag == "bfcl")
            if strategy is None:
                sels["TFIDF-Desc"] = ArgmaxFeatureSelector(fe, 0, seed)
                sels["DenseDescMatch"] = ArgmaxFeatureSelector(fed, 0, seed)
            for name, sel in sels.items():
                if strategy is None:
                    c, a = rollout(sel, env_cls(scen, seed=seed), len(scen))
                else:
                    env = AdversarialBFCLEnvironment(scen, strategy=strategy, seed=seed)
                    c, a = rollout(sel, env, len(scen), dict(env._mal_tool_ids))
                d = per.setdefault(name, {"tsa": [], "asr": []})
                d["tsa"].append(c); d["asr"].append(a)
        key = f"{tag}/{strategy or 'benign'}"
        res[key] = {n: {"tsa": summarize(v["tsa"]), "asr": summarize(v["asr"])} for n, v in per.items()}
        print(f"[{time.time()-t0:5.0f}s] {key:26s} " + "  ".join(
            f"{n}={v['tsa']['mean']:.1f}" + (f"/{v['asr']['mean']:.1f}" if strategy else "") for n, v in res[key].items()), flush=True)


def part_memory(seeds):
    emb = DenseEmbedder("all-MiniLM-L6-v2")
    res, t0 = {}, time.time()
    for K in [5, 10, 20, 50, 199]:
        _, ev = metatool_split(K)
        run_block(ev, MetaToolEnvironment, seeds, emb, [None] + (["desc_copy"] if K == 10 else []), f"metatool_K{K}", res, t0)
    scen = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios
    run_block(scen, BFCLEnvironment, seeds, emb, [None, "desc_copy", "full_adaptive"], "bfcl", res, t0)
    save("memory", res)
    from tats.toolret_adapter import load_queries, load_tools, scenarios_from_candidates
    from sentence_transformers import SentenceTransformer
    tools = load_tools(); queries = load_queries(tools); ids = list(tools)
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    kw = dict(batch_size=256, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    S = m.encode([q for q, _, _ in queries], **kw) @ m.encode([f"{tools[t].name}: {tools[t].description}" for t in ids], **kw).T
    top = np.argsort(-S, axis=1)[:, :50]
    tr = scenarios_from_candidates(queries, [[ids[j] for j in row] for row in top], tools)
    run_block(tr, BFCLEnvironment, seeds[:5], emb, [None, "desc_copy"], "toolret", res, t0)
    save("memory", res)


def part_ppl():
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); lm = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda").eval()

    @torch.no_grad()
    def ppl(texts):
        out = []
        for t in texts:
            ids = tok(t or " ", return_tensors="pt", truncation=True, max_length=256)["input_ids"].to("cuda")
            out.append(float(torch.exp(lm(ids, labels=ids).loss)) if ids.shape[1] > 1 else float("inf"))
        return np.array(out)

    res = {}
    for tag, scen in [("bfcl", BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios), ("metatool", metatool_split(10)[1])]:
        benign = ppl(sorted({t.description for s in scen for t in s.candidate_tools if len(t.description) > 10}))
        benign = benign[np.isfinite(benign)]
        r = {"benign_ppl_median": float(np.median(benign)), "n_benign": int(len(benign))}
        for atk in ["desc_copy", "full_adaptive"]:
            mal = ppl([craft_malicious_tool(s.scenario_id, s.query_text, atk).description for s in scen])
            mal = mal[np.isfinite(mal)]
            r[atk] = {"mal_ppl_median": float(np.median(mal)),
                      "tpr_at_1pct_fpr": float(100 * np.mean(mal > np.quantile(benign, 0.99))),
                      "tpr_at_5pct_fpr": float(100 * np.mean(mal > np.quantile(benign, 0.95))),
                      "auroc": float(np.mean(mal[:, None] > benign[None, :]) + 0.5 * np.mean(mal[:, None] == benign[None, :]))}
        res[tag] = r
        print(tag, r, flush=True)
    save("ppl", res)


def part_toolbench():
    from sentence_transformers import SentenceTransformer
    from tats.toolret_adapter import load_queries, load_tools
    m = SentenceTransformer("ToolBench/ToolBench_IR_bert_based_uncased", device="cuda")
    kw = dict(batch_size=128, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    res = {}
    for tag, scen in [("bfcl", BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios), ("metatool", metatool_split(10)[1])]:
        docs = sorted({t.description for s in scen for t in s.candidate_tools}); di = {d: i for i, d in enumerate(docs)}
        Q, D = m.encode([s.query_text for s in scen], **kw), m.encode(docs, **kw)
        res[tag] = 100 * float(np.mean([s.candidate_tools[int(np.argmax(D[[di[t.description] for t in s.candidate_tools]] @ Q[i]))].tool_id == s.ground_truth_tool_id
                                        for i, s in enumerate(scen)]))
    tools = load_tools(); queries = load_queries(tools); ids = list(tools); pos = {t: i for i, t in enumerate(ids)}
    gold = np.array([pos[g] for _, g, _ in queries])
    S = m.encode([q for q, _, _ in queries], **kw) @ m.encode([f"{tools[t].name}: {tools[t].description}" for t in ids], **kw).T
    rank = (S > S[np.arange(len(gold)), gold][:, None]).sum(1)
    res["toolret"] = {"hit@1": float(100 * np.mean(rank < 1)), "recall@10": float(100 * np.mean(rank < 10)), "recall@50": float(100 * np.mean(rank < 50))}
    print(res, flush=True)
    save("toolbench_ir", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["memory", "ppl", "toolbench"])
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    {"memory": lambda: part_memory(list(range(a.seeds))), "ppl": part_ppl, "toolbench": part_toolbench}[a.part]()
