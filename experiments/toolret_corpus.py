"""Revision 1, second batch: retrieval baselines and a 37k-tool catalog (ToolRet web corpus).

  retrievers - BM25, MiniLM, bge-small, e5-small as description retrievers on BFCL and MetaTool
  toolret    - 2,466 single-gold queries against all 37,292 web tools: first-stage retrieval
               (hit@1, recall@10/50), then selection among the 50 candidates returned by the
               MiniLM first stage (the gold tool is NOT forced into the candidate set), benign and
               with one injected L1 entry per query

Output: results/revision_r1/{retrievers,toolret}.json
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from label_baselines import metatool_split
from tats.adversarial import AdversarialBFCLEnvironment
from tats.bfcl_adapter import BFCLEnvironment, BFCLToolSelectionBenchmark
from tats.bfcl_selectors import DescriptionMatchSelector, ParameterMatchSelector
from tats.bilints_selector import LinUCBSelector
from tats.dense_features import DenseEmbedder, DenseFeatureExtractor
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.novel_selectors import ADABEmaSelector
from tats.rev1_common import ArgmaxFeatureSelector, CachedExtractor, rollout, save, summarize, tokens

MODELS = {"MiniLM-L6": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
          "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
          "e5-small": ("intfloat/e5-small-v2", "query: ", "passage: ")}


def embed(name, queries, docs):
    from sentence_transformers import SentenceTransformer
    repo, qp, dp = MODELS[name]
    m = SentenceTransformer(repo, device="cuda")
    kw = dict(batch_size=256, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    return m.encode([qp + q for q in queries], **kw), m.encode([dp + d for d in docs], **kw)


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        from scipy.sparse import csr_matrix
        self.vocab = {}
        rows, cols, vals, lens = [], [], [], []
        for i, d in enumerate(docs):
            tk = tokens(d)
            lens.append(len(tk))
            cnt = {}
            for t in tk:
                cnt[t] = cnt.get(t, 0) + 1
            for t, c in cnt.items():
                rows.append(i); cols.append(self.vocab.setdefault(t, len(self.vocab))); vals.append(c)
        n, avg = len(docs), max(np.mean(lens), 1e-9)
        tf = csr_matrix((vals, (rows, cols)), shape=(n, len(self.vocab)), dtype=np.float64)
        df = np.bincount(tf.indices, minlength=len(self.vocab))
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * np.asarray(lens) / avg)
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
        self.M = csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).T.tocsr()  # term x doc

    def scores(self, query):
        ids = [self.vocab[t] for t in set(tokens(query)) if t in self.vocab]
        return np.asarray(self.M[ids].sum(axis=0)).ravel() if ids else np.zeros(self.M.shape[1])


def part_retrievers():
    res = {}
    for tag, scen in [("bfcl", BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios),
                      ("metatool", metatool_split(10)[1])]:
        docs = sorted({t.description for s in scen for t in s.candidate_tools})
        di = {d: i for i, d in enumerate(docs)}
        qs = [s.query_text for s in scen]
        rng = np.random.default_rng(0)

        def tsa(score_fn):
            ok = 0
            for i, s in enumerate(scen):
                sc = np.asarray(score_fn(i, [di[t.description] for t in s.candidate_tools]))
                best = np.flatnonzero(sc >= sc.max() - 1e-12)
                ok += s.candidate_tools[int(rng.choice(best))].tool_id == s.ground_truth_tool_id
            return 100 * ok / len(scen)

        bm = BM25(docs)
        r = {"BM25": tsa(lambda i, idx: bm.scores(qs[i])[idx])}
        for name in MODELS:
            Q, D = embed(name, qs, docs)
            r[name] = tsa(lambda i, idx: D[idx] @ Q[i])
        res[tag] = r
        print(tag, {k: round(v, 1) for k, v in r.items()}, flush=True)
    save("retrievers", res)


def part_toolret(seeds, K=50):
    from tats.toolret_adapter import load_queries, load_tools, scenarios_from_candidates
    tools = load_tools()
    queries = load_queries(tools)
    ids = list(tools)
    pos = {t: i for i, t in enumerate(ids)}
    docs = [f"{tools[t].name}: {tools[t].description}" for t in ids]
    qs = [q for q, _, _ in queries]
    gold = np.array([pos[g] for _, g, _ in queries])
    res = {"n_tools": len(ids), "n_queries": len(queries), "K": K, "first_stage": {}, "second_stage": {}}
    t0 = time.time()

    def metrics(S):
        rank = (S > S[np.arange(len(gold)), gold][:, None]).sum(1)
        return {"hit@1": float(100 * np.mean(rank < 1)), "recall@10": float(100 * np.mean(rank < 10)),
                "recall@50": float(100 * np.mean(rank < 50)), "per_query_hit1": (rank < 1).astype(int).tolist()}

    bm = BM25(docs)
    res["first_stage"]["BM25"] = metrics(np.stack([bm.scores(q) for q in qs]))
    top = None
    for name in MODELS:
        Q, D = embed(name, qs, docs)
        S = Q @ D.T
        res["first_stage"][name] = metrics(S)
        if name == "MiniLM-L6":
            top = np.argsort(-S, axis=1)[:, :K]
    for k, v in res["first_stage"].items():
        print(f"[{time.time()-t0:4.0f}s] first stage {k:10s} hit@1 {v['hit@1']:.1f}  R@10 {v['recall@10']:.1f}  R@50 {v['recall@50']:.1f}", flush=True)

    scen = scenarios_from_candidates(queries, [[ids[j] for j in row] for row in top], tools)
    qt = build_query_texts_from_scenarios(scen)
    emb = DenseEmbedder("all-MiniLM-L6-v2")
    emb.encode_batch(qs + list({t.description for s in scen for t in s.candidate_tools}))
    fe = CachedExtractor(ToolFeatureExtractor(qt))
    fed = CachedExtractor(DenseFeatureExtractor(qt, emb))
    for strategy in [None, "desc_copy", "full_adaptive"]:
        per = {}
        for seed in seeds:
            sels = {"DescMatch": DescriptionMatchSelector(qt, seed=seed), "TFIDF-Desc": ArgmaxFeatureSelector(fe, 0, seed),
                    "ParamMatch": ParameterMatchSelector(qt, seed=seed),
                    "DenseDescMatch": ArgmaxFeatureSelector(fed, 0, seed),
                    "LinUCB": LinUCBSelector(fe, seed=seed), "ADAB-g2": ADABEmaSelector(fe, seed=seed, gamma=2.0, quantile=0.99, decay=0.99),
                    "LinUCB-Dense": LinUCBSelector(fed, seed=seed),
                    "ADAB-Dense-g2": ADABEmaSelector(fed, seed=seed, gamma=2.0, quantile=0.99, decay=0.99)}
            for name, sel in sels.items():
                if strategy is None:
                    c, a = rollout(sel, BFCLEnvironment(scen, seed=seed), len(scen))
                else:
                    env = AdversarialBFCLEnvironment(scen, strategy=strategy, seed=seed)
                    c, a = rollout(sel, env, len(scen), dict(env._mal_tool_ids))
                d = per.setdefault(name, {"tsa": [], "asr": []})
                d["tsa"].append(c); d["asr"].append(a)
            print(f"[{time.time()-t0:4.0f}s] {strategy or 'benign'} seed {seed}", flush=True)
        key = strategy or "benign"
        res["second_stage"][key] = {n: {"tsa": summarize(v["tsa"]), "asr": summarize(v["asr"])} for n, v in per.items()}
        for n, v in res["second_stage"][key].items():
            print(f"toolret {key:14s} {n:16s} TSA {v['tsa']['mean']:5.1f}±{v['tsa']['sd']:.1f}  ASR {v['asr']['mean']:6.2f}±{v['asr']['sd']:.2f}", flush=True)
    save("toolret", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["retrievers", "toolret"])
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    part_retrievers() if args.part == "retrievers" else part_toolret(list(range(args.seeds)))
