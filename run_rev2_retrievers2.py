"""Revision 1, second batch: retrieval baselines a reviewer would expect in 2026.

Contriever-msmarco (the shadow retriever of ToolHijacker), bge-large-en-v1.5, and the cross-encoder
re-ranker bge-reranker-v2-m3, on the BFCL / MetaTool candidate sets and on ToolRet (first stage over
37k tools; re-ranker on the MiniLM top-50).  Output: results/revision_r1/retrievers2.json
"""
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from run_rev1_baselines import metatool_split
from tats.bfcl_adapter import BFCLToolSelectionBenchmark
from tats.rev1_common import save
from tats.toolret_adapter import load_queries, load_tools

DENSE = {"contriever-msmarco": ("facebook/contriever-msmarco", "", False),
         "bge-large": ("BAAI/bge-large-en-v1.5", "Represent this sentence for searching relevant passages: ", True)}
KW = dict(batch_size=128, show_progress_bar=False, convert_to_numpy=True)


def enc(name, queries, docs):
    repo, qp, norm = DENSE[name]
    m = SentenceTransformer(repo, device="cuda")
    out = m.encode([qp + q for q in queries], normalize_embeddings=norm, **KW), m.encode(docs, normalize_embeddings=norm, **KW)
    del m; torch.cuda.empty_cache()
    return out


def main():
    res = {}
    rer = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512)
    for tag, scen in [("bfcl", BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios),
                      ("metatool", metatool_split(10)[1])]:
        docs = sorted({t.description for s in scen for t in s.candidate_tools}); di = {d: i for i, d in enumerate(docs)}
        qs = [s.query_text for s in scen]
        r = {}
        for name in DENSE:
            Q, D = enc(name, qs, docs)
            r[name] = 100 * float(np.mean([s.candidate_tools[int(np.argmax(D[[di[t.description] for t in s.candidate_tools]] @ Q[i]))].tool_id
                                           == s.ground_truth_tool_id for i, s in enumerate(scen)]))
        pairs, spans = [], []
        for s in scen:
            spans.append((len(pairs), len(s.candidate_tools)))
            pairs += [(s.query_text, t.description) for t in s.candidate_tools]
        sc = rer.predict(pairs, batch_size=128, show_progress_bar=False)
        r["bge-reranker-v2-m3"] = 100 * float(np.mean([s.candidate_tools[int(np.argmax(sc[a:a + n]))].tool_id == s.ground_truth_tool_id
                                                       for s, (a, n) in zip(scen, spans)]))
        res[tag] = r
        print(tag, {k: round(v, 1) for k, v in r.items()}, flush=True)

    tools = load_tools(); queries = load_queries(tools)
    ids = list(tools); pos = {t: i for i, t in enumerate(ids)}
    docs = [f"{tools[t].name}: {tools[t].description}" for t in ids]
    qs = [q for q, _, _ in queries]; gold = np.array([pos[g] for _, g, _ in queries])
    r = {}
    for name in DENSE:
        Q, D = enc(name, qs, docs)
        S = Q @ D.T
        rank = (S > S[np.arange(len(gold)), gold][:, None]).sum(1)
        r[name] = {"hit@1": float(100 * np.mean(rank < 1)), "recall@10": float(100 * np.mean(rank < 10)), "recall@50": float(100 * np.mean(rank < 50))}
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    S = m.encode(qs, normalize_embeddings=True, **KW) @ m.encode(docs, normalize_embeddings=True, **KW).T
    top = np.argsort(-S, axis=1)[:, :50]
    pairs = [(qs[i], docs[j]) for i in range(len(qs)) for j in top[i]]
    sc = np.asarray(rer.predict(pairs, batch_size=128, show_progress_bar=False)).reshape(len(qs), 50)
    r["MiniLM top-50 + bge-reranker-v2-m3"] = {"hit@1": float(100 * np.mean(top[np.arange(len(qs)), sc.argmax(1)] == gold))}
    res["toolret"] = r
    print("toolret", r, flush=True)
    save("retrievers2", res)


if __name__ == "__main__":
    main()
