"""Revision 1, experiments E2a / E6 / E7: stronger label-using baselines, catalog scale,
and feature importance.

Reviewer 2 (comments 2 and 6), Reviewer 1 (comments 2 and 3), Reviewer 3.
  ft        - a dense retriever fine-tuned on the SAME labelled data that Sup-Offline and
              the warm start use (BFCL: the other split; MetaTool: the 25% train split),
              plus an online re-ranker over embedding features that sees the same bandit
              feedback as LinUCB
  scale     - MetaTool K in {5,...,199 (full catalog)} and BFCL with distractors pooled
              from the whole benchmark (up to every other unique function)
  features  - learned weights and drop-one-feature ablation on BFCL

Output: results/revision_r1/{ft,scale,features}.json
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np

from tats.bfcl_adapter import BFCLEnvironment, BFCLScenario, BFCLToolSelectionBenchmark
from tats.bfcl_selectors import DescriptionMatchSelector, ParameterMatchSelector
from tats.bilints_selector import LinUCBSelector, SupervisedOfflineSelector
from tats.dense_features import DenseDescMatchSelector, DenseEmbedder, DenseFeatureExtractor
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.metatool_adapter import MetaToolBenchmark, MetaToolEnvironment
from tats.rev1_common import ArgmaxFeatureSelector, CachedExtractor, MaskedExtractor, rollout, save, summarize
from tats.warmstart import train_offline_logistic

FEATURES = ["desc_sim", "name_sim", "param_match", "n_params", "desc_len", "n_cand", "req_ratio", "has_desc"]


def metatool_split(K, max_queries=2000, train_frac=0.25):
    scen = MetaToolBenchmark(n_candidates=K, seed=0, max_queries=max_queries).scenarios
    idx = np.random.default_rng(0).permutation(len(scen))
    n = int(len(scen) * train_frac)
    return [scen[i] for i in idx[:n]], [scen[i] for i in idx[n:]]


# ---------------------------------------------------------------- fine-tuned retriever
def finetune(train, epochs, lr=2e-5, batch=16, temp=0.05, seed=0):
    """Contrastive fine-tuning of MiniLM: query vs. the candidates of its own scenario."""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(seed)
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    def enc(texts):
        f = model.tokenize(texts)
        f = {k: v.to(model.device) for k, v in f.items()}
        return torch.nn.functional.normalize(model(f)["sentence_embedding"], dim=-1)

    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(train))
        for s in range(0, len(order), batch):
            loss = 0.0
            for i in order[s:s + batch]:
                sc = train[i]
                docs = [t.description or t.name for t in sc.candidate_tools]
                gt = [t.tool_id for t in sc.candidate_tools].index(sc.ground_truth_tool_id)
                logits = enc([sc.query_text]) @ enc(docs).T / temp
                loss = loss + torch.nn.functional.cross_entropy(logits, torch.tensor([gt], device=model.device))
            opt.zero_grad()
            (loss / batch).backward()
            opt.step()
    model.eval()
    return model


class STEmbedder(DenseEmbedder):
    """DenseEmbedder around an already-loaded (fine-tuned) model."""

    def __init__(self, model):
        self._model = model
        self._cache = {}


class EmbRerankOnline:
    """Online re-ranker: linear UCB on [q_emb * d_emb, cosine] with the same bandit feedback as LinUCB."""

    def __init__(self, qt, emb, alpha, seed=0, lam=1.0, sigma2=0.25):
        self._qt, self._emb, self._alpha, self._s2 = qt, emb, alpha, sigma2
        d = 385
        self._Binv, self._f, self._theta = np.eye(d) / lam, np.zeros(d), np.zeros(d)
        self._x = {}

    def _feat(self, q, d):
        a, b = self._emb.encode(q), self._emb.encode(d)
        a, b = a / (np.linalg.norm(a) + 1e-9), b / (np.linalg.norm(b) + 1e-9)
        return np.append(a * b * np.sqrt(len(a)), float(a @ b))

    def select(self, query, env):
        cands = env.get_declarers(query.required_capability)
        q = self._qt[query.query_id]
        self._x = {t.tool_id: self._feat(q, t.description) for t in cands}
        X = np.array([self._x[t.tool_id] for t in cands])
        s = X @ self._theta + self._alpha * np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X, self._Binv, X), 0))
        return cands[int(np.argmax(s))].tool_id

    def update(self, tid, cap, success):
        x = self._x[tid]
        Bx = self._Binv @ x  # Sherman-Morrison for B += x x^T / sigma2
        self._Binv -= np.outer(Bx, Bx) / (self._s2 + x @ Bx)
        self._f += x * float(success) / self._s2
        self._theta = self._Binv @ self._f


def part_ft(seeds):
    res = {}
    bf_eval = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios
    bf_train = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="multiple").scenarios
    mt_train, mt_eval = metatool_split(10)
    for tag, train, ev, env_cls, epochs in [("bfcl", bf_train, bf_eval, BFCLEnvironment, 5),
                                           ("metatool", mt_train, mt_eval, MetaToolEnvironment, 3)]:
        qt = build_query_texts_from_scenarios(ev)
        base = DenseEmbedder("all-MiniLM-L6-v2")
        ft = STEmbedder(finetune(train, epochs))
        docs = list({t.description for s in ev for t in s.candidate_tools})
        for e in (base, ft):
            e.encode_batch(list(qt.values()) + docs)
        fe = CachedExtractor(ToolFeatureExtractor(qt))
        logit = train_offline_logistic(train, ToolFeatureExtractor(build_query_texts_from_scenarios(train)))
        per = {}
        for seed in seeds:
            sels = {"TFIDF-Desc": ArgmaxFeatureSelector(fe, 0, seed),
                    "DenseDescMatch": DenseDescMatchSelector(qt, base, seed=seed),
                    "DenseDescMatch-FT": DenseDescMatchSelector(qt, ft, seed=seed),
                    "Sup-Offline": SupervisedOfflineSelector(fe, logit, seed=seed),
                    "LinUCB": LinUCBSelector(fe, seed=seed),
                    "LinUCB-Dense": LinUCBSelector(DenseFeatureExtractor(qt, base), seed=seed),
                    "LinUCB-Dense-FT": LinUCBSelector(DenseFeatureExtractor(qt, ft), seed=seed),
                    "EmbRerank-Online-a1": EmbRerankOnline(qt, base, 1.0, seed),
                    "EmbRerank-Online-a0": EmbRerankOnline(qt, base, 0.0, seed)}
            for name, sel in sels.items():
                c, _ = rollout(sel, env_cls(ev, seed=seed), len(ev))
                per.setdefault(name, []).append(c)
        res[tag] = {k: summarize(v) for k, v in per.items()}
        res[tag]["_train_size"] = len(train)
        print(tag, "  ".join(f"{k}={v['mean']:.1f}±{v['sd']:.1f}" for k, v in res[tag].items() if k[0] != "_"), flush=True)
    save("ft", res)


# ---------------------------------------------------------------- catalog scale
def pooled(scen, n_extra, seed):
    """Add n_extra distractors drawn from every other function of the benchmark (unique by name)."""
    pool = {}
    for s in scen:
        for t in s.candidate_tools:
            pool.setdefault(t.name, t)
    names = sorted(pool)
    rng = np.random.default_rng(seed)
    out = []
    for s in scen:
        own = {t.name for t in s.candidate_tools}
        avail = [n for n in names if n not in own]
        pick = avail if n_extra >= len(avail) else list(rng.choice(avail, size=n_extra, replace=False))
        out.append(BFCLScenario(scenario_id=s.scenario_id, query_text=s.query_text,
                                candidate_tools=list(s.candidate_tools) + [pool[n] for n in pick],
                                ground_truth_tool_id=s.ground_truth_tool_id,
                                ground_truth_args=copy.deepcopy(s.ground_truth_args)))
    return out


def part_scale(seeds):
    res = {"metatool": {}, "bfcl_pooled": {}}
    emb = DenseEmbedder("all-MiniLM-L6-v2")
    t0 = time.time()
    for K in [5, 10, 20, 50, 100, 199]:
        _, ev = metatool_split(K)
        qt = build_query_texts_from_scenarios(ev)
        emb.encode_batch(list(qt.values()) + list({t.description for s in ev for t in s.candidate_tools}))
        fe = CachedExtractor(ToolFeatureExtractor(qt))
        fed = CachedExtractor(DenseFeatureExtractor(qt, emb))
        per = {}
        for seed in seeds:
            sels = {"DescMatch": DescriptionMatchSelector(qt, seed=seed), "TFIDF-Desc": ArgmaxFeatureSelector(fe, 0, seed),
                    "DenseDescMatch": DenseDescMatchSelector(qt, emb, seed=seed),
                    "LinUCB": LinUCBSelector(fe, seed=seed), "LinUCB-Dense": LinUCBSelector(fed, seed=seed)}
            for name, sel in sels.items():
                c, _ = rollout(sel, MetaToolEnvironment(ev, seed=seed), len(ev))
                per.setdefault(name, []).append(c)
        res["metatool"][str(K)] = {k: summarize(v) for k, v in per.items()}
        print(f"[{time.time()-t0:5.0f}s] metatool K={K:3d} " + "  ".join(f"{k}={v['mean']:.1f}±{v['sd']:.1f}" for k, v in res["metatool"][str(K)].items()), flush=True)
    scen = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios
    qt = build_query_texts_from_scenarios(scen)
    emb.encode_batch(list(qt.values()) + list({t.description for s in scen for t in s.candidate_tools}))
    base_fe, base_fed = ToolFeatureExtractor(qt), DenseFeatureExtractor(qt, emb)
    for n_extra in [0, 10, 50, 200, 10**6]:
        per, sizes = {}, []
        for seed in seeds[:5]:
            cur = pooled(scen, n_extra, seed)
            sizes.append(np.mean([len(s.candidate_tools) for s in cur]))
            fe, fed = CachedExtractor(base_fe), CachedExtractor(base_fed)
            sels = {"DescMatch": DescriptionMatchSelector(qt, seed=seed), "TFIDF-Desc": ArgmaxFeatureSelector(fe, 0, seed),
                    "DenseDescMatch": DenseDescMatchSelector(qt, emb, seed=seed),
                    "ParamMatch": ParameterMatchSelector(qt, seed=seed),
                    "LinUCB": LinUCBSelector(fe, seed=seed), "LinUCB-Dense": LinUCBSelector(fed, seed=seed)}
            for name, sel in sels.items():
                c, _ = rollout(sel, BFCLEnvironment(cur, seed=seed), len(scen))
                per.setdefault(name, []).append(c)
        key = "all" if n_extra >= 10**6 else str(n_extra)
        res["bfcl_pooled"][key] = {k: summarize(v) for k, v in per.items()}
        res["bfcl_pooled"][key]["_mean_candidates"] = float(np.mean(sizes))
        print(f"[{time.time()-t0:5.0f}s] bfcl +{key:>3s} (K̄={np.mean(sizes):.0f}) " + "  ".join(
            f"{k}={v['mean']:.1f}±{v['sd']:.1f}" for k, v in res["bfcl_pooled"][key].items() if k[0] != "_"), flush=True)
    save("scale", res)


# ---------------------------------------------------------------- feature importance
def part_features(seeds):
    scen = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios
    fe = CachedExtractor(ToolFeatureExtractor(build_query_texts_from_scenarios(scen)))
    per, thetas = {}, []
    for seed in seeds:
        for name, drop in [("Full", [])] + [(f"-{f}", [i]) for i, f in enumerate(FEATURES)]:
            sel = LinUCBSelector(MaskedExtractor(fe, drop), seed=seed)
            c, _ = rollout(sel, BFCLEnvironment(scen, seed=seed), len(scen))
            per.setdefault(name, []).append(c)
            if not drop:
                thetas.append(sel._theta_mean.copy())
    X = np.array([fe.extract(i, t, len(s.candidate_tools)) for i, s in enumerate(scen) for t in s.candidate_tools])
    res = {"drop_one": {k: summarize(v) for k, v in per.items()}, "features": FEATURES,
           "theta_mean": np.mean(thetas, 0).tolist(), "theta_sd": np.std(thetas, 0, ddof=1).tolist(),
           "feature_sd": X.std(0).tolist(), "feature_mean": X.mean(0).tolist()}
    for i, f in enumerate(FEATURES):
        print(f"{f:12s} theta {res['theta_mean'][i]:+.3f}±{res['theta_sd'][i]:.3f}  theta*sd(x) {res['theta_mean'][i]*res['feature_sd'][i]:+.3f}  "
              f"drop-one TSA {res['drop_one']['-'+f]['mean']:.1f} (full {res['drop_one']['Full']['mean']:.1f})", flush=True)
    save("features", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["ft", "scale", "features"])
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    {"ft": part_ft, "scale": part_scale, "features": part_features}[args.part](list(range(args.seeds)))
