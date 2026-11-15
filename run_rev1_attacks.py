"""Revision 1, experiment E4: metadata-realizable attacks at every knowledge level.

Reviewer 2 (comments 4 and 7) / Reviewer 3. Every attack below builds an actual catalog
entry (name, description, JSON schema); its features come from the unmodified feature
extractor, and the candidate-count coordinate is whatever the catalog gives. Nothing is
set directly in feature space, and there is no percentile budget.

  L1-desc_copy / L1-full   handcrafted, query words only (submitted Table 7 attacks)
  L2-snapshot              one defender snapshot at t=500; the attacker commits ONE
                           construction recipe (chosen on its own shadow queries = the
                           warm-up queries) and instantiates it with query words afterwards
  L3-live                  live bandit state (theta, B) but not the anomaly penalty:
                           per-query best recipe for the unpenalized LinUCB score
  L4R-live                 live state including thresholds and gamma: per-query best
                           recipe for the defended score (exact over the recipe grid)
  shadow                   static entry, no access to the target query: the attacker knows
                           only earlier queries of the same task (same ground-truth tool)
                           and optimizes one description/name/schema for them. This is
                           the retrieval objective of tool-document attacks such as
                           ToolHijacker, with the attacker's shadow queries taken from the
                           warm-up stream.

Protocol as in run_verified_l4.py: 500 benign warm-up queries, the rest attacked, defender
updates online, injected tool always returns reward 0, benign candidates win ties.

Output: results/revision_r1/attacks_{bfcl,metatool}.json
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

from tats.adversarial import _extract_query_tokens, craft_malicious_tool
from tats.bfcl_adapter import BFCLEnvironment, BFCLTool, BFCLToolSelectionBenchmark
from tats.bilints_selector import LinUCBSelector
from tats.dense_features import DenseEmbedder, DenseFeatureExtractor
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.metatool_adapter import MetaToolBenchmark
from tats.models import Observation, Query
from tats.novel_selectors import ADABEmaSelector
from tats.optimizer_attack import count_scores
from tats.rev1_common import save, tokens

MAL = "__INJECTED__"
FILL = "utility service module endpoint handler interface wrapper component resource adapter".split()
DESC_FRACS = [0.25, 0.5, 0.75, 1.0]
DESC_FILL = [0, 30, 80]
NAME_GRID = [(k, u) for k in range(5) for u in range(3)]
SCHEMA_GRID = [(m, u, r) for m in [0, 1, 2, 3, 4, 6] for u in [0, 1, 2, 4] for r in [0, 1, 2]]


def tool(name="tool", desc="", props=(), req=()):
    return BFCLTool(tool_id=MAL, name=name, declared_capabilities=[], true_success_rates={},
                    description_affinity={}, description=desc,
                    parameter_spec={"type": "object", "properties": {p: {"type": "string"} for p in props},
                                    "required": list(req)})


def recipe_parts(query_text):
    """Concrete (descriptions, names, schemas) for one query; fixed grid shape for every query."""
    words = query_text.split()
    content = _extract_query_tokens(query_text, max_tokens=8)
    descs = [""]
    for f in DESC_FRACS:
        head = " ".join(words[:max(1, int(np.ceil(f * len(words))))])
        for n in DESC_FILL:
            descs.append((head + " " + " ".join(FILL[i % len(FILL)] for i in range(n))).strip())
    names = ["_".join(content[:k] + [f"zq{i}" for i in range(u)]) or "tool" for k, u in NAME_GRID]
    schemas = []
    for m, u, r in SCHEMA_GRID:
        props = content[:m] + [f"zp{i}" for i in range(u)]
        n_req = [0, (len(props) + 1) // 2, len(props)][r]
        schemas.append((props, props[:n_req]))
    return descs, names, schemas


class Grid:
    """Feature matrix of every recipe for one query, assembled from separable components."""

    def __init__(self, fe, qid, text, n_cand):
        descs, names, schemas = recipe_parts(text)
        self.parts = (descs, names, schemas)
        fd = np.array([fe.extract(qid, tool(desc=d), n_cand) for d in descs])
        fn = np.array([fe.extract(qid, tool(name=n), n_cand) for n in names])
        fs = np.array([fe.extract(qid, tool(props=p, req=r), n_cand) for p, r in schemas])
        self.shape = (len(descs), len(names), len(schemas))
        idx = np.array(list(itertools.product(*map(range, self.shape))))
        X = np.zeros((len(idx), 8))
        X[:, [0, 4, 7]] = fd[idx[:, 0]][:, [0, 4, 7]]
        X[:, 1] = fn[idx[:, 1], 1]
        X[:, [2, 3, 6]] = fs[idx[:, 2]][:, [2, 3, 6]]
        X[:, 5] = fd[0, 5]
        self.X, self.idx = X, idx

    def build(self, i):
        d, n, s = self.idx[i]
        props, req = self.parts[2][s]
        return tool(self.parts[1][n], self.parts[0][d], props, req)


class InjectionEnv:
    """Benign scenarios plus at most one injected candidate per query (appended last)."""

    def __init__(self, scenarios, seed):
        self.sc = {s.scenario_id: s for s in scenarios}
        self.queries = [Query(query_id=i, required_capability=s.scenario_id, ground_truth_tool_id=s.ground_truth_tool_id)
                        for i, s in enumerate(scenarios)]
        np.random.default_rng(seed).shuffle(self.queries)
        self.tools = [t for s in scenarios for t in s.candidate_tools]
        self.active = None  # injected tool for the current query, or None

    def get_declarers(self, cap):
        c = list(self.sc[cap].candidate_tools)
        return c + [self.active] if self.active is not None else c

    def execute(self, tid, q):
        return Observation(query=q, tool_id=tid, capability_used=q.required_capability,
                           success=tid == q.ground_truth_tool_id)


def state(sel):
    return (sel._theta_mean, sel._get_B_inv(), sel._alpha, getattr(sel, "_gamma", 0.0),
            getattr(sel, "_ema_quantile", None))


def score(X, st, penal=True):
    th, Bi, al, g, tau = st
    return count_scores(X, th, Bi, al, g if penal else 0.0, tau if penal else None)


def shadow_tool(shadows, sim):
    """Static entry optimized for the attacker's shadow queries (no target-query access)."""
    bag = {}
    for s in shadows:
        for t in set(_extract_query_tokens(s, max_tokens=50)):
            bag[t] = bag.get(t, 0) + 1
    top = sorted(bag, key=lambda t: (-bag[t], t))
    cands = [" ".join(shadows)] + [" ".join(top[:m]) for m in (5, 10, 20)] + list(shadows)

    def fit(c):
        others = [s for s in shadows if s != c] or shadows
        return np.mean([sim(s, c) for s in others])

    desc = max(cands, key=fit)
    return tool("_".join(top[:3]) or "tool", desc, top[:4], top[:4])


def make_defenders(fe, seed):
    return {"LinUCB": LinUCBSelector(fe, seed=seed),
            "ADAB-g2": ADABEmaSelector(fe, seed=seed, gamma=2.0, quantile=0.99, decay=0.99),
            "ADAB-g5": ADABEmaSelector(fe, seed=seed, gamma=5.0, quantile=0.99, decay=0.99)}


def run(benchmark, seeds, warm=500):
    if benchmark == "bfcl":
        scen = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios
    else:
        scen = MetaToolBenchmark(n_candidates=10, seed=42, max_queries=1500, similarity_mode="embedding").scenarios
    texts = build_query_texts_from_scenarios(scen)
    emb = DenseEmbedder("all-MiniLM-L6-v2")
    emb.encode_batch(list(texts.values()) + list({t.description for s in scen for t in s.candidate_tools}))
    attacks = ["none", "L1-desc_copy", "L1-full", "L2-snapshot", "L3-live", "L4R-live", "shadow"]
    res = {}
    t0 = time.time()
    for seed in seeds:
        env = InjectionEnv(scen, seed)
        warm_q, eval_q = env.queries[:warm], env.queries[warm:]
        fe = ToolFeatureExtractor({q.query_id: texts[q.query_id] for q in warm_q})  # IDF from warm-up only
        fe._query_texts = texts
        fed = DenseFeatureExtractor(texts, emb)
        fed._idf = fe._idf
        grids = {q.query_id: Grid(fe, q.query_id, texts[q.query_id], len(scen[q.query_id].candidate_tools) + 1)
                 for q in env.queries}
        # shadow queries: earlier (warm-up) queries of the same task
        by_task = {}
        for q in warm_q:
            by_task.setdefault(q.ground_truth_tool_id, []).append(texts[q.query_id])
        tf_sim = lambda a, b: fe._tfidf_cosine(tokens(a), b)
        shadow = {"tfidf": {k: shadow_tool(v, tf_sim) for k, v in by_task.items()},
                  "dense": {k: shadow_tool(v, emb.cosine_similarity) for k, v in by_task.items()}}
        feature_sets = [("tfidf", fe)] + ([("dense", fed)] if True else [])
        for fname, f in feature_sets:
            for dname, sel0 in make_defenders(f, seed).items():
                benign = BFCLEnvironment(scen, seed=seed)
                for q in warm_q:  # benign warm-up, shared by every attack on this defender
                    tid = sel0.select(q, benign)
                    sel0.update(tid, q.required_capability, tid == q.ground_truth_tool_id)
                snap = tuple(np.copy(v) if isinstance(v, np.ndarray) else v for v in state(sel0))
                for atk in attacks:
                    if fname == "dense" and atk in ("L2-snapshot", "L3-live", "L4R-live"):
                        continue  # the recipe grid is built on the TF-IDF extractor
                    import copy
                    sel = copy.deepcopy(sel0)
                    sel._fe = f
                    committed = None
                    if atk == "L2-snapshot":  # choose one recipe on the attacker's shadow (warm-up) queries
                        wins = np.zeros(len(grids[warm_q[0].query_id].X))
                        margin = np.zeros_like(wins)
                        for q in warm_q:
                            g = grids[q.query_id]
                            cands = scen[q.query_id].candidate_tools
                            bmax = max(float(score(f.extract(q.query_id, t, len(cands) + 1), snap)) for t in cands)
                            s = score(g.X, snap)
                            wins += s > bmax
                            margin += s - bmax
                        committed = int(np.lexsort((margin, wins))[-1])
                    n = c = a = 0
                    for q in eval_q:
                        g = grids[q.query_id]
                        txt = texts[q.query_id]
                        if atk == "none":
                            env.active = None
                        elif atk == "L1-desc_copy":
                            env.active = craft_malicious_tool("x", txt, "desc_copy")
                        elif atk == "L1-full":
                            env.active = craft_malicious_tool("x", txt, "full_adaptive")
                        elif atk == "L2-snapshot":
                            env.active = g.build(committed)
                        elif atk == "L3-live":
                            env.active = g.build(int(np.argmax(score(g.X, state(sel), penal=False))))
                        elif atk == "L4R-live":
                            i = int(np.argmax(score(g.X, state(sel))))
                            env.active = g.build(i)
                            x = f.extract(q.query_id, env.active, len(scen[q.query_id].candidate_tools) + 1)
                            assert np.allclose(x, g.X[i]), "assembled features differ from the real extractor"
                        elif atk == "shadow":
                            env.active = shadow[fname].get(q.ground_truth_tool_id)
                            if env.active is None:
                                continue  # no shadow query for this task: not attacked, not scored
                        if env.active is not None:
                            env.active.tool_id = MAL
                        tid = sel.select(q, env)
                        n += 1
                        c += tid == q.ground_truth_tool_id
                        a += tid == MAL
                        sel.update(tid, q.required_capability, tid == q.ground_truth_tool_id)
                    d = res.setdefault(f"{fname}/{dname}/{atk}", {"tsa": [], "asr": [], "n": []})
                    d["tsa"].append(100 * c / n); d["asr"].append(100 * a / n); d["n"].append(n)
        print(f"[{time.time()-t0:5.0f}s] {benchmark} seed {seed} done", flush=True)
    out = {}
    for k, v in res.items():
        out[k] = {"tsa_mean": float(np.mean(v["tsa"])), "tsa_sd": float(np.std(v["tsa"], ddof=1)) if len(seeds) > 1 else 0.0,
                  "asr_mean": float(np.mean(v["asr"])), "asr_sd": float(np.std(v["asr"], ddof=1)) if len(seeds) > 1 else 0.0,
                  "n_mean": float(np.mean(v["n"])), "per_seed": v}
        print(f"{benchmark} {k:34s} TSA {out[k]['tsa_mean']:5.1f}±{out[k]['tsa_sd']:4.1f}  "
              f"ASR {out[k]['asr_mean']:6.2f}±{out[k]['asr_sd']:5.2f}  n={out[k]['n_mean']:.0f}", flush=True)
    save(f"attacks_{benchmark}", {"warm": warm, "seeds": seeds, "grid_size": int(len(next(iter(grids.values())).X)), "results": out})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=["bfcl", "metatool"])
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    run(args.benchmark, list(range(args.seeds)))
