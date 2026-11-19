"""Attribution of the usage-memory gain (Table 23): the same success-filtered memory without bandit updates.

Fixed-score selectors on MetaTool: score = desc_sim + usage_sim (equal fixed weights, no learning), with a
TF-IDF or a dense desc_sim, and a memory-only selector (usage_sim, ties broken by desc_sim).
Output: results/revision_r1/memattr.json
"""
import time

import numpy as np

from run_rev1_baselines import metatool_split
from run_rev4_memory import MemoryExtractor, WithMemory
from tats.bilints_selector import LinUCBSelector
from tats.dense_features import DenseEmbedder, DenseFeatureExtractor
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.metatool_adapter import MetaToolEnvironment
from tats.rev1_common import ArgmaxFeatureSelector, CachedExtractor, rollout, save, summarize


class FixedScore:
    """score = w . x with fixed weights; no learning."""

    def __init__(self, fe, w, seed=0):
        self._fe, self._w, self.rng = fe, w, np.random.default_rng(seed)

    def select(self, query, env):
        cands = env.get_declarers(query.required_capability)
        s = np.array([sum(c * self._fe.extract(query.query_id, t, len(cands))[i] for i, c in self._w.items()) for t in cands])
        best = np.flatnonzero(s >= s.max() - 1e-12)
        return cands[int(self.rng.choice(best))].tool_id

    def update(self, *a):
        pass


if __name__ == "__main__":
    emb = DenseEmbedder("all-MiniLM-L6-v2")
    res, t0 = {}, time.time()
    for K in [10, 199]:
        _, scen = metatool_split(K)
        qt = build_query_texts_from_scenarios(scen)
        emb.encode_batch(list(qt.values()) + list({t.description for s in scen for t in s.candidate_tools}))
        fe, fed = CachedExtractor(ToolFeatureExtractor(qt)), CachedExtractor(DenseFeatureExtractor(qt, emb))
        nf = fe.n_features
        per = {}
        for seed in range(10):
            sels = {"TFIDF-Desc": ArgmaxFeatureSelector(fe, 0, seed), "DenseDescMatch": ArgmaxFeatureSelector(fed, 0, seed)}
            for tag, f in [("TFIDF", fe), ("Dense", fed)]:
                m = MemoryExtractor(f, emb); sels[f"{tag}+Mem (fixed sum)"] = WithMemory(FixedScore(m, {0: 1.0, nf: 1.0}, seed), m)
                m = MemoryExtractor(f, emb); sels[f"Memory only ({tag} tie-break)"] = WithMemory(FixedScore(m, {0: 1e-3, nf: 1.0}, seed), m)
                m = MemoryExtractor(f, emb); sels[f"LinUCB{'-Dense' if tag == 'Dense' else ''}+Mem"] = WithMemory(LinUCBSelector(m, seed=seed), m)
                sels[f"LinUCB{'-Dense' if tag == 'Dense' else ''}"] = LinUCBSelector(f, seed=seed)
            for name, sel in sels.items():
                c, _ = rollout(sel, MetaToolEnvironment(scen, seed=seed), len(scen))
                per.setdefault(name, []).append(c)
        res[f"metatool_K{K}"] = {n: summarize(v) for n, v in per.items()}
        print(f"[{time.time()-t0:4.0f}s] K={K}  " + "  ".join(f"{n}={v['mean']:.1f}±{v['sd']:.1f}" for n, v in res[f"metatool_K{K}"].items()), flush=True)
    save("memattr", res)
