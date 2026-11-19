"""Revision 1, audit pass: re-run numbers carried over from the submitted version for which no
result file had been kept, and one consistency run.

  glm      - Logistic Thompson sampling vs Gaussian TS vs LinUCB on BFCL (Table A6), per-seed values
  idf      - global vs rolling IDF (recomputed every 100 queries from the queries seen so far)
  negvar   - TW-LinUCB and the consistency-gated bandit under L1 attacks (Table A9)
  mt_adab  - ADAB on the MetaTool 1,500-query evaluation split (so that Tables 4 and 5 use one split)

Output: results/revision_r1/legacy_checks.json
"""
import math
from collections import Counter

import numpy as np
from scipy import stats

from label_baselines import metatool_split
from tats.adversarial import AdversarialBFCLEnvironment
from tats.bfcl_adapter import BFCLEnvironment, BFCLToolSelectionBenchmark
from tats.bilints_selector import ColdStartLinTSSelector, LinUCBSelector, LogisticTSSelector
from tats.dense_features import DenseEmbedder, DenseFeatureExtractor
from tats.features import ToolFeatureExtractor, _tokenize, build_query_texts_from_scenarios
from tats.metatool_adapter import MetaToolEnvironment
from tats.novel_selectors import ADABEmaSelector, ConsistencyGatedSelector, TWLinUCBSelector
from tats.rev1_common import CachedExtractor, rollout, save, summarize

SEEDS = list(range(10))
res = {}
scen = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios
qt = build_query_texts_from_scenarios(scen)
fe = ToolFeatureExtractor(qt)

# ---- glm
per = {}
for seed in SEEDS:
    for name, sel in [("LinUCB", LinUCBSelector(fe, seed=seed)), ("BilinTS-Cold", ColdStartLinTSSelector(feature_extractor=fe, seed=seed)),
                      ("GLM-TS", LogisticTSSelector(fe, seed=seed))]:
        per.setdefault(name, []).append(rollout(sel, BFCLEnvironment(scen, seed=seed), len(scen))[0])
res["glm"] = {k: summarize(v) for k, v in per.items()}
s = {k: np.array(v["seeds"]) for k, v in res["glm"].items()}
res["glm"]["paired_t_over_seeds"] = {f"{a} vs {b}": float(stats.ttest_rel(s[a], s[b]).pvalue)
                                      for a, b in [("GLM-TS", "LinUCB"), ("BilinTS-Cold", "LinUCB"), ("GLM-TS", "BilinTS-Cold")]}
print("glm", {k: (round(v["mean"], 1), round(v["sd"], 1)) for k, v in res["glm"].items() if "mean" in v}, res["glm"]["paired_t_over_seeds"], flush=True)


# ---- rolling IDF
class RollingIDF(ToolFeatureExtractor):
    """IDF from the queries seen so far, refreshed every `every` queries (neutral IDF before the first refresh)."""

    def __init__(self, texts, every=100):
        super().__init__(texts)
        self._idf, self._seen, self._every, self._order = {}, [], every, set()

    def extract(self, qid, tool, n):
        if qid not in self._order:
            self._order.add(qid)
            self._seen.append(qid)
            if len(self._seen) % self._every == 0:
                df = Counter(t for q in self._seen for t in set(_tokenize(self._query_texts[q])))
                self._idf = {t: math.log(len(self._seen) / (1 + c)) + 1.0 for t, c in df.items()}
        return super().extract(qid, tool, n)


per = {}
for seed in SEEDS:
    for name, mk in [("LinUCB", lambda f: LinUCBSelector(f, seed=seed)), ("BilinTS-Cold", lambda f: ColdStartLinTSSelector(feature_extractor=f, seed=seed))]:
        per.setdefault(f"{name}/global", []).append(rollout(mk(fe), BFCLEnvironment(scen, seed=seed), len(scen))[0])
        per.setdefault(f"{name}/rolling", []).append(rollout(mk(RollingIDF(qt)), BFCLEnvironment(scen, seed=seed), len(scen))[0])
res["idf"] = {k: summarize(v) for k, v in per.items()}
print("idf", {k: (round(v["mean"], 1), round(v["sd"], 1)) for k, v in res["idf"].items()}, flush=True)

# ---- negative variants
per = {}
for strategy in [None, "desc_copy", "param_steal", "full_adaptive"]:
    for seed in SEEDS:
        for name, sel in [("LinUCB", LinUCBSelector(fe, seed=seed)), ("TW-LinUCB", TWLinUCBSelector(fe, seed=seed)),
                          ("CGB", ConsistencyGatedSelector(fe, seed=seed))]:
            if strategy is None:
                c, a = rollout(sel, BFCLEnvironment(scen, seed=seed), len(scen))
            else:
                env = AdversarialBFCLEnvironment(scen, strategy=strategy, seed=seed)
                c, a = rollout(sel, env, len(scen), dict(env._mal_tool_ids))
            d = per.setdefault(f"{strategy or 'benign'}/{name}", {"tsa": [], "asr": []})
            d["tsa"].append(c); d["asr"].append(a)
res["negvar"] = {k: {"tsa": summarize(v["tsa"]), "asr": summarize(v["asr"])} for k, v in per.items()}
for k, v in res["negvar"].items():
    print("negvar", k, f"TSA {v['tsa']['mean']:.1f}±{v['tsa']['sd']:.1f} ASR {v['asr']['mean']:.1f}±{v['asr']['sd']:.1f}", flush=True)

# ---- ADAB on the MetaTool evaluation split
_, ev = metatool_split(10)
mqt = build_query_texts_from_scenarios(ev)
emb = DenseEmbedder("all-MiniLM-L6-v2")
emb.encode_batch(list(mqt.values()) + list({t.description for s in ev for t in s.candidate_tools}))
mfe = CachedExtractor(ToolFeatureExtractor(mqt))
per = {}
for strategy in [None, "desc_copy"]:
    for seed in SEEDS:
        mfed = DenseFeatureExtractor(mqt, emb)
        for name, sel in [("LinUCB", LinUCBSelector(mfe, seed=seed)), ("ADAB-g2", ADABEmaSelector(mfe, seed=seed, gamma=2.0, quantile=0.99, decay=0.99)),
                          ("LinUCB-Dense", LinUCBSelector(mfed, seed=seed)), ("ADAB-Dense-g2", ADABEmaSelector(mfed, seed=seed, gamma=2.0, quantile=0.99, decay=0.99))]:
            if strategy is None:
                c, a = rollout(sel, MetaToolEnvironment(ev, seed=seed), len(ev))
            else:
                env = AdversarialBFCLEnvironment(ev, strategy=strategy, seed=seed)
                c, a = rollout(sel, env, len(ev), dict(env._mal_tool_ids))
            d = per.setdefault(f"{strategy or 'benign'}/{name}", {"tsa": [], "asr": []})
            d["tsa"].append(c); d["asr"].append(a)
res["mt_adab"] = {k: {"tsa": summarize(v["tsa"]), "asr": summarize(v["asr"])} for k, v in per.items()}
for k, v in res["mt_adab"].items():
    print("mt_adab", k, f"TSA {v['tsa']['mean']:.1f}±{v['tsa']['sd']:.1f} ASR {v['asr']['mean']:.1f}±{v['asr']['sd']:.1f}", flush=True)
save("legacy_checks", res)
