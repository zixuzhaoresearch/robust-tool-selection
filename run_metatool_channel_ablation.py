"""Per-channel feature ablation on MetaTool ambiguity.

[MetaTool ambiguity] clarifying how much of the gain comes from name_sim
versus n_cand versus the online learner would make the mechanism more
convincing."

For each K in {5, 10, 20, 50}, run LinUCB on MetaTool with:
  full   — all 8 features (schemas absent on MetaTool, so schema features are 0)
  -desc  — zero out desc_sim (idx 0)
  -name  — zero out name_sim (idx 1)
  -desc_len — zero out desc_len_norm (idx 4)
  -n_cand — zero out n_cand_norm (idx 5)
  -has_desc — zero out has_desc (idx 7)

Reports TSA per (K, ablation), 5 seeds. Baseline DescMatch reference is
included so we can attribute the LinUCB gap to feature channels.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

from tats.metatool_adapter import MetaToolBenchmark
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.bilints_selector import LinUCBSelector


# Feature indices in the 8-dim vector
IDX = {
    "desc_sim":     0,
    "name_sim":     1,
    "param_match":  2,   # schema — all zero on MetaTool
    "n_params":     3,   # schema
    "desc_len":     4,
    "n_cand":       5,
    "req_ratio":    6,   # schema
    "has_desc":     7,
}


class MaskedFeatureExtractor:
    """Wraps a base FE and zeroes out one or more feature dimensions."""
    def __init__(self, base_fe, mask_indices: list[int]):
        self._base = base_fe
        self._mask = mask_indices

    @property
    def n_features(self):
        return self._base.n_features

    @property
    def _query_texts(self):
        return self._base._query_texts

    def extract(self, query_id, tool, n_candidates):
        x = self._base.extract(query_id, tool, n_candidates).copy()
        for i in self._mask:
            x[i] = 0.0
        return x


def descmatch_select(query_id, candidates, fe):
    """Rank by desc_sim only — DescMatch reference selector."""
    best = None; best_s = -np.inf
    for t in candidates:
        x = fe.extract(query_id, t, len(candidates))
        if x[0] > best_s:
            best_s = x[0]; best = t.tool_id
    return best


def run_one(bm, fe, mask_name: str, mask_indices: list[int], seed: int) -> float:
    fe_wrap = MaskedFeatureExtractor(fe, mask_indices) if mask_indices else fe
    sel = LinUCBSelector(fe_wrap, seed=seed)
    correct = 0
    for q in bm.queries:
        decl = bm.get_declarers(q.required_capability)
        tid = sel.select(q, bm)
        obs = bm.execute(tid, q)
        if obs.success:
            correct += 1
        sel.update(tid, q.required_capability, obs.success)
    return correct / len(bm.queries)


def run_descmatch(bm, fe) -> float:
    correct = 0
    for q in bm.queries:
        decl = bm.get_declarers(q.required_capability)
        tid = descmatch_select(q.query_id, decl, fe)
        obs = bm.execute(tid, q)
        if obs.success:
            correct += 1
    return correct / len(bm.queries)


ABLATIONS = [
    ("full",         []),
    ("-desc_sim",    [IDX["desc_sim"]]),
    ("-name_sim",    [IDX["name_sim"]]),
    ("-desc_len",    [IDX["desc_len"]]),
    ("-n_cand",      [IDX["n_cand"]]),
    ("-has_desc",    [IDX["has_desc"]]),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Ks", type=int, nargs="+", default=[5, 10, 20, 50])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--max-queries", type=int, default=1500)
    p.add_argument("--output", default="results/metatool_channel_ablation.json")
    args = p.parse_args()

    t0 = time.time()
    out = {"Ks": args.Ks, "seeds": args.seeds, "results": {}}

    for K in args.Ks:
        print(f"\n=== K={K} ===")
        per_ablation = {name: [] for name, _ in ABLATIONS}
        descmatch_rows = []
        for seed in args.seeds:
            bench = MetaToolBenchmark(n_candidates=K, seed=seed, max_queries=args.max_queries,
                                       similarity_mode="embedding")
            bm = bench.to_sequential_env()
            qt = build_query_texts_from_scenarios(bench.scenarios)
            fe = ToolFeatureExtractor(qt)
            dm = run_descmatch(bm, fe)
            descmatch_rows.append(dm)
            print(f"  seed={seed} DescMatch: {dm*100:5.2f}", flush=True)
            for name, mask in ABLATIONS:
                tsa = run_one(bm, fe, name, mask, seed)
                per_ablation[name].append(tsa)
                delta_vs_dm = (tsa - dm) * 100
                print(f"    {name:12s}: TSA={tsa*100:5.2f}  vs DescMatch={delta_vs_dm:+5.2f} pp",
                      flush=True)

        agg = {}
        dm_m = float(np.mean(descmatch_rows) * 100)
        dm_s = float(np.std(descmatch_rows) * 100)
        agg["descmatch"] = {"mean": dm_m, "std": dm_s}
        for name in per_ablation:
            v = np.array(per_ablation[name])
            m = float(v.mean() * 100)
            s = float(v.std() * 100)
            agg[name] = {"mean": m, "std": s, "delta_pp_vs_descmatch": m - dm_m}
        out["results"][K] = agg
        print(f"\n  K={K} aggregate ({len(args.seeds)} seeds):")
        print(f"    {'DescMatch':<15}: {dm_m:5.2f} ± {dm_s:.2f}")
        for name, mask in ABLATIONS:
            a = agg[name]
            print(f"    {name:<15}: {a['mean']:5.2f} ± {a['std']:.2f}  "
                  f"(Δ vs DescMatch = {a['delta_pp_vs_descmatch']:+.2f} pp)")

    out["wall_time_sec"] = time.time() - t0
    Path(args.output).parent.mkdir(exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {args.output} ({out['wall_time_sec']:.1f}s)")


if __name__ == "__main__":
    main()
