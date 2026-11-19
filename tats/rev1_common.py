"""Shared helpers for the experiment scripts (rollout, seed summary, paired bootstrap, simple selectors).

Every run records, per scenario, whether the ground-truth tool was selected and
whether the injected tool was selected, so that paired bootstrap intervals over
scenarios can be computed afterwards (rev1_bootstrap in stats_bootstrap_gamma.py).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

OUT = Path("results/revision_r1")


def rollout(sel, env, n_scen, mal_ids=None, feedback=None):
    """Run one selector over env.queries.

    Returns (correct, attacked): float arrays of length n_scen indexed by
    query_id (scenario index). `feedback(obs_success, tid, q) -> bool | None`
    optionally rewrites the reward the selector sees (None = no update).
    """
    correct = np.zeros(n_scen)
    attacked = np.zeros(n_scen)
    for q in env.queries:
        tid = sel.select(q, env)
        obs = env.execute(tid, q)
        correct[q.query_id] = float(tid == q.ground_truth_tool_id)
        if mal_ids:
            attacked[q.query_id] = float(tid == mal_ids.get(q.required_capability, ""))
        r = obs.success if feedback is None else feedback(obs.success, tid, q)
        if r is not None and hasattr(sel, "update"):
            sel.update(tid, q.required_capability, bool(r))
    return correct, attacked


def summarize(per_seed):
    """per_seed: list of per-scenario arrays -> mean/sample-SD over seeds (%), per-scenario mean."""
    a = np.asarray(per_seed, dtype=float)
    seed_means = a.mean(axis=1) * 100
    return {
        "mean": float(seed_means.mean()),
        "sd": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
        "seeds": seed_means.tolist(),
        "per_scenario": a.mean(axis=0).round(4).tolist(),
    }


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    json.dump(obj, open(path, "w"))
    print(f"saved {path}", flush=True)


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """95% percentile CI of mean(a-b) in pp, resampling scenarios (a, b per-scenario means)."""
    d = (np.asarray(a) - np.asarray(b)) * 100
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


class ArgmaxFeatureSelector:
    """Non-learning baseline: pick the candidate with the largest value of one feature."""

    def __init__(self, fe, index, seed=0):
        self._fe, self._i = fe, index
        self.rng = np.random.default_rng(seed)

    def select(self, query, env):
        cands = env.get_declarers(query.required_capability)
        s = np.array([self._fe.extract(query.query_id, t, len(cands))[self._i] for t in cands])
        best = np.flatnonzero(s == s.max())
        return cands[int(self.rng.choice(best))].tool_id


class MaskedExtractor:
    """Zero out feature coordinates (drop-one ablations)."""

    def __init__(self, base, drop):
        self._base, self._drop = base, list(drop)
        self._query_texts = base._query_texts

    @property
    def n_features(self):
        return self._base.n_features

    def extract(self, qid, tool, n):
        x = self._base.extract(qid, tool, n).copy()
        x[self._drop] = 0.0
        return x


class CachedExtractor:
    """Memoize extract() per (query, tool content, n)."""

    def __init__(self, base):
        self._base = base
        self._query_texts = base._query_texts
        self._cache = {}

    @property
    def n_features(self):
        return self._base.n_features

    def extract(self, qid, tool, n):
        key = (qid, tool.tool_id, tool.name, tool.description, tuple(tool.parameter_spec.get("properties", {})),
               tuple(tool.parameter_spec.get("required", [])), n)
        x = self._cache.get(key)
        if x is None:
            x = self._cache[key] = self._base.extract(qid, tool, n)
        return x


def tokens(text):
    return re.findall(r"\w+", text.lower())
