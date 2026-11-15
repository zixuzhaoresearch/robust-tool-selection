"""5.9 Runtime/Latency comparison: wall-clock time per selection step."""
import time
import numpy as np
from tats.bfcl_adapter import BFCLEnvironment, BFCLToolSelectionBenchmark
from tats.bfcl_selectors import (
    DescriptionMatchSelector, ParameterMatchSelector, RandomBFCLSelector,
    EpsilonGreedyBFCLSelector, UCBBFCLSelector,
)
from tats.bilints_selector import (
    LinUCBSelector, ColdStartLinTSSelector, SupervisedOnlineSelector,
)
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.experiment import run_selector

# Load data
bm = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple")
scenarios = bm.scenarios
qt = build_query_texts_from_scenarios(scenarios)
fe = ToolFeatureExtractor(qt)
n_queries = len(BFCLEnvironment(scenarios, seed=42).queries)

print(f"Benchmarking on {n_queries} queries (BFCL live_multiple)")
print("=" * 70)

# Feature extraction setup time
t0 = time.perf_counter()
fe2 = ToolFeatureExtractor(qt)
fe_setup_time = time.perf_counter() - t0
print(f"TF-IDF feature extractor setup: {fe_setup_time*1000:.1f} ms")

selectors = [
    ("Random", lambda: RandomBFCLSelector(seed=42)),
    ("DescMatch", lambda: DescriptionMatchSelector(qt, seed=42)),
    ("ParamMatch", lambda: ParameterMatchSelector(qt, seed=42)),
    ("EpsGreedy", lambda: EpsilonGreedyBFCLSelector(epsilon=0.1, seed=42)),
    ("UCB", lambda: UCBBFCLSelector(seed=42)),
    ("LinUCB", lambda: LinUCBSelector(fe, seed=42)),
    ("BilinTS-Cold", lambda: ColdStartLinTSSelector(feature_extractor=fe, seed=42)),
    ("Sup-Online", lambda: SupervisedOnlineSelector(feature_extractor=fe, seed=42)),
]

# Try dense if available
try:
    from tats.dense_features import DenseEmbedder, DenseFeatureExtractor, DenseDescMatchSelector
    embedder = DenseEmbedder()

    t0 = time.perf_counter()
    fe_dense = DenseFeatureExtractor(qt, embedder)
    dense_setup_time = time.perf_counter() - t0
    print(f"Dense feature extractor setup (model load + batch encode): {dense_setup_time*1000:.1f} ms")

    selectors.extend([
        ("DenseDescMatch", lambda: DenseDescMatchSelector(qt, embedder, seed=42)),
        ("LinUCB-Dense", lambda: LinUCBSelector(fe_dense, seed=42)),
    ])
except ImportError:
    print("sentence-transformers not available, skipping dense selectors")

print()
print(f"{'Selector':<20} {'Total (s)':<12} {'Per-step (ms)':<15} {'TSA':<8}")
print("-" * 55)

for name, factory in selectors:
    # Run 3 times and take min (reduce noise)
    times = []
    for _ in range(3):
        sel2 = factory()
        env2 = BFCLEnvironment(scenarios, seed=42)
        t0 = time.perf_counter()
        r = run_selector(name, sel2, env2)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    best_time = min(times)
    per_step_ms = best_time / n_queries * 1000

    print(f"{name:<20} {best_time:<12.3f} {per_step_ms:<15.3f} {r.tsa:<8.4f}")

print()
print("NOTE: Times include both select() and update() per step.")
print("Dense setup includes model loading + batch encoding of all queries.")


if __name__ == "__main__":
    pass
