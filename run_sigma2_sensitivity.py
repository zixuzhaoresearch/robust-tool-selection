"""5.5 σ² Sensitivity Analysis: LinUCB and BilinTS-Cold with different σ² values."""
import numpy as np
from scipy import stats
from tats.bfcl_adapter import BFCLEnvironment, BFCLToolSelectionBenchmark
from tats.bilints_selector import LinUCBSelector, ColdStartLinTSSelector
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.experiment import run_selector

SEEDS = list(range(10))
SIGMA2_VALUES = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

bm = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple")
scenarios = bm.scenarios
qt = build_query_texts_from_scenarios(scenarios)
fe = ToolFeatureExtractor(qt)

print("=" * 70)
print("σ² Sensitivity Analysis (BFCL live_multiple, 10 seeds)")
print("=" * 70)

results = {}
for sigma2 in SIGMA2_VALUES:
    linucb_tsas = []
    bilints_tsas = []
    for seed in SEEDS:
        env = BFCLEnvironment(scenarios, seed=seed)

        sel = LinUCBSelector(fe, seed=seed, sigma2=sigma2)
        r = run_selector("LinUCB", sel, env)
        linucb_tsas.append(r.tsa)

        sel = ColdStartLinTSSelector(feature_extractor=fe, seed=seed, noise_var=sigma2)
        r = run_selector("BilinTS", sel, env)
        bilints_tsas.append(r.tsa)

    results[sigma2] = {
        "LinUCB": {"mean": np.mean(linucb_tsas), "std": np.std(linucb_tsas), "raw": linucb_tsas},
        "BilinTS-Cold": {"mean": np.mean(bilints_tsas), "std": np.std(bilints_tsas), "raw": bilints_tsas},
    }
    print(f"σ²={sigma2:<5} LinUCB={np.mean(linucb_tsas):.4f}±{np.std(linucb_tsas):.4f}  BilinTS-Cold={np.mean(bilints_tsas):.4f}±{np.std(bilints_tsas):.4f}")

# Print formatted table
print("\n" + "=" * 70)
print("Summary Table")
print("=" * 70)
print(f"{'σ²':<8} {'LinUCB':<18} {'BilinTS-Cold':<18} {'Δ(LinUCB-Best)':<18}")
print("-" * 62)
best_linucb = max(r["LinUCB"]["mean"] for r in results.values())
for sigma2 in SIGMA2_VALUES:
    lm = results[sigma2]["LinUCB"]["mean"]
    ls = results[sigma2]["LinUCB"]["std"]
    bm_ = results[sigma2]["BilinTS-Cold"]["mean"]
    bs = results[sigma2]["BilinTS-Cold"]["std"]
    delta = lm - best_linucb
    print(f"{sigma2:<8} {lm:.4f}±{ls:.4f}     {bm_:.4f}±{bs:.4f}     {delta:+.4f}")

# Significance test: σ²=0.25 vs others
ref_sigma2 = 0.25
ref_linucb = results[ref_sigma2]["LinUCB"]["raw"]
ref_bilints = results[ref_sigma2]["BilinTS-Cold"]["raw"]
print(f"\nSignificance vs σ²={ref_sigma2}:")
for sigma2 in SIGMA2_VALUES:
    if sigma2 == ref_sigma2:
        continue
    t_l, p_l = stats.ttest_rel(results[sigma2]["LinUCB"]["raw"], ref_linucb)
    t_b, p_b = stats.ttest_rel(results[sigma2]["BilinTS-Cold"]["raw"], ref_bilints)
    sig_l = "***" if p_l < 0.001 else "**" if p_l < 0.01 else "*" if p_l < 0.05 else "ns"
    sig_b = "***" if p_b < 0.001 else "**" if p_b < 0.01 else "*" if p_b < 0.05 else "ns"
    print(f"  σ²={sigma2}: LinUCB p={p_l:.4f} {sig_l}, BilinTS p={p_b:.4f} {sig_b}")


if __name__ == "__main__":
    pass
