"""Verified L4 suite: matched states, disjoint calibration, independent rollouts.

All eight feature coordinates are controlled by the box oracle, including
candidate-count and binary metadata coordinates. This is explicitly a relaxed
feature-space stress test, not a metadata-realizable attack.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from tats.bfcl_adapter import BFCLToolSelectionBenchmark, BFCLEnvironment
from tats.metatool_adapter import MetaToolBenchmark
from tats.features import ToolFeatureExtractor, build_query_texts_from_scenarios
from tats.novel_selectors import ADABEmaSelector
from tats.optimizer_attack import (
    WhiteboxAttacker, OptAttackFeatureExtractor, OptimizedAttackEnvironment,
    compute_benign_q99, count_scores, read_selector_state,
)

FEATURES = ["desc_sim", "name_sim", "param_match", "n_params",
            "desc_len", "n_cand", "req_ratio", "has_desc"]


def native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def encode(obj):
    return json.dumps(obj, default=native, allow_nan=False, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(encode(obj).encode()).hexdigest()


def score_state(sel):
    theta, inv, alpha, gamma, kind, params = read_selector_state(sel)
    if kind not in (None, "quantile"):
        raise ValueError("Suite supports count ADAB only")
    return dict(theta=theta, B_inv=inv, alpha=alpha, gamma=gamma,
                tau=params.get("tau_q"), n_updates=sel.n_updates)


def scored(x, state):
    return float(count_scores(x, state["theta"], state["B_inv"],
                              state["alpha"], state["gamma"], state["tau"]))


class RegistrySelector(ADABEmaSelector):
    """Same defender score/order as ADAB; protected bit penalizes injected IDs."""
    def select(self, query, env):
        super().select(query, env)  # extract once, initialize thresholds, cache
        state = score_state(self)
        best, winner = -np.inf, None
        for t in env.get_declarers(query.required_capability):
            s = scored(self._cached_features[t.tool_id], state)
            if t.tool_id in self.registry_untrusted:
                s -= self.registry_beta
            if s > best:  # strict >, benign candidates come first
                best, winner = s, t.tool_id
        return winner


def paired_attack(state, upper, seed, exact):
    attacker = WhiteboxAttacker(d=len(upper), seed=seed)
    return attacker.optimize(
        state["theta"], state["B_inv"], state["alpha"], state["gamma"],
        anomaly_kind="quantile" if state["tau"] is not None else None,
        anomaly_params={"tau_q": state["tau"]} if state["tau"] is not None else {},
        upper_bounds=upper, hard_indicator=exact)


def prepare(benchmark, seed, warm, limit):
    if benchmark == "bfcl":
        bm = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple")
    else:
        bm = MetaToolBenchmark(n_candidates=10, seed=42, max_queries=1500,
                              similarity_mode="embedding")
    scenarios = bm.scenarios
    env = OptimizedAttackEnvironment(scenarios, seed=seed)
    if warm <= 0 or warm >= len(env.queries):
        raise ValueError("Warm-up must leave nonempty disjoint evaluation queries")
    warm_queries = env.queries[:warm]
    evaluation = env.queries[warm:]
    if limit is not None:
        evaluation = evaluation[:limit]
    texts = build_query_texts_from_scenarios(scenarios)
    calibration_ids = [q.query_id for q in warm_queries]
    # Fit IDF only on the calibration split, retaining true IDs for extraction.
    fe = ToolFeatureExtractor({i: texts[i] for i in calibration_ids})
    fe._query_texts = texts
    upper = compute_benign_q99(fe, scenarios, n_samples=len(calibration_ids),
                              query_indices=calibration_ids, injected_candidates=1)
    sel = RegistrySelector(fe, seed=seed, gamma=2., quantile=.99, decay=.99)
    sel.registry_untrusted = set()
    sel.registry_beta = 0.
    benign_env = BFCLEnvironment(scenarios, seed=seed)
    for q in warm_queries:
        tid = sel.select(q, benign_env)
        sel.update(tid, q.required_capability, benign_env.execute(tid, q).success)
    assert not set(calibration_ids) & {q.query_id for q in evaluation}
    return sel, env, fe, upper, evaluation, calibration_ids


def run_variant(initial, env, fe, upper, queries, seed, mode, solver, beta, path, fn=0., fp=0.):
    sel = copy.deepcopy(initial)
    mal_ids = set(env._mal_tool_ids.values())
    wrapper = OptAttackFeatureExtractor(fe, mal_ids, constraint="realistic",
                                       benign_q99=upper, hard_indicator=(solver == "exact"))
    sel._fe = wrapper
    wrapper.set_selector(sel)
    sel.registry_untrusted = mal_ids
    sel.registry_beta = beta
    initial_hash = digest(score_state(sel))
    injected_trusted = benign_flagged = 0
    correct = attacked = 0
    safe_wins = independent_safe_wins = 0
    paired_wins = []
    score_gaps = []
    score_consistency_max = 0.
    t0 = time.time()
    with gzip.open(path, "wt") as out:
        for index, q in enumerate(queries):
            state = score_state(sel)
            # Per-query optimizer seed makes repeated matched-state evaluations reproducible.
            attack_seed = seed * 1000003 + q.query_id
            wrapper._attacker = WhiteboxAttacker(d=fe.n_features, seed=attack_seed)
            declared = env.get_declarers(q.required_capability)
            # Trust-label noise: fn = P(injected tool labeled trusted), fp = P(benign tool labeled untrusted).
            if fn > 0 or fp > 0:
                label_rng = np.random.default_rng(attack_seed * 7919 + 17)
                untrusted = {t.tool_id for t in declared[:-1] if label_rng.random() < fp}
                benign_flagged += len(untrusted)
                if label_rng.random() < fn:
                    injected_trusted += 1
                else:
                    untrusted.add(declared[-1].tool_id)
            else:
                untrusted = mal_ids
            sel.registry_untrusted = untrusted
            xs = [wrapper.extract(q.query_id, t, len(declared)) for t in declared]
            scores = np.asarray([scored(x, state) for x in xs])
            assert declared[-1].tool_id in mal_ids
            witness = xs[-1]
            assert np.all(witness >= 0) and np.all(witness <= upper)
            benign_max = float(scores[:-1].max())
            penalized = scores.copy()
            for i, t in enumerate(declared):
                if t.tool_id in untrusted:
                    penalized[i] -= beta
            expected = declared[int(np.argmax(penalized))].tool_id
            chosen = sel.select(q, env)
            if chosen != expected:
                raise AssertionError("Instrumented and actual defender selection disagree")
            assert digest(score_state(sel)) == digest(state), "State changed within decision"
            assert np.array_equal(sel._cached_features[declared[-1].tool_id], witness)
            real_score = scored(sel._cached_features[declared[-1].tool_id], state)
            score_consistency_max = max(score_consistency_max, abs(real_score - scores[-1]))
            paired = None
            # Compare soft and exact on IDENTICAL exact-path pre-decision states.
            if solver == "exact" and beta == 0:
                sx, sv = paired_attack(state, upper, attack_seed, exact=False)
                assert np.all(sx >= 0) and np.all(sx <= upper)
                gap = float(scores[-1] - scored(sx, state))
                if gap < -1e-9:
                    raise AssertionError(f"Soft witness beats exact oracle by {-gap}")
                score_gaps.append(gap)
                paired_wins.append(float(scored(sx, state) > benign_max))
                paired = {"soft_witness": sx, "soft_true_score": scored(sx, state),
                          "exact_minus_soft_score": gap}
            tau = state["tau"]
            safe_upper = upper if tau is None else np.minimum(upper, tau)
            safe_x, _ = WhiteboxAttacker(d=fe.n_features).optimize(
                state["theta"], state["B_inv"], state["alpha"],
                upper_bounds=safe_upper, hard_indicator=True)
            safe_value = scored(safe_x, state)
            unpenalized_max = max(float(count_scores(
                x, state["theta"], state["B_inv"], state["alpha"])) for x in xs[:-1])
            safe_wins += safe_value > benign_max
            independent_safe_wins += safe_value > unpenalized_max
            if solver == "exact" and scores[-1] + 1e-9 < safe_value:
                raise AssertionError("Safe-zone witness beats exact oracle")
            obs = env.execute(chosen, q)
            is_attack = chosen in mal_ids
            attacked += is_attack
            correct += obs.success
            out.write(encode(dict(
                query_id=q.query_id, scenario_id=q.required_capability, index=index,
                attack_seed=attack_seed, state=state, upper=upper, witness=witness,
                benign_ids=[t.tool_id for t in declared[:-1]], benign_features=xs[:-1],
                benign_scores=scores[:-1], malicious_true_score=real_score,
                benign_max=benign_max, beta=beta, selected=chosen,
                ground_truth=q.ground_truth_tool_id, success=obs.success, attack_selected=is_attack,
                safe_witness=safe_x, safe_score=safe_value, benign_unpenalized_max=unpenalized_max,
                paired=paired)) + "\n")
            if mode == "online":
                sel.update(chosen, q.required_capability, obs.success)
            if (index + 1) % 250 == 0:
                print(f"  {mode}/{solver}/beta={beta:g}: {index+1}/{len(queries)}", flush=True)
    if mode == "frozen":
        assert digest(score_state(sel)) == initial_hash
    n = len(queries)
    result = dict(mode=mode, solver=solver, beta=beta, trust_fn=fn, trust_fp=fp, n=n,
                  injected_trusted_rate=injected_trusted/n, benign_flagged_per_query=benign_flagged/n,
                  tsa_pct=100*correct/n, asr_pct=100*attacked/n,
                  safe_floor_pct=100*safe_wins/n,
                  independent_safe_floor_pct=100*independent_safe_wins/n,
                  initial_state_hash=initial_hash, final_state_hash=digest(score_state(sel)),
                  score_consistency_max=score_consistency_max,
                  artifact=str(path), seconds=time.time()-t0)
    if paired_wins:
        result.update(paired_soft_asr_pct=100*np.mean(paired_wins),
                      exact_minus_soft_score_mean=float(np.mean(score_gaps)),
                      exact_minus_soft_score_min=float(np.min(score_gaps)))
    return result


def main(default_benchmark="both", default_suite="full"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["bfcl", "metatool", "both"], default=default_benchmark)
    p.add_argument("--suite", choices=["full", "compare", "registry", "snapshot", "registry-noise"], default=default_suite)
    p.add_argument("--mode", choices=["smoke", "full"], default="full")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--warm-queries", type=int, default=500)
    p.add_argument("--betas", nargs="+", type=float, default=[1., 2., 5.], help="registry-noise suite only")
    p.add_argument("--trust-fn", nargs="*", type=float, default=[0.01, 0.05, 0.1], help="P(injected tool labeled trusted)")
    p.add_argument("--trust-fp", nargs="*", type=float, default=[0.05, 0.1], help="P(benign tool labeled untrusted)")
    p.add_argument("--max-queries", type=int, help="Evaluation query cap, only for smoke checks")
    p.add_argument("--output", default="paper_tmlr/results/l4_verified/summary.json")
    args = p.parse_args()
    if args.mode == "smoke":
        args.seeds = [0]
        args.max_queries = args.max_queries or 20
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a published or partial run accidentally.
    if target.exists():
        raise FileExistsError(f"Choose a fresh output location: {target}")
    root = Path(__file__).resolve().parent
    hashes = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in [Path(__file__), root/"tats/optimizer_attack.py",
                        root/"tats/novel_selectors.py", root/"tats/features.py"]}
    variants = [("frozen", "exact", 0.), ("frozen", "soft", 0.),
                ("online", "exact", 0.), ("online", "soft", 0.)]
    if args.suite == "snapshot":
        variants = variants[:1]
    if args.suite in ("full", "registry"):
        variants += [("online", "exact", b) for b in [.1, .25, .5, 1., 2., 5.]]
    if args.suite == "registry":
        variants = [v for v in variants if v[0] == "online" and v[1] == "exact"]
    variants = [(m, s, b, 0., 0.) for m, s, b in variants]
    if args.suite == "registry-noise":
        settings = [(0., 0.)] + [(fn, 0.) for fn in args.trust_fn] + [(0., fp) for fp in args.trust_fp]
        variants = [("online", "exact", b, fn, fp) for b in args.betas for fn, fp in settings]
    result = dict(protocol_version=1, configuration=vars(args), source_sha256=hashes,
                  controlled_coordinates=FEATURES,
                  scope="All-eight-coordinate continuous box oracle; metadata realizability NOT assumed",
                  threshold_rule="strict x > tau; no epsilon; benign wins score ties",
                  calibration="500 disjoint benign warm-up queries; IDF/budget fit only on calibration",
                  benchmarks={})
    started = time.time()
    names = ["bfcl", "metatool"] if args.benchmark == "both" else [args.benchmark]
    for benchmark in names:
        rows = []
        result["benchmarks"][benchmark] = rows
        for seed in args.seeds:
            initial, env, fe, upper, queries, calibration = prepare(
                benchmark, seed, args.warm_queries, args.max_queries)
            checkpoint = target.parent / f"{benchmark}-seed{seed}-initial.json"
            checkpoint.write_text(encode(dict(
                state=score_state(initial), B=initial._B, f=initial._f,
                success_buffer=initial._benign_X, upper=upper,
                calibration_query_ids=calibration,
                evaluation_query_ids=[q.query_id for q in queries],
                idf=fe._idf)))
            entry = dict(seed=seed, n_evaluation=len(queries), upper=upper,
                         initial_checkpoint=str(checkpoint), variants=[])
            rows.append(entry)
            for mode, solver, beta, fn, fp in variants:
                tag = f"-fn{fn:g}-fp{fp:g}" if (fn or fp) else ""
                path = target.parent / f"{benchmark}-s{seed}-{mode}-{solver}-b{beta:g}{tag}.jsonl.gz"
                stats = run_variant(initial, env, fe, upper, queries, seed, mode, solver, beta, path, fn, fp)
                entry["variants"].append(stats)
                result["wall_time_sec"] = time.time()-started
                target.write_text(encode(result))
                print(f"{benchmark} seed={seed} {mode}/{solver} beta={beta:g} fn={fn:g} fp={fp:g}: "
                      f"ASR={stats['asr_pct']:.2f}% TSA={stats['tsa_pct']:.2f}%", flush=True)
    result["complete"] = True
    result["wall_time_sec"] = time.time()-started
    target.write_text(encode(result))
    print(f"COMPLETE {target} ({result['wall_time_sec']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
