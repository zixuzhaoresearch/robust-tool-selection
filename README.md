# Robust Tool Selection Beyond Description Matching: code and results (TATS)

Supplementary material for the manuscript "Robust Tool Selection Beyond Description Matching" by Zixu Zhao, Wenjie Zhang, and Xin Cao
(School of Computer Science and Engineering, University of New South Wales, Sydney, Australia), submitted to *Applied Sciences* (MDPI).
Table, figure, and appendix numbers below follow that manuscript. Contact: zixu.zhao@unsw.edu.au.
It contains the selector library (`tats/`), the experiment scripts behind every table and figure of the paper,
and the JSON results those scripts produced (`results/`), so the reported numbers can be checked without re-running anything.

## Layout

| Path | Contents |
|---|---|
| `tats/` | Feature extraction (`features.py`, `dense_features.py`), benchmark adapters (`bfcl_adapter.py`, `metatool_adapter.py`), documentation degradation (`degradation.py`), L1 attacks (`adversarial.py`), selectors (`bilints_selector.py` for LinUCB/BilinTS, `novel_selectors.py` for ADAB and its variants, `robust_selectors.py`, `neural_bandits.py`), the L4 feature-space oracle (`optimizer_attack.py`), warm-start priors and supervised baselines (`warmstart.py`), plotting |
| `run_*.py` | One script per experiment family (map below); run from this directory |
| `results/` | JSON outputs behind the paper's tables (means and per-seed values) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

`torch` and `xgboost` are only needed for the non-linear baselines (`run_nonlinear_experiments.py`);
`sentence-transformers` only for the dense-embedding variants (`run_dense_full.py`, `run_adab_dense_adversarial.py`, `run_runtime_bench.py`), which download `all-MiniLM-L6-v2` on first use.

### Data

* **BFCL** (live_multiple, v4_multiple): bundled with the `bfcl-eval` package, which the adapters load by default.
  Alternatively set `BFCL_DATA_DIR` to a checkout of `berkeley-function-call-leaderboard/bfcl_eval/data`.
* **MetaTool**: clone https://github.com/HowieHwong/MetaTool next to this directory (so that `../MetaTool/dataset/` exists),
  or set `METATOOL_DATA_DIR` to the `dataset/` folder (it must contain `plugin_des.json` and `data/all_clean_data.csv`).

The compact selectors run on CPU. The LLM selectors (`run_rev1_llm.py`, `run_rev2_methods.py llmguard`) and the fine-tuned retriever need a GPU;
the ToolRet experiment downloads `mangopy/ToolRet-Tools` and `mangopy/ToolRet-Queries` from the Hugging Face Hub into `data/toolret/`
(`huggingface_hub.snapshot_download(repo, repo_type="dataset", local_dir="data/toolret/{tools,queries}")`).

## Reproducing the paper

Table numbers follow the revised manuscript (revision 1).

| Paper element | Script | Output |
|---|---|---|
| Table 4 (clean selection, BFCL rows), Table 18 left, Figure 2, Table A2 (repeated interaction) | `run_bilints_experiments.py` | `results/bilints_results.json` |
| Table 4 (MetaTool rows) | `run_metatool_experiments.py` | `results/metatool_results.json` |
| Table 4 (Sup-Offline / Sup-Online) | `run_supervised_full.py` | `results/supervised_results.json` |
| Table 4 (TF-IDF-Desc, DenseDescMatch-FT, EmbRerank-Online), Table 3 (feature weights, drop-one), Table 6 and Table A8 (candidate-set and pooled-catalog scaling) | `run_rev1_baselines.py {ft,features,scale}` | `results/revision_r1/{ft,features,scale}.json` |
| Table 5 (LLM selectors, Qwen2.5-1.5B/3B) | `run_rev1_llm.py` | `results/revision_r1/llm.json` |
| Table 5 (larger LLM selectors) | `run_rev2_llm_export.py`, then `katana_llm/llm_remote.py` on a GPU node | `results/revision_r1/llm_large.json` |
| Table 7 (ToolRet, 37k tools) and Table A11 (retrieval baselines) | `run_rev2_toolret.py {toolret,retrievers}`, `run_rev2_retrievers2.py` | `results/revision_r1/{toolret,retrievers,retrievers2}.json` |
| Tables 8 and 9 (description and schema degradation) | `run_rev1_degradation.py` | `results/revision_r1/degradation.json` |
| Table 11 (L1 attacks: desc_copy, param_generic, param_steal, full_adaptive) | `run_adversarial_experiments.py` | `results/adversarial_*.json`, `results/adv_*.json` |
| Table 12 (ADAB gamma frontier, Mahalanobis row) | `run_tab8_sweep.py` | `results/tab8_gamma_sweep_10seed.json` |
| Table 12 (dense rows), Table 13 (MetaTool injection) | `run_adab_dense_adversarial.py`, `run_adab_metatool.py`, `run_adversarial_metatool.py` | `results/adab_dense_adv.json`, `results/adab_ema_metatool.json`, `results/adab_metatool_10seed.json` |
| Held-out selection of gamma, per-scenario outcomes, paired bootstrap intervals (Sections 4.1 and 4.5) | `run_rev1_stats.py {core_bfcl,core_metatool,holdout,boot}` | `results/revision_r1/{core_bfcl,core_metatool,gamma_holdout,bootstrap}.json` |
| Table 14 (relaxed oracle), Table 16 (registry rollouts), Table A9 (safe-zone bounds), Table A10 (optimizer quality) | `run_verified_l4.py --suite full` | `results/l4_verified/summary.json` plus per-query traces |
| Table 15 (metadata-realizable attacks: L1, Shadow, L2, L3, L4-R) | `run_rev1_attacks.py {bfcl,metatool}` | `results/revision_r1/attacks_{bfcl,metatool}.json` |
| Table 17 (registry trust-label noise) | `run_verified_l4.py --suite registry-noise --betas 1 2 5 --trust-fn 0.01 0.05 0.1 --trust-fp 0.05 0.1` | `results/l4_registry_noise/summary.json` |
| Table 18 right (ADAB design ablation), Table A5 (Committee, per-feature gamma, SupOOD), Appendix F (quantile rows) | `run_novel_selectors.py` | `results/lab_iter{2,4,7,10,11}.json` (Mahalanobis variants: `results/novel_selectors_10seed.json`) |
| Table A4 (logistic TS), Table A5 (TW-LinUCB, CGB), IDF check in Appendix F, MetaTool ADAB rows of Tables 4 and 5 | `run_rev3_legacy_checks.py` | `results/revision_r1/legacy_checks.json` |
| Table 19 (noisy, missing, delayed feedback) | `run_rev1_feedback.py` | `results/revision_r1/feedback.json` |
| Table 20 (manipulated feedback) | `run_malicious_feedback.py` | `results/malicious_feedback_10seed.json` |
| Table 5 (Gorilla OpenFunctions v2, Hammer2.1-7b through their function-calling formats) | `run_rev4_fc_export.py`, then `katana_llm/fc_remote.py` on a GPU | `results/revision_r1/llm_fc.json` (every raw model output: `llm_fc_raw.jsonl`) |
| Table 23 (usage-memory channel), perplexity-filter baseline (Section 4.4), ToolBench API retriever (Table A11) | `run_rev4_memory.py {memory,ppl,toolbench}` | `results/revision_r1/{memory,ppl,toolbench_ir}.json` |
| Table 24 (real execution of BFCL's executable functions; the function implementations are recovered from the Gorilla repository as described at the top of the script) | `run_rev6_real_exec.py {prepare,run}` | `results/revision_r1/real_exec.json`, per-pair outcomes and LLM arguments in `real_exec_outcomes.json` |
| Figure 3 and Table 25 (adaptation to execution feedback: hidden reliability, mid-stream change, documentation drift) | `run_rev5_adaptation.py` | `results/revision_r1/adaptation.json` |
| Table 26 (multi-step agent on BFCL's stateful multi-turn backends with a 580-tool registry; `coverage` needs no LLM, `agent` runs a local Qwen2.5-Instruct model and checkpoints every 10 tasks) | `run_rev7_multistep.py coverage`; `run_rev7_multistep.py agent --model Qwen/Qwen2.5-{7B,14B,32B}-Instruct --passes 2 [--seed 1|2] [--only "LinUCB-Dense (text features)" | "Dense + failure blacklist"]` (task order = seed; the registry is built in `load()` from the eight BFCL multi-turn backends plus the BFCL live_multiple functions as failing distractors); `run_rev7_multistep.py table` prints Table 26 with its bootstrap intervals | `results/revision_r1/multistep_coverage_{full,connected}.json`, `multistep_agent_full_Qwen2.5-{7B,14B,32B}-Instruct.json` (per-task outcomes) |
| Usage-memory attribution in Section 4.7 (memory with fixed weights and no learning) | `run_rev8_memattr.py` | `results/revision_r1/memattr.json` |
| Table 21 (parameter-matching variants), Table 22 (penalty as a guard on a fine-tuned retriever and an LLM) | `run_rev2_methods.py {param,guard,guard2,llmguard}` | `results/revision_r1/{param,guard,guard2,llmguard}.json` |
| RobustLinUCB and symmetric label noise (Section 4.6, Appendix C.4) | `run_day3_experiments.py` | `results/day3_experiments.json` |
| Table A6 (MetaTool per-channel ablation) | `run_metatool_channel_ablation.py --Ks 5 10 20 50 --seeds 0 1 2 3 4` | `results/metatool_channel_ablation.json` |
| Table A7 (random-distractor construction) | `run_metatool_randneg.py` | `results/metatool_randneg_10seed.json` |
| Table A4 (non-linear baselines) | `run_nonlinear_experiments.py` | `results/nonlinear_results_10seed.json` |
| Appendix F (sigma^2 sensitivity) | `run_sigma2_sensitivity.py` | printed table |
| Runtime appendix, Table 5 latency of the feature-based selectors | `run_runtime_bench.py` | `results/revision_r1/runtime_bench.txt` |

`make_rev1_tables.py` prints the LaTeX rows of the revision tables from `results/revision_r1/`.
Every `results/revision_r1/*.json` stores per-scenario outcomes averaged over seeds, which is what the paired bootstrap uses.

Naming note: `DescMatch` in the code and result files is unweighted token overlap between query and description;
the TF-IDF cosine baseline is `TFIDF-Desc` (`tats.rev1_common.ArgmaxFeatureSelector` on feature 0).

Approximate wall-clock on one CPU core: `run_verified_l4.py --suite full` about 10 minutes (five seeds, both benchmarks),
`--suite registry-noise` about 3 minutes, `run_tab8_sweep.py` under 1 minute, `run_malicious_feedback.py` about 1 minute,
the 10-seed benign and attack sweeps a few minutes each, and the non-linear baselines the longest (NeuralUCB training).
`run_verified_l4.py` refuses to overwrite an existing output file, and its default `--output` is the authors' development-tree location;
pass a fresh path such as `--output results/l4_rerun/summary.json` for new runs.
Seeds are fixed per script (`--seeds`); tables report means and standard deviations over seeds.

## Audit hashes

Each `results/l4_*/summary.json` records the SHA-256 of the sources that produced it.
`tats/optimizer_attack.py`, `tats/novel_selectors.py`, and `tats/features.py` in this package are byte-identical to the versions recorded in both summaries.
`run_verified_l4.py` matches the hash recorded in `results/l4_registry_noise/summary.json`; `results/l4_verified/summary.json` was produced by the
revision of that script that preceded the addition of the `registry-noise` suite.

## License and citation

Released under the MIT License (see `LICENSE`). The official repository is https://github.com/zixuzhaoresearch/robust-tool-selection.
If you use it, please cite:

> Zhao, Z.; Zhang, W.; Cao, X. Robust Tool Selection Beyond Description Matching. Submitted to *Applied Sciences*, 2026.
