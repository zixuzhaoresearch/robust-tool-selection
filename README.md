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
| `experiments/` | One script per experiment family (map below). Run them from the repository root, e.g. `python experiments/adab_gamma_sweep.py` |
| `figures/` | Source of Figure 1 (`teaser.jsx`, rendered with `npm install && npm run teaser`; needs the Lato fonts and `rsvg-convert`). Figure 2 is drawn by `python experiments/adaptation_streams.py --figure-only` |
| `katana_llm/` | Stand-alone scripts that score exported prompts with the larger LLMs on a GPU server |
| `results/` | JSON outputs behind the paper's tables (means and per-seed values) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .        # installs the `tats` package, which the scripts in experiments/ import
```

`torch` and `xgboost` are only needed for the non-linear baselines (`experiments/nonlinear_models.py`);
`sentence-transformers` only for the dense-embedding variants (`experiments/dense_variants.py`, `experiments/adab_dense.py`, `experiments/runtime_bench.py`), which download `all-MiniLM-L6-v2` on first use.

### Data

* **BFCL** (live_multiple, v4_multiple): bundled with the `bfcl-eval` package, which the adapters load by default.
  Alternatively set `BFCL_DATA_DIR` to a checkout of `berkeley-function-call-leaderboard/bfcl_eval/data`.
* **MetaTool**: clone https://github.com/HowieHwong/MetaTool next to this directory (so that `../MetaTool/dataset/` exists),
  or set `METATOOL_DATA_DIR` to the `dataset/` folder (it must contain `plugin_des.json` and `data/all_clean_data.csv`).

The compact selectors run on CPU. The LLM selectors (`experiments/llm_selectors.py`, `experiments/extensions_param_guard.py llmguard`) and the fine-tuned retriever need a GPU;
the ToolRet experiment downloads `mangopy/ToolRet-Tools` and `mangopy/ToolRet-Queries` from the Hugging Face Hub into `data/toolret/`
(`huggingface_hub.snapshot_download(repo, repo_type="dataset", local_dir="data/toolret/{tools,queries}")`).

## Reproducing the paper

Table numbers follow the revised manuscript (revision 1). The paper's ADAB operating point is γ = 2; the constructor default of `ADABEmaSelector` is γ = 3, and every reported run passes γ explicitly.

| Paper element | Script | Output |
|---|---|---|
| Table 3 (clean selection, BFCL rows), Table 15 left, repeated interaction (Appendix C, text) | `experiments/clean_bfcl.py` | `results/bilints_results.json` |
| Table 3 (MetaTool rows) | `experiments/clean_metatool.py` | `results/metatool_results.json` |
| Table 3 (Sup-Offline / Sup-Online) | `experiments/supervised_baselines.py` | `results/supervised_results.json` |
| Table 3 (TF-IDF-Desc, DenseDescMatch-FT, EmbRerank-Online), Table 2 (feature weights, drop-one), Table 5 and Table A1 (candidate-set and pooled-catalog scaling) | `experiments/label_baselines.py {ft,features,scale}` | `results/revision_r1/{ft,features,scale}.json` |
| Table 4 (LLM selectors, Qwen2.5-1.5B/3B) | `experiments/llm_selectors.py` | `results/revision_r1/llm.json` |
| Table 4 (larger LLM selectors) | `experiments/llm_export_prompts.py`, then `katana_llm/llm_remote.py` on a GPU node | `results/revision_r1/llm_large.json` |
| Table 6 (ToolRet, 37k tools) and Table A2 (retrieval baselines) | `experiments/toolret_corpus.py {toolret,retrievers}`, `experiments/retriever_baselines.py` | `results/revision_r1/{toolret,retrievers,retrievers2}.json` |
| Tables 7 and 8 (description and schema degradation) | `experiments/degradation_levels.py` | `results/revision_r1/degradation.json` |
| Table 10 (L1 attacks: desc_copy, param_generic, param_steal, full_adaptive) | `experiments/attacks_l1_bfcl.py` | `results/adversarial_*.json`, `results/adv_*.json` |
| Table 11 (ADAB gamma frontier, Mahalanobis row) | `experiments/adab_gamma_sweep.py` | `results/tab8_gamma_sweep_10seed.json` |
| Table 11 (dense rows), MetaTool injection results in Section 4.5 (text) | `experiments/adab_dense.py`, `experiments/adab_metatool.py`, `experiments/attacks_l1_metatool.py` | `results/adab_dense_adv.json`, `results/adab_ema_metatool.json`, `results/adab_metatool_10seed.json` |
| Held-out selection of gamma, per-scenario outcomes, paired bootstrap intervals (Sections 4.1 and 4.5) | `experiments/stats_bootstrap_gamma.py {core_bfcl,core_metatool,holdout,boot}` | `results/revision_r1/{core_bfcl,core_metatool,gamma_holdout,bootstrap}.json` |
| relaxed-oracle ceiling in Section 4.5 (text, last rows of Table 12), Table 13 (registry rollouts), safe-zone bounds and optimizer quality (result files only; described in Appendix B) | `experiments/oracle_registry.py --suite full` | `results/l4_verified/summary.json` plus per-query traces |
| Table 12 (metadata-realizable attacks: L1, Shadow, L2, L3, L4-R) | `experiments/attacks_realizable.py {bfcl,metatool}` | `results/revision_r1/attacks_{bfcl,metatool}.json` |
| Table 14 (registry trust-label noise) | `experiments/oracle_registry.py --suite registry-noise --betas 1 2 5 --trust-fn 0.01 0.05 0.1 --trust-fp 0.05 0.1` | `results/l4_registry_noise/summary.json` |
| Table 15 right (ADAB design ablation), alternative defenses and quantile variants (Appendix C, text) | `experiments/adab_design_ablation.py` | `results/lab_iter{2,4,7,10,11}.json` (Mahalanobis variants: `results/novel_selectors_10seed.json`) |
| logistic TS, TW-LinUCB, CGB and the IDF check (Appendix C, text), MetaTool ADAB rows of Tables 3 and 4 | `experiments/secondary_checks.py` | `results/revision_r1/legacy_checks.json` |
| Table 16 (noisy, missing, delayed feedback) | `experiments/feedback_noise.py` | `results/revision_r1/feedback.json` |
| Table 17 (manipulated feedback) | `experiments/feedback_manipulated.py` | `results/malicious_feedback_10seed.json` |
| Table 4 (Gorilla OpenFunctions v2, Hammer2.1-7b through their function-calling formats) | `experiments/fc_export_prompts.py`, then `katana_llm/fc_remote.py` on a GPU | `results/revision_r1/llm_fc.json` (every raw model output: `llm_fc_raw.jsonl`) |
| Table 18 (usage-memory channel), perplexity-filter baseline (Section 4.4), ToolBench API retriever (Table A2) | `experiments/usage_memory.py {memory,ppl,toolbench}` | `results/revision_r1/{memory,ppl,toolbench_ir}.json` |
| Table A5 (real execution of BFCL's executable functions; the function implementations are recovered from the Gorilla repository as described at the top of the script) | `experiments/real_execution.py {prepare,run}` | `results/revision_r1/real_exec.json`, per-pair outcomes and LLM arguments in `real_exec_outcomes.json` |
| Figure 2 and Table 19 (adaptation to execution feedback: hidden reliability, mid-stream change, documentation drift) | `experiments/adaptation_streams.py` | `results/revision_r1/adaptation.json` |
| Table 20 (multi-step agent on BFCL's stateful multi-turn backends with a 580-tool registry; `coverage` needs no LLM, `agent` runs a local Qwen2.5-Instruct model and checkpoints every 10 tasks) | `experiments/multistep_agent.py coverage`; `experiments/multistep_agent.py agent --model Qwen/Qwen2.5-{7B,14B,32B}-Instruct --passes 2 [--seed 1|2] [--only "LinUCB-Dense (text features)" | "Dense + failure blacklist"]` (task order = seed; the registry is built in `load()` from the eight BFCL multi-turn backends plus the BFCL live_multiple functions as failing distractors); `experiments/multistep_agent.py table` prints Table 20 with its bootstrap intervals | `results/revision_r1/multistep_coverage_{full,connected}.json`, `multistep_agent_full_Qwen2.5-{7B,14B,32B}-Instruct.json` (per-task outcomes) |
| Usage-memory attribution in Section 4.7 (memory with fixed weights and no learning) | `experiments/usage_memory_attribution.py` | `results/revision_r1/memattr.json` |
| Table A3 (parameter-matching variants), Table A4 (penalty as a guard on a fine-tuned retriever and an LLM) | `experiments/extensions_param_guard.py {param,guard,guard2,llmguard}` | `results/revision_r1/{param,guard,guard2,llmguard}.json` |
| RobustLinUCB and symmetric label noise (Section 4.6, Appendix C) | `experiments/robust_cb_label_noise.py` | `results/day3_experiments.json` |
| MetaTool per-channel ablation (Appendix C, text) | `experiments/metatool_channel_ablation.py --Ks 5 10 20 50 --seeds 0 1 2 3 4` | `results/metatool_channel_ablation.json` |
| random-distractor construction (Appendix C, text) | `experiments/metatool_random_distractors.py` | `results/metatool_randneg_10seed.json` |
| non-linear baselines (Appendix C, text) | `experiments/nonlinear_models.py` | `results/nonlinear_results_10seed.json` |
| Appendix C (sigma^2 sensitivity) | `experiments/sigma2_sensitivity.py` | printed table |
| Runtime (Appendix B), Table 4 latency of the feature-based selectors | `experiments/runtime_bench.py` | `results/revision_r1/runtime_bench.txt` |

`experiments/make_tables.py` prints the LaTeX rows of the revision tables from `results/revision_r1/`.
Every `results/revision_r1/*.json` stores per-scenario outcomes averaged over seeds, which is what the paired bootstrap uses.

Naming note: `DescMatch` in the code and result files is unweighted token overlap between query and description;
the TF-IDF cosine baseline is `TFIDF-Desc` (`tats.rev1_common.ArgmaxFeatureSelector` on feature 0).

Approximate wall-clock on one CPU core: `experiments/oracle_registry.py --suite full` about 10 minutes (five seeds, both benchmarks),
`--suite registry-noise` about 3 minutes, `experiments/adab_gamma_sweep.py` under 1 minute, `experiments/feedback_manipulated.py` about 1 minute,
the 10-seed benign and attack sweeps a few minutes each, and the non-linear baselines the longest (NeuralUCB training).
`experiments/oracle_registry.py` refuses to overwrite an existing output file, and its default `--output` is the authors' development-tree location;
pass a fresh path such as `--output results/l4_rerun/summary.json` for new runs.
Seeds are fixed per script (`--seeds`); tables report means and standard deviations over seeds.

## Audit hashes

`experiments/oracle_registry.py` was named `run_verified_l4.py` when the L4 result files were produced; its content is unchanged, and the summaries record it under the old name.

Each `results/l4_*/summary.json` records the SHA-256 of the sources that produced it.
`tats/optimizer_attack.py`, `tats/novel_selectors.py`, and `tats/features.py` in this package are byte-identical to the versions recorded in both summaries.
`experiments/oracle_registry.py` matches the hash recorded in `results/l4_registry_noise/summary.json`; `results/l4_verified/summary.json` was produced by the
revision of that script that preceded the addition of the `registry-noise` suite.

## License and citation

Released under the MIT License (see `LICENSE`). The official repository is https://github.com/zixuzhaoresearch/robust-tool-selection.
If you use it, please cite:

> Zhao, Z.; Zhang, W.; Cao, X. Robust Tool Selection Beyond Description Matching. Submitted to *Applied Sciences*, 2026.
