"""Export the LLM-selector prompts of run_rev1_llm.py to one JSON file, so that larger models can be
scored on a cluster with nothing but torch + transformers (katana_llm/llm_remote.py)."""
import json

from run_rev1_baselines import metatool_split
from run_rev1_degradation import degrade_desc
from run_rev1_llm import SYSTEM, build
from tats.bfcl_adapter import BFCLToolSelectionBenchmark

bfcl = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios
mt = metatool_split(10)[1][:500]
conds = [("bfcl/full", bfcl, None), ("bfcl/D1", degrade_desc(bfcl, "D1", 0), None),
         ("bfcl/desc_copy", bfcl, "desc_copy"), ("bfcl/full_adaptive", bfcl, "full_adaptive"),
         ("metatool/full", mt, None), ("metatool/desc_copy", mt, "desc_copy")]
out = {"system": SYSTEM, "prefill": "The best tool is number ", "conditions": {}}
for name, scen, attack in conds:
    out["conditions"][name] = [[{"text": t, "gt": g, "mal": m} for t, g, m in build(scen, attack, o)] for o in range(3)]
json.dump(out, open("katana_llm/prompts.json", "w"))
print({k: (len(v), len(v[0])) for k, v in out["conditions"].items()})
