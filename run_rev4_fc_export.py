"""Export scenarios in function-calling format (query + JSON function specs) for tool-tuned LLM
selectors (Gorilla OpenFunctions, Hammer, native Qwen tool calling); scored by katana_llm/fc_remote.py."""
import json
import re

import numpy as np

from run_rev1_baselines import metatool_split
from run_rev1_degradation import degrade_desc
from tats.adversarial import craft_malicious_tool
from tats.bfcl_adapter import BFCLToolSelectionBenchmark

bfcl = BFCLToolSelectionBenchmark(bfcl_eval_package=True, dataset="live_multiple").scenarios
mt = metatool_split(10)[1][:500]
conds = [("bfcl/full", bfcl, None), ("bfcl/D1", degrade_desc(bfcl, "D1", 0), None),
         ("bfcl/desc_copy", bfcl, "desc_copy"), ("bfcl/full_adaptive", bfcl, "full_adaptive"),
         ("metatool/full", mt, None), ("metatool/desc_copy", mt, "desc_copy")]


TYPES = {"dict": "object", "float": "number", "tuple": "array", "list": "array", "bool": "boolean", "any": "string", "str": "string", "int": "integer"}


def norm(x):
    """Keep the full JSON schema of the benchmark (nested properties, enums, defaults); only map
    BFCL's Python-style type names to JSON-Schema names, as the BFCL harness does for OpenAI-format models."""
    if isinstance(x, dict):
        return {k: (TYPES.get(v, v) if k == "type" and isinstance(v, str) else norm(v)) for k, v in x.items()}
    if isinstance(x, list):
        return [norm(v) for v in x]
    return x


def build(scen, attack, order_seed):
    rng = np.random.default_rng(order_seed)
    items = []
    for sc in scen:
        cands = list(sc.candidate_tools)
        if attack:
            cands.append(craft_malicious_tool(sc.scenario_id, sc.query_text, attack))
        gt_i = [t.tool_id for t in cands].index(sc.ground_truth_tool_id)
        names, seen = [], set()
        for t in cands:  # function names must be identifier-safe and unique
            n = re.sub(r"[^A-Za-z0-9_]", "_", t.name)[:60] or "tool"
            while n in seen:
                n += "_"
            seen.add(n); names.append(n)
        perm = rng.permutation(len(cands))
        fns = []
        for i in perm:
            spec = cands[i].parameter_spec if isinstance(cands[i].parameter_spec, dict) else {}
            params = norm(spec) if spec.get("properties") else {"type": "object", "properties": {}, "required": []}
            params.setdefault("type", "object"); params.setdefault("properties", {}); params.setdefault("required", [])
            fns.append({"name": names[i], "description": cands[i].description[:600], "parameters": params})
        items.append({"query": sc.query_text, "functions": fns, "gt": names[gt_i], "mal": names[-1] if attack else ""})
    return items


out = {name: [build(scen, attack, o) for o in range(3)] for name, scen, attack in conds}
json.dump(out, open("katana_llm/fc_scenarios.json", "w"))
print({k: (len(v), len(v[0])) for k, v in out.items()})
