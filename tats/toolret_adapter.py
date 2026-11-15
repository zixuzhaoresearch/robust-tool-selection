"""ToolRet (web corpus) adapter: 37k heterogeneous tool documents -> BFCLTool, single-gold queries.

Data: HuggingFace datasets mangopy/ToolRet-Tools and mangopy/ToolRet-Queries, downloaded to
data/toolret/{tools,queries}. Only the `web` corpus is used (tools with API-style documentation),
and only queries with exactly one gold tool, so that tool-selection accuracy is well defined.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from .bfcl_adapter import BFCLScenario, BFCLTool

ROOT = Path("data/toolret")


def _schema(d):
    """Normalize the many parameter conventions in ToolRet to {properties, required}."""
    props, req = {}, []

    def add(name, spec=None, required=None):
        if not isinstance(name, str) or not name:
            return
        props[name] = {"type": str((spec or {}).get("type", "string"))} if isinstance(spec, dict) else {"type": "string"}
        if required:
            req.append(name)

    for key in ("parameters", "doc_arguments", "arguments", "api_arguments"):
        p = d.get(key)
        if isinstance(p, dict) and isinstance(p.get("properties"), dict):
            for k, v in p["properties"].items():
                add(k, v)
            req.extend(x for x in (p.get("required") or []) if isinstance(x, str))
        elif isinstance(p, dict):
            for k, v in p.items():
                add(k, v, isinstance(v, dict) and v.get("default") in (None, "") and "default" in v)
        elif isinstance(p, list):
            for it in p:
                if isinstance(it, dict):
                    add(it.get("name"), it.get("schema") if isinstance(it.get("schema"), dict) else it, it.get("required") is True)
                else:
                    add(it)
    for it in d.get("required_parameters") or []:
        if isinstance(it, dict):
            add(it.get("name"), it, True)
    for it in d.get("optional_parameters") or []:
        if isinstance(it, dict):
            add(it.get("name"), it, False)
    return {"type": "object", "properties": props, "required": [r for r in dict.fromkeys(req) if r in props]}


def load_tools():
    df = pd.read_parquet(ROOT / "tools/web/tools-00000-of-00001.parquet")
    tools = {}
    for tid, doc in zip(df["id"], df["documentation"]):
        try:
            d = json.loads(doc)
        except Exception:
            d = {}
        if not isinstance(d, dict):
            d = {}
        name = d.get("name") or d.get("tool_name") or d.get("api_name") or tid
        desc = d.get("description") or d.get("api_description") or d.get("tool_description") or ""
        tools[tid] = BFCLTool(tool_id=tid, name=str(name), declared_capabilities=[], true_success_rates={},
                              description_affinity={}, description=str(desc), parameter_spec=_schema(d))
    return tools


def load_queries(tools):
    """-> list of (query_text, gold_tool_id, subset) for single-gold queries whose gold tool is in the web corpus."""
    out = []
    for f in sorted(glob.glob(str(ROOT / "queries/*/*.parquet"))):
        subset = f.split("/")[-2]
        df = pd.read_parquet(f)
        for q, lab in zip(df["query"], df["labels"]):
            L = json.loads(lab)
            if len(L) == 1 and L[0]["id"] in tools:
                out.append((str(q).strip(), L[0]["id"], subset))
    return out


def scenarios_from_candidates(queries, cand_ids, tools):
    """queries[i] with candidate ids cand_ids[i] -> BFCLScenario list (gold is NOT forced into the set)."""
    return [BFCLScenario(scenario_id=f"toolret_{i}", query_text=q, candidate_tools=[tools[t] for t in c],
                         ground_truth_tool_id=g)
            for i, ((q, g, _), c) in enumerate(zip(queries, cand_ids))]
