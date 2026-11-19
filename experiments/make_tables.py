"""Print LaTeX-ready rows for the revision-1 tables from results/revision_r1/*.json."""
import json
from pathlib import Path
R = Path("results/revision_r1")
L = lambda n: json.load(open(R / f"{n}.json"))
f = lambda v, d=1: f"${v['mean']:.{d}f} \\pm {v['sd']:.{d}f}$"

print("% --- degradation (desc levels)")
d = L("degradation")["results"]
for m in ["DescMatch", "TFIDF-Desc", "DenseDescMatch", "NameMatch", "ParamMatch", "Sup-Offline", "LinUCB", "LinUCB-Dense"]:
    print(m, "&", " & ".join(f(d[f"{k}|intact"][m]) for k in ["Full", "D3", "D3s", "D2", "D1n", "D1"]), r"\\")
print("% --- degradation (schema levels): Full then D1")
for m in ["DenseDescMatch", "ParamMatch", "Sup-Offline", "LinUCB", "LinUCB-Dense"]:
    for base in ["Full", "D1"]:
        print(m, base, "&", " & ".join(f"${d[f'{base}|{s}'][m]['mean']:.1f}$" for s in ["intact", "S-add3", "S-abbrev", "S-rename", "S-drop50", "S-noreq", "S-none"]), r"\\")

print("% --- scale")
s = L("scale")
for m in ["DescMatch", "TFIDF-Desc", "LinUCB", "DenseDescMatch", "LinUCB-Dense"]:
    print(m, "&", " & ".join(f"${s['metatool'][k][m]['mean']:.1f}$" for k in ["5", "10", "20", "50", "100", "199"]), r"\\")
print("max sd metatool", max(v["sd"] for k in s["metatool"].values() for v in k.values()))
for m in ["DescMatch", "TFIDF-Desc", "ParamMatch", "LinUCB", "DenseDescMatch", "LinUCB-Dense"]:
    print(m, "&", " & ".join(f"${s['bfcl_pooled'][k][m]['mean']:.1f}$" for k in ["0", "10", "50", "200", "all"]), r"\\")
print("K-bar", [round(s["bfcl_pooled"][k]["_mean_candidates"], 1) for k in ["0", "10", "50", "200", "all"]],
      "max sd", max(v["sd"] for k in s["bfcl_pooled"].values() for n, v in k.items() if n[0] != "_"))

print("% --- ft")
t = L("ft")
for m in [k for k in t["bfcl"] if k[0] != "_"]:
    print(m, "&", f(t["bfcl"][m]), "&", f(t["metatool"][m]), r"\\")

print("% --- features")
x = L("features")
for i, n in enumerate(x["features"]):
    print(n, f"& ${x['theta_mean'][i]:+.2f}$ & ${x['drop_one']['-'+n]['mean']-x['drop_one']['Full']['mean']:+.1f}$", r"\\")
print("full", f(x["drop_one"]["Full"]))

print("% --- feedback")
fb = L("feedback")
for k, v in fb.items():
    b, a = v["benign"], v["full_adaptive"]
    print(k, "&", f(b["LinUCB"]["tsa"]), "&", f(b["ADAB-g2"]["tsa"]), "&", f(a["LinUCB"]["tsa"]), "&", f(a["LinUCB"]["asr"]),
          "&", f(a["ADAB-g2"]["tsa"]), "&", f(a["ADAB-g2"]["asr"]), r"\\")

print("% --- attacks")
for b in ["bfcl", "metatool"]:
    a = L(f"attacks_{b}")["results"]
    for fs in ["tfidf", "dense"]:
        for atk in ["none", "L1-desc_copy", "L1-full", "shadow", "L2-snapshot", "L3-live", "L4R-live"]:
            if f"{fs}/LinUCB/{atk}" not in a:
                continue
            row = " & ".join(f"${a[f'{fs}/{dn}/{atk}']['tsa_mean']:.1f}$ & ${a[f'{fs}/{dn}/{atk}']['asr_mean']:.1f} \\pm {a[f'{fs}/{dn}/{atk}']['asr_sd']:.1f}$"
                             for dn in ["LinUCB", "ADAB-g2", "ADAB-g5"])
            print(b, fs, atk, "&", row, r"\\", f"% n={a[f'{fs}/LinUCB/{atk}']['n_mean']:.0f}")

print("% --- holdout")
h = L("gamma_holdout")
print(h["chosen_counts"])
for k, v in h["aggregate"].items():
    print(k, f"TSA ${v['tsa_mean']:.1f} \\pm {v['tsa_sd']:.1f}$ ASR ${v['asr_mean']:.2f} \\pm {v['asr_sd']:.2f}$")

print("% --- core metatool")
c = L("core_metatool")
for m in ["DescMatch", "TFIDF-Desc", "DenseDescMatch", "LinUCB", "ADAB-g2", "ADAB-g3", "LinUCB-Dense", "ADAB-Dense-g2", "ADAB-Dense-g4"]:
    print(m, "&", f(c["benign"][m]["tsa"]), "&", f(c["desc_copy"][m]["tsa"]), "&", f(c["desc_copy"][m]["asr"]), r"\\")
print("LinUCB desc_copy ASR seeds", [round(v, 1) for v in c["desc_copy"]["LinUCB"]["asr"]["seeds"]])
print("LinUCB-Dense desc_copy ASR seeds", [round(v, 1) for v in c["desc_copy"]["LinUCB-Dense"]["asr"]["seeds"]])
cb = L("core_bfcl")
print("% --- core bfcl dense")
for cond in ["benign", "param_steal", "full_adaptive", "desc_copy"]:
    print(cond, {m: (round(cb[cond][m]["tsa"]["mean"], 1), round(cb[cond][m]["tsa"]["sd"], 1), round(cb[cond][m]["asr"]["mean"], 2), round(cb[cond][m]["asr"]["sd"], 2)) for m in ["LinUCB-Dense", "ADAB-Dense-g2", "TFIDF-Desc"]})

print("% --- bootstrap")
for r in L("bootstrap")["rows"]:
    print(f"{r['file'][5:]:9s} {r['condition']:13s} {r['metric']} {r['a']:14s} vs {r['b']:14s} {r['diff_pp']:+.1f} [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]")

if (R / "llm.json").exists():
    print("% --- llm")
    for m, v in L("llm").items():
        for k, r in v.items():
            print(m.split("/")[-1], k, r if not isinstance(r, dict) else f"TSA ${r['tsa_mean']:.1f} \\pm {r['tsa_sd']:.1f}$ ASR ${r['asr_mean']:.1f} \\pm {r['asr_sd']:.1f}$ tokens {r['prompt_tokens_mean']:.0f} n={r['n']}")
