"""E2 from the redesign sheet: do the positions where an instruction is decided predict whether edits work?

Reads results/E1_<task>.jsonl. Per instruction (mean over the 200 questions):
  d1  first position where the instructed and the uninstructed answer differ, divided by the reference length
  nD  number of reference positions where the uninstructed model, fed the instructed answer, would write another token
  dL  relative position of the last such position
and relates them to the compliance of the conditional edits (C5) and to the number of edits that were stacked.

    python -m wsp.redesign_e2
"""
import csv
import glob
import json

import numpy as np
from scipy.stats import spearmanr

KIND = {"json_answer": "prefix", "json_format": "prefix", "title": "prefix", "start_phrase": "prefix",
        "lowercase": "whole answer", "uppercase": "whole answer", "french": "whole answer", "no_comma": "whole answer",
        "end_phrase": "end or count", "postscript": "end or count", "three_bullets": "end or count",
        "quotation": "other", "three_words": "other"}
rows = []
for f in sorted(glob.glob("results/E1_*.jsonl")):
    R = [json.loads(l) for l in open(f)]
    task = R[0]["task"]
    c1 = [r for r in R if r["condition"] == "C1 prompt"]
    rate = lambda cond, k="follows": float(np.mean([r[k] for r in R if r["condition"] == cond]))
    rows.append(dict(task=task, kind=KIND[task], d1=float(np.mean([r["d1"] for r in c1])), nD=float(np.mean([r["nD"] for r in c1])),
                     nD_share=float(np.mean([r["nD_share"] for r in c1])), dL=float(np.mean([r["dL"] for r in c1])),
                     reference_length=float(np.mean([r["ref_len"] for r in c1])), edit_follows=rate("C5 edit"),
                     edit_follows_and_correct=float(np.mean([r["follows"] and r["correct"] for r in R if r["condition"] == "C5 edit"])),
                     prompt_follows=rate("C1 prompt"), edit_groups=R[0]["n_edit_groups"]))
with open("results/E2_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
S = {}
y = [r["edit_follows"] for r in rows]
for k in ("d1", "nD", "nD_share", "dL"):
    rho, p = spearmanr([r[k] for r in rows], y)
    S[f"{k} vs edit compliance"] = dict(rho=float(rho), p=float(p), n=len(rows))
    rho, p = spearmanr([r[k] for r in rows], [r["edit_groups"] for r in rows])
    S[f"{k} vs number of edits"] = dict(rho=float(rho), p=float(p), n=len(rows))
top_dL = [r["task"] for r in sorted(rows, key=lambda r: -r["dL"])[:3]]
S["three instructions with the latest last decision (dL)"] = top_dL
json.dump(S, open("results/E2_spearman.json", "w"), indent=1)
for r in sorted(rows, key=lambda r: -r["edit_follows"]):
    print(f"{r['task']:14s} {r['kind']:13s} d1 {r['d1']:.2f}  nD {r['nD']:5.1f}  nD/len {r['nD_share']:.2f}  dL {r['dL']:.2f}  len {r['reference_length']:5.1f} | "
          f"edit {100 * r['edit_follows']:5.1f}%  edits {r['edit_groups']}")
for k, v in S.items():
    print(k, v)

# ---------------------------------------------------------------- scatter: nD against edit compliance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"prefix": "#2a78d6", "whole answer": "#eb6834", "end or count": "#1baf7a", "other": "#eda100"}       # validated categorical slots 1-4
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
for kind, col in COL.items():
    pts = [r for r in rows if r["kind"] == kind]
    ax.scatter([r["nD"] for r in pts], [100 * r["edit_follows"] for r in pts], s=70, color=col, edgecolor=SURF, linewidth=2, label=kind, zorder=3)
OFFSET = {"json_answer": (7, -11), "three_words": (7, -11), "postscript": (-50, 8)}       # keep labels off each other
for r in rows:
    ax.annotate(r["task"], (r["nD"], 100 * r["edit_follows"]), xytext=OFFSET.get(r["task"], (6, 5)), textcoords="offset points", fontsize=7.5, color=MUTED)
ax.set_xlabel("positions where the uninstructed model would write another token (nD, mean per answer)", fontsize=8.5, color=INK)
ax.set_ylabel("answers that follow the instruction with conditional edits (%)", fontsize=8.5, color=INK)
ax.set_ylim(-5, 105)
ax.grid(True, color="#e6e5e1", linewidth=0.6, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#c9c8c3")
ax.tick_params(colors=MUTED, labelsize=8)
ax.legend(title="instruction kind", fontsize=8, title_fontsize=8, frameon=False, loc="upper right", labelcolor=INK)
rho = S["nD vs edit compliance"]
ax.set_title(f"Decision positions and edit success, 13 instructions (Spearman rho {rho['rho']:.2f}, p {rho['p']:.2f})", fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig("results/E2_scatter.png", facecolor=SURF)
print("wrote results/E2_scatter.png")
