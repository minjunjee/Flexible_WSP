"""E0 from the redesign sheet: does taking an edit out again give back the original weights?

The two JSON edits (answer start, key name; 24 rank-1 changes each) are added to the MLP output matrices and then
subtracted in the same order, in three number formats. float32: no rounding of the stored weights beyond float32
itself. float16 and bfloat16: the weights are rounded to that format after every addition and subtraction (bfloat16
is the format of the released checkpoint and what the experiments used).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m wsp.redesign_e0
"""
import csv
import json

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from wsp.edit import load_stacked
from wsp.lm import LM

lm = LM("qwen35-0.8b")
rounds = load_stacked("results/exp1/stack_json_onpolicy/edits.npz")
W0 = [w.clone() for w in lm.W0]                                    # float32 copies of the released bfloat16 values
tab = pq.read_table(hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")).to_pydict()
ids = lm.tok("".join(tab["text"]), add_special_tokens=False).input_ids[:10240]
chunks = [ids[i:i + 1024] for i in range(0, 10240, 1024)]
base_lp = [lm.logprobs(c) for c in chunks]


def kl_to_base():
    return float(np.mean([float((b.exp() * (b - lm.logprobs(c))).sum(-1).mean()) for c, b in zip(chunks, base_lp)]))


rows, hist = [], {}
for fmt, dt in (("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
    W = [w.to(dt) for w in W0]
    start = [w.clone() for w in W]
    for sign in (1.0, -1.0):
        for edits in rounds:
            for e in edits:
                l = e["layer"]
                W[l] = (W[l].float() + sign * torch.outer(e["u"].to(lm.device), e["v"].to(lm.device))).to(dt)
    n_diff = sum(int((a != b).sum()) for a, b in zip(W, start))
    total = sum(a.numel() for a in W)
    max_abs = max(float((a.float() - b.float()).abs().max()) for a, b in zip(W, start))
    steps = None
    if dt != torch.float32:
        # how many representable steps apart: spacing of the format at each original value
        counts = np.zeros(6, dtype=np.int64)
        for a, b in zip(W, start):
            up = torch.nextafter(b.float().abs().to(dt), torch.tensor(float("inf"), dtype=dt, device=b.device)).float() - b.float().abs()
            k = ((a.float() - b.float()).abs() / up).round().clamp(max=5).long().flatten()
            counts += torch.bincount(k, minlength=6).cpu().numpy()
        steps = {("5 or more" if i == 5 else str(i)): int(c) for i, c in enumerate(counts)}
        hist[fmt] = steps
    for m, w in zip(lm.mlp_out, W):
        m.weight.data.copy_(w.float())
    kl = kl_to_base()
    lm.restore()
    rows.append(dict(format=fmt, weights_total=total, weights_different_after_add_then_subtract=n_diff, share=n_diff / total,
                     largest_absolute_difference=max_abs, mean_kl_to_original_on_10240_tokens=kl))
    print(rows[-1], steps, flush=True)
with open("results/E0_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
json.dump(dict(rows=rows, difference_in_representable_steps=hist, edits="results/exp1/stack_json_onpolicy/edits.npz (two edits, 24 layers each)"),
          open("results/E0_detail.json", "w"), indent=1)
print("DONE")
