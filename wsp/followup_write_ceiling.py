"""Follow-up diagnostic: how far can the key-name edit push the next-token logits on Gemma 3 4B?

Gemma 3 passes the MLP output through an RMSNorm before it is added to the residual stream, so however large the
vector that an edit adds before that norm, what reaches the residual stream has a bounded length. Here, on target
items of the test episode where the edited model wrote the record's word as the key, the key-name edit is scaled and
two things are read at the position where the key is written:
  - the length of what the MLPs add to the residual stream after the norm (summed over layers), and
  - logit(asked key) - logit(the word the model wrote instead).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python -m wsp.followup_write_ceiling
"""
import json

import numpy as np
import torch

from wsp.edit import load_stacked
from wsp.lm import LM

RUN, SEED, KEY, OTHER = "results/iwibench/gemma3-4b_episodes_first_run_after_freezing_the_procedure/seed0_test", 0, "subject", "resident"
lm = LM("gemma3-4b")
rounds = load_stacked(f"{RUN}/edits_seed{SEED}.npz")
rows = [json.loads(l) for l in open(f"{RUN}/outputs_seed{SEED}.jsonl")]
bad = [r for r in rows if r["id"].startswith("target") and r["edited"].startswith('{"' + OTHER + '"')][:16]
good = [r for r in rows if r["id"].startswith("target") and r["edited"].startswith('{"' + KEY + '"')][:16]
first = lm.tok('{"', add_special_tokens=False).input_ids
t_key, t_other = (lm.tok('{"' + w, add_special_tokens=False).input_ids[len(first)] for w in (KEY, OTHER))
print("tokens:", repr(lm.tok.decode([t_key])), repr(lm.tok.decode([t_other])), "| items:", len(bad), "wrong key,", len(good), "asked key", flush=True)
R = {}
for name, items in (("items where the model wrote the record's word", bad), ("items where the model wrote the asked key", good)):
    R[name] = {}
    for f in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        lm.restore()
        for k, e in enumerate(rounds, 1):
            lm.apply(e, scale=f if k == 2 else 1.0)
        gaps, added, top = [], [], []
        for r in items:
            ids = lm.prompt_ids(r["prompt"]) + first                      # the position after '{"' is where the key is written
            post, pre = lm.mlp_stats(ids, len(ids) - 1)                      # per layer: output after the norm [L, d], length before the norm [L]
            lp = lm.logprobs(ids)[-1]
            gaps.append(float(lp[t_key] - lp[t_other]))
            added.append(float(post.sum(0).norm()))
            top.append(lm.tok.decode([int(lp.argmax())]))
        R[name][str(f)] = dict(mean_logit_gap_asked_minus_other=float(np.mean(gaps)), mean_length_of_what_the_mlps_add=float(np.mean(added)),
                               share_asked_key_is_top=float(np.mean([t.strip() == KEY for t in top])), top_tokens=sorted(set(top))[:5])
        print(name, "| key edit x", f, R[name][str(f)], flush=True)
json.dump(R, open("results/followup/write_ceiling_test_seed0.json", "w"), indent=1, ensure_ascii=False)
print("DONE")
