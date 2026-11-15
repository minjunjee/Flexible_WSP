"""Experiment T: teacher-forced effect of each edit on held-out text (half 1 of the A2 segments).

  productive segments : KL(unedited || edited) of the next-token distribution  -> how much normal reasoning is disturbed
  tight-loop segments : change in log-probability of the token that continues the loop -> how much the loop is discouraged
Note: the replication write-up found that teacher-forced numbers rank near-equivalent edits unreliably,
so these are used only as a coarse side-effect measure.
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/claude/poc")
import qw
from conds import all_conditions
from gen import Editor

N_EACH = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ALT = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [11, 23]
LEN, BURN, STEP = 640, 256, 4
torch.set_grad_enabled(False)
tok, model = qw.load_model()
lm = qw.lm_of(model)
ed = Editor(model)
conds, sets, S = all_conditions(model, ALT)
names = list(conds)
MATH = json.load(open(f"{qw.ROOT}/data/math500.json"))
GSM = json.load(open(f"{qw.ROOT}/data/gsm8k_test.json"))
meta = [m for m in json.load(open("/home/claude/poc/outA2/meta.json")) if m["half"] == 1]
out = {n: dict(kl_prod=[], dlogp_prod=[], dlogp_loop=[], kl_loop=[]) for n in names}
t0 = time.time()
for cls in ["prod_mid", "tight"]:
    for m in [x for x in meta if x["cls"] == cls][:N_EACH]:
        prob = (MATH if m["bench"] == "math500" else GSM)[m["id"]]
        p_ids = tok(qw.build_prompt(tok, prob["problem"]), add_special_tokens=False).input_ids
        seq = p_ids + m["seg"][:LEN]
        others = [n for n in names if n != "base"]
        for g0 in range(0, len(others), 4):
            grp = ["base"] + others[g0:g0 + 4]
            ed.set(len(grp), [conds[n] for n in grp])
            with torch.inference_mode():
                hid = lm(input_ids=torch.tensor([seq] * len(grp))).last_hidden_state
                pos = torch.arange(len(p_ids) + BURN, len(seq) - 1, STEP)
                tgt = torch.tensor(seq)[pos + 1]
                lp0 = torch.log_softmax(model.lm_head(hid[0:1, pos]).float(), -1)[0]
                for b, n in enumerate(grp):
                    if b == 0:
                        continue
                    lpb = torch.log_softmax(model.lm_head(hid[b:b + 1, pos]).float(), -1)[0]
                    kl = (lp0.exp() * (lp0 - lpb)).sum(-1).mean().item()
                    dl = (lpb.gather(1, tgt[:, None]) - lp0.gather(1, tgt[:, None])).mean().item()
                    key = ("kl_prod", "dlogp_prod") if cls == "prod_mid" else ("kl_loop", "dlogp_loop")
                    out[n][key[0]].append(kl)
                    out[n][key[1]].append(dl)
                    del lpb
            ed.clear()
            del hid, lp0
        print(f"{cls} {m['id']} elapsed {time.time() - t0:.0f}s", flush=True)
json.dump(out, open("/home/claude/poc/results_T.json", "w"))
print(f"{'condition':24s} {'KL on productive text':>22s} {'dlogp productive':>17s} {'dlogp loop continuation':>24s}")
for n in names:
    if n == "base":
        continue
    o = out[n]
    print(f"{n:24s} {np.mean(o['kl_prod']):22.4f} {np.mean(o['dlogp_prod']):17.4f} {np.mean(o['dlogp_loop']):24.4f}")
print("DONE", flush=True)
