"""Experiment H: is the instruction's effect at a position the same for every question, or does it depend on the question?

For each question the hidden-state difference (instructed minus not instructed, same answer tokens fed in) is taken at
one answer position. "Constant share" = squared norm of the mean difference / mean squared norm of the differences.
1.0 means every question gets the same difference (one fixed vector can reproduce it); a value near 1/Q means the
differences share nothing.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/claude/poc")
import qw
from gen import generate, END_IDS

torch.set_grad_enabled(False)
tok, model = qw.load_model()
lm = qw.lm_of(model)
PRE = {}
lm.norm.register_forward_hook(lambda mod, inp, out: PRE.__setitem__("x", inp[0].detach()))
NL = qw.N_LAYERS
exec(open("/home/claude/poc/expG_divergence.py").read().split("QA = ")[1].split("BUILD, EVAL")[0].join(["QA = ", ""]))
TASKS = {"json": 'Respond only with a JSON object of the form {"answer": "..."} and nothing else.',
         "brief": "Reply with the answer only, in as few words as possible. No explanation."}


def prompt_ids(q, instr=None):
    return tok(qw.build_prompt(tok, q if instr is None else q + " " + instr, thinking=False, raw=True), add_special_tokens=False).input_ids


def run(seq):
    out = lm(input_ids=torch.tensor([seq]), output_hidden_states=True)
    return torch.stack([h[0].float() for h in out.hidden_states[:NL]] + [PRE["x"][0].float()])


OUT = {}
for task, instr in TASKS.items():
    D = {"answer start": [], ("key name" if task == "json" else "end of answer"): []}
    second = list(D)[1]
    for q, gold in QA:
        ids0, ids1 = prompt_ids(q), prompt_ids(q, instr)
        y1 = [t for t in generate(model, ids1, 1, [0], 28, temperature=0, presence=0.0, stop_loop=False)[0]["tokens"] if t not in END_IDS]
        rU, rI = run(ids0 + y1), run(ids1 + y1)
        p0, p1 = len(ids0) - 1, len(ids1) - 1
        D["answer start"].append(rI[:, p1] - rU[:, p0])
        j = 1 if task == "json" else len(y1)
        D[second].append(rI[:, p1 + j] - rU[:, p0 + j])
    OUT[task] = {}
    for name, X in D.items():
        X = torch.stack(X)                                   # [Q, 25, 1024]
        sh = {L: float(X[:, L + 1].mean(0).norm() ** 2 / (X[:, L + 1].norm(dim=-1) ** 2).mean()) for L in (9, 13, 17, 21, 23)}
        OUT[task][name] = sh
        print(f"[{task}] {name:14s} constant share by layer: " + ", ".join(f"L{L} {v:.2f}" for L, v in sh.items()) + f"   (Q = {len(X)}, floor {1 / len(X):.2f})", flush=True)
json.dump(OUT, open("/home/claude/poc/results_H.json", "w"))
print("DONE")
