"""Experiment D: installing a behavior without training (an edit that adds, instead of erasing).

Two target behaviors that the editing benchmarks report as hard: a fixed output format (JSON) and answer length (brief).
The instruction is used only to measure what changes inside the model; at test time the prompt has no instruction.
  direction-level edit : add the mean residual difference (with instruction - without) at one layer, scaled by a dose
  neuron-level edit    : add the mean change of the MLP output carried by the top-k neurons of each layer
  baked edit           : a real rank-1 weight change  W += (dose * direction) v^T  with v fitted (ridge) so that v.h ~ 1
"""
import json
import re
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qw
from gen import Editor, generate

torch.set_grad_enabled(False)
tok, model = qw.load_model()
lm = qw.lm_of(model)
ed = Editor(model)
H = {}
for l, layer in enumerate(lm.layers):
    layer.mlp.down_proj.register_forward_hook(lambda mod, inp, out, l=l: H.__setitem__(l, inp[0].detach()))

QA = [("What is the capital of France?", "paris"), ("What is the capital of Japan?", "tokyo"), ("What is the chemical symbol for gold?", "au"),
      ("How many days are there in a week?", "7|seven"), ("What is the largest planet in the solar system?", "jupiter"),
      ("How many sides does a hexagon have?", "6|six"), ("What is the opposite of hot?", "cold"), ("Who wrote Romeo and Juliet?", "shakespeare"),
      ("What is the capital of Italy?", "rome"), ("What is the square root of 81?", "9|nine"), ("Which ocean is the largest?", "pacific"),
      ("How many minutes are in an hour?", "60|sixty"), ("What is the capital of Germany?", "berlin"), ("What gas do plants absorb from the air?", "carbon dioxide|co2"),
      ("What is 7 times 8?", "56"), ("What is the capital of Spain?", "madrid"), ("How many legs does a spider have?", "8|eight"),
      ("What is the boiling point of water in Celsius?", "100"), ("What planet is known as the Red Planet?", "mars"), ("What is the capital of Egypt?", "cairo"),
      ("How many continents are there?", "7|seven"), ("What is the chemical formula of water?", "h2o"), ("What is the capital of Canada?", "ottawa"),
      ("What is 12 plus 30?", "42"), ("Which animal is known as the king of the jungle?", "lion"), ("How many months are in a year?", "12|twelve"),
      ("What is the capital of Russia?", "moscow"), ("What is the largest mammal?", "whale"), ("What is the first letter of the Greek alphabet?", "alpha"),
      ("How many hours are in a day?", "24|twenty-four"), ("What is the capital of China?", "beijing"), ("What is the fastest land animal?", "cheetah"),
      ("What do bees make?", "honey"), ("What is the capital of the United Kingdom?", "london"), ("How many players are on a soccer team on the field?", "11|eleven"),
      ("What is the hardest natural substance?", "diamond")]
BUILD, EVAL = QA[:16], QA[16:36]
TASKS = {"json": 'Respond only with a JSON object of the form {"answer": "..."} and nothing else.',
         "brief": "Reply with the answer only, in as few words as possible. No explanation."}
if len(sys.argv) > 1:
    TASKS = {k: v for k, v in TASKS.items() if k in sys.argv[1].split(",")}
RESFILE = f"{qw.ROOT}/results_D.json" if len(sys.argv) <= 1 else f"{qw.ROOT}/results_D_{sys.argv[1]}.json"
LAYERS_RES, DOSES = [5, 9, 13], [1.0, 2.0, 4.0]
MLP_RANGE = list(range(2, 16))
NEW = 32


def check(task, txt):
    t = txt.strip()
    if task == "json":
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
        if not t.startswith("{"):
            return False
        depth = 0
        for i, ch in enumerate(t):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                try:
                    d = json.loads(t[: i + 1])
                    return isinstance(d, dict) and "answer" in d
                except Exception:
                    return False
        return False
    words = re.findall(r"[A-Za-z0-9]+", t)
    return 0 < len(words) <= 3


def clean(tokens):
    return tok.decode([t for t in tokens if t != 248044]).replace("<|endoftext|>", "").strip()


def prompt_ids(q, instr=None):
    return tok(qw.build_prompt(tok, q if instr is None else q + " " + instr, thinking=False, raw=True), add_special_tokens=False).input_ids


ALL = {}
t0 = time.time()
for task, instr in TASKS.items():
    # ---------------- contrast statistics from the build questions
    sr = torch.zeros(2, qw.N_LAYERS + 1, qw.D_MODEL, dtype=torch.float64)
    sh = torch.zeros(2, qw.N_LAYERS, qw.D_MLP, dtype=torch.float64)
    n = [0, 0]
    Hfit = {l: [] for l in LAYERS_RES}
    comply_build = []
    for q, gold in BUILD:
        for c, ins in enumerate([None, instr]):
            ids = prompt_ids(q, ins)
            g = generate(model, ids, 1, [0], NEW, temperature=0, presence=0.0, stop_loop=False)[0]
            if c == 1:
                comply_build.append(check(task, clean(g["tokens"])))
            seq = ids + [t for t in g["tokens"] if t != 248044]
            with torch.inference_mode():
                out = lm(input_ids=torch.tensor([seq]), output_hidden_states=True)
            p0 = len(ids) - 1
            for li in range(qw.N_LAYERS + 1):
                sr[c, li] += out.hidden_states[li][0, p0:].float().sum(0).double()
            for l in range(qw.N_LAYERS):
                sh[c, l] += H[l][0, p0:].float().sum(0).double()
                if c == 0 and l in Hfit:
                    Hfit[l].append(H[l][0].float().clone())
            n[c] += len(seq) - p0
    d_res = (sr[1] / n[1] - sr[0] / n[0]).float()
    d_h = (sh[1] / n[1] - sh[0] / n[0]).float()
    print(f"[{task}] contrast built in {time.time() - t0:.0f}s; instruction followed on build questions: {np.mean(comply_build):.2f}", flush=True)
    # ---------------- conditions
    conds = {"no_edit": []}
    for L in LAYERS_RES:
        for a in DOSES:
            conds[f"direction_L{L}_x{a:g}"] = [("push", L, a * d_res[L + 1])]
    Wn = {l: lm.layers[l].mlp.down_proj.weight.detach().float() for l in MLP_RANGE}
    for a in (1.0, 2.0):
        conds[f"neurons_all_x{a:g}"] = [("push", l, a * (Wn[l] @ d_h[l])) for l in MLP_RANGE]
        for k in (8, 64):
            ops = []
            for l in MLP_RANGE:
                idx = (d_h[l].abs() * Wn[l].norm(dim=0)).topk(k).indices
                ops.append(("push", l, a * (Wn[l][:, idx] @ d_h[l][idx])))
            conds[f"neurons_top{k}_x{a:g}"] = ops
    fitq = {}
    for L, a in ((9, 2.0), (5, 2.0)):
        Hm = torch.cat(Hfit[L], 0)
        G = Hm @ Hm.T
        lam = 1e-2 * G.diagonal().mean()
        v = Hm.T @ torch.linalg.solve(G + lam * torch.eye(len(G)), torch.ones(len(G)))
        fitq[L] = float(((Hm @ v) - 1).abs().mean())
        conds[f"baked_weight_L{L}_x{a:g}"] = [("rank1", L, a * d_res[L + 1], v)]
    names = list(conds)
    # explained share of the local MLP change by top-k neurons (how concentrated is the change?)
    conc = {}
    for k in (8, 64, 512):
        num = den = 0.0
        for l in MLP_RANGE:
            full = Wn[l] @ d_h[l]
            idx = (d_h[l].abs() * Wn[l].norm(dim=0)).topk(k).indices
            part = Wn[l][:, idx] @ d_h[l][idx]
            num += float((part @ full) / full.norm())
            den += float(full.norm())
        conc[k] = num / den
    print(f"[{task}] share of the mean MLP-output change carried by top-k neurons per layer: {conc}; baked-fit mean abs error {fitq}", flush=True)
    # ---------------- evaluation
    NEWSTYLE = [("Extract the person's name from this sentence: 'Dr. Alice Kim arrived late to the meeting.'", "alice"),
                ("Translate the word 'hello' into Spanish.", "hola"), ("Name the color of a ripe banana.", "yellow"),
                ("Extract the city from this sentence: 'The conference will be held in Vienna next May.'", "vienna"),
                ("Give the plural of the word 'mouse'.", "mice"), ("Which is bigger, an elephant or a cat?", "elephant"),
                ("Extract the year from this sentence: 'The bridge was completed in 1932 after four years.'", "1932"),
                ("What comes after Tuesday?", "wednesday"), ("Spell the word 'cat' backwards.", "tac"),
                ("Extract the product from this sentence: 'She bought a new laptop from the store.'", "laptop")]
    PROSE = " Answer in one full plain-English sentence, not in JSON." if task == "json" else " Explain your answer in two or three full sentences."
    sets = {"same_style": [(q, g, None) for q, g in EVAL], "new_style": [(q, g, None) for q, g in NEWSTYLE],
            "explicit_other_request": [(q + PROSE, g, "other") for q, g in EVAL[:8]]}
    rows = []
    for sname, items in sets.items():
        for qi, (q, gold, kind) in enumerate(items):
            ids = prompt_ids(q)
            B = len(names)
            ed.set(B, [conds[nm] for nm in names])
            res = generate(model, ids, B, list(range(B)), NEW if kind is None else 48, temperature=0, presence=0.0, stop_loop=False)
            ed.clear()
            for nm, x in zip(names, res):
                txt = clean(x["tokens"])
                rows.append(dict(task=task, set=sname, q=q, cond=nm, text=txt, fmt=bool(check(task, txt)),
                                 correct=any(g in txt.lower() for g in gold.split("|")), ntok=len(x["tokens"])))
            if kind is None:
                g = generate(model, prompt_ids(q, instr), 1, [0], NEW, temperature=0, presence=0.0, stop_loop=False)[0]
                txt = clean(g["tokens"])
                rows.append(dict(task=task, set=sname, q=q, cond="instruction_in_prompt", text=txt, fmt=bool(check(task, txt)),
                                 correct=any(gg in txt.lower() for gg in gold.split("|")), ntok=len(g["tokens"])))
        print(f"[{task}] finished set {sname} elapsed {time.time() - t0:.0f}s", flush=True)
    ALL[task] = dict(rows=rows, conc=conc, comply_build=float(np.mean(comply_build)), fitq=fitq)
    json.dump(ALL, open(RESFILE, "w"))
    for sname in sets:
        print(f"[{task}/{sname}] {'condition':28s} {'format ok':>10s} {'answer ok':>10s} {'both':>6s} {'tokens':>7s}")
        for nm in names + ["instruction_in_prompt"]:
            r = [x for x in rows if x["cond"] == nm and x["set"] == sname]
            if not r:
                continue
            print(f"[{task}/{sname}] {nm:28s} {np.mean([x['fmt'] for x in r]):10.2f} {np.mean([x['correct'] for x in r]):10.2f} "
                  f"{np.mean([x['fmt'] and x['correct'] for x in r]):6.2f} {np.mean([x['ntok'] for x in r]):7.1f}", flush=True)
print("DONE", flush=True)
