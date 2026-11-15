"""Experiment G: at which answer positions does a model without the instruction part ways with the instructed model,
and how much of that is left after the answer-start edit?

For each evaluation question the instructed model's own answer is fed to three or four models and the next-token
distributions are compared position by position (KL from the instructed model):
  U : no instruction, no edit
  P : no instruction, pushes at the last prompt token on all 24 layers (activation level)
  W : no instruction, the same as 24 real rank-1 weight edits
Positions are grouped by what the instructed model writes next (answer start, key, punctuation, content, close, end).
Two behaviors: JSON format and a brief answer. For the brief answer the script also lets the edited models write freely.
"""
import json
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/claude/poc")
import qw
from gen import Editor, generate, END_IDS

torch.set_grad_enabled(False)
tok, model = qw.load_model()
lm = qw.lm_of(model)
ed = Editor(model)
H, PRE = {}, {}
for l, layer in enumerate(lm.layers):
    layer.mlp.down_proj.register_forward_hook(lambda mod, inp, out, l=l: H.__setitem__(l, inp[0].detach()))
lm.norm.register_forward_hook(lambda mod, inp, out: PRE.__setitem__("x", inp[0].detach()))
NL = qw.N_LAYERS

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
NEW = 28


def prompt_ids(q, instr=None):
    return tok(qw.build_prompt(tok, q if instr is None else q + " " + instr, thinking=False, raw=True), add_special_tokens=False).input_ids


def strip(tokens):
    return [t for t in tokens if t not in END_IDS]


def greedy(ids, n=NEW):
    return generate(model, ids, 1, [0], n, temperature=0, presence=0.0, stop_loop=False)[0]


def run(seq):
    out = lm(input_ids=torch.tensor([seq]), output_hidden_states=True)
    res = torch.stack([h[0].float() for h in out.hidden_states[:NL]] + [PRE["x"][0].float()])
    hh = torch.stack([H[l][0].float() for l in range(NL)])
    return res, hh


def run_patched(ids, y, ops, want="h"):
    """prompt with position-0 pushes `ops`, then tokens y. want='h': MLP hidden activations; want='logp': next-token log-probs
    at the last prompt position and at every position of y"""
    ed.set(1, [ops])
    ed.ctx = ("prompt_last",)
    o1 = lm(input_ids=torch.tensor([ids]), use_cache=True)
    h1 = torch.stack([H[l][0].float() for l in range(NL)])
    last = [o1.last_hidden_state[0, -1:]]
    ed.ctx = None
    if y:
        o2 = lm(input_ids=torch.tensor([y]), past_key_values=o1.past_key_values, use_cache=True)
        h1 = torch.cat([h1, torch.stack([H[l][0].float() for l in range(NL)])], 1)
        last.append(o2.last_hidden_state[0])
    ed.clear()
    if want == "h":
        return h1
    return torch.log_softmax(model.lm_head(torch.cat(last)).float(), -1)


def logp_seq(ids, y, ops=None):
    if ops is not None:
        ed.set(1, [ops])
    o = lm(input_ids=torch.tensor([ids + y]))
    ed.clear()
    return torch.log_softmax(model.lm_head(o.last_hidden_state[0, len(ids) - 1:]).float(), -1)


def fit(Hx, yx, lam):
    G = Hx @ Hx.T
    return Hx.T @ torch.linalg.solve(G + lam * G.diagonal().mean() * torch.eye(len(G)), yx)


def bucket(task, j, n, has_end):
    if j == 0:
        return "1 answer start"
    if j == n and has_end:
        return "6 end of answer"
    if task == "json":
        if j == 1:
            return "2 key name"
        if j in (2, 3):
            return "3 punctuation after key"
        if j == n - 1:
            return "5 closing brace"
    return "4 content"


ALL = {}
t0 = time.time()
for task, instr in TASKS.items():
    # ---------------- build: mean difference at the last prompt token, and gates for the 24 baked edits
    D, build = [], []
    for q, gold in BUILD:
        ids0, ids1 = prompt_ids(q), prompt_ids(q, instr)
        y0, y1 = strip(greedy(ids0)["tokens"]), strip(greedy(ids1)["tokens"])
        resA0, hhA0 = run(ids0 + y0)
        resB, _ = run(ids1)
        p0 = len(ids0) - 1
        D.append(resB[:, -1] - resA0[:, p0])
        build.append(dict(ids0=ids0, y1=y1, p0=p0, hhA0=hhA0))
    d0 = torch.stack(D).mean(0)                                    # [25, 1024]; index l + 1 = after block l
    u = {l: d0[l + 1] - d0[l] for l in range(NL)}                  # d0[0] is zero (same token)
    act = [("sched", l, {0: u[l]}) for l in range(NL)]
    rows = []
    for qi, bq in enumerate(build):
        hp = run_patched(bq["ids0"], bq["y1"], act)
        for t in range(hp.shape[1]):
            rows.append((qi, 1.0 if t == bq["p0"] else 0.0, hp[:, t]))
        for t in range(bq["hhA0"].shape[1]):
            rows.append((qi, 1.0 if t == bq["p0"] else 0.0, bq["hhA0"][:, t]))
    del build
    qidx = torch.tensor([r[0] for r in rows])
    y = torch.tensor([r[1] for r in rows])
    tr = qidx < 12
    gate = {}
    for l in range(NL):
        Hm = torch.stack([r[2][l] for r in rows])
        best = None
        for lam in (1e-3, 1e-2, 1e-1):
            g = Hm[~tr] @ fit(Hm[tr], y[tr], lam)
            m = float(g[y[~tr] == 1].min() - g[y[~tr] == 0].abs().max())
            if best is None or m > best[0]:
                best = (m, lam)
        gate[l] = fit(Hm, y, best[1])
    del rows
    baked = [("rank1", l, u[l], gate[l]) for l in range(NL)]
    print(f"[{task}] build done at {time.time() - t0:.0f}s", flush=True)

    # ---------------- divergence profile along the instructed model's own answer
    prof = []
    for q, gold in EVAL:
        ids0, ids1 = prompt_ids(q), prompt_ids(q, instr)
        g1 = greedy(ids1)
        y1 = strip(g1["tokens"])
        has_end = g1["end"] == "eos"
        targets = y1 + ([g1["tokens"][-1]] if has_end else [])
        n = len(y1)
        LI, LU = logp_seq(ids1, y1), logp_seq(ids0, y1)
        LP = run_patched(ids0, y1, act, want="logp")
        LW = logp_seq(ids0, y1, ops=baked)
        for j, tgt in enumerate(targets):
            pI = LI[j].exp()
            rec = dict(q=q, j=j, bucket=bucket(task, j, n, has_end), target=tok.decode([tgt]), pI=float(pI[tgt]))
            for nm, L in (("U", LU), ("P", LP), ("W", LW)):
                rec["kl_" + nm] = float((pI * (LI[j] - L[j])).sum())
                rec["p_" + nm] = float(L[j][tgt].exp())
                rec["top_" + nm] = tok.decode([int(L[j].argmax())])
            prof.append(rec)
    print(f"[{task}] where the model without the instruction parts ways with the instructed model (KL in nats; p = probability of the instructed model's token)")
    print(f"[{task}] {'position group':26s} {'n':>4s} | {'KL no edit':>10s} {'KL pushes':>10s} {'KL weights':>10s} | {'p instr':>8s} {'p no edit':>9s} {'p pushes':>9s} {'p weights':>9s}")
    summ = {}
    for b in sorted(set(r["bucket"] for r in prof)):
        r = [x for x in prof if x["bucket"] == b]
        f = lambda k: float(np.mean([x[k] for x in r]))
        summ[b] = dict(n=len(r), kl_U=f("kl_U"), kl_P=f("kl_P"), kl_W=f("kl_W"), pI=f("pI"), p_U=f("p_U"), p_P=f("p_P"), p_W=f("p_W"))
        print(f"[{task}] {b:26s} {len(r):4d} | {f('kl_U'):10.2f} {f('kl_P'):10.2f} {f('kl_W'):10.2f} | {f('pI'):8.2f} {f('p_U'):9.2f} {f('p_P'):9.2f} {f('p_W'):9.2f}", flush=True)
    ALL[task] = dict(profile=prof, summary=summ)

    # ---------------- free writing for the brief answer (JSON free writing is in expF_gated.py)
    if task == "brief":
        conds = {"no_edit": [], "const_L13": [("push", 13, d0[14])], "const_L17": [("push", 17, d0[18])], "gate0_all24": act, "baked_all24": baked}
        names = list(conds)
        out = []
        for q, gold in EVAL:
            ids = prompt_ids(q)
            ed.set(len(names), [conds[nm] for nm in names])
            res = generate(model, ids, len(names), list(range(len(names))), NEW, temperature=0, presence=0.0, stop_loop=False, editor=ed)
            ed.clear()
            res.append(greedy(prompt_ids(q, instr)))
            for nm, x in zip(names + ["instruction_in_prompt"], res):
                txt = tok.decode(strip(x["tokens"])).strip()
                w = re.findall(r"[A-Za-z0-9]+", txt)
                out.append(dict(q=q, cond=nm, text=txt, brief=0 < len(w) <= 3, words=len(w), ended=x["end"] == "eos",
                                correct=any(g in txt.lower() for g in gold.split("|"))))
        print(f"[brief] {'condition':22s} {'<=3 words':>9s} {'answer':>7s} {'both':>6s} {'ended':>6s} {'median words':>12s}")
        for nm in names + ["instruction_in_prompt"]:
            r = [x for x in out if x["cond"] == nm]
            print(f"[brief] {nm:22s} {np.mean([x['brief'] for x in r]):9.2f} {np.mean([x['correct'] for x in r]):7.2f} "
                  f"{np.mean([x['brief'] and x['correct'] for x in r]):6.2f} {np.mean([x['ended'] for x in r]):6.2f} {np.median([x['words'] for x in r]):12.1f}", flush=True)
        ALL[task]["free"] = out
    json.dump(ALL, open("/home/claude/poc/results_G.json", "w"))
    print(f"[{task}] finished at {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
