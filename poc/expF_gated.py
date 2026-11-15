"""Experiment F: is the missing piece "when to push"?  (CPU check for the candidate-2 plan)

Target: JSON answers with no instruction in the prompt (same questions as expD_install.py).
The instruction's effect is measured at aligned answer positions. Position 0 is the last prompt token (the one that
predicts the first answer token). Position j >= 1 is where the j-th answer token is the input.
  const        : add the position-0 difference at every position of one layer (what activation-steering papers do)
  gate0_L      : add it only at position 0, at one layer (optionally with a larger dose)
  gate0_<set>  : at position 0, at every layer of a set, add what is still missing there so that the mean hidden state
                 matches the mean with the instruction (set = all 24 layers, the last 12, the last 6, the last 2, ...)
  baked_<set>  : the same as real weight edits. One rank-1 change W += u v^T per layer of the set, on the MLP output
                 matrix. u is that layer's missing difference; v is fitted so that v.h is 1 at position 0 and 0 at every
                 other position (rest of the prompt, prose answers, JSON answers). No hook is involved at test time.
"""
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qw
from gen import Editor, generate, END_IDS

torch.set_grad_enabled(False)
SMOKE = os.environ.get("SMOKE") == "1"
tok, model = qw.load_model()
lm = qw.lm_of(model)
ed = Editor(model)
H = {}
for l, layer in enumerate(lm.layers):
    layer.mlp.down_proj.register_forward_hook(lambda mod, inp, out, l=l: H.__setitem__(l, inp[0].detach()))

PRE = {}
assert hasattr(lm, "norm")
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
NEWSTYLE = [("Extract the person's name from this sentence: 'Dr. Alice Kim arrived late to the meeting.'", "alice"),
            ("Translate the word 'hello' into Spanish.", "hola"), ("Name the color of a ripe banana.", "yellow"),
            ("Extract the city from this sentence: 'The conference will be held in Vienna next May.'", "vienna"),
            ("Give the plural of the word 'mouse'.", "mice"), ("Which is bigger, an elephant or a cat?", "elephant"),
            ("Extract the year from this sentence: 'The bridge was completed in 1932 after four years.'", "1932"),
            ("What comes after Tuesday?", "wednesday"), ("Spell the word 'cat' backwards.", "tac"),
            ("Extract the product from this sentence: 'She bought a new laptop from the store.'", "laptop")]
BUILD, EVAL = QA[:16], QA[16:36]
if SMOKE:
    BUILD, EVAL, NEWSTYLE = QA[:4], QA[16:18], NEWSTYLE[:1]
INSTR = 'Respond only with a JSON object of the form {"answer": "..."} and nothing else.'
NEW = 12 if SMOKE else 28
J = 5                                   # aligned positions: last prompt token, then inputs '{"', 'answer', '":', ' "'
PREFIX = tok('{"answer": "', add_special_tokens=False).input_ids
RES = f"{qw.ROOT}/results_F_smoke.json" if SMOKE else f"{qw.ROOT}/results_F.json"


def prompt_ids(q, instr=None):
    return tok(qw.build_prompt(tok, q if instr is None else q + " " + instr, thinking=False, raw=True), add_special_tokens=False).input_ids


def strip(tokens):
    return [t for t in tokens if t not in END_IDS]


def greedy(ids, n=NEW):
    return generate(model, ids, 1, [0], n, temperature=0, presence=0.0, stop_loop=False)[0]


def run(seq):
    """res[l + 1] = residual after block l (l = 0..23), hh[l] = MLP hidden activations of layer l (inputs of down_proj)"""
    out = lm(input_ids=torch.tensor([seq]), output_hidden_states=True)
    res = torch.stack([h[0].float() for h in out.hidden_states[:NL]] + [PRE["x"][0].float()])     # [25, T, 1024]
    hh = torch.stack([H[l][0].float() for l in range(NL)])                                        # [24, T, 3584]
    return res, hh


def run_patched(ids, y, ops):
    """prompt with the position-0 pushes `ops` switched on, then the answer tokens y; returns MLP hidden activations"""
    ed.set(1, [ops])
    ed.ctx = ("prompt_last",)
    o1 = lm(input_ids=torch.tensor([ids]), use_cache=True)
    h1 = torch.stack([H[l][0].float() for l in range(NL)])
    ed.ctx = None
    lm(input_ids=torch.tensor([y]), past_key_values=o1.past_key_values, use_cache=True)
    h2 = torch.stack([H[l][0].float() for l in range(NL)])
    ed.clear()
    return torch.cat([h1, h2], 1)


def parse(txt):
    """returns (starts with '{', valid JSON object, has key 'answer', nothing after the object)"""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip()).strip()
    if not t.startswith("{"):
        return False, False, False, False
    depth = 0
    for i, ch in enumerate(t):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            try:
                d = json.loads(t[: i + 1])
                return True, isinstance(d, dict), isinstance(d, dict) and "answer" in d, t[i + 1:].strip() == ""
            except Exception:
                return True, False, False, False
    return True, False, False, False


# ------------------------------------------------------------------ contrast statistics from the build questions
t0 = time.time()
E, build = [], []                        # E: per build question [J, 25, 1024] instruction effect at aligned positions
aligned = 0
for qi, (q, gold) in enumerate(BUILD):
    ids0, ids1 = prompt_ids(q), prompt_ids(q, INSTR)
    y0, y1 = strip(greedy(ids0)["tokens"]), strip(greedy(ids1)["tokens"])
    p0, p1 = len(ids0) - 1, len(ids1) - 1
    resA0, hhA0 = run(ids0 + y0)         # no instruction, the model's own prose answer
    resA1, _ = run(ids0 + y1)            # no instruction, the JSON answer forced in
    resB1, _ = run(ids1 + y1)            # instruction present, JSON answer
    if y1[:len(PREFIX)] == PREFIX:
        aligned += 1
        E.append(torch.stack([resB1[:, p1 + j] - resA1[:, p0 + j] for j in range(J)]))
    build.append(dict(ids0=ids0, y1=y1, p0=p0, hhA0=hhA0))
E = torch.stack(E)                       # [Q, J, 25, 1024]
d_pos = E.mean(0)                        # [J, 25, 1024]
d0 = d_pos[0]                            # [25, 1024] mean difference at position 0; index l + 1 = after block l
print(f"contrast built in {time.time() - t0:.0f}s; JSON prefix aligned on {aligned}/{len(BUILD)} build questions", flush=True)

# ---- how much does the instruction's effect change from one answer position to the next?
stats = {}
for L in (5, 9, 13, 17, 21, 23):
    M = E[:, :, L + 1]                                             # [Q, J, 1024]
    m = M.mean(0)                                                  # [J, 1024]
    cos = [float(torch.nn.functional.cosine_similarity(m[0], m[j], dim=0)) for j in range(J)]
    norms = [float(m[j].norm()) for j in range(J)]
    S = torch.linalg.svdvals(M.reshape(-1, M.shape[-1]))
    ev = (S ** 2).cumsum(0) / (S ** 2).sum()
    stats[L] = dict(cos_to_pos0=cos, norm_of_mean=norms, var_top=[float(ev[k - 1]) for k in (1, 2, 3, 5)])
    print(f"layer {L}: cos(mean effect at position 0, at position j) = {[round(c, 2) for c in cos]}; norms {[round(x, 1) for x in norms]}; "
          f"share of squared norm in top 1/2/3/5 directions = {[round(float(ev[k - 1]), 2) for k in (1, 2, 3, 5)]}", flush=True)

# ---- layer sets, and for each set the pushes that make the mean hidden state at position 0 match the instructed mean
SETS = {"all24": list(range(NL)), "late12": list(range(12, NL)), "late6": list(range(18, NL)), "last2": [22, 23],
        "early12": list(range(12)), "coarse6": [3, 7, 11, 15, 19, 23], "coarse3": [7, 15, 23], "L23": [23]}


def writes(S):
    u, prev = {}, None
    for l in S:
        u[l] = d0[l + 1] - (d0[prev + 1] if prev is not None else 0)
        prev = l
    return u


def act_ops(S):
    return [("sched", l, {0: u}) for l, u in writes(S).items()]


# ---- gate vectors for the baked edits: one per layer of the set, fitted on activations of the patched model
def fit(Hx, yx, lam):
    G = Hx @ Hx.T
    return Hx.T @ torch.linalg.solve(G + lam * G.diagonal().mean() * torch.eye(len(G)), yx)


nq = len(BUILD)
held = set(range(nq - max(1, nq // 4), nq))
BAKED = ["all24", "coarse6", "late6", "L23"]
gate, gate_fit = {}, {}
for sname in BAKED:
    S = SETS[sname]
    ops = act_ops(S)
    rows = []                                                      # (question, label, [24, 3584])
    for qi, bq in enumerate(build):
        hp = run_patched(bq["ids0"], bq["y1"], ops)                # [24, T, 3584]; JSON answer after the patched prompt
        for t in range(hp.shape[1]):
            rows.append((qi, 1.0 if t == bq["p0"] else 0.0, hp[:, t]))
        hA = bq["hhA0"]                                            # unpatched prompt, prose answer
        for t in range(hA.shape[1]):
            rows.append((qi, 1.0 if t == bq["p0"] else 0.0, hA[:, t]))
    qidx = torch.tensor([r[0] for r in rows])
    y = torch.tensor([r[1] for r in rows])
    tr = torch.tensor([int(i) not in held for i in qidx])
    gate[sname], rec = {}, []
    for l in S:
        Hm = torch.stack([r[2][l] for r in rows])
        best = None
        for lam in (1e-3, 1e-2, 1e-1):
            g = Hm[~tr] @ fit(Hm[tr], y[tr], lam)
            pos, neg = g[y[~tr] == 1], g[y[~tr] == 0]
            r = dict(layer=l, lam=lam, pos_mean=float(pos.mean()), pos_min=float(pos.min()), neg_max=float(neg.abs().max()),
                     neg_p99=float(neg.abs().quantile(0.99)), margin=float(pos.min() - neg.abs().max()))
            if best is None or r["margin"] > best["margin"]:
                best = r
        gate[sname][l] = fit(Hm, y, best["lam"])
        rec.append(best)
    gate_fit[sname] = rec
    print(f"gates for {sname}: held-out build questions, over layers: gate at position 0 mean {np.mean([r['pos_mean'] for r in rec]):.2f} "
          f"(lowest single value {min(r['pos_min'] for r in rec):.2f}); elsewhere largest |gate| {max(r['neg_max'] for r in rec):.2f}, "
          f"mean 99th pct {np.mean([r['neg_p99'] for r in rec]):.3f}   [{time.time() - t0:.0f}s]", flush=True)
del build

# ------------------------------------------------------------------ conditions
conds = {"no_edit": []}
for L in (9, 13, 17):
    conds[f"const_L{L}"] = [("push", L, d0[L + 1])]
for L in (13, 17, 21):
    conds[f"gate0_L{L}"] = [("sched", L, {0: d0[L + 1]})]
for L in (13, 17):
    conds[f"gate0_L{L}_x4"] = [("sched", L, {0: 4 * d0[L + 1]})]
for sname, S in SETS.items():
    conds[f"gate0_{sname}"] = act_ops(S)
for sname in BAKED:
    u = writes(SETS[sname])
    conds[f"baked_{sname}"] = [("rank1", l, u[l], gate[sname][l]) for l in SETS[sname]]
names = list(conds)
B = len(names)
print(f"{B} conditions; starting evaluation at {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ evaluation (no instruction in the prompt)
out_rows, gate_eval = [], {sn: dict(pos0=[], elsewhere=[]) for sn in BAKED}
sets = {"same_style": EVAL, "new_style": NEWSTYLE}
for sname, items in sets.items():
    for q, gold in items:
        ids = prompt_ids(q)
        ed.set(B, [conds[nm] for nm in names])
        res = generate(model, ids, B, list(range(B)), NEW, temperature=0, presence=0.0, stop_loop=False, editor=ed)
        ed.clear()
        for nm, x in zip(names, res):
            txt = tok.decode(strip(x["tokens"])).strip()
            a, b, c, d = parse(txt)
            out_rows.append(dict(set=sname, q=q, cond=nm, text=txt, starts=a, valid=b, strict=c, clean_end=bool(d and x["end"] == "eos"),
                                 correct=any(g in txt.lower() for g in gold.split("|")), ntok=len(x["tokens"])))
        for nm, x in zip(names, res):
            if nm.startswith("baked_"):
                # gate values inside the edited model, on the text it wrote
                sn = nm[6:]
                ed.set(1, [conds[nm]])
                _, hh = run(ids + strip(x["tokens"]))
                ed.clear()
                p0 = len(ids) - 1
                for l in SETS[sn]:
                    g = hh[l] @ gate[sn][l]
                    gate_eval[sn]["pos0"].append(float(g[p0]))
                    gate_eval[sn]["elsewhere"].extend([float(v) for i, v in enumerate(g) if i != p0])
        g = greedy(prompt_ids(q, INSTR))
        txt = tok.decode(strip(g["tokens"])).strip()
        a, b, c, d = parse(txt)
        out_rows.append(dict(set=sname, q=q, cond="instruction_in_prompt", text=txt, starts=a, valid=b, strict=c, clean_end=bool(d and g["end"] == "eos"),
                             correct=any(gg in txt.lower() for gg in gold.split("|")), ntok=len(g["tokens"])))
    print(f"finished set {sname} elapsed {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(rows=out_rows, effect_stats=stats, gate_fit=gate_fit, gate_eval=gate_eval, aligned=aligned), open(RES, "w"))

for sname in sets:
    print(f"[{sname}] {'condition':16s} {'starts {':>9s} {'valid':>6s} {'strict':>7s} {'clean':>6s} {'answer':>7s} {'strict+ans':>10s}")
    for nm in names + ["instruction_in_prompt"]:
        r = [x for x in out_rows if x["cond"] == nm and x["set"] == sname]
        f = lambda k: np.mean([x[k] for x in r])
        print(f"[{sname}] {nm:16s} {f('starts'):9.2f} {f('valid'):6.2f} {f('strict'):7.2f} {f('clean_end'):6.2f} {f('correct'):7.2f} "
              f"{np.mean([x['strict'] and x['correct'] for x in r]):10.2f}", flush=True)
for sn in BAKED:
    e = np.abs(np.array(gate_eval[sn]["elsewhere"]))
    print(f"gates of baked_{sn} on evaluation questions: at position 0 mean {np.mean(gate_eval[sn]['pos0']):.2f} min {np.min(gate_eval[sn]['pos0']):.2f}; "
          f"elsewhere max |gate| {e.max():.2f}, 99th pct {np.quantile(e, 0.99):.3f}, n {len(e)}")
print("DONE", flush=True)
