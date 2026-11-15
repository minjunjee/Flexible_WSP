"""Experiment E: does an edit change answers? Thinking off, greedy decoding, short direct-answer questions.
Every edited model is compared with the unedited model inside the same batch (bf16 results depend on batch shape).
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qw
from conds import all_conditions
from gen import Editor, generate

ALT = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [11, 23]
torch.set_grad_enabled(False)
tok, model = qw.load_model()
ed = Editor(model)
conds, sets, S = all_conditions(model, ALT)
names = list(conds)
rng = np.random.default_rng(3)
Q = []
for _ in range(14):
    a, b = int(rng.integers(11, 90)), int(rng.integers(11, 90))
    Q.append((f"What is {a} + {b}? Answer with just the number.", str(a + b)))
for _ in range(10):
    a, b = int(rng.integers(40, 99)), int(rng.integers(11, 40))
    Q.append((f"What is {a} - {b}? Answer with just the number.", str(a - b)))
for _ in range(10):
    a, b = int(rng.integers(3, 10)), int(rng.integers(11, 30))
    Q.append((f"What is {a} times {b}? Answer with just the number.", str(a * b)))
FACT = [("What is the capital of France?", "paris"), ("What is the capital of Japan?", "tokyo"), ("What is the chemical symbol for gold?", "au"),
        ("How many days are there in a week?", "7|seven"), ("What is the largest planet in the solar system?", "jupiter"),
        ("What color do you get when you mix red and white?", "pink"), ("How many sides does a hexagon have?", "6|six"),
        ("What is the opposite of hot?", "cold"), ("Who wrote Romeo and Juliet?", "shakespeare"), ("What is the capital of Italy?", "rome"),
        ("What is the freezing point of water in Celsius?", "0|zero"), ("What is the square root of 81?", "9|nine"),
        ("Which ocean is the largest?", "pacific"), ("How many minutes are in an hour?", "60|sixty"),
        ("What is the capital of Germany?", "berlin"), ("What gas do plants absorb from the air?", "carbon dioxide|co2")]
Q += [(q + " Answer in a few words.", a) for q, a in FACT]
print("questions", len(Q), "conditions", names, flush=True)
rows = []
t0 = time.time()
for qi, (q, gold) in enumerate(Q):
    ids = tok(qw.build_prompt(tok, q, thinking=False, raw=True), add_special_tokens=False).input_ids
    B = len(names)
    ed.set(B, [conds[n] for n in names])
    res = generate(model, ids, B, list(range(B)), 20, temperature=0, presence=0.0, stop_loop=False)
    ed.clear()
    base_txt = None
    for n, x in zip(names, res):
        txt = tok.decode([t for t in x["tokens"] if t not in (248044, 248046)]).strip().lower()
        if n == "base":
            base_txt = txt
        ok = any(g in txt for g in gold.split("|"))
        rows.append(dict(q=q, cond=n, text=txt, correct=bool(ok), same_as_base=(txt == base_txt)))
    if qi % 10 == 0:
        print(f"[{qi + 1}/{len(Q)}] elapsed {time.time() - t0:.0f}s", flush=True)
json.dump(rows, open(f"{qw.ROOT}/results_E.json", "w"))
print(f"{'condition':24s} {'correct':>8s} {'same output as unedited':>26s}")
for n in names:
    r = [x for x in rows if x["cond"] == n]
    print(f"{n:24s} {np.mean([x['correct'] for x in r]):8.3f} {np.mean([x['same_as_base'] for x in r]):26.3f}")
print("DONE", flush=True)
