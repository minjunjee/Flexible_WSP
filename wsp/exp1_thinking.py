"""Experiment 1, step 7 (PLAN.md): the same edits when the model thinks first.

The edits were built with thinking switched off, where the prompt ends in '<think>\\n\\n</think>\\n\\n' and the answer
starts right there. With thinking on, the prompt ends in '<think>\\n', the model writes its reasoning, closes it with
'</think>' and only then starts the answer. The question: do the edits still fire at that later answer start, and do
they stay silent while the model is thinking?

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m wsp.exp1_thinking --out results/exp1/thinking
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from wsp.edit import load_stacked
from wsp.lm import LM, load_questions
from wsp.score import boot_ci, score_json

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/exp1/thinking")
ap.add_argument("--edits", default="results/exp1/stack_json_onpolicy/edits.npz")
ap.add_argument("--max_new", type=int, default=512)
ap.add_argument("--per_type", type=int, default=40)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
INSTR = 'Respond only with a JSON object of the form {"answer": "..."} and nothing else.'
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)
lm = LM("qwen35-0.8b")
test, seen = [], {}
for q in load_questions():
    if q["split"] == "test" and seen.get(q["type"], 0) < args.per_type:
        seen[q["type"]] = seen.get(q["type"], 0) + 1
        test.append(q)
rounds = load_stacked(args.edits)
CLOSE = lm.tok.convert_tokens_to_ids("</think>")
P0 = [lm.prompt_ids(q["question"], enable_thinking=True) for q in test]
P1 = [lm.prompt_ids(q["question"], INSTR, enable_thinking=True) for q in test]
log(f"{len(test)} test questions; prompt ends with {lm.tok.decode(P0[0][-3:])!r}")

conds = {"no edit": (P0, []), "instruction in the prompt": (P1, []),
         "answer-start edit (weights only)": (P0, rounds[:1]), "answer-start edit and key-name edit (weights only)": (P0, rounds[:2])}
S = {}
for name, (prompts, eds) in conds.items():
    lm.restore()
    for e in eds:
        lm.apply(e)
    outs = lm.generate(prompts, args.max_new, batch=32)
    rows, gate = [], dict(at_answer_start=[], while_thinking_max=[], while_thinking_share_above_half=[])
    for q, p, toks in zip(test, prompts, outs):
        closed = CLOSE in toks
        ans = lm.decode(toks[toks.index(CLOSE) + 1:]) if closed else ""
        r = dict(id=q["id"], type=q["type"], question=q["question"], gold=q["gold"], closed_thinking=closed,
                 thinking_tokens=toks.index(CLOSE) if closed else len(toks), answer=ans, thinking=lm.tok.decode(toks[:toks.index(CLOSE)] if closed else toks)[:600],
                 **{k: (v and closed) for k, v in score_json(ans, q["gold"]).items()})
        rows.append(r)
        if eds and closed:
            # read-vector values of the answer-start edit along the model's own text
            _, hh = lm.run(p + toks)
            V = torch.stack([e["v"] for e in eds[0]]).to(hh.device)                      # [L, d_mlp]
            g = (hh * V[:, None, :]).sum(-1).mean(0)                                      # mean over layers, per position
            c = len(p) + toks.index(CLOSE)                                                # position of '</think>'
            start = c + 1 if c + 1 < len(g) else c                                        # the token after it predicts the answer
            think = g[len(p) - 1:c]
            gate["at_answer_start"].append(float(g[start]))
            gate["while_thinking_max"].append(float(think.max()))
            gate["while_thinking_share_above_half"].append(float((think > 0.5).float().mean()))
    with open(f"{args.out}/outputs_{name.replace(' ', '_').replace('(', '').replace(')', '')}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cl = [r for r in rows if r["closed_thinking"]]
    S[name] = dict(n=len(rows), closed_thinking=boot_ci([r["closed_thinking"] for r in rows]),
                   median_thinking_tokens=float(np.median([r["thinking_tokens"] for r in rows])),
                   valid=boot_ci([r["valid"] for r in rows]), key_answer=boot_ci([r["key_answer"] for r in rows]), correct=boot_ci([r["correct"] for r in rows]),
                   valid_among_closed=float(np.mean([r["valid"] for r in cl])) if cl else None,
                   key_answer_among_closed=float(np.mean([r["key_answer"] for r in cl])) if cl else None, n_closed=len(cl),
                   read_vector={k: (float(np.mean(v)) if v else None) for k, v in gate.items()})
    log(f"{name}: closed thinking {100 * S[name]['closed_thinking'][0]:.1f}%, valid JSON {100 * S[name]['valid'][0]:.1f}% "
        f"(among closed {100 * (S[name]['valid_among_closed'] or 0):.1f}%), key 'answer' {100 * S[name]['key_answer'][0]:.1f}%, "
        f"correct {100 * S[name]['correct'][0]:.1f}%, read vector {S[name]['read_vector']}")
    json.dump(S, open(f"{args.out}/summary.json", "w"), indent=1)
log("DONE")
