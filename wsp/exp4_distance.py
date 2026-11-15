"""Experiment 4 (PLAN.md), parts 2 and 4, plus the side-effect count of experiment 3: what the edited model does far
from where it was built.

A. Ordinary text. The edited model reads Wikipedia text with no chat template. Counted: how often the read vectors of
   the edits are above 0.5 (they should stay near 0), and how far the edited model's next-token distribution is from
   the original model's (KL in nats, and the change in the loss on the text).
B. Distance from the instruction. An unrelated document of N tokens is placed in the user message before the question.
     - instruction first:  instruction, document, question      (no edit)
     - instruction last:   document, question, instruction      (no edit)
     - edits only:         document, question                   (edited weights, no instruction)
   The edits were built on short prompts without any document.
C. Undoing the edit: subtract the same outer products again and compare with the original weights.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python -m wsp.exp4_distance --out results/exp4/distance
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from wsp.edit import load_stacked
from wsp.lm import LM, load_questions
from wsp.onpolicy import gate_values
from wsp.score import boot_ci, score_json

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/exp4/distance")
ap.add_argument("--edits", default="results/exp1/stack_json_onpolicy/edits.npz")
ap.add_argument("--per_type", type=int, default=20)
ap.add_argument("--lengths", default="0,1000,4000,16000")
ap.add_argument("--text_tokens", type=int, default=100000)
ap.add_argument("--max_new", type=int, default=40)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
INSTR = 'Respond only with a JSON object of the form {"answer": "..."} and nothing else.'
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)
lm = LM("qwen35-0.8b")
rounds = load_stacked(args.edits)
R = dict(edits=args.edits, n_edits=len(rounds))

# ---------------------------------------------------------------- the text
tab = pq.read_table(hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")).to_pydict()
text = "".join(tab["text"])
ids = lm.tok(text, add_special_tokens=False).input_ids
log(f"wikitext-103 test: {len(ids):,} tokens")

# ---------------------------------------------------------------- A. ordinary text
CH = 1024
chunks = [ids[i:i + CH] for i in range(0, args.text_tokens, CH)]
base_lp = []
for c in chunks:
    base_lp.append(lm.logprobs(c).cpu())
for e in rounds:
    lm.apply(e)
n, above, big, kls, d_nll, peak = 0, np.zeros(len(rounds)), np.zeros(len(rounds)), [], [], np.zeros(len(rounds))
for c, lp0 in zip(chunks, base_lp):
    res, hh, logits = lm.run(c, logits=True)
    lp1 = torch.log_softmax(logits, -1).cpu()
    g = torch.stack([gate_values(e, hh) for e in rounds]).cpu().numpy()          # [edits, T]
    above += (g > 0.5).sum(1)
    big += (np.abs(g) > 0.1).sum(1)
    peak = np.maximum(peak, g.max(1))
    kls.append((lp0.exp() * (lp0 - lp1)).sum(-1).numpy())
    tgt = torch.tensor(c[1:])
    d_nll.append((lp0[:-1].gather(1, tgt[:, None]) - lp1[:-1].gather(1, tgt[:, None])).squeeze(1).numpy())
    n += len(c)
kls, d_nll = np.concatenate(kls), np.concatenate(d_nll)
R["ordinary_text"] = dict(tokens=n, read_vector_above_half=[int(x) for x in above], share_above_half=[float(x / n) for x in above],
                          share_abs_above_0p1=[float(x / n) for x in big], largest_value=[float(x) for x in peak],
                          mean_kl_original_to_edited=float(kls.mean()), share_kl_above_0p1=float((kls > 0.1).mean()), largest_kl=float(kls.max()),
                          mean_increase_of_loss_on_the_text=float(d_nll.mean()))
log("ordinary text:", R["ordinary_text"])

# ---------------------------------------------------------------- C. undoing the edit
changed = lm.changed_weights()
for e in reversed(rounds):
    lm.apply(e, scale=-1.0)
left = lm.changed_weights()
R["undo"] = dict(weights_changed_by_the_edits=changed, weights_still_different_after_subtracting=left,
                 note="each rank-1 change is rounded to bfloat16 when it is added and again when it is subtracted")
lm.restore()
log("undo:", R["undo"])

# ---------------------------------------------------------------- B. distance from the instruction
test, seen = [], {}
for q in load_questions():
    if q["split"] == "test" and seen.get(q["type"], 0) < args.per_type:
        seen[q["type"]] = seen.get(q["type"], 0) + 1
        test.append(q)
R["distance"] = {}
for N in [int(x) for x in args.lengths.split(",")]:
    docs = [lm.tok.decode(ids[200000 + i * 17001: 200000 + i * 17001 + N]) for i in range(len(test))] if N else [""] * len(test)
    sep = "\n\n" if N else ""
    P = {"instruction first": [lm.prompt_ids(INSTR + sep + d + sep + q["question"]) for d, q in zip(docs, test)] if N else [lm.prompt_ids(INSTR + " " + q["question"]) for q in test],
         "instruction last": [lm.prompt_ids(d + sep + q["question"], INSTR) for d, q in zip(docs, test)],
         "edits only": [lm.prompt_ids(d + sep + q["question"]) for d, q in zip(docs, test)],
         "no edit and no instruction": [lm.prompt_ids(d + sep + q["question"]) for d, q in zip(docs, test)]}
    R["distance"][str(N)] = {}
    for name, prompts in P.items():
        lm.restore()
        if name == "edits only":
            for e in rounds:
                lm.apply(e)
        outs = lm.generate(prompts, args.max_new, batch=8 if N <= 4000 else 2)
        rows = [dict(id=q["id"], type=q["type"], gold=q["gold"], text=lm.decode(t), **score_json(lm.decode(t), q["gold"])) for q, t in zip(test, outs)]
        gate = None
        if name == "edits only":
            # read vector of the first edit at the answer start, with the document in front
            vals = []
            for p in prompts:
                _, hh = lm.run(p)
                vals.append(float(gate_values(rounds[0], hh[:, -1:])[0]))
            gate = dict(mean=float(np.mean(vals)), share_above_half=float(np.mean([v > 0.5 for v in vals])))
        with open(f"{args.out}/outputs_{N}_{name.replace(' ', '_')}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        R["distance"][str(N)][name] = dict(n=len(rows), prompt_tokens=int(np.mean([len(p) for p in prompts])), valid=boot_ci([r["valid"] for r in rows]),
                                           key_answer=boot_ci([r["key_answer"] for r in rows]), correct=boot_ci([r["correct"] for r in rows]),
                                           read_vector_at_answer_start=gate)
        s = R["distance"][str(N)][name]
        log(f"document of {N} tokens, {name}: valid JSON {100 * s['valid'][0]:.0f}%, key 'answer' {100 * s['key_answer'][0]:.0f}%, correct {100 * s['correct'][0]:.0f}%"
            + (f", read vector at the answer start {gate}" if gate else ""))
        json.dump(R, open(f"{args.out}/summary.json", "w"), indent=1)
lm.restore()
log("DONE")
