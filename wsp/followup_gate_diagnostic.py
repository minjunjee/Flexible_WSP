"""Why do the edits fail on an episode they were not developed on? (follow-up of 2026-11-16)

For every target item of one finished IWIBench run, the edited model answers again (greedy, same edits) and the value
of each edit's read vector is recorded at every answer position. An item is then one of:
  - the read vector of the key-name edit stayed low at the position where the key is written (the edit did not fire), or
  - it fired, and the model still wrote another key (the vector that is added was not enough).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python -m wsp.followup_gate_diagnostic \
        --run results/iwibench/gemma3-4b_episodes_first_run_after_freezing_the_procedure/seed0_test --seed 0 --key subject
"""
import argparse
import json

import numpy as np
import torch

from wsp.edit import load_stacked
from wsp.lm import LM
from wsp.onpolicy import gate_values

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--key", required=True, help="the key the episode asks for")
ap.add_argument("--model", default="gemma3-4b")
ap.add_argument("--out", required=True)
args = ap.parse_args()

lm = LM(args.model)
rounds = load_stacked(f"{args.run}/edits_seed{args.seed}.npz")
for e in rounds:
    lm.apply(e)
rows = [json.loads(l) for l in open(f"{args.run}/outputs_seed{args.seed}.jsonl")]
out = []
for kind in ("target", "nearby", "general", "override"):
    items = [r for r in rows if r["id"].startswith(kind)]
    prompts = [lm.prompt_ids(r["prompt"]) for r in items]
    for r, p, toks in zip(items, prompts, lm.generate(prompts, 48)):
        _, hh = lm.run(p + toks)
        g = torch.stack([gate_values(e, hh[:, len(p) - 1:]) for e in rounds]).cpu().numpy()        # [edits, answer positions + 1]
        pieces = [lm.tok.decode([t]) for t in toks]
        out.append(dict(id=r["id"], kind=kind, success=r["edited_check"]["success"], text=lm.decode(toks), first_tokens=pieces[:6],
                        gates_first_positions=[[round(float(x), 3) for x in g[k][:6]] for k in range(len(rounds))]))
json.dump(out, open(args.out, "w"), indent=0, ensure_ascii=False)

t = [o for o in out if o["kind"] == "target"]
right = [o for o in t if o["text"].startswith('{"' + args.key + '"')]
wrong = [o for o in t if not o["text"].startswith('{"' + args.key + '"')]
print(f"targets: {len(right)} start with the asked key, {len(wrong)} do not")
for name, grp in (("asked key", right), ("another key", wrong)):
    if grp:
        for k in range(len(rounds)):
            m = np.array([o["gates_first_positions"][k] + [np.nan] * (6 - len(o["gates_first_positions"][k])) for o in grp])
            print(f"   {name}: edit {k + 1}, mean read-vector value at answer positions 0..5: {np.round(np.nanmean(m, 0), 2).tolist()}")
for name, kind in (("general questions", "general"), ("prose requests", "override")):
    grp = [o for o in out if o["kind"] == kind]
    for k in range(len(rounds)):
        m = np.array([o["gates_first_positions"][k] + [np.nan] * (6 - len(o["gates_first_positions"][k])) for o in grp])
        print(f"   {name}: edit {k + 1}, mean read-vector value at answer positions 0..5: {np.round(np.nanmean(m, 0), 2).tolist()}; share of items with position 0 above 0.5: {float(np.mean(m[:, 0] > 0.5)):.2f}")
print("examples of another key:", [(o["first_tokens"][:4], o["gates_first_positions"][1][:3]) for o in wrong[:4]])
print("DONE")
