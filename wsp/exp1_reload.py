"""Experiment 1, step 6 (PLAN.md): the edited model as an ordinary checkpoint.

    python -m wsp.exp1_reload save  --edits results/exp1/stack_json_onpolicy/edits.npz --ckpt <dir>
    python -m wsp.exp1_reload check --ckpt <dir> --reference results/exp1/stack_json_onpolicy/outputs_after_2_edits.jsonl --out results/exp1/reload

"save" loads the released model, adds the edits to down_proj.weight (rounded to bfloat16 after each one, in the order
they were stacked) and writes a safetensors checkpoint. "check" is meant for a new process: it uses nothing from this
project except the question file and the scorer, loads the checkpoint with from_pretrained and answers the test
questions with model.generate.
"""
import argparse
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["save", "check"])
ap.add_argument("--edits")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--reference")
ap.add_argument("--out")
ap.add_argument("--max_new", type=int, default=40)
args = ap.parse_args()
BASE = os.environ.get("WSP_DATA_ROOT", "/data/minjun_dev/Introduction_LLM/Flexible_WSP") + "/models/qwen35_08b"
from transformers import AutoModelForImageTextToText, AutoTokenizer

if args.mode == "save":
    from wsp.edit import load_stacked
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16)
    layers = model.model.language_model.layers
    n = 0
    for edits in load_stacked(args.edits):
        for e in edits:
            W = layers[e["layer"]].mlp.down_proj.weight
            before = W.data.clone()
            W.data.copy_((W.data.float() + torch.outer(e["u"], e["v"])).to(torch.bfloat16))
            n += int((W.data != before).sum())
    model.save_pretrained(args.ckpt, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(args.ckpt)
    print(f"saved {args.ckpt}; weight entries changed by the edits (counted per edit): {n:,}")
else:
    from wsp.score import boot_ci, score_json
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForImageTextToText.from_pretrained(args.ckpt, dtype=torch.float32).eval().cuda()
    ref = [json.loads(l) for l in open(args.reference)]
    rows = []
    for r in ref:
        text = tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        with torch.inference_mode():
            out = model.generate(input_ids=ids, max_new_tokens=args.max_new, do_sample=False)
        txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        rows.append(dict(id=r["id"], type=r["type"], question=r["question"], gold=r["gold"], text=txt, same_as_reference=txt == r["text"],
                         **score_json(txt, r["gold"])))
    with open(f"{args.out}/outputs_reloaded_checkpoint.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    S = dict(checkpoint=args.ckpt, reference=args.reference, n=len(rows), same_text_as_reference=int(sum(r["same_as_reference"] for r in rows)))
    for k in ("valid", "key_answer", "correct"):
        S[k] = boot_ci([r[k] for r in rows])
        S[k + "_reference"] = float(np.mean([r[k] for r in ref]))
    json.dump(S, open(f"{args.out}/summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))
