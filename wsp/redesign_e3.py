"""E3 from the redesign sheet: a trained baseline. LoRA on the MLP output matrices, for two instructions.

Training pairs: the 300 test questions that E1 does not use, each without the instruction as input and the
instructed model's answer as target. Rank 4 on every down_proj (the matrices the conditional edits change).
Two settings: the one in the sheet (lr 1e-4, 1 epoch, batch 8 = 38 steps) and a longer one that we added
(lr 1e-3, 5 epochs), because 38 steps at 1e-4 barely move the model.
Evaluation on the same 200 questions as E1, with the LoRA applied through a hook.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m wsp.redesign_e3
"""
import csv
import json
import math
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from wsp.instructions import TASKS
from wsp.lm import LM, load_questions
from wsp.score import is_correct

torch.manual_seed(0)
lm = LM("qwen35-0.8b")
for p in lm.model.parameters():
    p.requires_grad_(False)
test, rest, seen = [], [], {}
for q in load_questions():
    if q["split"] == "test":
        if seen.get(q["type"], 0) < 40:
            seen[q["type"]] = seen.get(q["type"], 0) + 1
            test.append(q)
        else:
            rest.append(q)
tab = pq.read_table(hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")).to_pydict()
ids = lm.tok("".join(tab["text"]), add_special_tokens=False).input_ids[:10240]
chunks = [ids[i:i + 1024] for i in range(0, 10240, 1024)]
base_lp = [lm.logprobs(c) for c in chunks]
END = lm.tok.convert_tokens_to_ids("<|im_end|>")
strip = lambda toks: [t for t in toks if t not in lm.end_ids]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


rows = []
for task in ("json_answer", "lowercase"):
    INSTR, check = TASKS[task]["text"], TASKS[task]["check"]
    refs = [strip(t) for t in lm.generate([lm.prompt_ids(q["question"], INSTR) for q in rest], 96)]
    data = [(lm.prompt_ids(q["question"]), y + [END]) for q, y in zip(rest, refs)]
    P0 = [lm.prompt_ids(q["question"]) for q in test]
    for setting, lr, epochs in (("as in the sheet: lr 1e-4, 1 epoch", 1e-4, 1), ("added by us: lr 1e-3, 5 epochs", 1e-3, 5)):
        A = [torch.nn.Parameter(torch.randn(4, lm.d_mlp, device=lm.device) / math.sqrt(lm.d_mlp)) for _ in range(lm.L)]
        B = [torch.nn.Parameter(torch.zeros(lm.d_model, 4, device=lm.device)) for _ in range(lm.L)]
        hooks = [m.register_forward_hook(lambda mod, inp, out, l=l: out + (inp[0] @ A[l].T) @ B[l].T) for l, m in enumerate(lm.mlp_out)]
        opt = torch.optim.AdamW(A + B, lr=lr, weight_decay=0.0)
        t0, steps, order = time.time(), 0, np.random.default_rng(0)
        for ep in range(epochs):
            perm = order.permutation(len(data))
            for s in range(0, len(perm), 8):
                batch = [data[i] for i in perm[s:s + 8]]
                T = max(len(p) + len(y) for p, y in batch)
                x = torch.full((len(batch), T), END, device=lm.device)
                lab = torch.full((len(batch), T), -100, device=lm.device)
                for b, (p, y) in enumerate(batch):
                    x[b, :len(p) + len(y)] = torch.tensor(p + y, device=lm.device)
                    lab[b, len(p):len(p) + len(y)] = torch.tensor(y, device=lm.device)
                with torch.enable_grad():
                    logits = lm.head(lm.body(input_ids=x).last_hidden_state).float()
                    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                steps += 1
        train_s = time.time() - t0
        outs = lm.generate(P0, 96, batch=32)
        res = []
        for q, toks in zip(test, outs):
            txt = lm.decode(toks)
            res.append(dict(task=task, setting=setting, id=q["id"], type=q["type"], question=q["question"], gold=q["gold"], text=txt,
                            follows=bool(check(txt)), correct=is_correct(txt, q["gold"])))
        kl = float(np.mean([float((b.exp() * (b - lm.logprobs(c))).sum(-1).mean()) for c, b in zip(chunks, base_lp)]))
        for h in hooks:
            h.remove()
        with open(f"results/E3_{task}_{'sheet' if epochs == 1 else 'longer'}.jsonl", "w") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        fo, co, bo = (wilson(sum(k(r) for r in res), len(res)) for k in (lambda r: r["follows"], lambda r: r["correct"], lambda r: r["follows"] and r["correct"]))
        rows.append(dict(task=task, setting=setting, rank=4, trained_parameters=sum(a.numel() + b.numel() for a, b in zip(A, B)), n_train=len(data), steps=steps,
                         final_loss=float(loss), train_seconds=round(train_s, 1), n_test=len(res), follows=fo[0], follows_lo=fo[1], follows_hi=fo[2],
                         correct=co[0], follows_and_correct=bo[0], mean_kl_to_original_on_10240_tokens=kl))
        print(rows[-1], flush=True)
with open("results/E3_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("DONE")
