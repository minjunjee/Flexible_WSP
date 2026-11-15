"""Follow-up to E3: LoRA trained on the SAME questions the conditional edits are built from.

E3 trained LoRA on 300 question-answer pairs and the edits were built from 64 questions, so the two had different
amounts of data. Here the LoRA sees only those 64 build questions (input: the question without the instruction,
target: the instructed model's answer). Rank 4 on every down_proj, as in E3. Three settings:
  - lr 1e-4, 1 epoch   (8 steps: the sheet's learning rate and epoch count)
  - lr 1e-4, 5 epochs  (40 steps: about the number of steps of the sheet's setting on 300 pairs)
  - lr 1e-3, 5 epochs  (40 steps: the longer setting of E3)
Evaluated on the same 200 test questions as E1 and E3, with the LoRA applied through a hook.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m wsp.followup_lora_same_data
"""
import csv
import json
import math
import os

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from wsp.instructions import TASKS
from wsp.lm import LM, load_questions
from wsp.lora import LoRA
from wsp.score import is_correct

OUT = "results/followup"
os.makedirs(OUT, exist_ok=True)
lm = LM("qwen35-0.8b")
QS = load_questions()
build = [q for q in QS if q["split"] == "build"]
test, seen = [], {}
for q in QS:
    if q["split"] == "test" and seen.get(q["type"], 0) < 40:
        seen[q["type"]] = seen.get(q["type"], 0) + 1
        test.append(q)
tab = pq.read_table(hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")).to_pydict()
ids = lm.tok("".join(tab["text"]), add_special_tokens=False).input_ids[:10240]
chunks = [ids[i:i + 1024] for i in range(0, 10240, 1024)]
base_lp = [lm.logprobs(c) for c in chunks]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


rows = []
P0 = [lm.prompt_ids(q["question"]) for q in test]
for task in ("json_answer", "lowercase"):
    INSTR, check = TASKS[task]["text"], TASKS[task]["check"]
    data = [(lm.prompt_ids(q["question"]), y) for q, y in zip(build, lm.generate([lm.prompt_ids(q["question"], INSTR) for q in build], 96))]
    for name, lr, epochs in (("lr 1e-4, 1 epoch", 1e-4, 1), ("lr 1e-4, 5 epochs", 1e-4, 5), ("lr 1e-3, 5 epochs", 1e-3, 5)):
        lora = LoRA(lm, rank=4)
        st = lora.train(data, lr, epochs)
        res = []
        for q, toks in zip(test, lm.generate(P0, 96, batch=32)):
            txt = lm.decode(toks)
            res.append(dict(task=task, setting=name, id=q["id"], type=q["type"], question=q["question"], gold=q["gold"], text=txt,
                            follows=bool(check(txt)), correct=is_correct(txt, q["gold"])))
        kl = float(np.mean([float((b.exp() * (b - lm.logprobs(c))).sum(-1).mean()) for c, b in zip(chunks, base_lp)]))
        n_par = lora.n_parameters
        lora.remove()
        with open(f"{OUT}/lora_same_64_questions_{task}_{name.replace(' ', '_').replace(',', '')}.jsonl", "w") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        fo, co, bo = (wilson(sum(k(r) for r in res), len(res)) for k in (lambda r: r["follows"], lambda r: r["correct"], lambda r: r["follows"] and r["correct"]))
        rows.append(dict(task=task, setting=name, rank=4, trained_parameters=n_par, n_train=len(data), steps=st["steps"], final_loss=st["final_loss"],
                         train_seconds=st["seconds"], n_test=len(res), follows=fo[0], follows_lo=fo[1], follows_hi=fo[2], correct=co[0],
                         follows_and_correct=bo[0], mean_kl_to_original_on_10240_tokens=kl))
        print(rows[-1], flush=True)
        with open(f"{OUT}/lora_same_64_questions.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
print("DONE")
