"""Build the input files that the early scripts read but the original bundle did not ship.

  data/math500.json     {unique_id: {problem, answer, level, subject}}       from HuggingFaceH4/MATH-500 (test)
  data/gsm8k_test.json  {"gsm8k_test_<i>": {problem, answer, level, subject}} from openai/gsm8k (main, test)
  released_stats.json   {"<bench>_unedited": [{id, n_tok, closed, cut, onset}, ...]}

Problem ids and the GSM8K answer string follow nathanrs/weight-space-lp (src/wslp/eval/datasets.py), which matches the
ids in the released traces. released_stats.json is a reconstruction: the original file was not in the bundle, so the
definitions below are the ones the other scripts in this folder use.
  n_tok  : response length in tokens
  closed : "</think>" appears in the response
  cut    : n_tok >= 32000 (same test as the `cut` field that expA_record.py writes to its meta.json)
  onset  : qw.loop_onset(qw.dup_mask(ids)), None when no loop is found
Written on 2026-11-15 (KST) for the server run; not part of the original CPU run.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qw

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

os.makedirs(f"{qw.ROOT}/data", exist_ok=True)

# ---------------------------------------------------------------- problems
p = hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset")
MATH = {}
for line in open(p):
    r = json.loads(line)
    MATH[r["unique_id"]] = dict(problem=r["problem"], answer=r["answer"], level=int(r["level"]), subject=r["subject"])
json.dump(MATH, open(f"{qw.ROOT}/data/math500.json", "w"))

import pyarrow.parquet as pq

p = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
t = pq.read_table(p).to_pydict()
GSM = {}
for i, (q, a) in enumerate(zip(t["question"], t["answer"])):
    GSM[f"gsm8k_test_{i}"] = dict(problem=q, answer=a.split("####")[-1].strip().replace(",", ""), level=None, subject=None)
json.dump(GSM, open(f"{qw.ROOT}/data/gsm8k_test.json", "w"))
print("problems:", len(MATH), "MATH-500,", len(GSM), "GSM8K test", flush=True)

# ---------------------------------------------------------------- check against the released traces
for bench, P in (("math500", MATH), ("gsm8k", GSM)):
    for run in ("unedited", "p50c"):
        rows = qw.load_released(run, bench)
        missing = [r["id"] for r in rows if r["id"] not in P]
        differ = [r["id"] for r in rows if r["id"] in P and str(r["answer"]) != str(P[r["id"]]["answer"])]
        print(f"check {bench}/{run}: {len(rows)} traces, {len(missing)} ids missing, {len(differ)} answers differ", flush=True)
        assert not missing and not differ, (missing[:3], differ[:3])

# ---------------------------------------------------------------- loop statistics of the unedited traces
tok = AutoTokenizer.from_pretrained(qw.MP)
stats = {}
for bench in ("math500", "gsm8k"):
    out = []
    for r in qw.load_released("unedited", bench):
        ids = np.array(tok(r["text"], add_special_tokens=False).input_ids)
        onset = qw.loop_onset(qw.dup_mask(ids))
        out.append(dict(id=r["id"], n_tok=int(len(ids)), closed="</think>" in r["text"], cut=bool(len(ids) >= 32000), onset=onset))
    stats[f"{bench}_unedited"] = out
    sel = lambda s, cut: cut and s["onset"] is not None and 300 <= s["onset"] <= 1150
    by_len = {s["id"] for s in out if sel(s, s["cut"])}
    by_think = {s["id"] for s in out if sel(s, not s["closed"])}
    print(f"{bench}: {len(out)} traces, cut {sum(s['cut'] for s in out)}, think never closed {sum(not s['closed'] for s in out)}, "
          f"loop found {sum(s['onset'] is not None for s in out)}; "
          f"usable by expA_record (cut, onset in [300, 1150]) {len(by_len)}; "
          f"same rule with cut = think never closed {len(by_think)}; in one set only {len(by_len ^ by_think)}", flush=True)
json.dump(stats, open(f"{qw.ROOT}/released_stats.json", "w"))
print("wrote released_stats.json")
