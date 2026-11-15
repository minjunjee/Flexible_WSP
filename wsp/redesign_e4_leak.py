"""E4a leak check: how close are the prompt sets to the benchmark's test items?

Word 4-grams (lower-cased, punctuation dropped). For each prompt of a set, the share of its 4-grams that also occur
anywhere in the other set; then the mean over the prompts. Also the closest pairs by Jaccard similarity of 4-gram sets.

    python -m wsp.redesign_e4_leak
"""
import glob
import json
import re

import numpy as np


def grams(s, n=4):
    w = re.findall(r"[a-z0-9']+", s.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


new = [p["prompt"] for p in json.load(open("results/E4/E4a_prompts_written_without_seeing_the_benchmark_code.json"))]
seen = [p["prompt"] for p in json.load(open("results/iwibench/qwen35-0.8b_wide/our_build_prompts_seed0.json"))]
held = [p["prompt"] for p in json.load(open("results/iwibench/qwen35-0.8b_wide_heldout_wording/our_build_prompts_seed0.json"))]
small = [p["prompt"] for p in json.load(open("results/iwibench/qwen35-0.8b/our_build_prompts_seed0.json"))]
targets = []
for f in sorted(glob.glob("results/iwibench/qwen35-0.8b_wide/outputs_seed*.jsonl")):
    targets += [json.loads(l)["prompt"] for l in open(f) if json.loads(l)["id"].startswith("target")]
print(len(new), "prompts written without seeing the code;", len(seen), "written after reading the code;", len(targets), "benchmark target items")


def overlap(A, B):
    pool = set().union(*[grams(b) for b in B])
    return float(np.mean([len(grams(a) & pool) / max(len(grams(a)), 1) for a in A]))


def closest(A, B, k=5):
    gb = [grams(b) for b in B]
    out = []
    for a in A:
        ga = grams(a)
        j = [len(ga & g) / max(len(ga | g), 1) for g in gb]
        i = int(np.argmax(j))
        out.append((j[i], a, B[i]))
    return sorted(out, key=lambda x: -x[0])[:k]


R = dict(sets=dict(written_without_seeing_the_code=len(new), written_after_reading_the_code=len(seen), same_without_the_benchmark_words=len(held),
                   first_small_set=len(small), benchmark_target_items=len(targets)),
         share_of_4grams_also_in_the_benchmark_targets={
             "written without seeing the code": overlap(new, targets), "written after reading the code": overlap(seen, targets),
             "after reading the code, benchmark words left out": overlap(held, targets), "first small set (80 prompts)": overlap(small, targets)},
         share_of_4grams_of_the_new_prompts_also_in_the_code_seen_prompts=overlap(new, seen),
         closest_pairs_new_vs_targets=[dict(jaccard=j, new=a, target=b) for j, a, b in closest(new, targets)],
         closest_pairs_new_vs_code_seen=[dict(jaccard=j, new=a, code_seen=b) for j, a, b in closest(new, seen)],
         closest_pairs_code_seen_vs_targets=[dict(jaccard=j, code_seen=a, target=b) for j, a, b in closest(seen, targets)])
json.dump(R, open("results/E4/E4a_leak_check.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in R.items() if not k.startswith("closest")}, indent=1))
for k in ("closest_pairs_new_vs_targets", "closest_pairs_code_seen_vs_targets"):
    print(k)
    for p in R[k][:3]:
        print("  ", round(p["jaccard"], 2), "|", list(p.values())[1][:90].replace("\n", " / "), "||", list(p.values())[2][:90])
