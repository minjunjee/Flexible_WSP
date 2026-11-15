"""Counts on the released reasoning traces (Weight-Space Programming logs, GSM8K, 200 first samples per run).

How often is the correct number written during thinking, compared with the final accuracy, and what would
happen if thinking stopped at the first wrap-up phrase? Used as background for the stop-thinking experiment.
"""
import collections
import json
import re
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qw
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(qw.MP)
NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def tonum(s):
    try:
        return float(s.replace(",", "").replace("$", ""))
    except ValueError:
        return None


PATS = {"final answer/output/result": r"final\s+(answer|output|result|response|check|verification)",
        "drafting": r"draft(ing)?\s+(the\s+)?(response|answer|final|solution|output)",
        "looks solid/correct": r"(looks|seems)\s+(solid|good|correct|right)",
        "the answer is": r"(the|final)\s+answer\s+(is|=|:)",
        "conclusion/formatting": r"conclusion|construct(ing)? the (final|output)|formatting"}
MARK = {k: re.compile(v, re.I) for k, v in PATS.items()}
res = {}
for run in ["unedited", "p50c"]:
    rows = qw.load_released(run, "gsm8k")
    n = len(rows)
    stats = collections.defaultdict(lambda: dict(found=0, last_number_correct=0))
    gold_appears, total_tok = 0, []
    for r in rows:
        t = r["text"]
        te = t.find("</think>")
        body = t if te < 0 else t[:te]
        g = tonum(str(r["answer"]))
        gold_appears += any(tonum(m.group()) == g for m in NUM.finditer(body))
        total_tok.append(len(tok(t, add_special_tokens=False).input_ids))
        for name, pat in MARK.items():
            m = pat.search(body)
            if not m:
                continue
            nums = list(NUM.finditer(body[:m.start()]))
            if not nums:
                continue
            stats[name]["found"] += 1
            stats[name]["last_number_correct"] += tonum(nums[-1].group()) == g
    res[run] = dict(n=n, final_correct=int(sum(r["correct_first"] for r in rows)), correct_number_written_in_thinking=gold_appears,
                    median_response_tokens=float(np.median(total_tok)), stop_at_first_phrase=dict(stats))
    print(run, json.dumps(res[run]))
json.dump(res, open(f"{qw.ROOT}/results_oracle_stop.json", "w"), indent=1)
