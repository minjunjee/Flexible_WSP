"""Build the question set shared by the experiments (PLAN.md, experiment 1 step 2).

Five question types, all from public data, all with a short gold answer:
  fact       country-capital, country-currency, landmark-country   (ericwtodd/function_vectors)
  extract    one person or one location named in a sentence        (tner/conll2003, test split)
  arith      2-digit addition, 2-digit subtraction, 3-digit addition (EleutherAI/arithmetic)
  translate  one English word into French, German or Spanish       (ericwtodd/function_vectors)
  sentiment  positive / negative for a short review                (ericwtodd/function_vectors)  -- test only

The first four types get 16 questions for building edits, 25 for tuning and 100 for testing; sentiment gets 100 for
testing only. Splits never share a question. Output: wsp/data/questions_v1.json

    python -m wsp.build_questions
"""
import json
import os
import random
import re
import urllib.request

from huggingface_hub import hf_hub_download

FV_COMMIT = "fb9eac7b6dc707ea1475a717379916007fe448d5"
FV = f"https://raw.githubusercontent.com/ericwtodd/function_vectors/{FV_COMMIT}/dataset_files/abstractive"
N_BUILD, N_TUNE, N_TEST = 16, 25, 100
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "questions_v1.json")
rng = random.Random(0)


def fv(name):
    with urllib.request.urlopen(f"{FV}/{name}.json") as r:
        return json.load(r)


def take(items, n):
    """n distinct items in a fixed random order"""
    items = sorted(items, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
    rng.shuffle(items)
    seen, out = set(), []
    for q, g, src in items:
        if q not in seen:
            seen.add(q)
            out.append((q, g, src))
        if len(out) == n:
            return out
    raise ValueError(f"only {len(out)} of {n}")


def mix(pools, n):
    """n items, taken in turn from each pool"""
    per = [take(p, (n + len(pools) - 1) // len(pools)) for p in pools]
    out = [x for group in zip(*per) for x in group]
    return out[:n]


N = N_BUILD + N_TUNE + N_TEST
types = {}

# ---------------------------------------------------------------- fact
plain = lambda s: re.fullmatch(r"[A-Za-z .'\-]+", s) is not None
cap = [(f"What is the capital of {d['input']}?", d["output"].lower(), "function_vectors/country-capital") for d in fv("country-capital")
       if plain(d["input"]) and plain(d["output"])]
cur = [(f"What is the currency of {d['input']}?", d["output"].lower(), "function_vectors/country-currency") for d in fv("country-currency")
       if plain(d["input"]) and plain(d["output"])]
lmk = [(f"In which country is {d['input']} located?", d["output"].lower(), "function_vectors/landmark-country") for d in fv("landmark-country")
       if plain(d["input"]) and plain(d["output"]) and d["output"].lower() not in d["input"].lower()]
types["fact"] = mix([cap, cur, lmk], N)

# ---------------------------------------------------------------- extract
label = json.load(open(hf_hub_download("tner/conll2003", "dataset/label.json", repo_type="dataset")))
inv = {v: k for k, v in label.items()}
per, loc = [], []
for line in open(hf_hub_download("tner/conll2003", "dataset/test.json", repo_type="dataset")):
    r = json.loads(line)
    toks, tags = r["tokens"], [inv[t] for t in r["tags"]]
    if not 8 <= len(toks) <= 24 or any("'" in t or '"' in t for t in toks) or not any(t.islower() for t in toks):
        continue
    ents, i = [], 0
    while i < len(toks):
        if tags[i].startswith("B-"):
            j = i + 1
            while j < len(toks) and tags[j] == "I-" + tags[i][2:]:
                j += 1
            ents.append((tags[i][2:], " ".join(toks[i:j])))
            i = j
        else:
            i += 1
    sent = re.sub(r" ([.,;:!?%)])", r"\1", " ".join(toks)).replace("( ", "(")
    for kind, pool, word in (("PER", per, "person's name"), ("LOC", loc, "location")):
        mine = [e for k, e in ents if k == kind]
        if len(mine) == 1 and re.fullmatch(r"[A-Za-z .\-]+", mine[0]) and len(mine[0]) >= 4:
            pool.append((f"Extract the {word} from this sentence: '{sent}'", mine[0].lower(), "tner/conll2003:test"))
types["extract"] = mix([per, loc], N)

# ---------------------------------------------------------------- arith
pools = []
for name in ("two_digit_addition", "two_digit_subtraction", "three_digit_addition"):
    pool = []
    for line in open(hf_hub_download("EleutherAI/arithmetic", f"data/{name}.jsonl", repo_type="dataset")):
        r = json.loads(line)
        q = re.search(r"What is [^?]*\?", r["context"]).group(0)
        a = r["completion"].strip()
        if re.fullmatch(r"\d+", a):                      # non-negative results only
            pool.append((q, a, f"EleutherAI/arithmetic:{name}"))
    pools.append(pool)
types["arith"] = mix(pools, N)

# ---------------------------------------------------------------- translate
pools = []
for lang in ("french", "german", "spanish"):
    pool = [(f"Translate the word '{d['input']}' into {lang.capitalize()}.", d["output"].lower(), f"function_vectors/english-{lang}")
            for d in fv(f"english-{lang}")
            if re.fullmatch(r"[a-z]{4,}", d["input"]) and re.fullmatch(r"[^\W\d_]{4,}", d["output"]) and d["output"].lower() != d["input"].lower()]
    pools.append(pool)
types["translate"] = mix(pools, N)

# ---------------------------------------------------------------- sentiment (test only)
sen = [(f"Is the sentiment of this review positive or negative? '{d['input']}'", d["output"].lower(), "function_vectors/sentiment")
       for d in fv("sentiment") if 20 <= len(d["input"]) <= 120 and "'" not in d["input"] and '"' not in d["input"]]
pos, neg = [x for x in sen if x[1] == "positive"], [x for x in sen if x[1] == "negative"]
types["sentiment"] = mix([pos, neg], N_TEST)

rows = []
for t, items in types.items():
    for i, (q, g, src) in enumerate(items):
        split = "test" if t == "sentiment" else ("build" if i < N_BUILD else "tune" if i < N_BUILD + N_TUNE else "test")
        rows.append(dict(id=f"{t}_{i:03d}", type=t, split=split, question=q, gold=g, source=src))
assert len({r["question"] for r in rows}) == len(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(dict(function_vectors_commit=FV_COMMIT, seed=0, questions=rows), open(OUT, "w"), ensure_ascii=False, indent=1)
for s in ("build", "tune", "test"):
    print(s, {t: sum(r["type"] == t and r["split"] == s for r in rows) for t in types})
for t in types:
    r = next(x for x in rows if x["type"] == t and x["split"] == "test")
    print(f"  {t}: {r['question']}  ->  {r['gold']}")
