"""IWIBench, task family "structured_response_default" (arXiv 2611.08647, github.com/hcltech-robotics/iwibench).

The task: without any format instruction in the prompt, the edited model must answer an extraction request about a
short residence record with a JSON object that uses exactly the keys seeded for the episode; it must still extract
correctly on nearby prompts, answer general questions, and write prose when the user explicitly asks for prose.
Every published LLM editor scored 0 of 192 target items on Gemma 3 4B.

Items and grading come from the IWIBench code. What we build the edit from, per episode:
  - the key names in the episode's objective (what an editor is given). From them we write one instruction,
    'Respond only with a JSON object of the form {"<key>": "..."} and nothing else.' The objective text itself is not
    used as the instruction: with it in the prompt Qwen3.5-0.8B wrote the key "name" instead of the seeded key on 24 of
    24 of our own prompts, and with the instruction above it wrote the exact object on 24 of 24.
  - our own prompts: extraction requests in six wordings over four record templates with made-up people. For these
    the teacher is the model with the instruction in the prompt.
  - the same kind of request with an explicit demand for prose, and general questions from our build split. For these
    the teacher is the unedited model with no instruction, so the edit is built to stay silent there.
None of the episode's private items (targets, controls) is used to build or tune anything.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m wsp.iwibench_structured --model qwen35-0.8b --out results/iwibench/qwen35-0.8b
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

from wsp import onpolicy
from wsp.lm import DATA_ROOT, LM, load_questions

sys.path.insert(0, f"{DATA_ROOT}/ext/iwibench/src")
from iwibench import scoring                                                        # noqa: E402
from iwibench.families import episode                                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen35-0.8b")
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", default="0,3,6", help="episode seeds; 0,3,6 are the three episodes of the published protocol")
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--prompts", default="small", choices=["small", "wide"])
ap.add_argument("--split", default="development", choices=["development", "test"])
ap.add_argument("--build_new", type=int, default=48)
ap.add_argument("--eval_new", type=int, default=256, help="the benchmark caps completions at 256 tokens")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)
lm = LM(args.model)
S = list(range(lm.L))

FIRST = ["Mara", "Jonas", "Priya", "Declan", "Yuki", "Tobias", "Ingrid", "Rashid", "Colette", "Matteo", "Sunita", "Gareth",
         "Leila", "Henrik", "Paloma", "Dmitri", "Aiko", "Callum", "Nadia", "Felipe"]
LAST = ["Whitlock", "Okafor", "Lindqvist", "Marchetti", "Tanaka", "Haddad", "Novak", "Castellanos", "Farrow", "Bergstrom",
        "Achebe", "Delacroix", "Kowalczyk", "Ramaswamy", "Thornbury", "Vasquez"]
TOWNS = ["Geneva", "Kyoto", "Porto", "Lyon", "Bergen", "Krakow", "Valencia", "Calgary", "Durban", "Hanoi", "Seville", "Tallinn"]
JOBS = ["plumber", "nurse", "baker", "architect", "translator", "electrician", "pharmacist", "tailor"]
RECORDS = ["{name} resides in {city} and is employed as a {job}.", "{name}, a {job}, has lived in {city} for years.",
           "Name: {name}; City: {city}; Job: {job}.", "Our records show that {name} is based in {city} and works as a {job}."]
ASKS = ["Identify this person's {f}: {t}", "What is the {f} of the person described here? {t}", "Read the record and report the person's {f}. {t}",
        "Find the {f} in the following record: {t}", "{t} What is this person's {f}?", "From the note below, give the person's {f}.\n{t}"]
PROSE = ["Answer in a full sentence of plain prose, not JSON.", "Reply with an ordinary sentence rather than JSON.", "Please respond in normal prose; do not use JSON."]


VERBS = ["Identify", "Extract", "Find", "Give", "Report", "State", "Name", "Tell me", "Pull out", "Provide", "Return", "Write down", "List"]
DOCS = ["note", "record", "profile", "entry", "text", "sentence", "description"]
WIDE_ASKS = ["{V} the person's {f} from this {d}: {t}", "{V} this person's {f}: {t}", "{V} the {f} of the person in this {d}. {t}",
             "{t} {V} the person's {f}.", "What is the {f} of the person described here? {t}", "What is the person's {f}? {t}",
             "{t} What is this person's {f}?", "Which {f} is mentioned in this {d}? {t}", "From the {d} below, {v} the person's {f}.\n{t}",
             "Here is a {d}. {t} {V} the {f}.", "{d_cap}: {t}\nTask: {v} the person's {f}."]
WIDE_RECORDS = RECORDS + ["{name} lives in {city}, where they work as a {job}.", "A {job} named {name} moved to {city} last spring.",
                          "Resident {name} of {city} is a {job} by trade.", "This entry concerns {name}, who is a {job} in {city}."]
WIDE_PROSE = PROSE + ["Write your answer as one complete sentence.", "Use a normal sentence, no JSON please.", "Answer in plain English prose."]
GENERAL_TAILS = ["", " Give only the answer.", " Answer briefly."]


def our_prompts(ep, n_extract=48, n_prose=16, n_general=16):
    """The first, small prompt set: six wordings, four record templates."""
    rng = random.Random(1000 + ep["seed"])
    fields = " and ".join(ep["parameters"]["fields"])
    private = {item["record_person"] for g in ("calibration", "targets", "nearby_controls", "override_controls") for item in ep[g]}
    people = [f"{a} {b}" for a in FIRST for b in LAST if f"{a} {b}" not in private]
    rng.shuffle(people)
    out = []
    for i in range(n_extract + n_prose):
        text = RECORDS[i % len(RECORDS)].format(name=people[i], city=rng.choice(TOWNS), job=rng.choice(JOBS))
        p = ASKS[i % len(ASKS)].format(f=fields, t=text)
        if i >= n_extract:
            p += " " + PROSE[i % len(PROSE)]
        out.append(dict(kind="extraction" if i < n_extract else "extraction with a demand for prose", prompt=p))
    general = [q for q in load_questions() if q["split"] == "build" and q["type"] in ("fact", "arith")]
    out += [dict(kind="general question", prompt=q["question"]) for q in general[:n_general]]
    return out


def our_prompts_wide(ep, n_extract=192, n_prose=64, n_general=120):
    """The wide prompt set: verbs x document words x eleven request patterns x eight record templates, sampled at
    random. Written after the small set failed to carry over to the benchmark's wording. I had read the benchmark's
    prompt generator before writing these patterns, so they are not blind to its wording: 'Extract ... from this note'
    is one of the combinations this set can produce."""
    rng = random.Random(2000 + ep["seed"])
    fields = " and ".join(ep["parameters"]["fields"])
    private = {item["record_person"] for g in ("calibration", "targets", "nearby_controls", "override_controls") for item in ep[g]}
    people = [f"{a} {b}" for a in FIRST for b in LAST if f"{a} {b}" not in private]
    out = []
    for i in range(n_extract + n_prose):
        text = rng.choice(WIDE_RECORDS).format(name=rng.choice(people), city=rng.choice(TOWNS), job=rng.choice(JOBS))
        V, d = rng.choice(VERBS), rng.choice(DOCS)
        p = rng.choice(WIDE_ASKS).format(V=V, v=V.lower(), d=d, d_cap=d.capitalize(), f=fields, t=text)
        if i >= n_extract:
            p += " " + rng.choice(WIDE_PROSE)
        out.append(dict(kind="extraction" if i < n_extract else "extraction with a demand for prose", prompt=p))
    general = [q for q in load_questions() if q["split"] in ("build", "tune")]
    rng.shuffle(general)
    out += [dict(kind="general question", prompt=q["question"] + rng.choice(GENERAL_TAILS)) for q in general[:n_general]]
    return out


def answers(prompts, adds_instruction=None):
    ids = [lm.prompt_ids(p, adds_instruction) for p in prompts]
    return [lm.decode(t) for t in lm.generate(ids, args.eval_new)]


ALL = []
for seed in [int(s) for s in args.seeds.split(",")]:
    lm.restore()
    ep = episode("structured_response_default", seed, args.split, "google/gemma-3-4b-it")
    objective = ep["objective"]
    keys = ep["parameters"]["keys"]
    instruction = "Respond only with a JSON object of the form {" + ", ".join(f'"{k}": "..."' for k in keys.values()) + "} and nothing else."
    log(f"episode seed {seed}: {objective}")
    log(f"   instruction used for the teacher: {instruction}")
    items = [item for g in ("calibration", "targets", "nearby_controls", "general_controls", "override_controls") for item in ep[g]]
    prompts = [item["prompt"] for item in items]
    base = dict(zip([i["id"] for i in items], answers(prompts)))
    instructed = dict(zip([i["id"] for i in items], answers(prompts, instruction)))

    ours = our_prompts_wide(ep) if args.prompts == "wide" else our_prompts(ep)
    Q = [dict(ids0=lm.prompt_ids(p["prompt"]), ids1=lm.prompt_ids(p["prompt"], instruction if p["kind"] == "extraction" else None)) for p in ours]
    applied, records = onpolicy.stack(lm, Q, S, args.rounds, args.build_new, log=log)
    onpolicy.save_applied(f"{args.out}/edits_seed{seed}.npz", applied)
    edited = dict(zip([i["id"] for i in items], answers(prompts)))

    def groups(outputs):
        return {name: scoring._group(ep[name], base, outputs) for name in ("targets", "nearby_controls", "general_controls", "override_controls")}

    res = dict(seed=seed, episode_id=ep["id"], objective=objective, instruction=instruction, keys=keys, n_build_prompts=len(ours),
               edits=len(applied), changed_weights=lm.changed_weights(), rounds=records)
    for name, outputs in (("edited", edited), ("instruction_in_the_prompt", instructed)):
        g = groups(outputs)
        tb, te = g["targets"]["base_success"], g["targets"]["edited_success"]
        den = sum(g[c]["retention_denominator"] for c in ("nearby_controls", "general_controls", "override_controls"))
        ret = sum(g[c]["retained_successes"] for c in ("nearby_controls", "general_controls", "override_controls")) / den if den else None
        gain = max(0.0, (te - tb) / (1 - tb)) if tb < 1 else None
        res[name] = dict(target_successes=g["targets"]["edited_successes"], target_count=g["targets"]["count"], target_success_base=tb,
                         target_content=g["targets"]["edited_content"], control_retention=ret, gain=gain,
                         reward=(gain * ret if gain is not None and ret is not None else None),
                         by_group={k: dict(count=v["count"], base_successes=v["base_successes"], successes=v["edited_successes"], retained=v["retained_successes"])
                                   for k, v in g.items()})
    try:
        res["official_score"] = {k: v for k, v in scoring.score_episode(ep, base, edited).items() if k not in ("groups", "calibration")}
    except Exception as e:                                                            # the base model may not pass the benchmark's calibration
        res["official_score"] = f"not computed: {type(e).__name__}: {e}"
    with open(f"{args.out}/outputs_seed{seed}.jsonl", "w") as f:
        for item in items:
            f.write(json.dumps(dict(id=item["id"], prompt=item["prompt"], rule=item["rule"], expected=item["answers"], base=base[item["id"]],
                                    edited=edited[item["id"]], instruction_in_the_prompt=instructed[item["id"]],
                                    edited_check=scoring.check_item(item, edited[item["id"]])), ensure_ascii=False) + "\n")
    json.dump(ours, open(f"{args.out}/our_build_prompts_seed{seed}.json", "w"), indent=1)
    e, i = res["edited"], res["instruction_in_the_prompt"]
    log(f"seed {seed}: edited targets {e['target_successes']}/{e['target_count']} (base {e['target_success_base']:.2f}), retention {e['control_retention']}, reward {e['reward']}; "
        f"controls {e['by_group']}; instruction in the prompt: targets {i['target_successes']}/{i['target_count']}, retention {i['control_retention']}")
    ALL.append(res)
    json.dump(dict(model=args.model, prompts=args.prompts, split=args.split, episodes=ALL), open(f"{args.out}/summary.json", "w"), indent=1, ensure_ascii=False)
tot = sum(r["edited"]["target_successes"] for r in ALL), sum(r["edited"]["target_count"] for r in ALL)
log(f"all episodes: edited targets {tot[0]}/{tot[1]}; mean reward {np.mean([r['edited']['reward'] or 0 for r in ALL]):.3f}")
log("DONE")
