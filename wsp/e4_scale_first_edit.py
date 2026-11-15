"""Outside the redesign sheet: can the near-miss first token on unseen wordings be fixed by a stronger first edit?

With the prompts written without seeing the benchmark code, 29 of 64 target answers started with '{' instead of '{"'
and then got the key wrong. Here the answer-start edit of that run is scaled by s; the other edits are left as built.
s is chosen on prompts that are neither the build prompts nor benchmark items (our code-seen prompt set): the share of
extraction prompts where the edited model's answer equals the instructed model's answer. The chosen s is then
evaluated once on the benchmark episode.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m wsp.e4_scale_first_edit
"""
import json
import sys

from wsp.edit import load_stacked
from wsp.lm import DATA_ROOT, LM

sys.path.insert(0, f"{DATA_ROOT}/ext/iwibench/src")
from iwibench import scoring                                                        # noqa: E402
from iwibench.families import episode                                               # noqa: E402

lm = LM("qwen35-0.8b")
OUT = {}
for seed in (0,):
    ep = episode("structured_response_default", seed, "development", "google/gemma-3-4b-it")
    key = list(ep["parameters"]["keys"].values())[0]
    instr = 'Respond only with a JSON object of the form {"' + key + '": "..."} and nothing else.'
    rounds = load_stacked(f"results/E4/E4b_qwen35-0.8b/edits_seed{seed}.npz")
    val = [p for p in json.load(open(f"results/iwibench/qwen35-0.8b_wide/our_build_prompts_seed{seed}.json"))]
    vx = [p["prompt"] for p in val if p["kind"] == "extraction"][:96]
    vo = [p["prompt"] for p in val if p["kind"] != "extraction"][:96]
    gen = lambda prompts, ins=None: [lm.decode(t) for t in lm.generate([lm.prompt_ids(p, ins) for p in prompts], 64)]
    ref_x, ref_o = gen(vx, instr), gen(vo)
    items = [i for g in ("targets", "nearby_controls", "general_controls", "override_controls") for i in ep[g]]
    base = dict(zip([i["id"] for i in items], gen([i["prompt"] for i in items])))
    res = {}
    for s in (1.0, 1.25, 1.5, 2.0, 3.0):
        lm.restore()
        lm.apply(rounds[0], scale=s)
        for e in rounds[1:]:
            lm.apply(e)
        ax, ao = gen(vx), gen(vo)
        res[s] = dict(validation_extraction_same_as_instructed=sum(a == b for a, b in zip(ax, ref_x)) / len(vx),
                      validation_other_same_as_unedited=sum(a == b for a, b in zip(ao, ref_o)) / len(vo))
        print(seed, s, res[s], flush=True)
    best = max(res, key=lambda s: (res[s]["validation_extraction_same_as_instructed"] + res[s]["validation_other_same_as_unedited"], -s))
    lm.restore()
    lm.apply(rounds[0], scale=best)
    for e in rounds[1:]:
        lm.apply(e)
    edited = dict(zip([i["id"] for i in items], gen([i["prompt"] for i in items])))
    g = {name: scoring._group(ep[name], base, edited) for name in ("targets", "nearby_controls", "general_controls", "override_controls")}
    den = sum(g[c]["retention_denominator"] for c in ("nearby_controls", "general_controls", "override_controls"))
    ret = sum(g[c]["retained_successes"] for c in ("nearby_controls", "general_controls", "override_controls")) / den
    OUT[seed] = dict(scale_grid=res, chosen_scale=best, targets=g["targets"]["edited_successes"], target_count=g["targets"]["count"], control_retention=ret,
                     reward=g["targets"]["edited_success"] * ret)
    print("seed", seed, "chosen scale", best, "targets", OUT[seed]["targets"], "/", OUT[seed]["target_count"], "retention", round(ret, 3), "reward", round(OUT[seed]["reward"], 3), flush=True)
    json.dump(OUT, open("results/E4/extra_scaled_first_edit.json", "w"), indent=1)
print("DONE")
