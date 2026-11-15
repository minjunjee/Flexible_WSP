"""Follow-up diagnostic: is the strength of the key-name edit what limits an episode the edits were not developed on?

The read vector of the key-name edit fires on the failing items (wsp/followup_gate_diagnostic.py), and the model still
writes a word from the record as the key. Here the saved edits are applied again with ONE edit multiplied by a factor
and the episode is scored again. This uses the episode's own items to look at the factor, so it is a diagnostic of
the failure, not a way to choose the factor.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python -m wsp.followup_scale_key_edit \
        --run results/iwibench/gemma3-4b_episodes_first_run_after_freezing_the_procedure/seed0_test --seed 0 --split test --out results/followup/scale_key_edit_test_seed0.json
"""
import argparse
import json
import sys

from wsp.edit import load_stacked
from wsp.lm import DATA_ROOT, LM

sys.path.insert(0, f"{DATA_ROOT}/ext/iwibench/src")
from iwibench import scoring                                                        # noqa: E402
from iwibench.families import episode                                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--model", default="gemma3-4b")
ap.add_argument("--edit", type=int, default=2, help="which edit to scale, counted from 1 (2 is the one that writes the key name)")
ap.add_argument("--factors", default="1,1.5,2,3")
ap.add_argument("--out", required=True)
args = ap.parse_args()

lm = LM(args.model)
rounds = load_stacked(f"{args.run}/edits_seed{args.seed}.npz")
ep = episode("structured_response_default", args.seed, args.split, "google/gemma-3-4b-it")
saved = {r["id"]: r for r in map(json.loads, open(f"{args.run}/outputs_seed{args.seed}.jsonl"))}
items = [item for g in ("targets", "nearby_controls", "general_controls", "override_controls") for item in ep[g]]
base = {i["id"]: saved[i["id"]]["base"] for i in items}
prompts = [lm.prompt_ids(i["prompt"]) for i in items]
R = dict(run=args.run, seed=args.seed, split=args.split, scaled_edit=args.edit, n_edits=len(rounds), factors={})
for f in [float(x) for x in args.factors.split(",")]:
    lm.restore()
    for k, e in enumerate(rounds, 1):
        lm.apply(e, scale=f if k == args.edit else 1.0)
    outs = dict(zip([i["id"] for i in items], [lm.decode(t) for t in lm.generate(prompts, 256)]))
    g = {name: scoring._group(ep[name], base, outs) for name in ("targets", "nearby_controls", "general_controls", "override_controls")}
    den = sum(g[c]["retention_denominator"] for c in ("nearby_controls", "general_controls", "override_controls"))
    ret = sum(g[c]["retained_successes"] for c in ("nearby_controls", "general_controls", "override_controls")) / den
    tgt = g["targets"]
    json_on_general = sum(outs[i["id"]].strip().startswith(("{", "```")) for i in ep["general_controls"])
    json_on_prose = sum(outs[i["id"]].strip().startswith(("{", "```")) for i in ep["override_controls"])
    R["factors"][str(f)] = dict(targets=tgt["edited_successes"], n_targets=tgt["count"], retention=ret, reward=tgt["edited_success"] * ret,
                                nearby_kept=g["nearby_controls"]["retained_successes"], general_kept=g["general_controls"]["retained_successes"],
                                general_answered_with_json=json_on_general, prose_requests_answered_with_json=json_on_prose,
                                examples=[outs[i["id"]][:60] for i in ep["targets"][:3]])
    print(f"edit {args.edit} x {f}: targets {tgt['edited_successes']}/{tgt['count']}, retention {ret:.3f}, reward {tgt['edited_success'] * ret:.3f}, "
          f"nearby kept {g['nearby_controls']['retained_successes']}, general kept {g['general_controls']['retained_successes']}, "
          f"JSON on general {json_on_general}, JSON on prose requests {json_on_prose}; {R['factors'][str(f)]['examples'][:2]}", flush=True)
    json.dump(R, open(args.out, "w"), indent=1, ensure_ascii=False)
print("DONE")
