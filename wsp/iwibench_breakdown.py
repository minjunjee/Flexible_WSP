"""What an edited model did on each group of IWIBench items, read from the saved outputs.

The benchmark's reward leaves a control group out when the unedited model fails it. On Gemma 3 4B the unedited model
fails all eight prompts that ask for prose (its sentence is not the expected one), so the reward does not show whether
the edited model still honours a request for prose. This prints that count next to the scores.

    python -m wsp.iwibench_breakdown results/iwibench/gemma3-4b_prompts_written_without_seeing_the_code
"""
import glob
import json
import sys


def looks_like_json(text):
    t = text.strip()
    return t.startswith("{") or t.startswith("```")


def breakdown(folder):
    out = []
    for f in sorted(glob.glob(f"{folder}/outputs_seed*.jsonl") + glob.glob(f"{folder}/seed*/outputs_seed*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        grp = lambda name: [r for r in rows if r["id"].startswith(name)]
        sumf = f.rsplit("/", 1)[0] + "/summary.json"
        seed = int(f.rsplit("outputs_seed", 1)[1].split(".")[0])
        ep = next(e for e in json.load(open(sumf))["episodes"] if e["seed"] == seed)
        out.append(dict(file=f, seed=seed, edits=ep["edits"], targets=ep["edited"]["target_successes"], n_targets=ep["edited"]["target_count"],
                        retention=ep["edited"]["control_retention"], reward=ep["edited"]["reward"],
                        prose_requests_answered_with_json=sum(looks_like_json(r["edited"]) for r in grp("override")), n_prose_requests=len(grp("override")),
                        prose_requests_the_unedited_model_passes=ep["edited"]["by_group"]["override_controls"]["base_successes"],
                        general_answered_with_json=sum(looks_like_json(r["edited"]) for r in grp("general")), n_general=len(grp("general")),
                        general_kept=ep["edited"]["by_group"]["general_controls"]["retained"],
                        nearby_kept=ep["edited"]["by_group"]["nearby_controls"]["retained"], n_nearby=len(grp("nearby"))))
    return out


if __name__ == "__main__":
    for folder in sys.argv[1:]:
        print(folder)
        for b in breakdown(folder):
            print(f"  seed {b['seed']}: {b['edits']} edits, targets {b['targets']}/{b['n_targets']}, retention {b['retention']:.3f}, reward {b['reward']:.3f}; "
                  f"prose requests answered with JSON {b['prose_requests_answered_with_json']}/{b['n_prose_requests']} "
                  f"(the unedited model passes {b['prose_requests_the_unedited_model_passes']}); "
                  f"general questions answered with JSON {b['general_answered_with_json']}/{b['n_general']}, kept {b['general_kept']}; nearby kept {b['nearby_kept']}/{b['n_nearby']}")
