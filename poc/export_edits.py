"""Export the edit conditions of this proof of concept so they can be evaluated at full length on a GPU.

    python export_edits.py                 # writes edits_poc.npz (all conditions, small)
    python export_edits.py <condition> <out.safetensors>   # writes a full edited checkpoint for vLLM

Conditions: astra_edit, remove_direction, gc_128, gc_1024, random_1024, recipe_own_15_19, recipe_own_21_23
(lp_push is a steering vector, not a weight edit; its vectors are in nathanrs/weight-space-lp/steer/realshift_k8.npz)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qw
from conds import all_conditions


def edited_tensors(model, ops):
    lm = qw.lm_of(model)
    out = {}
    for op in ops:
        l = op[1]
        name = qw.down_proj_name(l)
        W = out.get(name, lm.layers[l].mlp.down_proj.weight.detach().float().clone())
        if op[0] == "zero":
            W[:, op[2]] = 0
        elif op[0] == "delta":
            W[:, op[2]] += op[3]
        elif op[0] == "rank1":
            W += torch.outer(op[2], op[3])
        else:
            raise ValueError("steering vectors are not weight edits")
        out[name] = W
    return {k: v.to(torch.bfloat16) for k, v in out.items()}


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    tok, model = qw.load_model()
    conds, sets, S = all_conditions(model, [21, 23])
    if len(sys.argv) == 3:
        from safetensors.torch import load_file, save_file
        full = load_file(f"{qw.MP}/model.safetensors-00001-of-00001.safetensors")
        full.update(edited_tensors(model, conds[sys.argv[1]]))
        save_file(full, sys.argv[2], metadata={"format": "pt"})
        print("wrote", sys.argv[2])
    else:
        z = {}
        for name, ops in conds.items():
            for op in ops:
                if op[0] == "zero":
                    z[f"{name}|L{op[1]}|zero_columns"] = np.asarray(op[2])
                elif op[0] == "delta":
                    z[f"{name}|L{op[1]}|columns"] = np.asarray(op[2])
                    z[f"{name}|L{op[1]}|delta"] = op[3].numpy()
                elif op[0] == "rank1":
                    z[f"{name}|L{op[1]}|u"] = op[2].numpy()
                    z[f"{name}|L{op[1]}|v"] = op[3].numpy()
                elif op[0] == "push":
                    z[f"{name}|L{op[1]}|push"] = op[2].numpy()
        np.savez_compressed(f"{qw.ROOT}/edits_poc.npz", **z)
        print("wrote edits_poc.npz with", len(z), "arrays")
