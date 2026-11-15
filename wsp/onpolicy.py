"""Stack conditional edits along the model's OWN answers (second version of the stacking procedure).

The first version (wsp/stack.py, the procedure written in PLAN.md) aligns positions on the instructed model's answer.
On the JSON instruction it failed in two ways (results/exp1/stack_json_v1):
  - After the answer-start edit the model writes ' {"' (with a leading space) where the instructed model writes '{"'.
    The text is still valid JSON, but the next edit was built for the position after '{"' and does not fire after ' {"'.
  - A second edit aimed at a position that already has one changes the activations that the first edit's read vectors
    were fitted on. Valid JSON fell from 83% to 1%.

This version never needs the two models to write the same tokens:
  1. Let the current model answer the build questions without the instruction (its own answers).
  2. Feed each own answer to the instructed model (original weights, instruction in the prompt) and to the current
     model, and measure KL(instructed || current) at every position.
  3. For each question take the first position with KL above the threshold that no edit has been aimed at yet. This is
     where the two models part ways next; later differences may only be consequences of it.
  4. Group these positions, pick the group with the largest summed KL, and build one conditional edit for it.
     A position is never targeted twice.
"""
import contextlib

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage

from wsp import edit


def strip(lm, toks):
    return [t for t in toks if t not in lm.end_ids]


def aligned(lm, ids, answer, adds=(), original=False):
    """log-probabilities [n, V] and residuals [L, n, d] at the answer-start position and at every answer position"""
    with (lm.original() if original else contextlib.nullcontext()):
        res, _, logits = lm.run(ids + answer, adds=adds, logits=True)
    p, n = len(ids) - 1, len(answer) + 1
    return torch.log_softmax(logits[p:p + n], -1), res[:, p:p + n].clone()


def kl(lp1, lp0):
    return (lp1.exp() * (lp1 - lp0)).sum(-1)


def measure(lm, Q, max_new):
    """Q: dicts with ids0 (prompt) and ids1 (prompt with the instruction). Rolls out the current model and compares it
    with the instructed model on every position of its own answer."""
    own = [strip(lm, t) for t in lm.generate([q["ids0"] for q in Q], max_new)]
    info = []
    for q, o in zip(Q, own):
        lp1, r1 = aligned(lm, q["ids1"], o, original=True)
        lp0, r0 = aligned(lm, q["ids0"], o)
        info.append(dict(own=o, kl=kl(lp1, lp0), r1=r1, r0=r0, top_instr=lp1.argmax(-1).tolist(), top_cur=lp0.argmax(-1).tolist()))
    return info


def next_points(info, done, kl_min):
    """for each question, the first diverging position that has not been targeted (identified by the answer prefix)"""
    pts = []
    for qi, x in enumerate(info):
        for j in range(len(x["kl"])):
            if float(x["kl"][j]) > kl_min and (qi, tuple(x["own"][:j])) not in done:
                pts.append((qi, j))
                break
    return pts


def candidate_groups(lm, info, pts, cos_dist=0.5):
    """Two ways of grouping the positions: the instructed model wants the same next token, or the last-block
    hidden-state differences point the same way (average linkage, cosine distance). Sorted by summed KL."""
    if not pts:
        return []
    X = torch.stack([info[qi]["r1"][-1, j] - info[qi]["r0"][-1, j] for qi, j in pts])
    kls = np.array([float(info[qi]["kl"][j]) for qi, j in pts])
    want = np.array([info[qi]["top_instr"][j] for qi, j in pts])
    parts = [("same next token for the instructed model", [np.flatnonzero(want == t) for t in np.unique(want)])]
    if len(pts) > 1:
        lab = fcluster(linkage(X.cpu().numpy(), "average", "cosine"), cos_dist, "distance")
        parts.append(("similar direction of the hidden-state difference", [np.flatnonzero(lab == c) for c in np.unique(lab)]))
    out = []
    for kind, groups in parts:
        for idx in groups:
            toks, cnt = np.unique(want[idx], return_counts=True)
            hist = sorted(([lm.tok.decode([int(t)]), int(n)] for t, n in zip(toks, cnt)), key=lambda x: -x[1])[:6]
            have = [lm.tok.decode([info[pts[i][0]]["top_cur"][pts[i][1]]]) for i in idx]
            h2, c2 = np.unique(have, return_counts=True)
            out.append(dict(kind=kind, size=len(idx), kl_sum=float(kls[idx].sum()), kl_mean=float(kls[idx].mean()),
                            median_position=float(np.median([pts[i][1] for i in idx])),
                            shared_fraction=edit.shared_fraction(X[torch.as_tensor(idx, device=X.device)]),
                            instructed_next_tokens=hist,
                            current_next_tokens=sorted(([str(t), int(n)] for t, n in zip(h2, c2)), key=lambda x: -x[1])[:6],
                            members=[pts[i] for i in idx]))
    out.sort(key=lambda g: -g["kl_sum"])
    return out


def build_edit(lm, Q, info, members, S, max_new, kl_min, lams=(1e-3, 1e-2, 1e-1)):
    """One conditional edit (a rank-1 change per layer of S) that fires at the member positions.

    The vector: mean hidden-state difference (instructed minus current) at the member positions, split over the layers.
    The read vectors are fitted on two kinds of sequences, all without the instruction:
      - for a member question, its own answer up to the member position followed by what the instructed model writes
        from there, with the vector switched on at the member position. This is the path the edited model should take.
      - for every question, its own answer as it is.
    Positions where the two models still disagree (other than the targets) are left out of the fit: the edit is not
    required to stay silent there."""
    d = torch.stack([info[qi]["r1"][:, j] - info[qi]["r0"][:, j] for qi, j in members]).mean(0)
    u = edit.writes(d, S)
    with lm.original():
        conts = lm.generate([Q[qi]["ids1"] + info[qi]["own"][:j] for qi, j in members], max_new)
    at = {qi: (j, strip(lm, c)) for (qi, j), c in zip(members, conts)}
    seqs = []
    for qi, q in enumerate(Q):
        o, p0 = info[qi]["own"], len(q["ids0"]) - 1
        if qi in at:
            j, cont = at[qi]
            ans = o[:j] + cont
            adds = [(l, p0 + j, u[l]) for l in S]
            k = kl(aligned(lm, q["ids1"], ans, original=True)[0], aligned(lm, q["ids0"], ans, adds=adds)[0])
            seqs.append(dict(q=qi, ids=q["ids0"] + ans, adds=adds, targets=[p0 + j],
                             ignore=[p0 + i for i in range(len(k)) if i != j and float(k[i]) > kl_min]))
        k = info[qi]["kl"]
        seqs.append(dict(q=qi, ids=q["ids0"] + o, targets=[p0 + at[qi][0]] if qi in at else [],
                         ignore=[p0 + i for i in range(len(k)) if float(k[i]) > kl_min]))
    H, y, qidx = edit.collect(lm, seqs, S)
    V, stats = edit.fit_gates(H, y, qidx, lams)
    return [dict(layer=l, u=u[l].float().cpu(), v=V[l].cpu()) for l in S], stats


def profile(info, kl_min):
    """how much of the answers the two models disagree on"""
    n = sum(len(x["kl"]) for x in info)
    div = sum(int((x["kl"] > kl_min).sum()) for x in info)
    return dict(positions=n, diverging=div, share=div / n, questions_with_divergence=sum(bool((x["kl"] > kl_min).any()) for x in info),
                mean_kl_at_answer_start=float(np.mean([float(x["kl"][0]) for x in info])),
                mean_kl_elsewhere=float(np.mean([float(x["kl"][1:].mean()) for x in info if len(x["kl"]) > 1])))
