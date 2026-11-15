"""Find the positions where the model without the instruction parts ways with the instructed model, and stack
conditional edits there (PLAN.md, "갈라지는 자리를 찾아 편집을 쌓는 법").

Positions are aligned on the instructed model's own answer y. Position 0 is the answer-start position (it predicts
y[0]); position j >= 1 has y[j-1] as its input and predicts y[j]; position len(y) predicts the end of the answer.
"""
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage

from wsp import edit


def strip(lm, toks):
    return [t for t in toks if t not in lm.end_ids]


def instructed_cache(lm, questions, instruction, max_new):
    """Call with the ORIGINAL weights. Per question: prompt without instruction, the instructed answer, and the
    instructed model's log-probabilities and residuals at the aligned positions."""
    ids0 = [lm.prompt_ids(q) for q in questions]
    ids1 = [lm.prompt_ids(q, instruction) for q in questions]
    ys = [strip(lm, t) for t in lm.generate(ids1, max_new)]
    cache = []
    for a, b, y in zip(ids0, ids1, ys):
        res, _, logits = lm.run(b + y, logits=True)
        p1, n = len(b) - 1, len(y) + 1
        cache.append(dict(ids0=a, y=y, lp1=torch.log_softmax(logits[p1:p1 + n], -1), r1=res[:, p1:p1 + n].clone()))
    return cache


def divergence(lm, cache):
    """With the current weights and no instruction: KL(instructed || current) in nats at every aligned position,
    the current residuals there, and the two models' most likely next tokens."""
    out = []
    for c in cache:
        res, _, logits = lm.run(c["ids0"] + c["y"], logits=True)
        p0, n = len(c["ids0"]) - 1, len(c["y"]) + 1
        lp0 = torch.log_softmax(logits[p0:p0 + n], -1)
        kl = (c["lp1"].exp() * (c["lp1"] - lp0)).sum(-1)
        out.append(dict(kl=kl, r0=res[:, p0:p0 + n].clone(), top_instr=c["lp1"].argmax(-1).tolist(), top_plain=lp0.argmax(-1).tolist()))
    return out


def kl_by_position(cache, div, first=5):
    """mean KL at aligned positions 0..first-1, at the positions between, and at the end-of-answer position"""
    rows = {f"position {j}": [] for j in range(first)}
    rows["middle of the answer"], rows["end of the answer"] = [], []
    for c, dv in zip(cache, div):
        n = len(dv["kl"])
        for j in range(n):
            k = float(dv["kl"][j])
            if j == n - 1 and j >= 1:
                rows["end of the answer"].append(k)
            elif j < first:
                rows[f"position {j}"].append(k)
            else:
                rows["middle of the answer"].append(k)
    return {k: dict(mean_kl=float(np.mean(v)) if v else None, n=len(v)) for k, v in rows.items()}


def choose_cluster(lm, cache, div, kl_min=1.0, cos_dist=0.5):
    """Positions with KL above kl_min, grouped by the direction of the last-block hidden-state difference (average
    linkage, cosine distance). Returns the group with the largest summed KL, or None when no position diverges."""
    pts = [(qi, j) for qi, dv in enumerate(div) for j in range(len(dv["kl"])) if float(dv["kl"][j]) > kl_min]
    n_pos = sum(len(dv["kl"]) for dv in div)
    if not pts:
        return dict(n_positions=n_pos, n_diverging=0, clusters=[], members=[], diverging=[])
    X = torch.stack([cache[qi]["r1"][-1, j] - div[qi]["r0"][-1, j] for qi, j in pts])
    kls = np.array([float(div[qi]["kl"][j]) for qi, j in pts])
    lab = np.ones(len(pts), int) if len(pts) == 1 else fcluster(linkage(X.cpu().numpy(), "average", "cosine"), cos_dist, "distance")
    clusters = []
    for c in np.unique(lab):
        idx = np.flatnonzero(lab == c)
        js = [pts[i][1] for i in idx]
        at_end = sum(pts[i][1] == len(div[pts[i][0]]["kl"]) - 1 and pts[i][1] >= 1 for i in idx)
        want = [lm.tok.decode([div[pts[i][0]]["top_instr"][pts[i][1]]]) for i in idx[:200]]
        clusters.append(dict(size=len(idx), kl_sum=float(kls[idx].sum()), kl_mean=float(kls[idx].mean()),
                             questions=len({pts[i][0] for i in idx}), median_position=float(np.median(js)),
                             share_at_answer_start=float(np.mean([j == 0 for j in js])), share_at_answer_end=float(at_end / len(idx)),
                             shared_fraction=edit.shared_fraction(X[torch.as_tensor(idx, device=X.device)]),
                             instructed_next_tokens=sorted(([str(t), int(n)] for t, n in zip(*np.unique(want, return_counts=True))), key=lambda x: -x[1])[:8],
                             members=[pts[i] for i in idx]))
    clusters.sort(key=lambda c: -c["kl_sum"])
    return dict(n_positions=n_pos, n_diverging=len(pts), clusters=clusters, members=clusters[0]["members"], diverging=pts)


def build_edit(lm, cache, div, members, S, max_new, diverging=(), lams=(1e-3, 1e-2, 1e-1)):
    """One conditional edit (one rank-1 change per layer of S) that fires at the member positions.
    Diverging positions outside the chosen group are left out of the fit: they are not places where the edit must stay
    silent, they are places a later edit (or this one) still has to fix."""
    targets, skip = {}, {}
    for qi, j in members:
        targets.setdefault(qi, []).append(j)
    for qi, j in diverging:
        if j not in targets.get(qi, []):
            skip.setdefault(qi, []).append(j)
    d = torch.stack([cache[qi]["r1"][:, j] - div[qi]["r0"][:, j] for qi, j in members]).mean(0)       # [L, d_model]
    u = edit.writes(d, S)
    own = [strip(lm, t) for t in lm.generate([c["ids0"] for c in cache], max_new)]
    seqs = []
    for qi, c in enumerate(cache):
        p0, js, sk = len(c["ids0"]) - 1, targets.get(qi, []), skip.get(qi, [])
        # the instructed answer forced in, with the differences switched on at the target positions
        seqs.append(dict(q=qi, ids=c["ids0"] + c["y"], adds=[(l, p0 + j, u[l]) for j in js for l in S], targets=[p0 + j for j in js],
                         ignore=[p0 + j for j in sk]))
        # the current model's own answer. Position j there is the same position when the first j answer tokens agree.
        o = own[qi]
        same = lambda j: j <= len(o) and o[:j] == c["y"][:j]
        seqs.append(dict(q=qi, ids=c["ids0"] + o, targets=[p0 + j for j in js if same(j)], ignore=[p0 + j for j in sk if same(j)]))
    H, y, q = edit.collect(lm, seqs, S)
    V, stats = edit.fit_gates(H, y, q, lams)
    return [dict(layer=l, u=u[l].float().cpu(), v=V[l].cpu()) for l in S], stats
