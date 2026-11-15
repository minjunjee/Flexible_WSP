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

Two rules keep the edits from undoing each other. Both came from failures (results/exp5/first_attempt_lowercase: a
second edit aimed at the answer starts of other questions took the lowercase instruction from 72% to 8%).
  - A position where the read vector of an earlier edit already fires (above 0.5) is not targeted again.
  - After the read vector of a new edit is fitted, the other diverging positions where it fires anyway are added to
    the group, and the edit is rebuilt for the larger group. An edit cannot tell those positions apart, so its
    vector should be the mean over all of them.
"""
import contextlib

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage

from wsp import edit


def strip(lm, toks):
    return [t for t in toks if t not in lm.end_ids]


def aligned(lm, ids, answer, adds=(), original=False, gates=()):
    """log-probabilities [n, V] and residuals [L, n, d] at the answer-start position and at every answer position.
    With gates (a list of edits), also the value of each edit's read vectors there, averaged over its layers [edits, n]."""
    with (lm.original() if original else contextlib.nullcontext()):
        res, hh, logits = lm.run(ids + answer, adds=adds, logits=True)
    p, n = len(ids) - 1, len(answer) + 1
    out = (torch.log_softmax(logits[p:p + n], -1), res[:, p:p + n].clone())
    if gates:
        out += (torch.stack([gate_values(e, hh[:, p:p + n]) for e in gates]),)
    return out


def gate_values(edits, hh):
    """hh [L, n, d_mlp] -> [n]: read-vector value of one edit, averaged over the layers it changes"""
    return torch.stack([hh[e["layer"]] @ e["v"].to(hh.device) for e in edits]).mean(0)


def kl(lp1, lp0):
    return (lp1.exp() * (lp1 - lp0)).sum(-1)


def measure(lm, Q, max_new, applied=()):
    """Q: dicts with ids0 (prompt) and ids1 (prompt with the instruction). Rolls out the current model and compares it
    with the instructed model on every position of its own answer. applied: the edits already in the weights (a list
    of edits); a position where one of their read vectors is above 0.5 is marked as taken."""
    # an entry may carry "prefix": answer tokens that are forced before the model continues (see `explore` in stack)
    own = [q.get("prefix", []) + strip(lm, t) for q, t in zip(Q, lm.generate([q["ids0"] + q.get("prefix", []) for q in Q], max_new))]
    info = []
    for q, o in zip(Q, own):
        lp1, r1 = aligned(lm, q["ids1"], o, original=True)
        cur = aligned(lm, q["ids0"], o, gates=applied)
        lp0, r0 = cur[0], cur[1]
        taken = (cur[2] > 0.5).any(0) if applied else torch.zeros(len(o) + 1, dtype=torch.bool, device=lp0.device)
        info.append(dict(own=o, kl=kl(lp1, lp0), r1=r1, r0=r0, taken=taken, top_instr=lp1.argmax(-1).tolist(), top_cur=lp0.argmax(-1).tolist()))
    return info


def next_points(info, kl_min, Q=None, done=()):
    """for each question, the first diverging position where no earlier edit fires.
    done: questions whose own answer is already the text the instructed model writes. They are not targeted. What is
    left there is a difference that does not show in the answer (on Gemma 3 4B the instructed model puts a newline
    after the closing brace and the edited model ends the turn; an edit built for that newline has nothing to gain
    and, with a poorly fitted read vector, a model to lose).
    A question whose entry in Q has follow=False is one where the model should keep behaving as it does without any
    edit (its teacher is the unedited model). Such questions are never targeted: if the model diverges there, an
    earlier edit leaked, and the cure is not another edit aimed at the damage (on Gemma 3 4B that took one episode
    from a working two-edit model to 0 of 64)."""
    pts = []
    for qi, x in enumerate(info):
        if (Q is not None and not Q[qi].get("follow", True)) or qi in done:
            continue
        for j in range(len(x["kl"])):
            if float(x["kl"][j]) > kl_min and not bool(x["taken"][j]):
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


def build_edit(lm, Q, info, members, S, max_new, kl_min, candidates=(), lams=(1e-3, 1e-2, 1e-1), each=False):
    """One conditional edit for the member positions, widened once to the other candidate positions (first diverging
    positions of other questions) where its read vectors fire anyway. Returns (edits, fit statistics, final members)."""
    edits, stats = _build(lm, Q, info, members, S, max_new, kl_min, lams, each)
    extra = []
    for qi, j in candidates:
        if (qi, j) not in members:
            _, hh = lm.run(Q[qi]["ids0"] + info[qi]["own"])
            if float(gate_values(edits, hh[:, len(Q[qi]["ids0"]) - 1 + j][:, None])[0]) > 0.5:
                extra.append((qi, j))
    if extra:
        members = list(members) + extra
        edits, stats = _build(lm, Q, info, members, S, max_new, kl_min, lams, each)
    return edits, stats, members


def _build(lm, Q, info, members, S, max_new, kl_min, lams, each=False):
    """One conditional edit (a rank-1 change per layer of S) that fires at the member positions.

    The vector: mean hidden-state difference (instructed minus current) at the member positions, split over the layers.
    The read vectors are fitted on two kinds of sequences, all without the instruction:
      - for a member question, its own answer up to the member position followed by what the instructed model writes
        from there, with the vector switched on at the member position. This is the path the edited model should take.
      - for every question, its own answer as it is.
    Positions where the two models still disagree (other than the targets) are left out of the fit: the edit is not
    required to stay silent there. The exception is a question the model should answer as it does without any edit
    (follow=False): there every position is a place to stay silent, also the ones an earlier edit has damaged."""
    diffs = {qi: info[qi]["r1"][:, j] - info[qi]["r0"][:, j] for qi, j in members}
    d = torch.stack(list(diffs.values())).mean(0)
    u = edit.writes(d, S)
    # each=True: every member question keeps its OWN difference as the vector to add (edit.fit_maps), instead of the
    # mean over the members. Added on 2026-11-16 after the diagnosis in results/followup/tables.md: the mean vector
    # leaves about 1 nat of margin and a word in the record can outweigh it.
    u_each = {qi: edit.writes(dq, S) for qi, dq in diffs.items()} if each else None
    if lm.has_post_norm:
        # Gemma: the MLP output goes through a norm before it reaches the residual stream, so the shift u cannot be
        # added as it is. Convert it to the vector that has to be added before the norm (wsp/lm.py, to_pre_norm).
        st = [lm.mlp_stats(Q[qi]["ids0"] + info[qi]["own"], len(Q[qi]["ids0"]) - 1 + j) for qi, j in members]
        ybar, mnorm = torch.stack([a for a, _ in st]).mean(0), torch.stack([b for _, b in st]).mean(0)
        u = {l: lm.to_pre_norm(l, u[l], ybar[l], mnorm[l]) for l in S}
        if each:
            u_each = {qi: {l: lm.to_pre_norm(l, u_each[qi][l], a[l], b[l]) for l in S} for (qi, _), (a, b) in zip(members, st)}
    if each and each is not True:
        # each = k: keep, per layer, only the k leading directions of what differs between the questions (the mean plus
        # k principal directions). With one free vector per question the edit solved every target item on Qwen3.5-0.8B
        # but also changed the answers to general questions (results/followup). The weight change then has rank k + 1.
        order = list(u_each)
        for l in S:
            D = torch.stack([u_each[qi][l] for qi in order]).float()
            m = D.mean(0)
            P = torch.linalg.svd(D - m, full_matrices=False).Vh[:int(each)]
            Dk = m + ((D - m) @ P.T) @ P
            for i, qi in enumerate(order):
                u_each[qi][l] = Dk[i]
    with lm.original():
        conts = lm.generate([Q[qi]["ids1"] + info[qi]["own"][:j] for qi, j in members], max_new)
    at = {qi: (j, strip(lm, c)) for (qi, j), c in zip(members, conts)}
    seqs = []
    for qi, q in enumerate(Q):
        o, p0 = info[qi]["own"], len(q["ids0"]) - 1
        if qi in at:
            j, cont = at[qi]
            ans = o[:j] + cont
            adds = [(l, p0 + j, (u_each[qi][l] if each else u[l])) for l in S]
            k = kl(aligned(lm, q["ids1"], ans, original=True)[0], aligned(lm, q["ids0"], ans, adds=adds)[0])
            seqs.append(dict(q=qi, ids=q["ids0"] + ans, adds=adds, targets=[p0 + j],
                             ignore=[p0 + i for i in range(len(k)) if i != j and float(k[i]) > kl_min]))
        k = info[qi]["kl"]
        # where the model must stay as it is, every position counts as a place to stay silent, diverging or not
        skip = [p0 + i for i in range(len(k)) if float(k[i]) > kl_min] if q.get("follow", True) else []
        seqs.append(dict(q=qi, ids=q["ids0"] + o, targets=[p0 + at[qi][0]] if qi in at else [], ignore=skip))
    H, y, qidx = edit.collect(lm, seqs, S)
    V, stats = edit.fit_gates(H, y, qidx, lams)
    edits = [dict(layer=l, u=u[l].float().cpu(), v=V[l].cpu()) for l in S]
    if each:
        dW = edit.fit_maps(H, y, qidx, {l: {qi: u_each[qi][l] for qi in u_each} for l in S}, {s["layer"]: s["lam"] for s in stats})
        for e in edits:
            e["dW"] = dW[e["layer"]]
    return edits, stats


def profile(info, kl_min):
    """how much of the answers the two models disagree on"""
    n = sum(len(x["kl"]) for x in info)
    div = sum(int((x["kl"] > kl_min).sum()) for x in info)
    return dict(positions=n, diverging=div, share=div / n, questions_with_divergence=sum(bool((x["kl"] > kl_min).any()) for x in info),
                mean_kl_at_answer_start=float(np.mean([float(x["kl"][0]) for x in info])),
                mean_kl_elsewhere=float(np.mean([float(x["kl"][1:].mean()) for x in info if len(x["kl"]) > 1])))


def near_miss_variants(lm, Q, min_prob):
    """For every prompt, the second most likely first answer token of the current model, when its probability is at
    least min_prob. Returned as extra entries whose answer is forced to start with that token."""
    out = []
    for q in Q:
        if "prefix" in q:
            continue
        p = lm.logprobs(q["ids0"])[-1].exp()
        top = p.topk(2)
        t = int(top.indices[1])
        if float(top.values[1]) >= min_prob and t not in lm.end_ids:
            out.append(dict(ids0=q["ids0"], ids1=q["ids1"], prefix=[t], follow=q.get("follow", True)))
    return out


def stack(lm, Q, S, rounds, max_new, kl_min=1.0, min_group=0.1, log=print, explore=0.0, skip_done=True, each=False):
    """The whole loop: measure, pick the next group, build an edit, add it to the weights; at most `rounds` times.
    Returns (applied edits as a list of edits, one record per round).

    explore > 0: after the first edit, answers that start with the model's second most likely first token (when its
    probability is at least `explore`) are added to the answers that are followed. An edited model is less sure of its
    first token than the instructed model, and on new wordings it sometimes takes the runner-up (' {' or '{' for '{"');
    without this the later edits have never seen what follows such a start.

    skip_done: a question whose own answer already reads exactly as the instructed model's answer is not targeted
    (see next_points)."""
    applied, records = [], []
    Q = list(Q)
    n_base = len(Q)
    text = lambda toks: lm.decode(strip(lm, toks)).strip()
    teacher = {}
    if skip_done:
        with lm.original():
            for q, t in zip(Q, lm.generate([q["ids1"] for q in Q], max_new)):
                teacher[tuple(q["ids1"])] = text(t)
    for r in range(1, rounds + 1):
        if explore > 0 and r > 1:
            Q = Q[:n_base] + near_miss_variants(lm, Q[:n_base], explore)
            log(f"   round {r}: {len(Q) - n_base} answers with the runner-up first token added")
        info = measure(lm, Q, max_new, applied)
        done = {qi for qi, q in enumerate(Q) if q.get("follow", True) and teacher.get(tuple(q["ids1"])) == text(info[qi]["own"])}
        pts = next_points(info, kl_min, Q, done)
        groups = candidate_groups(lm, info, pts)
        n_follow = sum(q.get("follow", True) for q in Q[:n_base])
        rec = dict(round=r, own_answers=profile(info, kl_min), questions_with_an_untargeted_divergence=len(pts),
                   questions_already_answered_as_the_instructed_model_answers=len(done) if skip_done else None,
                   candidate_groups=[{k: v for k, v in g.items() if k != "members"} for g in groups[:6]])
        records.append(rec)
        if skip_done:
            log(f"   round {r}: {len(done)} of {n_follow} questions are already answered as the instructed model answers them")
        for g in groups[:3]:
            log(f"   round {r} group ({g['kind']}): {g['size']} positions, median position {g['median_position']}, mean KL {g['kl_mean']:.2f}, "
                f"shared fraction {g['shared_fraction']:.2f}, instructed wants {g['instructed_next_tokens'][:3]}, current writes {g['current_next_tokens'][:3]}")
        if not groups or groups[0]["size"] < min_group * n_follow:
            rec["stopped"] = "no group is large enough"
            break
        edits, stats, members = build_edit(lm, Q, info, groups[0]["members"], S, max_new, kl_min, pts, each=each)
        lm.apply(edits)
        applied.append(edits)
        ok = [s for s in stats if not np.isnan(s["margin"])]
        rec["chosen_group"] = {k: v for k, v in groups[0].items() if k != "members"}
        rec["positions_in_the_group"] = dict(chosen=groups[0]["size"], after_adding_positions_where_the_edit_fires_anyway=len(members))
        rec["fit"] = dict(target_mean=float(np.mean([s["target_mean"] for s in ok])), target_min=float(min(s["target_min"] for s in ok)),
                          other_max_abs=float(max(s["other_max_abs"] for s in ok))) if ok else None
        rec["changed_weights"] = lm.changed_weights()
        log(f"   round {r}: edit built for {len(members)} positions (group had {groups[0]['size']}); fit {rec['fit']}")
    return applied, records


def save_applied(path, applied):
    z = {}
    for r, edits in enumerate(applied, 1):
        for e in edits:
            z[f"round{r}_u_L{e['layer']}"] = np.asarray(e["u"], np.float32)
            z[f"round{r}_v_L{e['layer']}"] = np.asarray(e["v"], np.float32)
    np.savez_compressed(path, **z)
