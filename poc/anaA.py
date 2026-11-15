"""Analysis of Experiment A recordings. Prints tables and saves results/anaA.json."""
import json
import os
import sys
import warnings

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qw

OUT = f"{qw.ROOT}/outA"
meta = json.load(open(f"{OUT}/meta.json"))
CKPT = [128, 256, 384, 512, 768, 1024]
R = {}
rng = np.random.default_rng(0)


def load(m):
    return np.load(f"{OUT}/trace_{m['ji']:04d}.npz")


def auc(y, x):
    y, x = np.asarray(y), np.asarray(x, float)
    ok = ~np.isnan(x)
    if len(set(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok], x[ok])


def boot_auc(y, x, n=1000):
    y, x = np.asarray(y), np.asarray(x, float)
    a = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            a.append(roc_auc_score(y[i], x[i]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def strat_auc(xp, pp, xn, pn, width=100):
    """Position-stratified AUC: positives vs negatives compared only within the same position bin."""
    num = den = 0.0
    for b in np.unique(pp // width):
        a, c = xp[pp // width == b], xn[pn // width == b]
        if len(a) < 20 or len(c) < 20:
            continue
        u = stats.mannwhitneyu(a, c).statistic / (len(a) * len(c))
        w = min(len(a), len(c))
        num, den = num + w * u, den + w
    return num / den if den else np.nan


# =============================================================== A-0 length dominates raw counts
print("=" * 100)
print("A-0  How much of 'total complexity' (distinct active neurons) is just length?")
U = {s: [] for s in ["base", "p50c_math", "p50c_gsm"]}
for m in meta:
    z = load(m)
    U[m["set"]].append(z["union_ck"].sum(1))
for s, v in U.items():
    v = np.array(v, float)
    row = []
    for k, L in enumerate(CKPT):
        col = v[:, k][v[:, k] > 0]
        if len(col) > 5:
            row.append((L, col.mean(), col.std() / col.mean(), len(col)))
    print(f"  {s}: " + " | ".join(f"L={L}: mean {mu:,.0f} cv {cv:.3f} (n={n})" for L, mu, cv, n in row))
    if len(row) >= 3:
        Ls, mus = np.array([r[0] for r in row]), np.array([r[1] for r in row])
        beta = np.polyfit(np.log(Ls), np.log(mus), 1)[0]
        R[f"A0_{s}"] = dict(beta=float(beta), cv_at_768=float([r[2] for r in row if r[0] == 768][0]) if any(r[0] == 768 for r in row) else None,
                            rows=[(int(a), float(b), float(c), int(d)) for a, b, c, d in row])
        print(f"     growth exponent beta (U ~ L^beta) = {beta:.3f}")

# =============================================================== A-1 per-token signals around the published loop onset
print("=" * 100)
print("A-1  Base model: tokens after the published loop onset vs. tokens before it (position-matched)")
feat_names = ["novelty", "inst", "partic", "ret_max(component)", "res_max(residual)", "proj_u15", "proj_u19", "dup16", "dup8"]
POS, NEG = {f: [] for f in feat_names}, {f: [] for f in feat_names}
PP, PN = [], []
align = {f: [] for f in ["novelty", "ret_max(component)", "res_max(residual)", "proj_u15", "dup16", "inst"]}
W = 300
for m in [x for x in meta if x["set"] == "base"]:
    z = load(m)
    T, o = m["T"], m["onset"]
    inst = z["inst"].sum(1).astype(float)
    f = {"novelty": z["newc"].sum(1) / np.maximum(inst, 1), "inst": inst, "partic": z["pr"].sum(1),
         "ret_max(component)": z["ret_max"], "res_max(residual)": z["res_max"], "proj_u15": z["proj"][:, 0],
         "proj_u19": z["proj"][:, 1], "dup16": z["dup"].astype(float), "dup8": qw.dup_mask(z["ids"], 8).astype(float)}
    t = np.arange(T)
    pos = (t >= o) & (t < o + 320)
    neg = (t >= 32) & (t < o - 64)
    for k in feat_names:
        POS[k].append(f[k][pos])
        NEG[k].append(f[k][neg])
    PP.append(t[pos])
    PN.append(t[neg])
    if o >= W and T >= o + W:
        for k in align:
            align[k].append(f[k][o - W:o + W])
PP, PN = np.concatenate(PP), np.concatenate(PN)
R["A1"] = {}
print(f"  tokens: after-onset {len(PP):,}  before-onset {len(PN):,}")
print(f"  {'feature':24s} {'raw AUC':>8s} {'position-matched AUC':>22s}")
for k in feat_names:
    xp, xn = np.concatenate(POS[k]), np.concatenate(NEG[k])
    raw = roc_auc_score(np.r_[np.ones(len(xp)), np.zeros(len(xn))], np.r_[xp, xn])
    sa = strat_auc(xp, PP, xn, PN)
    R["A1"][k] = dict(raw=float(raw), matched=float(sa))
    print(f"  {k:24s} {raw:8.3f} {sa:22.3f}")
R["A1_align"] = {k: np.mean(v, 0).tolist() for k, v in align.items()}
R["A1_align_n"] = len(align["novelty"])

# =============================================================== A-2 early prediction from a fixed-length prefix (edited model)
print("=" * 100)
print("A-2  Edited model: do measures on the first 768 tokens predict what happens later?")


def prefix_features(m, L):
    z = load(m)
    k = CKPT.index(L)
    inst = z["inst"][:L].sum(1).astype(float)
    newc = z["newc"][:L].sum(1).astype(float)
    h = L // 2
    ids = z["ids"][:L]
    comp = dict(total=float(z["union_ck"][k].sum()), inst_mean=float(inst.mean()), novelty_late=float((newc[h:] / np.maximum(inst[h:], 1)).mean()),
                eff_components=float(z["usepr_ck"][k].sum()), partic_mean=float(z["pr"][:L].sum(1).mean()),
                ret_mean=float(z["ret_max"][64:L].mean()), ret_hi=float((z["ret_max"][64:L] > 0.5).mean()),
                ret_cnt=float(z["ret_cnt"][64:L, 1].mean()))
    actv = dict(specdim12=float(z["specdim"][k, 0]), specdim15=float(z["specdim"][k, 1]), res_rec=float(z["res_max"][64:L].mean()),
                proj15=float(z["proj"][:L, 0].mean()), proj19=float(z["proj"][:L, 1].mean()))
    text = dict(dup16=float(z["dup"][:L].mean()), dup8=float(qw.dup_mask(ids, 8).mean()), ttr=float(len(set(ids.tolist())) / L))
    return comp, actv, text


rows = []
for m in meta:
    if m["set"] in ("p50c_math", "p50c_gsm") and m["T"] >= 768:
        c, a, t = prefix_features(m, 768)
        rows.append(dict(m=m, comp=c, actv=a, text=t))
print(f"  traces: {len(rows)}  (math {sum(r['m']['set'] == 'p50c_math' for r in rows)}, gsm {sum(r['m']['set'] == 'p50c_gsm' for r in rows)})")
is_math = np.array([r["m"]["set"] == "p50c_math" for r in rows], float)
y_cut = np.array([int(r["m"]["cut"]) for r in rows])
y_cor = np.array([int(r["m"]["correct_first"] or 0) for r in rows])
n_full = np.array([r["m"]["n_full"] for r in rows], float)
level = np.array([r["m"]["level"] if r["m"]["level"] is not None else np.nan for r in rows], float)
print(f"  outcome rates: later hits the token limit {y_cut.mean():.2f} | answer correct {y_cor.mean():.2f}")
groups = {"component": list(rows[0]["comp"]), "activation": list(rows[0]["actv"]), "text": list(rows[0]["text"])}
X = {g: np.array([[r[{"component": "comp", "activation": "actv", "text": "text"}[g]][k] for k in ks] for r in rows]) for g, ks in groups.items()}
R["A2_single"] = {}
print(f"  {'group':10s} {'feature':16s} {'AUC limit':>10s} {'AUC correct':>12s} {'rho level(math)':>16s} {'rho length':>11s}")
mm = is_math == 1
for g, ks in groups.items():
    for j, k in enumerate(ks):
        x = X[g][:, j]
        # remove benchmark mean so that math/gsm offset does not create a spurious signal
        xr = x - np.where(mm, x[mm].mean(), x[~mm].mean())
        a1, a2 = auc(y_cut, xr), auc(y_cor, xr)
        rl = stats.spearmanr(x[mm & ~np.isnan(level)], level[mm & ~np.isnan(level)]).statistic
        rn = stats.spearmanr(xr, np.log(n_full)).statistic
        R["A2_single"][f"{g}:{k}"] = dict(auc_cut=float(a1), auc_correct=float(a2), rho_level=float(rl), rho_len=float(rn))
        print(f"  {g:10s} {k:16s} {a1:10.3f} {a2:12.3f} {rl:16.3f} {rn:11.3f}")


def cv_auc(Xs, y, reps=20):
    Xs = np.column_stack(Xs)
    out = []
    for rep in range(reps):
        skf = StratifiedKFold(5, shuffle=True, random_state=rep)
        p = np.zeros(len(y))
        for tr, te in skf.split(Xs, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
            clf.fit(Xs[tr], y[tr])
            p[te] = clf.predict_proba(Xs[te])[:, 1]
        out.append(roc_auc_score(y, p))
    return float(np.mean(out)), float(np.std(out))


bench = is_math[:, None]
R["A2_cv"] = {}
print("  Cross-validated AUC of feature groups (logistic regression, 5-fold x 20 repeats; benchmark indicator always included)")
print(f"  {'features':40s} {'hits token limit':>18s} {'answer correct':>16s}")
combos = {"benchmark indicator only": [bench], "text": [bench, X["text"]], "activation": [bench, X["activation"]],
          "component": [bench, X["component"]], "text + activation": [bench, X["text"], X["activation"]],
          "text + activation + component": [bench, X["text"], X["activation"], X["component"]]}
for name, xs in combos.items():
    c1, s1 = cv_auc(xs, y_cut)
    c2, s2 = cv_auc(xs, y_cor)
    R["A2_cv"][name] = dict(cut=c1, cut_sd=s1, correct=c2, correct_sd=s2)
    print(f"  {name:40s} {c1:10.3f} ±{s1:.3f} {c2:9.3f} ±{s2:.3f}")
# difficulty (math only)
lv_ok = mm & ~np.isnan(level)
print(f"  Difficulty level (MATH-500 only, n={lv_ok.sum()}): cross-validated Spearman of ridge prediction")
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold


def cv_rho(Xs, y, reps=20):
    Xs = np.column_stack(Xs)
    out = []
    for rep in range(reps):
        p = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(Xs):
            reg = make_pipeline(StandardScaler(), RidgeCV(alphas=[1, 10, 100]))
            reg.fit(Xs[tr], y[tr])
            p[te] = reg.predict(Xs[te])
        out.append(stats.spearmanr(p, y).statistic)
    return float(np.mean(out)), float(np.std(out))


R["A2_level"] = {}
for name, xs in {"text": [X["text"]], "activation": [X["activation"]], "component": [X["component"]],
                 "text + activation": [X["text"], X["activation"]],
                 "text + activation + component": [X["text"], X["activation"], X["component"]]}.items():
    r, s = cv_rho([x[lv_ok] for x in xs], level[lv_ok])
    R["A2_level"][name] = dict(rho=r, sd=s)
    print(f"    {name:36s} rho = {r:.3f} ±{s:.3f}")

# =============================================================== A-3 tokens before vs after the answer first appears
print("=" * 100)
print("A-3  Edited model on GSM8K: tokens written before vs. after the correct answer first appears (position-matched)")
f3 = ["novelty", "inst", "partic", "ret_max(component)", "res_max(residual)", "dup16"]
POS, NEG = {f: [] for f in f3}, {f: [] for f in f3}
PP, PN = [], []
n_use = 0
reach_list = []
for m in [x for x in meta if x["set"] == "p50c_gsm"]:
    if m["reach"] is None:
        continue
    reach_list.append(m["reach"])
    z = load(m)
    T = m["T"]
    inst = z["inst"].sum(1).astype(float)
    f = {"novelty": z["newc"].sum(1) / np.maximum(inst, 1), "inst": inst, "partic": z["pr"].sum(1),
         "ret_max(component)": z["ret_max"], "res_max(residual)": z["res_max"], "dup16": z["dup"].astype(float)}
    t = np.arange(T)
    r = m["reach"] if m["reach"] >= 0 else 10 ** 9
    pos = (t > r + 16) & (t >= 64)
    neg = (t < r) & (t >= 64)
    if m["correct_first"] != 1:
        continue
    n_use += 1
    for k in f3:
        POS[k].append(f[k][pos])
        NEG[k].append(f[k][neg])
    PP.append(t[pos])
    PN.append(t[neg])
reach_arr = np.array(reach_list)
print(f"  traces with a usable gold number: {len(reach_arr)}; answer appears within the first 1,152 tokens in {np.mean(reach_arr >= 0):.2f}; "
      f"median position {np.median(reach_arr[reach_arr >= 0]) if (reach_arr >= 0).any() else None}")
PP, PN = np.concatenate(PP), np.concatenate(PN)
print(f"  correct traces used: {n_use}; tokens after {len(PP):,} / before {len(PN):,}")
R["A3"] = {"n_traces": n_use, "n_after": int(len(PP)), "n_before": int(len(PN)), "feat": {}}
print(f"  {'feature':24s} {'raw AUC':>8s} {'position-matched AUC':>22s}   (AUC > 0.5: larger after the answer)")
for k in f3:
    xp, xn = np.concatenate(POS[k]), np.concatenate(NEG[k])
    raw = roc_auc_score(np.r_[np.ones(len(xp)), np.zeros(len(xn))], np.r_[xp, xn])
    sa = strat_auc(xp, PP, xn, PN)
    R["A3"]["feat"][k] = dict(raw=float(raw), matched=float(sa))
    print(f"  {k:24s} {raw:8.3f} {sa:22.3f}")
json.dump(R, open(f"{qw.ROOT}/results_anaA.json", "w"))
print("saved results_anaA.json")
