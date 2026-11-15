"""Verification of A-2: stronger baselines and paired bootstrap for the gain from component-level features."""
import json, sys, warnings
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/poc")
import qw
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(qw.MP)
OUT = "/home/claude/poc/outA"
meta = json.load(open(f"{OUT}/meta.json"))
CKPT = [128, 256, 384, 512, 768, 1024]
MATH = json.load(open(f"{qw.ROOT}/data/math500.json")); GSM = json.load(open(f"{qw.ROOT}/data/gsm8k_test.json"))
rng = np.random.default_rng(0)
vocab_flag = {}
def tok_flags(ids):
    out = np.zeros((len(ids), 3))
    for i, t in enumerate(ids):
        t = int(t)
        if t not in vocab_flag:
            s = tok.decode([t])
            vocab_flag[t] = (any(c.isdigit() for c in s), ("\\" in s) or ("$" in s), "\n" in s)
        out[i] = vocab_flag[t]
    return out.mean(0)
def feats(m, L):
    z = np.load(f"{OUT}/trace_{m['ji']:04d}.npz"); k = CKPT.index(L)
    inst = z["inst"][:L].sum(1).astype(float); newc = z["newc"][:L].sum(1).astype(float); h = L // 2; ids = z["ids"][:L]
    comp = [z["union_ck"][k].sum(), inst.mean(), (newc[h:] / np.maximum(inst[h:], 1)).mean(), z["usepr_ck"][k].sum(), z["pr"][:L].sum(1).mean(),
            z["ret_max"][64:L].mean(), (z["ret_max"][64:L] > 0.5).mean(), z["ret_cnt"][64:L, 1].mean()]
    actv = [z["specdim"][k, 0], z["specdim"][k, 1], z["res_max"][64:L].mean(), z["proj"][:L, 0].mean(), z["proj"][:L, 1].mean()]
    prob = (MATH if m["bench"] == "math500" else GSM)[m["id"]]
    fl = tok_flags(ids)
    text = [z["dup"][:L].mean(), qw.dup_mask(ids, 8).mean(), len(set(ids.tolist())) / L, len(tok(prob["problem"]).input_ids), fl[0], fl[1], fl[2]]
    return comp, actv, text, z["mean_res15"]
def oof(Xs, y, pca=None, reps=20):
    X = np.column_stack(Xs); P = np.zeros(len(y))
    for rep in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            Xtr, Xte = X[tr], X[te]
            if pca is not None:
                sc = StandardScaler().fit(pca[tr]); pc = PCA(pca_k, random_state=0).fit(sc.transform(pca[tr]))
                Xtr = np.column_stack([Xtr, pc.transform(sc.transform(pca[tr]))]); Xte = np.column_stack([Xte, pc.transform(sc.transform(pca[te]))])
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(Xtr, y[tr])
            P[te] += clf.predict_proba(Xte)[:, 1]
    return P / reps
def boot_diff(y, pa, pb, n=4000):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d); return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d <= 0))
pca_k = 10
RES = {}
for L in (768, 512):
    rows = [m for m in meta if m["set"] in ("p50c_math", "p50c_gsm") and m["T"] >= L]
    F = [feats(m, L) for m in rows]
    C, A, T, M = (np.array([f[i] for f in F], float) for i in range(4))
    bench = np.array([m["set"] == "p50c_math" for m in rows], float)[:, None]
    y_cut = np.array([int(m["cut"]) for m in rows]); y_cor = np.array([int(m["correct_first"] or 0) for m in rows])
    print("=" * 100); print(f"prefix length {L}: n={len(rows)}; hits limit {y_cut.mean():.2f}; correct {y_cor.mean():.2f}")
    for yname, y in (("hits token limit", y_cut), ("answer correct", y_cor)):
        P = {"text (7 features)": oof([bench, T], y), "activation scalars": oof([bench, A], y), "activation probe (PCA-10 of mean residual)": oof([bench], y, pca=M),
             "component": oof([bench, C], y), "text + activation scalars + probe": oof([bench, T, A], y, pca=M),
             "text + activation scalars + probe + component": oof([bench, T, A, C], y, pca=M),
             "component: recurrence only": oof([bench, C[:, 5:8]], y), "component: counts only": oof([bench, C[:, 0:5]], y)}
        print(f"  outcome: {yname}")
        for k, p in P.items():
            print(f"     {k:48s} AUC {roc_auc_score(y, p):.3f}")
        for a, b in (("component", "text (7 features)"), ("component", "activation scalars"), ("component", "activation probe (PCA-10 of mean residual)"),
                     ("text + activation scalars + probe + component", "text + activation scalars + probe")):
            mu, lo, hi, p = boot_diff(y, P[a], P[b])
            print(f"     gain [{a}] over [{b}]: {mu:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  share of resamples with no gain {p:.3f}")
            RES[f"{L}|{yname}|{a}|{b}"] = dict(gain=mu, lo=lo, hi=hi, p=p)
        RES[f"{L}|{yname}|auc"] = {k: float(roc_auc_score(y, p)) for k, p in P.items()}
    # difficulty
    mm = np.array([m["set"] == "p50c_math" for m in rows]); level = np.array([m["level"] or 0 for m in rows], float)
    subj = sorted({m["subject"] for m in rows if m["subject"]}); S = np.array([[float(m["subject"] == s) for s in subj] for m in rows])
    def cvr(Xs, reps=20):
        X = np.column_stack(Xs)[mm]; yv = level[mm]; out = np.zeros(len(yv))
        for rep in range(reps):
            for tr, te in KFold(5, shuffle=True, random_state=rep).split(X):
                out[te] += make_pipeline(StandardScaler(), RidgeCV(alphas=[1, 10, 100])).fit(X[tr], yv[tr]).predict(X[te])
        return out / reps
    preds = {"subject only": cvr([S]), "subject + text": cvr([S, T]), "subject + text + activation scalars": cvr([S, T, A]),
             "subject + text + activation scalars + component": cvr([S, T, A, C]), "component only": cvr([C])}
    print(f"  difficulty level (MATH-500, n={mm.sum()}): Spearman of cross-validated prediction")
    for k, p in preds.items():
        r = stats.spearmanr(p, level[mm]).statistic
        bs = []
        for _ in range(2000):
            i = rng.integers(0, mm.sum(), mm.sum()); bs.append(stats.spearmanr(p[i], level[mm][i]).statistic)
        print(f"     {k:50s} rho {r:.3f}  95% CI [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
        RES[f"{L}|level|{k}"] = dict(rho=float(r), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
json.dump(RES, open("/home/claude/poc/results_anaA_check.json", "w"))
