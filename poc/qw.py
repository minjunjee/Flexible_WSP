"""Shared helpers for the Qwen3.5-0.8B proof-of-concept experiments (CPU, bf16).

Loop labeling follows nathanrs/weight-space-lp (MIT license): a token is a duplicate if the 16-token window ending
at it already appeared earlier in the response; a loop starts at the first duplicate after which >= 80% of the
remaining tokens are duplicates.
"""
import glob
import gzip
import json
import os
import re

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
MP = f"{ROOT}/models/qwen35_08b"
LOGS = f"{ROOT}/ext/lm_understand-logs/exp8_math"
SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
N_LAYERS, D_MODEL, D_MLP = 24, 1024, 3584

torch.set_num_threads(int(os.environ.get("POC_THREADS", "2")))


def load_model():
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MP)
    model = AutoModelForImageTextToText.from_pretrained(MP, dtype=torch.bfloat16).eval()
    model.model.visual = None
    return tok, model


def lm_of(model):
    return model.model.language_model


def build_prompt(tok, problem, thinking=True, raw=False):
    content = problem if raw else problem + "\n" + SUFFIX
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


# ---------------------------------------------------------------- loop labels
K_DUP, FRAC, GAP = 16, 0.8, 256


def dup_mask(ids, k=K_DUP):
    ids = np.asarray(ids, dtype=np.int64)
    seen, out = set(), np.zeros(len(ids), bool)
    for t in range(k - 1, len(ids)):
        w = ids[t - k + 1:t + 1].tobytes()
        out[t] = w in seen
        seen.add(w)
    return out


def loop_onset(dup, frac=FRAC, min_tail=512):
    n = len(dup)
    if n < min_tail:
        return None
    suffix = np.cumsum(dup[::-1])[::-1] / np.arange(n, 0, -1)
    ok = np.flatnonzero((suffix >= frac) & dup & (np.arange(n) <= n - min_tail))
    return int(ok[0]) if len(ok) else None


def online_loop_onset(dup, run=96):
    """Onset usable on short generations: first index that starts `run` consecutive duplicate tokens."""
    n, c = len(dup), 0
    for t in range(n):
        c = c + 1 if dup[t] else 0
        if c >= run:
            return t - run + 1
    return None


# ---------------------------------------------------------------- released traces
def load_released(run, bench="math500"):
    """run in {unedited, p50c, p10c, p25c}. Returns list of dicts(id, answer, text, correct_first)."""
    rows, corr = [], {}
    for part in sorted(glob.glob(f"{LOGS}/eval/{run}/parts/{bench}_s*")):
        pp = glob.glob(f"{part}/{bench}_per_problem.json")
        if pp:
            for r in json.load(open(pp[0])):
                corr[r["id"]] = r["correct"]
        fs = glob.glob(f"{part}/{bench}_first_samples.jsonl.gz")
        if fs:
            with gzip.open(fs[0], "rt") as fh:
                for line in fh:
                    rows.append(json.loads(line))
    for r in rows:
        c = corr.get(r["id"])
        r["correct_all"] = c
        r["correct_first"] = (c[0] if c else None)
    return rows


# ---------------------------------------------------------------- answers
def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def to_number(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\d?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", lambda m: str(float(m.group(1)) / float(m.group(2))), s)
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def numeric_correct(text, gold):
    """Grade the last \\boxed{} answer after </think> (or anywhere if thinking never closed) against a number."""
    pred = to_number(last_boxed(text))
    g = to_number(str(gold))
    return pred is not None and g is not None and abs(pred - g) < 1e-6 * max(1.0, abs(g))


# ---------------------------------------------------------------- published edits
def astra_edit():
    """Exact p50c edit as {layer: (cols[int], delta[1024, ncols] float32)} with delta = W_edited - W_base (bf16)."""
    z = np.load(f"{LOGS}/runs/p50c/edit/edit.npz")
    return z


def down_proj_name(l):
    return f"model.language_model.layers.{l}.mlp.down_proj.weight"


# ---------------------------------------------------------------- weights / edits
def bf16_bits(t):
    return t.detach().contiguous().view(torch.int16).numpy().view(np.uint16)


def astra_deltas(model, verify=True):
    """Return {layer: (cols LongTensor[64], delta float32 [1024, 64])} for the published p50c edit,
    where delta = W_edited(bf16) - W_base(bf16) on the changed columns."""
    import hashlib
    z = np.load(f"{LOGS}/runs/p50c/edit/edit.npz")
    man = json.load(open(f"{LOGS}/runs/p50c/edit/manifest.json"))
    lm = lm_of(model)
    out = {}
    for i, t in enumerate(man["tensors"]):
        l = int(t["name"].split(".layers.")[1].split(".")[0])
        W = lm.layers[l].mlp.down_proj.weight
        base = bf16_bits(W).copy()
        if verify:
            assert hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest() == t["base_sha256"], "base mismatch"
        e = base.reshape(-1).copy()
        e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
        e = e.reshape(base.shape)
        if verify:
            assert hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest() == t["edited_sha256"], "edit mismatch"
        We = torch.from_numpy(e.view(np.int16).copy()).view(torch.bfloat16)
        d = We.float() - W.detach().float()
        cols = torch.nonzero(d.abs().sum(0) > 0).squeeze(1)
        out[l] = (cols, d[:, cols].clone())
    return out


class SwapWeights:
    """Context manager: temporarily add sparse column deltas {layer: (cols, delta)} to down_proj (in bf16)."""

    def __init__(self, model, deltas):
        self.lm, self.deltas, self.saved = lm_of(model), deltas, {}

    def __enter__(self):
        for l, (cols, d) in self.deltas.items():
            W = self.lm.layers[l].mlp.down_proj.weight
            self.saved[l] = W.data[:, cols].clone()
            W.data[:, cols] = (W.data[:, cols].float() + d).to(torch.bfloat16)
        return self

    def __exit__(self, *a):
        for l, (cols, d) in self.deltas.items():
            self.lm.layers[l].mlp.down_proj.weight.data[:, cols] = self.saved[l]
