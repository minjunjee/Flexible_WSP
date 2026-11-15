"""Scoring of answers and bootstrap intervals over questions."""
import json
import re

import numpy as np


def parse_json(txt):
    """(starts with '{', is a JSON object, has the key 'answer', nothing follows the object, parsed object or None)"""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip()).strip()
    if not t.startswith("{"):
        return False, False, False, False, None
    depth = 0
    for i, ch in enumerate(t):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            try:
                d = json.loads(t[: i + 1])
            except Exception:
                return True, False, False, False, None
            ok = isinstance(d, dict)
            return True, ok, ok and "answer" in d, t[i + 1:].strip() == "", d if ok else None
    return True, False, False, False, None


def is_correct(txt, gold, obj=None):
    """gold: alternatives separated by '|'. A number must appear as a whole number; anything else as a substring.
    The text of a parsed JSON object is searched as well, because JSON may escape non-ASCII letters."""
    hay = txt.lower()
    if obj is not None:
        hay += " " + " ".join(str(v) for v in obj.values()).lower()
    for g in gold.lower().split("|"):
        if re.fullmatch(r"\d+", g):
            if re.search(rf"(?<![\d.]){g}(?![\d]|\.\d)", hay.replace(",", "")):
                return True
        elif g in hay:
            return True
    return False


def score_json(txt, gold):
    starts, valid, key, clean, obj = parse_json(txt)
    return dict(starts=starts, valid=valid, key_answer=key, clean_end=clean, correct=is_correct(txt, gold, obj))


def boot_ci(x, n_boot=2000, seed=0):
    """mean and 95% interval from resampling questions"""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff_ci(x, y, n_boot=2000, seed=0):
    """mean(x) - mean(y) on the same questions, with a paired 95% interval"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(x), (n_boot, len(x)))
    d = x[i].mean(1) - y[i].mean(1)
    return float(x.mean() - y.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fmt_ci(x):
    m, lo, hi = boot_ci(x)
    return f"{100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
