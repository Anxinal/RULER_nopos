"""
Classify NIAH prediction errors and test whether they are positional.

Built to check one claim: a causal + future-only model (e.g. CCCCFFFF) fails because it
cannot tell nearby positions apart far from the sequence edges, since position reaches
it only as roughly 1/distance from the start (causal heads) or the end (future-only
heads). That claim makes two predictions, and two other explanations make different
ones:

  P1  accuracy dips mid-context: it falls with the gold needle's distance to the
      nearest edge (permutation test on the correlation).
  P2  the wrong answers are mostly the gold needle's neighbours (within NEAR lines),
      well above the rate a random pick of another needle would give.
  A   key confusion: the picked needle's key looks like the queried key (a shared
      word, or the same first or last GPT-2 token) well above chance.
  B   copy failure: the answer is not in the context at all, or is a near-copy of the
      gold value. If these sit mid-context and start with the gold value's first token,
      the model found the right needle but lost the order of its tokens -- a local
      position failure inside the value rather than in choosing the needle.

Every prediction is put in exactly one class: correct, adjacent needle, far needle,
near copy, not in context, or no answer. Positions are character fractions of the
context, which ends where the question begins.

Running
```
python scripts/eval/error_taxonomy.py --pred pred=CCCCFFFF_new
python scripts/eval/error_taxonomy.py --results_dir /path/to/run/results --bins 8
```
"""

import os
import re
import sys
import glob
import math
import argparse
import collections

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.manifest_utils import read_manifest  # noqa: E402
from synthetic.constants import string_match_all  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--pred", action="append", default=[],
                    help="folder of <task>.jsonl predictions, optionally DIR=LABEL; repeatable")
parser.add_argument("--results_dir", type=str, default=None,
                    help="also read every <model>/synthetic/<seq_len>/pred folder under this")
parser.add_argument("--tasks", type=str, default="niah_multikey_1,niah_multikey_2,niah_multikey_3")
parser.add_argument("--bins", type=int, default=8, help="depth bins over the context")
parser.add_argument("--near", type=int, default=2,
                    help="a wrong needle within this many lines of the gold one is 'adjacent'")
parser.add_argument("--permutations", type=int, default=10000)
parser.add_argument("--tokenizer", type=str, default="gpt2",
                    help="tokenizer for the look-alike key check (first/last token)")
parser.add_argument("--out", type=str, default=None, help="csv path for the per-bin table")
args = parser.parse_args()

QUESTION_MARKER = "What is the special magic"
NEEDLE = re.compile(r"One of the special magic (?:numbers|uuids|words) for (.+?) is: (\S+?)\.(?=\s|$)")
QUERY = re.compile(r"What is the special magic \w+ for (.+?) mentioned in the provided text\?")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
NUMBER = re.compile(r"\d+")
CLASSES = ("correct", "adjacent needle", "far needle", "near copy", "not in context", "no answer")


def postprocess_pred(predict_str: str):
    # Same cleaning as evaluate.postprocess_pred, which cannot be imported because
    # evaluate.py parses its own arguments at import time.
    predict_str = predict_str.strip()
    return re.sub(r"[\x00-\x1f]", "\n", predict_str).strip()


_tok_cache = {}


def key_tokens(tokenizer, key):
    if key not in _tok_cache:
        _tok_cache[key] = tokenizer.encode(" " + key)
    return _tok_cache[key]


def looks_alike(tokenizer, key, query):
    """A shared whole word, or the same first or last token."""
    if set(re.split(r"[- ]", key)) & set(re.split(r"[- ]", query)):
        return True
    a, b = key_tokens(tokenizer, key), key_tokens(tokenizer, query)
    return a[0] == b[0] or a[-1] == b[-1]


def near_copy(picked, gold):
    """Same length and at least 4/7 of the characters (hyphens aside) in place."""
    if len(picked) != len(gold):
        return False
    pairs = [(p, g) for p, g in zip(picked, gold) if g != "-"]
    return sum(p == g for p, g in pairs) >= math.ceil(4 / 7 * len(pairs))


def analyse_row(row, tokenizer):
    """One record per prediction: class, gold depth, line offset, look-alike flag, baselines."""
    text = row["input"]
    end = text.rfind(QUESTION_MARKER)
    end = end if end > 0 else len(text)
    context = text[:end]
    needles = [(m.group(1), m.group(2), m.start()) for m in NEEDLE.finditer(context)]
    gold = str(row["outputs"][0])
    query = QUERY.search(text)
    query = query.group(1) if query else ""
    gold_idx = next((i for i, n in enumerate(needles) if n[1] == gold), None)
    if gold_idx is None:
        return None                                   # needle not parseable; reported as skipped
    pred = postprocess_pred(row["pred"])
    rec = dict(depth=needles[gold_idx][2] / end, n_needles=len(needles), offset=None,
               lookalike=None, p_adjacent=None, p_lookalike=None, pred=pred,
               starts_right=None)
    if gold.lower() in pred.lower():
        rec["cls"] = "correct"
        return rec
    found = (UUID if UUID.fullmatch(gold) else NUMBER).findall(pred)
    picked = found[0] if found else None
    hits = [i for i, n in enumerate(needles) if picked is not None and n[1] == picked]
    if picked is None:
        rec["cls"] = "no answer"
    elif hits:
        offset = hits[0] - gold_idx
        rec["offset"] = offset
        rec["cls"] = "adjacent needle" if abs(offset) <= args.near else "far needle"
        rec["lookalike"] = looks_alike(tokenizer, needles[hits[0]][0], query)
        # What a random pick of another needle would give, for this sample.
        others = [i for i in range(len(needles)) if i != gold_idx]
        rec["p_adjacent"] = sum(abs(i - gold_idx) <= args.near for i in others) / len(others)
        rec["p_lookalike"] = sum(looks_alike(tokenizer, needles[i][0], query) for i in others) / len(others)
    else:
        rec["cls"] = "near copy" if near_copy(picked, gold) else "not in context"
        # Did the copy begin on the right needle? Same first token as the gold value.
        rec["starts_right"] = (tokenizer.encode(" " + picked)[0]
                               == tokenizer.encode(" " + gold)[0])
    return rec


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def permutation_corr(x, y, n_perm, rng):
    """Correlation of x and y, and the one-sided p-value that it is this negative by chance."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    obs = np.corrcoef(x, y)[0, 1]
    perms = np.array([np.corrcoef(rng.permutation(x), y)[0, 1] for _ in range(n_perm)])
    return obs, (np.sum(perms <= obs) + 1) / (n_perm + 1)


def length_label(path, rows):
    """The sequence length from a .../synthetic/<len>/pred path, else from the rows."""
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 2 and parts[-1] == "pred" and parts[-2].isdigit():
        return int(parts[-2])
    longest = max((r.get("length", 0) for r in rows), default=0)
    return 2 ** math.ceil(math.log2(longest)) if longest > 0 else -1


def sources():
    out = []
    for spec in args.pred:
        path, _, label = spec.partition("=")
        out.append((label or os.path.basename(os.path.normpath(path)), path))
    if args.results_dir:
        for d in sorted(glob.glob(os.path.join(args.results_dir, "*", "synthetic", "*", "pred"))):
            out.append((os.path.relpath(d, args.results_dir).split(os.sep)[0], d))
    return out


def report(model, length, task, recs, n_skipped, rng, csv_rows):
    n = len(recs)
    counts = collections.Counter(r["cls"] for r in recs)
    print(f"\n=== {model} | {length} | {task} | {n} samples"
          + (f" ({n_skipped} skipped: gold needle not found)" if n_skipped else "") + " ===")
    print("  outcomes: " + ", ".join(f"{c} {counts[c]}" for c in CLASSES if counts[c]))

    # Accuracy and error mix per depth bin.
    print(f"  {'depth bin':>11s} {'n':>5s} {'acc %':>6s} {'95% CI':>13s} {'adjacent':>8s} {'far':>5s} {'copy':>5s}")
    for b in range(args.bins):
        lo, hi = b / args.bins, (b + 1) / args.bins
        cell = [r for r in recs if lo <= r["depth"] < hi or (b == args.bins - 1 and r["depth"] == 1.0)]
        k = sum(r["cls"] == "correct" for r in cell)
        cl, ch = wilson(k, len(cell))
        c = collections.Counter(r["cls"] for r in cell)
        acc = 100 * k / len(cell) if cell else float("nan")
        print(f"  {lo:5.3f}-{hi:5.3f} {len(cell):5d} {acc:6.1f} {100*cl:5.1f}-{100*ch:5.1f}  "
              f"{c['adjacent needle']:8d} {c['far needle']:5d} {c['near copy'] + c['not in context']:5d}")
        csv_rows.append(dict(model=model, length=length, task=task, bin=f"{lo:.3f}-{hi:.3f}",
                             n=len(cell), accuracy=acc, ci_low=100 * cl, ci_high=100 * ch,
                             **{c_: c[c_] for c_ in CLASSES}))
    k = counts["correct"]
    cl, ch = wilson(k, n)
    csv_rows.append(dict(model=model, length=length, task=task, bin="all", n=n,
                         accuracy=100 * k / n, ci_low=100 * cl, ci_high=100 * ch,
                         **{c_: counts[c_] for c_ in CLASSES}))

    # P1: does accuracy fall with distance to the nearest edge?
    correct = [r["cls"] == "correct" for r in recs]
    edge = [min(r["depth"], 1 - r["depth"]) for r in recs]
    r_edge, p_edge = permutation_corr(edge, correct, args.permutations, rng)
    r_depth, p_depth = permutation_corr([r["depth"] for r in recs], correct, args.permutations, rng)
    print(f"  corr(correct, distance to nearest edge) = {r_edge:+.3f}, one-sided p = {p_edge:.4f}"
          f"   | corr(correct, depth) = {r_depth:+.3f}, one-sided p = {p_depth:.4f}")

    # P2 and A: wrong needles against a random-pick baseline.
    wrong = [r for r in recs if r["offset"] is not None]
    adj = sum(abs(r["offset"]) <= args.near for r in wrong)
    adj_exp = sum(r["p_adjacent"] for r in wrong)
    mid = [r for r in wrong if min(r["depth"], 1 - r["depth"]) >= 0.25]
    adj_mid = sum(abs(r["offset"]) <= args.near for r in mid)
    adj_mid_exp = sum(r["p_adjacent"] for r in mid)
    lex = sum(r["lookalike"] for r in wrong)
    lex_exp = sum(r["p_lookalike"] for r in wrong)
    offsets = collections.Counter(r["offset"] for r in wrong)
    hist = " ".join(f"{o:+d}:{offsets[o]}" for o in sorted(offsets, key=lambda o: (abs(o), o))
                    if abs(o) <= 5)
    beyond = sum(v for o, v in offsets.items() if abs(o) > 5)
    if wrong:
        med = float(np.median([abs(r["offset"]) for r in wrong]))
        print(f"  wrong needles {len(wrong)} | line offsets {hist or '-'} | beyond ±5: {beyond} | median |offset| {med:.0f} lines")
    ratio = lambda o, e: (o / e) if e > 0 else float("nan")
    print(f"  adjacent (|offset| <= {args.near}): {adj} vs {adj_exp:.1f} by chance ({ratio(adj, adj_exp):.1f}x)"
          f" | in the middle half: {adj_mid} vs {adj_mid_exp:.1f} ({ratio(adj_mid, adj_mid_exp):.1f}x)")
    print(f"  look-alike key: {lex} vs {lex_exp:.1f} by chance ({ratio(lex, lex_exp):.1f}x)")

    # B, located: are the copy failures concentrated mid-context?
    in_mid = lambda r: min(r["depth"], 1 - r["depth"]) >= 0.25
    copies = [r for r in recs if r["cls"] in ("near copy", "not in context")]
    share_mid = sum(map(in_mid, recs)) / n
    k_mid = sum(map(in_mid, copies))
    p_copy_mid = sum(math.comb(len(copies), i) * share_mid ** i * (1 - share_mid) ** (len(copies) - i)
                     for i in range(k_mid, len(copies) + 1)) if copies else float("nan")
    started = sum(bool(r["starts_right"]) for r in copies)
    print(f"  copy failures: {len(copies)}, in the middle half {k_mid} (it holds {100*share_mid:.0f}% of "
          f"samples; one-sided binomial p = {p_copy_mid:.1e}) | starting with the gold first token: {started}")

    # Verdict.
    errors = n - k
    copy = counts["near copy"] + counts["not in context"] + counts["no answer"]
    edge_bins = [r["cls"] == "correct" for r in recs if min(r["depth"], 1 - r["depth"]) < 0.25]
    mid_bins = [r["cls"] == "correct" for r in recs if min(r["depth"], 1 - r["depth"]) >= 0.25]
    acc_edge = 100 * np.mean(edge_bins) if edge_bins else float("nan")
    acc_mid = 100 * np.mean(mid_bins) if mid_bins else float("nan")
    p1 = p_edge < 0.05 and acc_mid < acc_edge
    p2 = adj_mid >= 5 and ratio(adj_mid, adj_mid_exp) >= 3 and adj_mid >= 0.25 * max(len(mid), 1)
    a = lex >= 5 and ratio(lex, lex_exp) >= 3
    b = errors > 0 and copy >= 0.25 * errors
    b_mid = len(copies) >= 5 and p_copy_mid < 0.05
    print(f"  accuracy near the edges {acc_edge:.1f}% vs middle half {acc_mid:.1f}%")
    print(f"  verdict: P1 (mid-context dip) {'met' if p1 else 'not met'} | "
          f"P2 (neighbour errors) {'met' if p2 else 'not met'} | "
          f"A (look-alike keys) {'met' if a else 'not met'} | "
          f"B (copy failures, {copy} of {errors} errors) {'met' if b else 'not met'}, "
          f"{'concentrated mid-context' if b_mid else 'not concentrated mid-context'} | "
          f"positional claim {'SUPPORTED' if p1 and p2 else 'not supported'}")


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = np.random.default_rng(0)
    tasks = [t for t in args.tasks.split(",") if t]
    srcs = sources()
    if not srcs:
        raise SystemExit("give at least one --pred folder or --results_dir")

    csv_rows = []
    for model, path in srcs:
        for task in tasks:
            f = os.path.join(path, f"{task}.jsonl")
            if not os.path.exists(f):
                continue
            rows = read_manifest(f)
            recs = [analyse_row(r, tokenizer) for r in rows]
            kept = [r for r in recs if r is not None]
            # The aggregate must equal evaluate.py's score for the same file.
            score = string_match_all([postprocess_pred(r["pred"]) for r in rows],
                                     [r["outputs"] for r in rows])
            acc = 100 * sum(r["cls"] == "correct" for r in kept) / len(kept)
            report(model, length_label(path, rows), task, kept, len(recs) - len(kept), rng, csv_rows)
            if not math.isclose(acc, score, abs_tol=0.01) and len(kept) == len(rows):
                print(f"  WARNING: accuracy {acc:.2f} differs from string_match_all {score:.2f}")

    if args.out:
        pd.DataFrame(csv_rows).to_csv(args.out, index=False)
        print(f"\nSaved per-bin table to {args.out}")


if __name__ == "__main__":
    main()
