"""
Score NIAH predictions grouped by where the needle sits in the context.

The prediction files carry no depth metadata, so the needle position is recovered
from the input: the answer value appears exactly once in each NIAH input, and the
context ends where the question ("What is the special magic ...") begins. Positions
are character fractions of the context, binned into equal groups (quarters by
default; Q1 is farthest from the question).

Running
```
python scripts/eval/position_breakdown.py \
    --results_dir /path/to/run/results \
    --seq_len 4096
```
"""

import os
import re
import sys
import glob
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.manifest_utils import read_manifest  # noqa: E402
from synthetic.constants import string_match_all  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default='/Users/anxiao/experiments/run_5/results',
                    help='folder holding one <model>/synthetic/<seq_len>/pred/ tree per model')
parser.add_argument("--seq_len", type=int, default=4096)
parser.add_argument("--tasks", type=str, default='niah_multikey_1,niah_multikey_2,niah_multikey_3')
parser.add_argument("--num_bins", type=int, default=4)
parser.add_argument("--out", type=str, default=None,
                    help='csv path; defaults to <results_dir>/position_breakdown_<seq_len>.csv')
args = parser.parse_args()

QUESTION_MARKER = 'What is the special magic'


def postprocess_pred(predict_str: str):
    # Same cleaning as evaluate.postprocess_pred, which cannot be imported because
    # evaluate.py parses its own arguments at import time.
    predict_str = predict_str.strip()
    np_pattern = re.compile(r'[\x00-\x1f]')
    return np_pattern.sub('\n', predict_str).strip()


def needle_fraction(line: dict):
    """Relative position of the answer within the context, or None if not found."""
    text = line['input']
    context_end = text.rfind(QUESTION_MARKER)
    if context_end <= 0:
        context_end = len(text)
    pos = text.find(line['outputs'][0])
    if pos < 0 or pos >= context_end:
        return None
    return pos / context_end


def score_rows(rows):
    preds = [postprocess_pred(r['pred']) for r in rows]
    refs = [r['outputs'] for r in rows]
    return string_match_all(preds, refs) if rows else float('nan')


def main():
    tasks = [t for t in args.tasks.split(',') if t]
    pred_dirs = sorted(glob.glob(os.path.join(args.results_dir, '*', 'synthetic', str(args.seq_len), 'pred')))
    if not pred_dirs:
        raise SystemExit(f'No */synthetic/{args.seq_len}/pred folders under {args.results_dir}')

    records = []
    for pred_dir in pred_dirs:
        model = os.path.relpath(pred_dir, args.results_dir).split(os.sep)[0]
        for task in tasks:
            path = os.path.join(pred_dir, f'{task}.jsonl')
            if not os.path.exists(path):
                print(f'{model}: {task}.jsonl not found, skipping')
                continue

            lines = read_manifest(path)
            bins = [[] for _ in range(args.num_bins)]
            unlocated = 0
            for line in lines:
                frac = needle_fraction(line)
                if frac is None:
                    unlocated += 1
                    continue
                bins[min(int(frac * args.num_bins), args.num_bins - 1)].append(line)
            if unlocated:
                print(f'{model}/{task}: {unlocated}/{len(lines)} needles not located, excluded from bins')

            for b, rows in enumerate(bins):
                records.append({'model': model, 'task': task, 'quarter': f'Q{b + 1}',
                                'n': len(rows), 'score': score_rows(rows)})
            records.append({'model': model, 'task': task, 'quarter': 'all',
                            'n': len(lines), 'score': score_rows(lines)})

    df = pd.DataFrame(records)
    out = args.out or os.path.join(args.results_dir, f'position_breakdown_{args.seq_len}.csv')
    df.to_csv(out, index=False)

    pivot = df.pivot_table(index=['task', 'quarter'], columns='model', values='score', sort=False)
    counts = df.pivot_table(index=['task', 'quarter'], columns='model', values='n', sort=False)
    pivot['n'] = counts.iloc[:, 0]
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(f'\nScore by needle position (seq_len={args.seq_len}, Q1 = start of context)\n')
        print(pivot)
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
