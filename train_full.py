"""Stage 2 of the starter curriculum: continue from ``starter.pt`` on the full task set.

Loads the checkpoint written by ``train_start.py`` and runs ``train.main`` from it: the
model weights, the AdamW state and the learning-rate schedule (its step count and its
warmup) all carry over, so the two stages together are one continuous run whose data
changes at the switch. Everything after that -- early stopping, ``best.pt``,
``metrics.jsonl``, ``TRAINING_COMPLETE.json`` -- is ``train.py``'s, and prediction
reads the result exactly as it reads a run trained from scratch.

``--epochs`` counts full-stage epochs only. See ``train_start.py`` for usage.
"""

import torch

import train

# Must agree between the stages. The first four fix the model's shape and vocabulary,
# so a mismatch would fail in load_state_dict anyway, only later and less clearly. lr
# and weight_decay come back from the AdamW state, so a different value here would be
# silently ignored rather than applied.
MUST_MATCH = ("model", "encoder_mask", "pe", "tokenizer", "lr", "weight_decay")


def resolved_max_len(a):
    """The max_len build_model_from_args gives the model, for args or a saved dict."""
    get = a.get if isinstance(a, dict) else lambda k: getattr(a, k)
    return get("max_len") or get("src_len") + get("tgt_len")


def check_compatible(args, starter_args):
    """Raise if this stage cannot continue the starter stage's run."""
    diffs = [f"--{k}: starter {starter_args.get(k)!r}, now {getattr(args, k)!r}"
             for k in MUST_MATCH if starter_args.get(k) != getattr(args, k)]
    if resolved_max_len(starter_args) != resolved_max_len(args):
        diffs.append(f"max_len: starter {resolved_max_len(starter_args)}, "
                     f"now {resolved_max_len(args)}")
    if diffs:
        raise ValueError("train_full.py must continue the starter stage with the same "
                         "settings:\n  " + "\n  ".join(diffs))


def main(args, wandb_run=None):
    init_state = torch.load(args.init_from, map_location="cpu", weights_only=False)
    check_compatible(args, init_state["args"])
    train.main(args, wandb_run=wandb_run, init_state=init_state)


def add_full_args(p):
    """Flags for the full stage, on top of every train.py flag."""
    g = p.add_argument_group("starter curriculum")
    g.add_argument("--init_from", required=True,
                   help="starter.pt written by train_start.py.")


def parse_args():
    p = train.build_parser()
    p.description = "Starter curriculum, stage 2: continue on the full task set"
    add_full_args(p)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
