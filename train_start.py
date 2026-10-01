"""Stage 1 of the starter curriculum: train on the easy rungs first.

Every arm that was trained on the full suite from scratch floored at 0.0 on the needle
tasks, emitting numbers that appear nowhere in the context -- a copy circuit that never
formed. This stage gives it somewhere easy to form, training only on the three tasks
that each need a single retrieval:

* ``niah_single_1``  noise haystack, one needle, 7-digit answer.
* ``niah_single_2``  the same needle in an essay haystack, so the circuit cannot lean on
  the haystack being repetitive noise.
* ``vt``             one variable chain: the same job for following a chain.

It runs for ``--starter_epochs`` and saves ``starter.pt`` -- model weights, AdamW state,
the learning-rate schedule and its warmup -- to ``--output_dir``. ``train_full.py``
continues from that file on the full task set, so the two stages together are one run
whose training data changes at the switch.

Both stages read the same ``--data_dir``, which holds every task of either stage, and
each loads only its own list: ``--starter_tasks`` here, ``--tasks`` in the full stage.
run_seed_experiments.sh generates the union and runs both stages per cell.

Usage
-----
    python train_start.py --data_dir train_data/2048/data \\
        --model transformer_mask --encoder_mask CCCCFFFF --starter_epochs 4 --lr 1.5e-4 \\
        --output_dir experiments/tm_CCCCFFFF_s42 --bf16
    python train_full.py --init_from experiments/tm_CCCCFFFF_s42/starter.pt \\
        --data_dir train_data/2048/data --tasks <TRAIN_TASKS> --model transformer_mask \\
        --encoder_mask CCCCFFFF --lr 1.5e-4 --output_dir experiments/tm_CCCCFFFF_s42 --bf16
"""

import json
import math
import os
import time

import torch
import torch.nn as nn

import train
from train import log

STARTER_TASKS = ("niah_single_1", "niah_single_2", "vt")


def main(args, wandb_run=None):
    """Train the starter stage and save ``starter.pt``; return its path."""
    if args.data_format != "ruler":
        raise ValueError("the starter curriculum is defined over RULER tasks; "
                         "use --data_format ruler")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s | starter stage: %s for %d epoch(s)",
             device, ", ".join(args.starter_tasks), args.starter_epochs)

    tokenizer = train.build_tokenizer(args.tokenizer)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id
    model = train.build_model_from_args(args, vocab_size, pad_id).to(device)
    log.info("Model: %.1fM params | %s",
             sum(p.numel() for p in model.parameters()) / 1e6, model.description)

    train_loader, val_loader = train.build_dataloaders(args, tokenizer, pad_id,
                                                       tasks=args.starter_tasks)
    # Warmup is set in starter epochs, not --warmup_steps: build_optimizer would cap a
    # step count at a tenth of this short stage, and an epoch count stays right if the
    # batch size or the sample count changes. train_full.py inherits this value.
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    optimizer, scheduler, warmup = train.build_optimizer(
        args, model, len(train_loader), args.starter_epochs,
        warmup=max(1, round(args.starter_warmup_epochs * steps_per_epoch)))
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    amp_dtype, scaler = train.resolve_precision(args, device)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config_starter.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    metrics_path = os.path.join(args.output_dir, "starter_metrics.jsonl")

    val_loss = float("nan")
    for epoch in range(1, args.starter_epochs + 1):
        t0 = time.time()
        train_loss, train_stats = train.run_epoch(
            model, train_loader, criterion, vocab_size, device,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            amp_dtype=amp_dtype, grad_clip=args.grad_clip, log_every=args.log_every,
            lr=args.lr, accum_steps=args.grad_accum,
            wandb_run=wandb_run, step_offset=(epoch - 1) * len(train_loader))
        with torch.no_grad():
            # fp32, as train.main validates, so the two stages' val losses compare.
            val_loss, val_stats = train.run_epoch(model, val_loader, criterion,
                                                  vocab_size, device)
        elapsed = time.time() - t0

        log.info("Starter epoch %d/%d | train %.4f (ppl %.1f) | val %.4f (ppl %.1f) | "
                 "%d non-finite batch(es), %d skipped step(s) | %.0fs",
                 epoch, args.starter_epochs, train_loss, math.exp(min(train_loss, 20)),
                 val_loss, math.exp(min(val_loss, 20)), train_stats["n_nonfinite"],
                 train_stats["n_skipped_steps"], elapsed)
        record = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                      train_recent_loss=train_stats["recent_loss"],
                      train_nonfinite=train_stats["n_nonfinite"],
                      val_nonfinite=val_stats["n_nonfinite"],
                      opt_steps=train_stats["n_opt_steps"],
                      skipped_steps=train_stats["n_skipped_steps"],
                      zero_grad_steps=train_stats["n_zero_grad"])
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if wandb_run is not None:
            wandb_run.log({f"starter/{k}": v for k, v in record.items()},
                          step=epoch * len(train_loader))

    # The LAST state, not the best: the full stage continues this exact optimizer and
    # schedule, and AdamW moments from a later step do not belong with earlier weights.
    path = os.path.join(args.output_dir, "starter.pt")
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), warmup=warmup,
                    starter_epochs=args.starter_epochs,
                    starter_tasks=list(args.starter_tasks), val_loss=val_loss,
                    args=vars(args), vocab_size=vocab_size, pad_token_id=pad_id),
               path)
    log.info("Starter stage done: val_loss %.4f after %d optimizer steps. Saved %s",
             val_loss, scheduler.last_epoch, path)
    return path


def add_starter_args(p):
    """Flags for the starter stage, on top of every train.py flag."""
    g = p.add_argument_group("starter curriculum")
    g.add_argument("--starter_epochs", type=int, default=4,
                   help="Epochs over the starter tasks before train_full.py takes over.")
    g.add_argument("--starter_warmup_epochs", type=float, default=2.0,
                   help="Learning-rate warmup, in starter epochs. Replaces --warmup_steps "
                        "for both stages: train_full.py continues this schedule.")
    g.add_argument("--starter_tasks", nargs="+", default=list(STARTER_TASKS),
                   help="Task directories under --data_dir to train on. Each must exist.")


def parse_args():
    p = train.build_parser()
    p.description = "Starter curriculum, stage 1: train on the easy tasks"
    add_starter_args(p)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
