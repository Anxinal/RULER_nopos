"""Train a tmodel model with Weights & Biases tracking.

Takes every ``train.py`` flag plus the wandb flags below, opens one wandb run, and
trains into it. ``train.py`` does the training; the run receives

* every ``--log_every`` batches: ``train/loss_window``, ``train/loss_epoch_avg``,
  ``train/ppl_window``, ``train/grad_norm``, ``train/lr``, ``train/nonfinite_batches``
* every epoch: ``epoch/*``, the same record ``train.py`` writes to ``metrics.jsonl``
  (train/val loss and perplexity, spikes, skipped and zero-gradient steps, ...)
* in the summary: parameter count, initial loss, best val loss and epoch, stop reason.

The wandb step is the global training-batch count, so step and epoch metrics share
one x-axis.

``--stage starter`` / ``--stage full`` run the two starter-curriculum stages
(``train_start.py``, ``train_full.py``) instead, each as its own run; the starter run's
name ends in ``-starter`` and logs ``starter/*`` per epoch.

Usage
-----
    wandb login        # once per machine (or export WANDB_API_KEY)

    python train_wandb.py --model transformer_mask --pe sinusoidal --encoder_mask CCCCFFFF \\
        --data_dir experiments/train_data --output_dir experiments/tm_sin_CCCCFFFF_s42 \\
        --seed 42 --group seed_sweep
    python train_wandb.py --model alibi --data_dir ... --output_dir ... --mode offline
"""

import argparse

import wandb

import train
import train_full
import train_start

WANDB_FLAGS = ("project", "entity", "group", "tags", "run_name", "mode")

# --stage picks the entry point and the extra flags it takes. "scratch" is a plain
# train.py run; "starter" and "full" are the two curriculum stages, each its own wandb
# run (the wandb step restarts at 0 in each).
STAGES = {
    "scratch": (train.main, None),
    "starter": (train_start.main, train_start.add_starter_args),
    "full":    (train_full.main, train_full.add_full_args),
}


def default_run_name(args):
    """Model, plus pe and mask for transformer_mask, plus the seed and a starter tag."""
    if args.model == "transformer_mask":
        name = f"transformer_mask-pe{args.pe}-enc{args.encoder_mask}-s{args.seed}"
    else:
        name = f"{args.model}-s{args.seed}"
    return name + ("-starter" if args.stage == "starter" else "")


def run_config(args):
    """Every train.py flag, minus the wandb ones and those this model does not use."""
    config = {k: v for k, v in vars(args).items() if k not in WANDB_FLAGS}
    if args.model != "transformer_mask":
        config.pop("encoder_mask")   # only transformer_mask has these two
        config.pop("pe")
    return config


def parse_args():
    # --stage first: it decides which flags the rest of the command line may carry.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--stage", choices=STAGES, default="scratch")
    stage = pre.parse_known_args()[0].stage

    p = train.build_parser()
    p.description = "Train a tmodel model with wandb tracking"
    p.add_argument("--stage", choices=STAGES, default="scratch",
                   help="scratch: train.py from scratch. starter / full: the two stages "
                        "of the starter curriculum (train_start.py, train_full.py).")
    add_stage_args = STAGES[stage][1]
    if add_stage_args is not None:
        add_stage_args(p)
    g = p.add_argument_group("wandb")
    g.add_argument("--project", default="Ruler_nopos")
    g.add_argument("--entity", default=None, help="wandb team or user; default is yours.")
    g.add_argument("--group", default=None, help="Group runs together in the UI, e.g. one sweep.")
    g.add_argument("--tags", nargs="*", default=None)
    g.add_argument("--run_name", default=None,
                   help="Default: model, pe and mask (transformer_mask), and seed.")
    g.add_argument("--mode", default="online", choices=["online", "offline", "disabled"],
                   help="offline writes the run locally for a later `wandb sync`.")
    return p.parse_args()


def main():
    args = parse_args()
    # Leaving the block finishes the run; if training raises, wandb marks it failed
    # and the exception still propagates, so the job exits non-zero.
    with wandb.init(project=args.project, entity=args.entity, group=args.group,
                    name=args.run_name or default_run_name(args), tags=args.tags,
                    job_type=args.stage, mode=args.mode, config=run_config(args)) as run:
        STAGES[args.stage][0](args, wandb_run=run)


if __name__ == "__main__":
    main()
