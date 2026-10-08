"""Train position probes on the layers of one trained model, one probe per layer.

    python train_position_probe.py experiments/<run> 0 1 2 3 4 5 --data_dir experiments/train_data/2048/data

One example per RULER niah sample. Input: a layer's hidden state at the last position,
where the model is about to write the answer. Target: the token position in the prompt
where the answer starts. Every layer sees the same samples and the same split. Writes
<run>/position_probe_layer<L>.pt for each layer.

With ``--mode online`` (or ``offline``) the probes are tracked in Weights & Biases, one
run per (model run, stack) holding every layer:

* every epoch, per layer: ``layer<L>/train/*`` and ``layer<L>/val/*`` (loss, accuracy,
  within_8, mean_abs_error), plotted against ``epoch``;
* once per layer, plotted against ``layer``: ``by_layer/val_mean_abs_error`` (the best
  epoch's) and ``by_layer/test_*``, so one chart shows the error across layers;
* in the summary: ``layer<L>/test/*`` and ``layer<L>/best_epoch``.

The default ``--mode disabled`` logs nothing and needs no login.
"""

import argparse
import copy
import glob
import json
import os

import torch
import wandb
from transformers import AutoTokenizer
import torch.nn as nn
from tmodel.CrossEntropyLossWithPositionBias import CrossEntropyLossWithPositionBias
from tmodel.models import build_model
from tmodel.probe import PositionalProbe
from tmodel.util import extract_hidden_state


def load_model(folder, device):
    ckpt = torch.load(os.path.join(folder, "best.pt"), map_location=device, weights_only=False)
    a = ckpt["args"]
    model = build_model(a["model"], ckpt["vocab_size"], ckpt["pad_token_id"],
                        mask_spec=a.get("encoder_mask", "B"), pe=a.get("pe", "none"),
                        max_len=a.get("max_len") or a["src_len"] + a["tgt_len"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), AutoTokenizer.from_pretrained(folder), ckpt


def collect(model, tokenizer, ckpt, data_dir, layers, stack, num_samples, device):
    """(hidden states ``[layer, sample, d_model]``, answer positions) for up to num_samples
    niah samples. One forward pass per sample yields every layer's state."""
    src_len = ckpt["args"]["src_len"]
    bos = torch.tensor([[ckpt["bos_token_id"]]], device=device)
    # niah only: a needle sits at one position; vt answers are spread over the prompt.
    files = sorted(glob.glob(os.path.join(data_dir, "**", "niah*", "*.jsonl"), recursive=True))
    xs, ys = [[] for _ in layers], []
    for path in files:
        with open(path, encoding="utf-8") as f:
            for _, line in zip(range(num_samples // len(files)), f):
                row = json.loads(line)
                text = (row["input"] + row.get("answer_prefix", "")).rstrip()
                char = text.find(str(row["outputs"][0]))
                enc = tokenizer(text, return_offsets_mapping=True)
                # First token that covers the answer's first character.
                token = next((i for i, (_, end) in enumerate(enc["offset_mapping"]) if end > char), -1)
                ids = enc["input_ids"][-src_len:]
                token -= len(enc["input_ids"]) - len(ids)       # after left truncation
                if char < 0 or token < 0:
                    continue                                    # answer absent or cut off
                src = torch.tensor([ids], device=device)
                # clone: indexing returns a view that would keep the whole
                # [seq, d_model] state of every sample alive (~4 MB each for the encoder).
                for got, hidden in zip(xs, extract_hidden_state(model, src, bos, layers, stack)):
                    got.append(hidden[0, -1].clone())
                ys.append(token)
    return torch.stack([torch.stack(got) for got in xs]), torch.tensor(ys)


@torch.no_grad()
def evaluate(probe, x, y):
    logits = probe(x)
    error = (logits.argmax(-1) - y).abs().float()
    return dict(loss=nn.functional.cross_entropy(logits, y).item(),
                accuracy=(error == 0).float().mean().item(),
                within_8=(error <= 8).float().mean().item(),
                mean_abs_error=error.mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="run folder holding best.pt and its tokenizer")
    p.add_argument("layers", type=int, nargs="+", help="layer indices, 0-based")
    p.add_argument("--data_dir", required=True, help="RULER training data root")
    p.add_argument("--stack", default="decoder", choices=["encoder", "decoder"])
    p.add_argument("--num_samples", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("wandb")
    g.add_argument("--project", default="Ruler_nopos")
    g.add_argument("--entity", default=None, help="wandb team or user; default is yours.")
    g.add_argument("--group", default="position_probe")
    g.add_argument("--run_name", default=None,
                   help="Default: probe-<run folder>-<stack>.")
    g.add_argument("--mode", default="disabled", choices=["online", "offline", "disabled"],
                   help="offline writes the run locally for a later `wandb sync`.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, ckpt = load_model(args.folder, device)
    source = os.path.basename(os.path.normpath(args.folder))
    # mode="disabled" gives a no-op run, so nothing below depends on the mode. Leaving
    # the block finishes the run; if the probe raises, wandb marks it failed.
    with wandb.init(project=args.project, entity=args.entity, group=args.group,
                    name=args.run_name or f"probe-{source}-{args.stack}",
                    mode=args.mode, job_type="position_probe",
                    config=dict(vars(args), source_run=source, source_args=ckpt["args"],
                                description=model.description)) as run:
        # Per-layer curves share the epoch axis; the by_layer metrics get one point
        # per layer. Without these every log call would advance wandb's own step.
        run.define_metric("epoch")
        run.define_metric("layer")
        for layer in args.layers:       # wandb globs match a suffix only
            run.define_metric(f"layer{layer}/*", step_metric="epoch")
        run.define_metric("by_layer/*", step_metric="layer")

        xs, y = collect(model, tokenizer, ckpt, args.data_dir, args.layers, args.stack,
                        args.num_samples, device)
        # 80/10/10 split: validation picks the epoch, test is only reported.
        order = torch.randperm(len(y))
        n = len(y) // 10
        split = order[:n], order[n:2 * n], order[2 * n:]
        run.summary.update(dict(n_train=len(split[2]), n_val=n, n_test=n))
        for layer, x in zip(args.layers, xs):
            print(f"=== layer {layer} ===")
            train_probe(args, layer, x, y, split, ckpt, device, run)


def train_probe(args, layer, x, y, split, ckpt, device, run):
    test, val, train = split
    # Standardise: the pre-norm residual stream grows with depth.
    mean, std = x[train].mean(0), x[train].std(0) + 1e-6
    x = ((x - mean) / std).to(device)
    y = y.to(device)

    probe = PositionalProbe(x.size(1), output_dim=ckpt["args"]["src_len"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        probe.train()
        for batch in train[torch.randperm(len(train))].split(args.batch_size):
            loss = criterion(probe(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        train_metrics, val_metrics = evaluate(probe, x[train], y[train]), evaluate(probe, x[val], y[val])
        val_error = val_metrics["mean_abs_error"]
        # Mean abs error picks the epoch: exact accuracy is too noisy at a few percent.
        if val_error < best_val:
            best_val, best_epoch = val_error, epoch
            best_state = copy.deepcopy(probe.state_dict())
            metrics = evaluate(probe, x[test], y[test])
        print(f"epoch {epoch:3d} | train loss {loss.item():.3f} | val mean abs error {val_error:.1f} "
              f"| best epoch {best_epoch}")
        run.log({**{f"layer{layer}/train/{k}": v for k, v in train_metrics.items()},
                 **{f"layer{layer}/val/{k}": v for k, v in val_metrics.items()},
                 "epoch": epoch})

    run.log({"by_layer/val_mean_abs_error": best_val,
             **{f"by_layer/test_{k}": v for k, v in metrics.items()}, "layer": layer})
    run.summary.update({**{f"layer{layer}/test/{k}": v for k, v in metrics.items()},
                        f"layer{layer}/best_epoch": best_epoch,
                        f"layer{layer}/best_val_mean_abs_error": best_val})

    print(f"test at epoch {best_epoch}: acc {metrics['accuracy']:.3f} | within 8: "
          f"{metrics['within_8']:.3f} | mean abs error {metrics['mean_abs_error']:.1f}")
    # Write to a temp file and rename, so a killed run never leaves a partial probe.
    out = os.path.join(args.folder, f"position_probe_layer{layer}.pt")
    torch.save(dict(probe=best_state, mean=mean, std=std, metrics=metrics,
                    best_epoch=best_epoch, layer=layer, args=vars(args)), out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"saved {out} ({len(train)} train / {len(val)} val / {len(test)} test samples)")


if __name__ == "__main__":
    main()
