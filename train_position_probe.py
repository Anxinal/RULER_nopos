"""Train one position probe on one layer of one trained model.

    python train_position_probe.py experiments/<run> 3 --data_dir experiments/train_data/2048/data

One example per RULER niah sample. Input: the layer's hidden state at the last position,
where the model is about to write the answer. Target: the token position in the prompt
where the answer starts. Writes <run>/position_probe_layer<L>.pt.
"""

import argparse
import glob
import json
import os

import torch
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


def collect(model, tokenizer, ckpt, data_dir, layer, stack, num_samples, device):
    """(hidden state, answer position) for up to num_samples niah samples."""
    src_len = ckpt["args"]["src_len"]
    bos = torch.tensor([[ckpt["bos_token_id"]]], device=device)
    # niah only: a needle sits at one position; vt answers are spread over the prompt.
    files = sorted(glob.glob(os.path.join(data_dir, "**", "niah*", "*.jsonl"), recursive=True))
    xs, ys = [], []
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
                xs.append(extract_hidden_state(model, src, bos, layer, stack)[0, -1])
                ys.append(token)
    return torch.stack(xs), torch.tensor(ys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="run folder holding best.pt and its tokenizer")
    p.add_argument("layer", type=int)
    p.add_argument("--data_dir", required=True, help="RULER training data root")
    p.add_argument("--stack", default="decoder", choices=["encoder", "decoder"])
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, ckpt = load_model(args.folder, device)
    x, y = collect(model, tokenizer, ckpt, args.data_dir, args.layer, args.stack,
                   args.num_samples, device)

    # 90/10 split, then standardise: the pre-norm residual stream grows with depth.
    order = torch.randperm(len(x))
    test, train = order[:len(x) // 10], order[len(x) // 10:]
    mean, std = x[train].mean(0), x[train].std(0) + 1e-6
    x = ((x - mean) / std).to(device)
    y = y.to(device)

    probe = PositionalProbe(x.size(1), output_dim=ckpt["args"]["src_len"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        probe.train()
        for batch in train[torch.randperm(len(train))].split(args.batch_size):
            loss = criterion(probe(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            error = (probe(x[test]).argmax(-1) - y[test]).abs().float()
        metrics = dict(accuracy=(error == 0).float().mean().item(),
                       within_8=(error <= 8).float().mean().item(),
                       mean_abs_error=error.mean().item())
        print(f"epoch {epoch:3d} | train loss {loss.item():.3f} | test acc {metrics['accuracy']:.3f} "
              f"| within 8: {metrics['within_8']:.3f} | mean abs error {metrics['mean_abs_error']:.1f}")

    # Write to a temp file and rename, so a killed run never leaves a partial probe.
    out = os.path.join(args.folder, f"position_probe_layer{args.layer}.pt")
    torch.save(dict(probe=probe.state_dict(), mean=mean, std=std, metrics=metrics,
                    args=vars(args)), out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"saved {out} ({len(train)} train / {len(test)} test samples)")


if __name__ == "__main__":
    main()
