"""Train a MaskedTransformer on RULER synthetic data or plain text.

Data formats
------------
* ``--data_format ruler``  (recommended for RULER experiments)
    Reads JSONL files produced by ``data/prepare.py``.  Each sample has a
    long ``input`` (encoder source) and short ``outputs`` (decoder target).
    Pass ``--data_dir`` pointing to the root that contains the generated
    ``*/data/*/validation.jsonl`` tree.

* ``--data_format text``
    Language-modelling on a HuggingFace dataset (e.g. WikiText-103).
    Source = token chunk, target = next chunk.

Usage
-----
    # Train on RULER data
    python scripts/tmodel/train.py \\
        --data_format ruler --data_dir experiments/train_data \\
        --pe_type rope --encoder_mask B --decoder_mask C \\
        --output_dir experiments/pe_rope_encB_decC

    # Train on WikiText
    python scripts/tmodel/train.py \\
        --data_format text --dataset wikitext --dataset_subset wikitext-103-raw-v1 \\
        --pe_type sinusoidal --output_dir experiments/pe_sin_lm

Dependencies: torch, transformers   (+ datasets for --data_format text)
"""

import argparse
import glob
import json
import logging
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Resolve package path so `from tmodel import …` works regardless of cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from tmodel import MaskedTransformer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------

class RulerDataset(Dataset):
    """Load RULER JSONL data for supervised seq2seq training.

    Walks *data_dir* for every ``.jsonl`` file, reads ``input`` / ``outputs``
    pairs, and tokenizes them on the fly.

    The source string is ``input`` **plus** ``answer_prefix``. The task generators
    split the prefix out of ``input`` into its own field, and the prediction path
    concatenates it back on before calling the model (see ``pred/call_api.py``).
    Reading ``input`` alone here would train the model on a prompt that never occurs
    at evaluation time.
    """

    def __init__(self, data_dir, tokenizer, max_src_len=2048, max_tgt_len=128):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.samples = []

        n_missing_prefix = 0
        pattern = os.path.join(data_dir, "**", "*.jsonl")
        for fpath in sorted(glob.glob(pattern, recursive=True)):
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    outputs = item.get("outputs", [])
                    if outputs and outputs[0]:
                        prefix = item.get("answer_prefix", "")
                        if not prefix:
                            n_missing_prefix += 1
                        self.samples.append((item["input"] + prefix, outputs[0]))

        log.info("RulerDataset: loaded %d samples from %s", len(self.samples), data_dir)
        if n_missing_prefix:
            log.warning(
                "RulerDataset: %d samples had no answer_prefix field; those prompts will "
                "not match the ones used at prediction time.", n_missing_prefix,
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_text, tgt_text = self.samples[idx]
        src_ids = self.tokenizer.encode(src_text, truncation=True,
                                        max_length=self.max_src_len)
        # Reserve two slots for the BOS/EOS wrapper below.
        tgt_ids = self.tokenizer.encode(tgt_text, truncation=True,
                                        max_length=max(1, self.max_tgt_len - 2))
        # Wrap the target so the shift in the training loop teaches the model both to
        # start from BOS (which is what generation feeds it) and to emit EOS (which is
        # what tells generation to stop).
        tgt_ids = [self.bos_id] + tgt_ids + [self.eos_id]
        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_ids, dtype=torch.long)


def ruler_collate(batch, pad_id=0):
    """Pad variable-length (src, tgt) pairs to the batch maximum."""
    src_list, tgt_list = zip(*batch)
    max_src = max(s.size(0) for s in src_list)
    max_tgt = max(t.size(0) for t in tgt_list)

    src = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        src[i, : s.size(0)] = s
        tgt[i, : t.size(0)] = t
    return src, tgt


class ChunkedTextDataset(Dataset):
    """Slice a flat token-id list into fixed-length (src, tgt) pairs."""

    def __init__(self, token_ids, src_len: int, tgt_len: int, stride: int = 0):
        self.token_ids = token_ids
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.stride = stride or src_len
        self.n_samples = max(0, (len(token_ids) - src_len - tgt_len) // self.stride)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s = idx * self.stride
        src = self.token_ids[s : s + self.src_len]
        tgt = self.token_ids[s + self.src_len : s + self.src_len + self.tgt_len]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def tokenize_split(tokenizer, dataset_name, subset, split, max_tokens=None):
    """Download *split*, tokenize every non-empty line, return flat id list."""
    from datasets import load_dataset

    log.info("Loading %s/%s [%s] ...", dataset_name, subset, split)
    ds = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
    ids = []
    for row in ds:
        text = row.get("text", "")
        if text.strip():
            ids.extend(tokenizer.encode(text))
            if max_tokens and len(ids) >= max_tokens:
                ids = ids[:max_tokens]
                break
    log.info("  -> %s tokens", f"{len(ids):,}")
    return ids


# ------------------------------------------------------------------
# LR schedule
# ------------------------------------------------------------------

def cosine_with_warmup(optimizer, warmup: int, total: int):
    def _lr(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)


# ------------------------------------------------------------------
# Training & evaluation loops
# ------------------------------------------------------------------

def run_epoch(model, loader, criterion, vocab_size, device, optimizer=None,
              scheduler=None, scaler=None, grad_clip=1.0, log_every=100, lr=0,
              accum_steps=1):
    """Run one training or validation epoch.  Pass *optimizer=None* for eval.

    ``accum_steps`` batches are accumulated before each optimizer step, so the
    effective batch size is ``batch_size * accum_steps`` at the memory cost of one
    batch. The learning-rate scheduler advances once per optimizer step, not once
    per batch.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    use_amp = scaler is not None
    n_batches = len(loader)

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, (src, tgt) in enumerate(loader, 1):
        src, tgt = src.to(device), tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # A fresh autocast context per batch; reusing one instance across the loop
        # relies on re-entrancy that is not guaranteed.
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))

        n_tok = (tgt_out != criterion.ignore_index).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

        if is_train:
            # Scale down so accumulated gradients average rather than sum.
            scaled_loss = loss / accum_steps
            if scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            is_last_batch = step == n_batches
            if step % accum_steps == 0 or is_last_batch:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()

            if step % log_every == 0:
                avg = total_loss / max(total_tokens, 1)
                ppl = math.exp(min(avg, 20))
                cur_lr = scheduler.get_last_lr()[0] if scheduler else lr
                log.info("  step %5d/%d | loss %.4f | ppl %7.1f | lr %.2e",
                         step, n_batches, avg, ppl, cur_lr)

    return total_loss / max(total_tokens, 1)


# ------------------------------------------------------------------
# Data builder
# ------------------------------------------------------------------

def build_dataloaders(args, tokenizer, pad_id):
    """Return (train_loader, val_loader) based on ``args.data_format``."""
    if args.data_format == "ruler":
        train_ds = RulerDataset(args.data_dir, tokenizer,
                                max_src_len=args.src_len, max_tgt_len=args.tgt_len)
        # Use 10 % of samples as validation (deterministic split)
        n_val = max(1, len(train_ds) // 10)
        n_train = len(train_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        collate = lambda batch: ruler_collate(batch, pad_id)  # noqa: E731
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True,
                                  collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                num_workers=args.workers, pin_memory=True,
                                collate_fn=collate)
    else:  # text
        train_ids = tokenize_split(tokenizer, args.dataset, args.dataset_subset,
                                   "train", args.max_train_tokens)
        val_ids = tokenize_split(tokenizer, args.dataset, args.dataset_subset,
                                 "validation", args.max_train_tokens // 10)
        train_ds = ChunkedTextDataset(train_ids, args.src_len, args.tgt_len)
        val_ds = ChunkedTextDataset(val_ids, args.src_len, args.tgt_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                num_workers=args.workers, pin_memory=True)

    log.info("Samples: train=%d  val=%d", len(train_ds), len(val_ds))
    return train_loader, val_loader


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ---- tokenizer ------------------------------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # GPT-2 ships no pad token. Aliasing pad to EOS (the previous behaviour) makes the
    # pad id, the BOS id and the EOS id all the same token -- and since the loss ignores
    # the pad id, the model could never be taught to emit EOS and so never learned to
    # stop. Add a dedicated pad token instead.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    # NOTE: len(tokenizer) counts added special tokens; tokenizer.vocab_size does not.
    # The embedding must be sized from the former or the new pad id is out of range.
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id == tokenizer.eos_token_id:
        raise ValueError(
            "pad_token_id must differ from eos_token_id, otherwise the loss ignores "
            "every EOS and the model cannot learn to stop generating."
        )
    log.info(
        "Tokenizer: vocab=%d pad=%d bos=%d eos=%d",
        vocab_size, pad_id, tokenizer.bos_token_id, tokenizer.eos_token_id,
    )

    # ---- model ----------------------------------------------------
    model = MaskedTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        pe_type=args.pe_type,
        encoder_mask_type=args.encoder_mask,
        decoder_mask_type=args.decoder_mask,
        pad_token_id=pad_id,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("Model: %.1fM params | pe=%s enc_mask=%s dec_mask=%s",
             n_params, args.pe_type, args.encoder_mask, args.decoder_mask)

    # ---- data -----------------------------------------------------
    train_loader, val_loader = build_dataloaders(args, tokenizer, pad_id)

    # ---- optimiser ------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.98))
    # The scheduler counts optimizer steps, not batches, so accumulation divides it.
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(args.warmup_steps, total_steps // 10)
    log.info(
        "Schedule: %d batches/epoch | accum %d -> %d steps/epoch | %d total | warmup %d "
        "| effective batch %d",
        len(train_loader), args.grad_accum, steps_per_epoch, total_steps, warmup,
        args.batch_size * args.grad_accum,
    )
    scheduler = cosine_with_warmup(optimizer, warmup, total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- output dir -----------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    # Save the tokenizer beside the checkpoint. It carries the added pad token, so
    # rebuilding it from the bare model name at inference would give a different
    # vocabulary size and a different pad id.
    tokenizer.save_pretrained(args.output_dir)

    # Recorded in every checkpoint so the prediction path can rebuild the model exactly
    # rather than re-deriving these from a tokenizer it constructed independently.
    token_config = dict(
        vocab_size=vocab_size,
        pad_token_id=pad_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # ---- train ----------------------------------------------------
    best_val = float("inf")
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, vocab_size, device,
                               optimizer=optimizer, scheduler=scheduler,
                               scaler=scaler if use_amp else None,
                               grad_clip=args.grad_clip, log_every=args.log_every,
                               lr=args.lr, accum_steps=args.grad_accum)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion, vocab_size, device)
        elapsed = time.time() - t0

        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))
        log.info("Epoch %d/%d | train %.4f (ppl %.1f) | val %.4f (ppl %.1f) | %.0fs",
                 epoch, args.epochs, train_loss, train_ppl, val_loss, val_ppl, elapsed)

        # checkpoint
        ckpt = dict(epoch=epoch, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), val_loss=val_loss,
                    args=vars(args), **token_config)
        torch.save(ckpt, os.path.join(args.output_dir, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
            log.info("  * new best checkpoint (val_loss=%.4f)", val_loss)

        with open(metrics_path, "a") as f:
            f.write(json.dumps(dict(epoch=epoch, train_loss=train_loss,
                                    train_ppl=train_ppl, val_loss=val_loss,
                                    val_ppl=val_ppl)) + "\n")

    log.info("Done. Best val_loss=%.4f  Saved to %s", best_val, args.output_dir)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train MaskedTransformer")

    # Model architecture
    g = p.add_argument_group("model")
    g.add_argument("--pe_type", default="sinusoidal",
                   choices=["none", "sinusoidal", "learned", "rope", "alibi"])
    g.add_argument("--encoder_mask", default="B", choices=["B", "C", "F"])
    g.add_argument("--decoder_mask", default="C", choices=["B", "C", "F"])
    g.add_argument("--d_model", type=int, default=512)
    g.add_argument("--num_heads", type=int, default=8)
    g.add_argument("--num_layers", type=int, default=8)
    g.add_argument("--d_ff", type=int, default=2048)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--max_len", type=int, default=8192)

    # Data
    g = p.add_argument_group("data")
    g.add_argument("--data_format", default="ruler", choices=["ruler", "text"],
                   help="'ruler' = JSONL from data/prepare.py; 'text' = HF dataset")
    g.add_argument("--data_dir", default=None,
                   help="Root dir with RULER JSONL files (for --data_format ruler)")
    g.add_argument("--tokenizer", default="gpt2")
    g.add_argument("--dataset", default="wikitext",
                   help="HuggingFace dataset name (for --data_format text)")
    g.add_argument("--dataset_subset", default="wikitext-103-raw-v1")
    g.add_argument("--src_len", type=int, default=2048,
                   help="Max encoder input length (tokens)")
    g.add_argument("--tgt_len", type=int, default=128,
                   help="Max decoder target length (tokens)")
    g.add_argument("--max_train_tokens", type=int, default=50_000_000,
                   help="Cap on training tokens (for --data_format text)")

    # Training
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--batch_size", type=int, default=8)
    g.add_argument("--grad_accum", type=int, default=1,
                   help="Batches to accumulate before each optimizer step. Effective "
                        "batch size is batch_size * grad_accum.")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--weight_decay", type=float, default=0.01)
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--warmup_steps", type=int, default=1000)
    g.add_argument("--fp16", action="store_true")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--log_every", type=int, default=100)

    # Output
    p.add_argument("--output_dir", required=True)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
