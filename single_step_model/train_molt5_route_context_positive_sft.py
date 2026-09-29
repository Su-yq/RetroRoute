#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Continue SFT from route-context SFT checkpoint using:
  gold samples + U-guided positive pseudo samples.

This is positive-only training:
  no rejected samples
  no DPO loss
  no preference objective

Save every epoch.
Use eval_molt5_topk.py separately to evaluate valid/test exact@k.
"""

import argparse
import gc
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

try:
    from rdkit import Chem
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False
    Chem = None


ATOM_MAP_RE = re.compile(r":\d+(?=\])")
TASK_PREFIX = "Please predict the reactant of the product:\n"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_atom_mapping(smi: str) -> str:
    if smi is None:
        return ""
    return ATOM_MAP_RE.sub("", str(smi).strip())


def canonicalize_smiles(smi: str) -> str:
    if smi is None:
        return ""

    smi = strip_atom_mapping(str(smi).strip().replace(" ", ""))

    if not smi:
        return ""

    if not HAS_RDKIT:
        return smi

    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return smi


def canonicalize_reactants(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip().replace(" ", "").rstrip(".")
    if not text:
        return ""

    parts = [p for p in text.split(".") if p.strip()]
    parts = [canonicalize_smiles(p) for p in parts]
    parts = [p for p in parts if p]

    if not parts:
        return ""

    return ".".join(sorted(parts))


def build_route_context_prompt(sample: Dict[str, Any], max_depth: int) -> str:
    target = canonicalize_smiles(sample.get("target_smiles", ""))
    current = canonicalize_smiles(
        sample.get("current_smiles")
        or sample.get("input_current")
        or ""
    )

    try:
        step_order = int(sample.get("step_order", 1))
    except Exception:
        step_order = 1

    current_depth = max(step_order - 1, 0)

    return (
        f"{TASK_PREFIX}"
        f"<target> {target}\n"
        f"<current> {current}\n"
        f"<current_depth> {current_depth}\n"
        f"<max_depth> {max_depth}\n"
        f"<goal> purchasable_starting_materials"
    )


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class RouteContextSFTDataset(Dataset):
    def __init__(self, path: str, max_depth: int, max_samples: int = -1):
        self.path = path
        self.max_depth = max_depth
        rows = load_json(Path(path))

        self.items = []

        for i, sample in enumerate(rows):
            if max_samples > 0 and i >= max_samples:
                break

            prompt = build_route_context_prompt(sample, max_depth=max_depth)
            target = canonicalize_reactants(sample.get("reactants_str", ""))

            if not prompt or not target:
                continue

            self.items.append({
                "id": sample.get("id"),
                "original_id": sample.get("original_id"),
                "target_source": sample.get("target_source", "unknown"),
                "prompt": prompt,
                "target": target,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_sft(batch, tokenizer, max_source_length: int, max_target_length: int):
    prompts = [x["prompt"] for x in batch]
    targets = [x["target"] for x in batch]

    src = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    )

    tgt = tokenizer(
        targets,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_target_length,
    )

    labels = tgt["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": src["input_ids"],
        "attention_mask": src["attention_mask"],
        "labels": labels,
        "metadata": batch,
    }


def label_smoothed_loss(model, batch, label_smoothing: float):
    labels = batch["labels"]

    decoder_input_ids = model._shift_right(labels)

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
        return_dict=True,
    )

    logits = outputs.logits

    log_probs = F.log_softmax(logits, dim=-1)

    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)

    nll = -torch.gather(
        log_probs,
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    smooth = -log_probs.mean(dim=-1)

    loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    loss = loss.masked_select(mask).mean()

    return loss


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate_loss(model, dataloader, device, fp16: bool, label_smoothing: float):
    model.eval()

    losses = []

    for batch in tqdm(dataloader, desc="valid CE loss"):
        batch = to_device(batch, device)

        if fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = label_smoothed_loss(model, batch, label_smoothing=label_smoothing)
        else:
            loss = label_smoothed_loss(model, batch, label_smoothing=label_smoothing)

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else None


def save_checkpoint(model, tokenizer, checkpoint_dir: Path):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init_model_dir",
        type=str,
        default="./molt5_route_context_sft_maxdepth14/checkpoint-best",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="./sft_u_positive_v1/train_u_positive_sft.json",
    )
    parser.add_argument(
        "--valid_file",
        type=str,
        default="./sft_u_positive_v1/valid_gold_sft.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./molt5_route_context_sft_u_positive_v1",
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4)

    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--max_depth", type=int, default=14)

    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_valid_samples", type=int, default=-1)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=100)

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(vars(args), output_dir / "train_config.json")

    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device.startswith("cuda")
        else "cpu"
    )

    print("===== Train MolT5 Route-Context U-positive SFT =====")
    print(f"init_model_dir   : {args.init_model_dir}")
    print(f"train_file       : {args.train_file}")
    print(f"valid_file       : {args.valid_file}")
    print(f"output_dir       : {args.output_dir}")
    print(f"epochs           : {args.epochs}")
    print(f"lr               : {args.lr}")
    print(f"batch_size       : {args.batch_size}")
    print(f"grad_accum_steps : {args.grad_accum_steps}")
    print(f"label_smoothing  : {args.label_smoothing}")
    print(f"device           : {device}")
    print(f"fp16             : {args.fp16}")

    tokenizer = T5Tokenizer.from_pretrained(
        args.init_model_dir,
        model_max_length=args.max_source_length,
    )

    model = T5ForConditionalGeneration.from_pretrained(args.init_model_dir)
    model.to(device)

    train_ds = RouteContextSFTDataset(
        args.train_file,
        max_depth=args.max_depth,
        max_samples=args.max_train_samples,
    )

    valid_ds = RouteContextSFTDataset(
        args.valid_file,
        max_depth=args.max_depth,
        max_samples=args.max_valid_samples,
    )

    print(f"train samples: {len(train_ds)}")
    print(f"valid samples: {len(valid_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_sft(
            b,
            tokenizer=tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        ),
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_sft(
            b,
            tokenizer=tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = args.epochs * num_update_steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    log_rows = []
    global_step = 0
    optimizer_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        losses = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"SFT epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(pbar, start=1):
            batch = to_device(batch, device)

            if args.fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = label_smoothed_loss(
                        model,
                        batch,
                        label_smoothing=args.label_smoothing,
                    )
                    loss_to_backward = loss / args.grad_accum_steps

                scaler.scale(loss_to_backward).backward()

            else:
                loss = label_smoothed_loss(
                    model,
                    batch,
                    label_smoothing=args.label_smoothing,
                )
                loss_to_backward = loss / args.grad_accum_steps
                loss_to_backward.backward()

            losses.append(float(loss.item()))

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                if args.fp16 and device.type == "cuda":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            global_step += 1

            if global_step % args.logging_steps == 0:
                pbar.set_postfix({
                    "loss": f"{np.mean(losses[-args.logging_steps:]):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        valid_loss = evaluate_loss(
            model,
            dataloader=valid_loader,
            device=device,
            fp16=args.fp16,
            label_smoothing=args.label_smoothing,
        )

        checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
        save_checkpoint(model, tokenizer, checkpoint_dir)

        row = {
            "epoch": epoch,
            "checkpoint_dir": str(checkpoint_dir),
            "train_loss": float(np.mean(losses)) if losses else None,
            "valid_loss": valid_loss,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "lr": scheduler.get_last_lr()[0],
        }

        log_rows.append(row)
        write_json(log_rows, output_dir / "training_log.json")

        print("\n===== Epoch Summary =====")
        print(json.dumps(row, ensure_ascii=False, indent=2))

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    save_checkpoint(model, tokenizer, output_dir / "checkpoint-last")

    print("\n===== Finished U-positive SFT =====")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()