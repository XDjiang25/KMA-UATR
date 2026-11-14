import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (PackedMelDataset, ordered_class_names,
                     stratified_train_val_split)
from model import PaperKMA, default_description


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--packed-root", default=None,
                   help="directory containing train/ and test/")
    p.add_argument("--train-dir", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--gpt2-path", default="gpt2")
    p.add_argument("--gpt2-local-files-only", action="store_true")
    p.add_argument("--text-layers", type=int, default=2)
    p.add_argument("--description", default=None)
    p.add_argument("--ckpt-dir", default="runs/ckpts")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_data(train_dir, batch_size, val_ratio, seed, num_workers):
    full = PackedMelDataset(train_dir, load_data=False)
    train_paths, val_paths = stratified_train_val_split(
        full.rel_paths, full.labels(), val_ratio=val_ratio, seed=seed)

    train_set = PackedMelDataset(
        train_dir, rel_paths=train_paths, standardize=True)
    val_set = PackedMelDataset(
        train_dir, rel_paths=val_paths, data=train_set.X)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, full


def train_epoch(model, loader, criterion, optimizer, device, max_batches=0):
    model.train()
    losses, correct, total = [], 0, 0
    for i, (feats, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return sum(losses) / len(losses), correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, max_batches=0):
    model.eval()
    losses, correct, total = [], 0, 0
    for i, (feats, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        losses.append(criterion(logits, labels).item())
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return sum(losses) / len(losses), correct / max(total, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    if device.type == "cuda":
        index = (device.index if device.index is not None
                 else torch.cuda.current_device())
        free, total = torch.cuda.mem_get_info(index)
        print(f"device=cuda:{index} "
              f"({torch.cuda.get_device_name(index)}) "
              f"free={free / 2**30:.1f}/{total / 2**30:.1f} GiB",
              flush=True)

    packed_root = Path(args.packed_root) if args.packed_root else None
    train_dir = Path(args.train_dir) if args.train_dir else (
        packed_root / "train" if packed_root else None)
    if train_dir is None:
        raise SystemExit("provide --packed-root or --train-dir")
    name = args.name or (
        packed_root.name if packed_root else train_dir.parent.name)

    train_loader, val_loader, full_train = load_data(
        train_dir, args.batch_size, args.val_ratio, args.seed,
        args.num_workers)
    class_names = ordered_class_names(full_train.meta)
    num_classes = len(class_names)
    description = args.description or default_description(class_names)

    print(f"dataset={name} seed={args.seed} | "
          f"train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} | classes={num_classes}",
          flush=True)
    print(f"train_dir={train_dir}", flush=True)

    model = PaperKMA(
        classes=num_classes,
        description=description,
        gpt2_path=args.gpt2_path,
        local_files_only=args.gpt2_local_files_only,
        text_layers=args.text_layers,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"params total={total_params:,} trainable={trainable_params:,}",
          flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10)

    tag = f"{name}_s{args.seed}"
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{tag}_best.pt"

    history = []
    best_val_acc, best_epoch = -1.0, -1
    best_state = None
    no_improve = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            args.max_train_batches)
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, args.max_val_batches)
        scheduler.step(val_acc)
        dt = time.time() - t0

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        history.append({
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "sec": dt,
        })
        print(f"epoch {epoch:3d} | lr {scheduler.get_last_lr()[0]:.2e} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"{dt:.1f}s{'  *best' if improved else ''}", flush=True)

        if args.patience and no_improve >= args.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("no checkpoint saved")
    train_minutes = (time.time() - t_start) / 60.0

    torch.save({
        "model_state_dict": best_state,
        "dataset": name,
        "seed": args.seed,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "num_classes": num_classes,
        "class_to_idx": full_train.class_to_idx,
        "class_names": class_names,
        "description": description,
        "gpt2_path": args.gpt2_path,
        "text_layers": args.text_layers,
        "val_split": {"train": len(train_loader.dataset),
                      "val": len(val_loader.dataset)},
        "args": vars(args),
        "history": history,
        "train_minutes": train_minutes,
    }, ckpt_path)

    print(f"training done: best val acc={best_val_acc:.4f} "
          f"@epoch {best_epoch}, {len(history)} epochs, "
          f"{train_minutes:.1f} min", flush=True)
    print(f"saved ckpt -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
