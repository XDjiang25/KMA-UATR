import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader

from dataset import PackedMelDataset
from model import PaperKMA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--packed-root", default=None,
                   help="directory containing train/ and test/")
    p.add_argument("--test-dir", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--gpt2-path", default=None)
    p.add_argument("--gpt2-local-files-only", action="store_true")
    p.add_argument("--result-dir", default="runs/results")
    return p.parse_args()


def get_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_test_data(test_dir, batch_size, num_workers):
    test_set = PackedMelDataset(test_dir, standardize=True)
    loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return loader, test_set


def init_model(ckpt_path, device, gpt2_path=None, local_files_only=False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    local_files_only = local_files_only or bool(
        ckpt.get("args", {}).get("gpt2_local_files_only", False))
    model = PaperKMA(
        classes=ckpt["num_classes"],
        description=ckpt["description"],
        gpt2_path=gpt2_path or ckpt["gpt2_path"],
        local_files_only=local_files_only,
        text_layers=ckpt.get("text_layers", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    preds, true = [], []
    for feats, labels in loader:
        logits = model(feats.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        true.append(labels)
    return torch.cat(true).numpy(), torch.cat(preds).numpy()


def save_confusion_matrix(cm, class_names, path):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.75), max(4, n * 0.65)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
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
    test_dir = Path(args.test_dir) if args.test_dir else (
        packed_root / "test" if packed_root else None)
    if test_dir is None:
        raise SystemExit("provide --packed-root or --test-dir")

    loader, test_set = load_test_data(
        test_dir, args.batch_size, args.num_workers)
    model, ckpt = init_model(args.ckpt, device, args.gpt2_path,
                             args.gpt2_local_files_only)

    class_names = list(ckpt["class_names"])
    num_classes = int(ckpt["num_classes"])
    name = args.name or ckpt["dataset"]
    seed = int(ckpt["seed"])
    tag = f"{name}_s{seed}"

    true, pred = infer(model, loader, device)
    acc = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(true, pred, average="weighted", zero_division=0)
    precision = precision_score(true, pred, average="macro", zero_division=0)
    recall = recall_score(true, pred, average="macro", zero_division=0)
    labels = list(range(num_classes))
    cm = confusion_matrix(true, pred, labels=labels)
    report = classification_report(true, pred, labels=labels,
                                   target_names=class_names,
                                   output_dict=True, zero_division=0)

    print("\n=== TEST RESULT ===", flush=True)
    print(f"accuracy    = {acc:.4f}", flush=True)
    print(f"macro-F1    = {macro_f1:.4f}", flush=True)
    print(f"weighted-F1 = {weighted_f1:.4f}", flush=True)
    print(f"macro-P     = {precision:.4f}", flush=True)
    print(f"macro-R     = {recall:.4f}", flush=True)
    print(classification_report(true, pred, labels=labels,
                                target_names=class_names, digits=4,
                                zero_division=0), flush=True)
    print("confusion matrix (rows=true, cols=pred):", flush=True)
    print(cm, flush=True)

    history = ckpt.get("history", [])
    result = {
        "dataset": name,
        "seed": seed,
        "run_tag": tag,
        "model": "PaperKMA",
        "description": ckpt["description"],
        "test_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "per_class": {
            class_name: {
                "precision": float(report[class_name]["precision"]),
                "recall": float(report[class_name]["recall"]),
                "f1": float(report[class_name]["f1-score"]),
                "support": int(report[class_name]["support"]),
            }
            for class_name in class_names
        },
        "confusion_matrix": cm.tolist(),
        "class_to_idx": dict(ckpt["class_to_idx"]),
        "best_val_acc": float(ckpt["best_val_acc"]),
        "best_epoch": int(ckpt["best_epoch"]),
        "epochs_run": int(ckpt.get("epochs_run", len(history))),
        "train_minutes": float(ckpt.get(
            "train_minutes", sum(e["sec"] for e in history) / 60)),
        "params_total": int(sum(p.numel() for p in model.parameters())),
        "val_split": ckpt["val_split"],
        "test_size": len(test_set),
        "args": vars(args),
    }

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{tag}_test.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    pd.DataFrame(report).transpose().to_csv(
        result_dir / f"{tag}_classification_report.csv")
    save_confusion_matrix(cm, class_names, result_dir / f"{tag}_cm.png")

    print(f"\nsaved result -> {result_path}", flush=True)


if __name__ == "__main__":
    main()
