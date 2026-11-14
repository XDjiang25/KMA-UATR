import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_meta(split_dir):
    meta_path = Path(split_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}")
    return json.loads(meta_path.read_text())


def ordered_class_names(meta):
    return [name for name, _ in sorted(meta["class_to_idx"].items(),
                                       key=lambda item: item[1])]


def stratified_train_val_split(rel_paths, labels, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    by_class = {}
    for path, label in zip(rel_paths, labels):
        by_class.setdefault(int(label), []).append(path)

    train_paths, val_paths = [], []
    for label in sorted(by_class):
        paths = list(by_class[label])
        rng.shuffle(paths)
        n_val = max(1, round(len(paths) * val_ratio))
        n_val = min(n_val, len(paths) - 1) if len(paths) > 1 else 0
        val_paths.extend(paths[:n_val])
        train_paths.extend(paths[n_val:])

    rng.shuffle(train_paths)
    rng.shuffle(val_paths)
    return train_paths, val_paths


class PackedMelDataset(Dataset):
    def __init__(self, split_dir, rel_paths=None, standardize=True,
                 data=None, load_data=True):
        split_dir = Path(split_dir)
        self.meta = load_meta(split_dir)
        self.shape = tuple(self.meta["shape"])
        n = int(self.meta["n"])

        if data is not None:
            self.X = data
        elif not load_data:
            self.X = None
        else:
            self.X = np.fromfile(split_dir / "mel.bin",
                                 dtype=np.float32).reshape(n, *self.shape)
            if standardize:
                mean = self.X.mean(axis=(1, 2), keepdims=True)
                std = self.X.std(axis=(1, 2), keepdims=True) + 1e-6
                self.X = ((self.X - mean) / std).astype(
                    np.float32, copy=False)

        self.y = np.asarray(self.meta["labels"], dtype=np.int64)
        self.class_to_idx = dict(self.meta["class_to_idx"])
        self.class_names = ordered_class_names(self.meta)
        self.rel_paths_all = [Path(p) for p in self.meta["rel_paths"]]

        if rel_paths is None:
            self.rel_paths = self.rel_paths_all
            self._sel = np.arange(n, dtype=np.int64)
        else:
            lookup = {str(p): i for i, p in enumerate(self.rel_paths_all)}
            missing = [str(p) for p in rel_paths if str(p) not in lookup]
            if missing:
                raise KeyError(f"{len(missing)} paths not in packed file, "
                               f"e.g. {missing[:3]}")
            self.rel_paths = [Path(p) for p in rel_paths]
            self._sel = np.asarray([lookup[str(p)] for p in rel_paths],
                                   dtype=np.int64)

    def __len__(self):
        return len(self._sel)

    def labels(self):
        return [int(v) for v in self.y[self._sel]]

    def __getitem__(self, idx):
        j = self._sel[idx]
        return torch.from_numpy(self.X[j]).unsqueeze(0), int(self.y[j])
