import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio


_MEL = None
_DB = None
_RESAMPLERS = {}
_CFG = {}


def _build_transforms(cfg):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg["sample_rate"], n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"], n_mels=cfg["n_mels"],
        f_min=cfg["f_min"], f_max=cfg["f_max"], power=2.0,
    )
    db = torchaudio.transforms.AmplitudeToDB(
        stype="power", top_db=cfg["top_db"])
    return mel, db


def _init_worker(cfg):
    global _MEL, _DB, _CFG
    torch.set_num_threads(1)
    _CFG = cfg
    _MEL, _DB = _build_transforms(cfg)


def _load_audio(path, sr):
    wav, file_sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if file_sr != sr:
        key = (file_sr, sr)
        if key not in _RESAMPLERS:
            _RESAMPLERS[key] = torchaudio.transforms.Resample(file_sr, sr)
        wav = _RESAMPLERS[key](wav)
    return wav.squeeze(0)


def _pad_or_trim(spec, frames):
    _, t = spec.shape
    if t == frames:
        return spec
    if t < frames:
        return np.pad(spec, ((0, 0), (0, frames - t)))
    start = (t - frames) // 2
    return spec[:, start:start + frames]


@torch.no_grad()
def _extract_mel(path):
    wav = _load_audio(path, _CFG["sample_rate"])
    mel = _DB(_MEL(wav))
    arr = _pad_or_trim(mel.numpy(), _CFG["frames"])
    return np.ascontiguousarray(arr, dtype=np.float32)


def _extract_one(item):
    idx, path = item
    return idx, _extract_mel(path)


def _wav_list(root, split):
    wavs = sorted(root.glob(f"{split}/*/*.wav"))
    if not wavs:
        raise RuntimeError(f"{split}: no .wav files found under {root}")
    return wavs


def pack_split(root, out_root, split, class_to_idx, cfg, workers):
    wavs = _wav_list(root, split)
    unknown = sorted({p.parent.name for p in wavs} - set(class_to_idx))
    if unknown:
        raise RuntimeError(
            f"{split}: classes missing from train split: {unknown}")

    rel_paths = [str(p.relative_to(root)) for p in wavs]
    labels = [class_to_idx[p.parent.name] for p in wavs]
    n = len(wavs)

    print(f"[{split}] {n} wavs, {len(set(labels))} classes, "
          f"extracting with {workers} workers ...", flush=True)
    t0 = time.time()
    arr = np.empty((n, cfg["n_mels"], cfg["frames"]), dtype=np.float32)

    tasks = [(i, str(p)) for i, p in enumerate(wavs)]
    done = 0
    with mp.Pool(processes=workers, initializer=_init_worker,
                 initargs=(cfg,)) as pool:
        for idx, mel in pool.imap_unordered(_extract_one, tasks, chunksize=32):
            arr[idx] = mel
            done += 1
            if done % 1000 == 0 or done == n:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"[{split}] {done}/{n} ({rate:.0f}/s)", flush=True)

    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    bin_path = split_dir / "mel.bin"
    arr.tofile(bin_path)

    meta = {
        "dataset": root.name,
        "split": split,
        "shape": [cfg["n_mels"], cfg["frames"]],
        "n": n,
        "rel_paths": rel_paths,
        "labels": labels,
        "class_to_idx": class_to_idx,
        "mel_config": dict(cfg),
        "source_name": root.name,
    }
    (split_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[{split}] wrote {bin_path} "
          f"({bin_path.stat().st_size / 2**30:.2f} GiB) "
          f"in {time.time() - t0:.1f}s", flush=True)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="wav root containing <split>/<class>/*.wav")
    p.add_argument("--out", required=True,
                   help="output directory; writes <out>/<split>/")
    p.add_argument("--splits", default="train,test")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--f-min", type=float, default=0.0)
    p.add_argument("--f-max", type=float, default=8000.0)
    p.add_argument("--top-db", type=float, default=80.0)
    p.add_argument("--frames", type=int, default=157)
    return p


def main():
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if not (root / splits[0]).is_dir():
        raise FileNotFoundError(f"missing {root / splits[0]}")

    class_names = sorted(
        d.name for d in (root / splits[0]).iterdir() if d.is_dir())
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    cfg = {
        "sample_rate": args.sample_rate,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "f_min": args.f_min,
        "f_max": args.f_max,
        "top_db": args.top_db,
        "frames": args.frames,
        "scale": "dB",
    }

    for split in splits:
        pack_split(root, out_root, split, class_to_idx, cfg, args.workers)


if __name__ == "__main__":
    main()
