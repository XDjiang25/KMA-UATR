import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def query_free_gpus():
    code = (
        "import json,torch;"
        "print(json.dumps([(i,torch.cuda.mem_get_info(i)[0]) "
        "for i in range(torch.cuda.device_count())]))"
    )
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", code], text=True,
            stderr=subprocess.DEVNULL, timeout=120)
        data = json.loads(out.strip().splitlines()[-1])
        return {int(i): int(free) for i, free in data}
    except Exception as exc:
        print(f"[launcher] GPU query failed ({exc})", flush=True)
        return {}


def pick_gpus(max_concurrent, min_free_gib):
    free = query_free_gpus()
    ok = [(gpu, nbytes) for gpu, nbytes in free.items()
          if nbytes >= min_free_gib * 1024**3]
    ok.sort(key=lambda item: -item[1])
    gpus = [gpu for gpu, _ in ok[:max_concurrent]]
    print("[launcher] free: "
          + ", ".join(f"cuda{gpu}={nbytes / 1024**3:.1f}GiB"
                      for gpu, nbytes in sorted(free.items())), flush=True)
    print(f"[launcher] selected {gpus}", flush=True)
    return gpus


def parse_datasets(values):
    datasets = []
    for value in values:
        if "=" not in value:
            raise SystemExit(
                f"--dataset expects NAME=PACKED_ROOT, got {value!r}")
        name, packed_root = value.split("=", 1)
        name, packed_root = name.strip(), packed_root.strip()
        if not name or not packed_root:
            raise SystemExit(f"invalid --dataset {value!r}")
        datasets.append((name, packed_root))
    return datasets


def build_jobs(datasets, seeds, args):
    jobs = []
    out_root = Path(args.out_root)
    result_dir = out_root / "results"
    ckpt_dir = out_root / "ckpts"
    log_dir = out_root / "logs"

    for name, packed_root in datasets:
        for seed in seeds:
            tag = f"{name}_s{seed}"
            result = result_dir / f"{tag}_test.json"
            ckpt = ckpt_dir / f"{tag}_best.pt"
            log = log_dir / f"{tag}.log"
            train_cmd = [
                sys.executable, "train.py",
                "--packed-root", packed_root,
                "--name", name,
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--patience", str(args.patience),
                "--val-ratio", str(args.val_ratio),
                "--num-workers", str(args.num_workers),
                "--gpt2-path", args.gpt2_path,
                "--text-layers", str(args.text_layers),
                "--ckpt-dir", str(ckpt_dir),
            ]
            test_cmd = [
                sys.executable, "test.py",
                "--ckpt", str(ckpt),
                "--packed-root", packed_root,
                "--name", name,
                "--num-workers", str(args.num_workers),
                "--result-dir", str(result_dir),
            ]
            if args.gpt2_local_files_only:
                train_cmd.append("--gpt2-local-files-only")
            jobs.append({"tag": tag, "cmds": [train_cmd, test_cmd],
                         "result": result, "log": log})
    return jobs


def launch(job, gpu, stage):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "4")

    job["log"].parent.mkdir(parents=True, exist_ok=True)
    fh = open(job["log"], "a" if stage else "w")
    if stage == 0:
        fh.write(f"# tag={job['tag']} gpu={gpu}\n")
    fh.write(f"# cmd={' '.join(job['cmds'][stage])}\n\n")
    fh.flush()
    proc = subprocess.Popen(job["cmds"][stage], stdout=fh,
                            stderr=subprocess.STDOUT, env=env,
                            cwd=str(ROOT), start_new_session=True)
    return proc, fh


def tail(path, n=15):
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return "<no log>"


def dump_status(path, pending, running, done, failed, skipped):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "pending": [job["tag"] for job in pending],
        "running": [{"tag": item["job"]["tag"], "gpu": item["gpu"],
                     "stage": item["stage"],
                     "min": round((time.time() - item["t0"]) / 60, 1)}
                    for item in running],
        "done": done,
        "failed": failed,
        "skipped": skipped,
    }, indent=2, ensure_ascii=False))


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", required=True,
                   metavar="NAME=PACKED_ROOT")
    p.add_argument("--seeds", required=True)
    p.add_argument("--max-concurrent", type=int, default=4)
    p.add_argument("--min-free-gib", type=float, default=15.0)
    p.add_argument("--gpus", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--gpt2-path", default="gpt2")
    p.add_argument("--gpt2-local-files-only", action="store_true")
    p.add_argument("--text-layers", type=int, default=2)
    p.add_argument("--out-root", default="runs")
    p.add_argument("--force", action="store_true")
    p.add_argument("--retries", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()

    datasets = parse_datasets(args.dataset)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    jobs = build_jobs(datasets, seeds, args)

    out_root = Path(args.out_root)
    for sub in ("logs", "results", "ckpts"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    gpus = ([int(x) for x in args.gpus.split(",")]
            if args.gpus else pick_gpus(args.max_concurrent,
                                        args.min_free_gib))
    gpus = gpus[:args.max_concurrent]
    if not gpus:
        raise SystemExit("no free GPU above --min-free-gib")

    pending, running, done, failed, skipped = [], [], [], [], []
    for job in jobs:
        if not args.force and job["result"].exists():
            skipped.append(job["tag"])
        else:
            pending.append(job)

    status = out_root / "logs" / "status.json"
    print(f"[launcher] {len(jobs)} jobs, concurrency={len(gpus)}, "
          f"skip={skipped}", flush=True)
    dump_status(status, pending, running, done, failed, skipped)

    attempts = {}
    while pending or running:
        while pending and len(running) < len(gpus):
            used = {item["gpu"] for item in running}
            available = [gpu for gpu in gpus if gpu not in used]
            if not available:
                break
            job = pending.pop(0)
            gpu = available[0]
            proc, fh = launch(job, gpu, 0)
            running.append({"job": job, "gpu": gpu, "proc": proc, "fh": fh,
                            "stage": 0, "t0": time.time()})
            attempts[job["tag"]] = attempts.get(job["tag"], 0) + 1
            print(f"[launcher] START {job['tag']} on cuda{gpu} "
                  f"-> {job['log']}", flush=True)
            dump_status(status, pending, running, done, failed, skipped)

        time.sleep(10)

        still = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                still.append(item)
                continue
            item["fh"].close()
            job = item["job"]
            if rc == 0 and item["stage"] + 1 < len(job["cmds"]):
                proc, fh = launch(job, item["gpu"], item["stage"] + 1)
                item.update(proc=proc, fh=fh, stage=item["stage"] + 1)
                still.append(item)
                continue

            minutes = (time.time() - item["t0"]) / 60
            tag = job["tag"]
            if rc == 0 and job["result"].exists():
                done.append(tag)
                print(f"[launcher] DONE {tag} ({minutes:.1f} min)",
                      flush=True)
            elif attempts[tag] <= args.retries:
                print(f"[launcher] RETRY {tag} rc={rc}", flush=True)
                pending.append(job)
            else:
                failed.append({"tag": tag, "rc": rc,
                               "min": round(minutes, 1)})
                print(f"[launcher] FAIL {tag} rc={rc} "
                      f"({minutes:.1f} min)\n{tail(job['log'])}", flush=True)
            dump_status(status, pending, running, done, failed, skipped)
        running = still

    dump_status(status, pending, running, done, failed, skipped)
    print(f"[launcher] ALL FINISHED done={done} failed={failed} "
          f"skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
