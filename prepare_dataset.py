"""
Generate a manifest CSV + params JSON for HCP resting-state fMRI training,
and optionally download + preprocess sessions from S3.

Usage — manifest only:
  python prepare_dataset.py --subjects 100307 --sessions REST1_LR --out data/manifest.csv

Usage — manifest + download + preprocess:
  python prepare_dataset.py --subjects 100307 --sessions REST1_LR --out data/manifest.csv \\
      --download --cache_dir data/
"""

import argparse
import csv
import json
import random
from datetime import date
from pathlib import Path

from preprocess import run as preprocess_run


HCP_SESSIONS = ["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"]
S3_TEMPLATE  = (
    "s3://hcp-openaccess/HCP_1200/{subject}/MNINonLinear/Results/"
    "rfMRI_{session}/rfMRI_{session}_Atlas_MSMAll.dtseries.nii"
)
HCP_N_TIMEPOINTS = 1200


def parse_args():
    p = argparse.ArgumentParser()

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--subjects",     nargs="+", help="Subject IDs (e.g. 100307 100408)")
    g.add_argument("--subject_list", help="Text file with one subject ID per line")

    p.add_argument("--n_subjects",   type=int, default=None,
                   help="Randomly sample N subjects")
    p.add_argument("--sessions",     nargs="+", default=HCP_SESSIONS, choices=HCP_SESSIONS)
    p.add_argument("--n_timepoints", type=int, default=HCP_N_TIMEPOINTS)

    p.add_argument("--val_frac",     type=float, default=0.1)
    p.add_argument("--val_temporal", action="store_true",
                   help="Temporal split within sessions. Auto-enabled for single subject.")
    p.add_argument("--seed",         type=int, default=42)

    p.add_argument("--tr",           type=float, default=0.72)
    p.add_argument("--lowcut",       type=float, default=0.01)
    p.add_argument("--highcut",      type=float, default=0.1)
    p.add_argument("--filter_order", type=int,   default=4)

    p.add_argument("--out",          default="data/manifest.csv")

    p.add_argument("--download",     action="store_true",
                   help="Download and preprocess all sessions after writing the manifest")
    p.add_argument("--cache_dir",    default="data/",
                   help="Directory for preprocessed npy files (used with --download)")

    return p.parse_args()


def _download_s3(s3_uri, local_path):
    try:
        import boto3
    except ImportError:
        raise ImportError("pip install boto3")
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_uri = s3_uri.removeprefix("s3://")
    bucket, key = s3_uri.split("/", 1)
    print(f"  S3 → {local_path.name}")
    boto3.client("s3").download_file(bucket, key, str(local_path))


def _preprocess_session(s3_path, subj, sess, cache_dir):
    cache_dir = Path(cache_dir)
    stem      = f"{subj}_{sess}"
    parc_path = cache_dir / f"{stem}_p.npy"
    raw_path  = cache_dir / f"{stem}_raw.npy"

    if parc_path.exists() and raw_path.exists():
        print(f"  {stem}: already preprocessed, skipping")
        return

    cifti_path = cache_dir / f"{stem}.dtseries.nii"
    if not cifti_path.exists():
        _download_s3(s3_path, cifti_path)

    print(f"  Preprocessing {stem}...")
    preprocess_run(str(cifti_path), str(parc_path))
    cifti_path.unlink()
    print(f"  Deleted {cifti_path.name}")


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.subjects:
        subjects = list(args.subjects)
    else:
        with open(args.subject_list) as f:
            subjects = [l.strip() for l in f if l.strip()]

    if args.n_subjects:
        subjects = random.sample(subjects, min(args.n_subjects, len(subjects)))

    if len(subjects) == 1:
        args.val_temporal = True

    if args.val_temporal:
        val_subjects = set()
        val_start    = int(args.n_timepoints * (1 - args.val_frac))
    else:
        n_val        = max(1, round(len(subjects) * args.val_frac))
        val_subjects = set(random.sample(subjects, n_val))
        val_start    = None

    rows = []
    for subj in subjects:
        for sess in args.sessions:
            s3 = S3_TEMPLATE.format(subject=subj, session=sess)
            if args.val_temporal:
                rows.append(dict(subject_id=subj, session=sess, s3_path=s3,
                                 tp_start=0,         tp_end=val_start,         split="train"))
                rows.append(dict(subject_id=subj, session=sess, s3_path=s3,
                                 tp_start=val_start, tp_end=args.n_timepoints, split="val"))
            else:
                split = "val" if subj in val_subjects else "train"
                rows.append(dict(subject_id=subj, session=sess, s3_path=s3,
                                 tp_start=0, tp_end=args.n_timepoints, split=split))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id","session","s3_path",
                                                "tp_start","tp_end","split"])
        writer.writeheader()
        writer.writerows(rows)

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows   = [r for r in rows if r["split"] == "val"]
    params = {
        "created":         str(date.today()),
        "seed":            args.seed,
        "val_frac":        args.val_frac,
        "val_mode":        "temporal" if args.val_temporal else "subject",
        "tr":              args.tr,
        "lowcut":          args.lowcut,
        "highcut":         args.highcut,
        "filter_order":    args.filter_order,
        "n_subjects":      len(subjects),
        "sessions":        args.sessions,
        "n_train_rows":    len(train_rows),
        "n_val_rows":      len(val_rows),
        "total_train_tps": sum(r["tp_end"] - r["tp_start"] for r in train_rows),
        "total_val_tps":   sum(r["tp_end"] - r["tp_start"] for r in val_rows),
    }
    params_path = out.with_name(out.stem + "_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Manifest : {out}  ({len(rows)} rows)")
    print(f"Params   : {params_path}")
    print(f"Subjects : {len(subjects)}  |  Sessions: {args.sessions}")
    print(f"Train    : {len(train_rows)} rows, {params['total_train_tps']} TPs")
    print(f"Val      : {len(val_rows)} rows, {params['total_val_tps']} TPs")

    if args.download:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        seen = set()
        for row in rows:
            key = (row["subject_id"], row["session"])
            if key not in seen:
                seen.add(key)
                _preprocess_session(row["s3_path"], row["subject_id"], row["session"], cache_dir)


if __name__ == "__main__":
    main()
