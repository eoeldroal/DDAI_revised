#!/usr/bin/env python3
"""Sample images and collect blank/near-solid suspects for manual review.

This does NOT delete anything. It only copies suspects into a review folder.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root, e.g. data/VDR_processed_filtered_1")
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--var-thr", type=float, default=300.0)
    parser.add_argument("--white-thr", type=float, default=0.90)
    parser.add_argument("--edge-thr", type=float, default=0.005)
    parser.add_argument("--bright-thr", type=int, default=245)
    parser.add_argument("--out-dir", default="filters/blank_like_sample")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    root = Path(args.root)
    idx_path = root / "corpus_index/OpenDocVQA.parquet"
    df = pd.read_parquet(idx_path, columns=["path"])

    paths = df["path"].tolist()
    random.seed(args.seed)
    sample_paths = random.sample(paths, min(args.sample, len(paths)))

    suspects = []
    for rel in sample_paths:
        img_path = root / rel
        try:
            with Image.open(img_path) as im:
                arr = np.array(im.convert("L"))
            h, w = arr.shape
            if h == 0 or w == 0:
                continue
            var = float(arr.var())
            white_ratio = float((arr > args.bright_thr).mean())
            gx = np.abs(np.diff(arr, axis=1))
            gy = np.abs(np.diff(arr, axis=0))
            edge_density = float(((gx > 20).mean() + (gy > 20).mean()) / 2.0)
            if (var < args.var_thr and white_ratio > args.white_thr) or (
                edge_density < args.edge_thr and white_ratio > 0.96
            ):
                suspects.append((rel, var, white_ratio, edge_density))
        except Exception:
            continue

    out_dir = root / args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rel, _, _, _ in suspects:
        src = root / rel
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())

    print(f"sampled={len(sample_paths)} suspects={len(suspects)}")
    print(f"Copied suspects to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
