#!/usr/bin/env python3
"""Collect low-resolution images into a review folder.

This does NOT delete anything. It only copies candidates for manual review.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root, e.g. data/VDR_processed_filtered_1")
    parser.add_argument("--min-side", type=int, default=256)
    parser.add_argument("--out-dir", default="filters/min_res")
    args = parser.parse_args()

    root = Path(args.root)
    index_dir = root / "corpus_index"
    out_dir = root / args.out_dir / f"min_res_{args.min_side}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ["OpenDocVQA", "SlideVQA", "VDR_ibm"]:
        idx_path = index_dir / f"{name}.parquet"
        if not idx_path.exists():
            print(f"[WARN] missing index: {idx_path}")
            continue
        df = pd.read_parquet(idx_path, columns=["doc_id", "path", "width", "height"])
        low = df[(df["width"] < args.min_side) | (df["height"] < args.min_side)].copy()
        print(f"{name}: total={len(df)}, low_res={len(low)}")
        for rel in low["path"]:
            src = root / rel
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists() and not dst.exists():
                dst.write_bytes(src.read_bytes())

    print(f"Copied candidates to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
