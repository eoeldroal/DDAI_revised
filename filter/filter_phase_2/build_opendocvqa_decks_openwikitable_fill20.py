#!/usr/bin/env python3
"""Build OpenDocVQA decks for openwikitable and ensure min size 20 by filling from adjacent docids."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def doc_key_openwikitable(doc_id: str) -> str:
    # openwikitable/204-942.jpg -> 204
    name = doc_id.split("/")[-1]
    if "-" not in name:
        return ""
    return name.split("-")[0]


def normalize_gti(gti):
    if isinstance(gti, np.ndarray):
        gti = gti.tolist()
    elif isinstance(gti, tuple):
        gti = list(gti)
    return gti


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--qa-in",
        default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_2/qa_gti_filtered_2plus/OpenDocVQA/opendocvqa_train.parquet",
    )
    ap.add_argument(
        "--corpus-index",
        default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_1/corpus_index/OpenDocVQA.parquet",
    )
    ap.add_argument(
        "--out",
        default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_2/qa_gti_filtered_2plus_deck_openwikitable_fill20/opendocvqa_train.parquet",
    )
    ap.add_argument(
        "--stats-out",
        default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_2/filters/deck_stats/opendocvqa_openwikitable_deck_stats_fill20.json",
    )
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--max-adjacent", type=int, default=50)
    args = ap.parse_args()

    qa = pd.read_parquet(args.qa_in)
    idx = pd.read_parquet(args.corpus_index, columns=["doc_id", "dataset_name"])

    # openwikitable-only corpus
    idx_openw = idx[idx["dataset_name"].astype(str) == "openwikitable"].copy()
    idx_openw = idx_openw[idx_openw["doc_id"].str.startswith("openwikitable/")]

    # build buckets by docid number
    bucket = defaultdict(list)
    for doc_id in idx_openw["doc_id"]:
        docnum = doc_key_openwikitable(doc_id)
        if not docnum:
            continue
        bucket[int(docnum)].append(doc_id)

    for k in list(bucket.keys()):
        bucket[k] = sorted(bucket[k])

    # helper: fill deck to min size by adjacent docnum
    def fill_to_min(docnum: int, deck: list[str]) -> list[str]:
        if len(deck) >= args.min_size:
            return deck
        # walk neighbors by distance
        needed = args.min_size - len(deck)
        for dist in range(1, args.max_adjacent + 1):
            for neighbor in (docnum - dist, docnum + dist):
                if neighbor in bucket:
                    for d in bucket[neighbor]:
                        if d not in deck:
                            deck.append(d)
                            needed -= 1
                            if needed <= 0:
                                return deck
        return deck

    keep_rows = []
    decks = []
    deck_keys = []
    deck_sizes = []
    missing_key = 0

    for row in qa.itertuples(index=False):
        gti = normalize_gti(row.gti)
        if not isinstance(gti, list) or not gti:
            continue
        doc_id = gti[0]
        if not isinstance(doc_id, str) or not doc_id.startswith("openwikitable/"):
            continue
        docnum_str = doc_key_openwikitable(doc_id)
        if not docnum_str:
            missing_key += 1
            continue
        docnum = int(docnum_str)
        deck = list(bucket.get(docnum, [doc_id]))
        if doc_id not in deck:
            deck.insert(0, doc_id)
        deck = fill_to_min(docnum, deck)

        keep_rows.append(row)
        decks.append(deck)
        deck_keys.append(f"openwikitable/{docnum_str}")
        deck_sizes.append(len(deck))

    if not keep_rows:
        raise SystemExit("No openwikitable rows found.")

    out_df = pd.DataFrame(keep_rows)
    out_df["document_images"] = decks
    out_df["deck_key"] = deck_keys
    out_df["deck_size"] = deck_sizes

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    sizes = pd.Series(deck_sizes)
    stats = {
        "total": int(len(sizes)),
        "min": int(sizes.min()),
        "max": int(sizes.max()),
        "mean": float(sizes.mean()),
        "median": float(sizes.median()),
        "lt20": int((sizes < 20).sum()),
        "eq20": int((sizes == 20).sum()),
        "gt20": int((sizes > 20).sum()),
        "missing_key": int(missing_key),
        "min_size": int(args.min_size),
        "max_adjacent": int(args.max_adjacent),
        "top_sizes": dict(Counter(deck_sizes).most_common(10)),
    }

    stats_out = Path(args.stats_out)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
