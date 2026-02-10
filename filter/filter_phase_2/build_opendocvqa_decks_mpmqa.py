#!/usr/bin/env python3
"""Build OpenDocVQA decks for mpmqa only (raw grouping, no padding/truncation)."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def doc_key_mpmqa(doc_id: str) -> str:
    # mpmqa/<docid>/<docid>_00004.jpg -> mpmqa/<docid>
    parts = doc_id.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return doc_id


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
        default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_2/qa_gti_filtered_2plus_deck_mpmqa/opendocvqa_train.parquet",
    )
    ap.add_argument(
        "--stats-out",
        default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_2/filters/deck_stats/opendocvqa_mpmqa_deck_stats.json",
    )
    args = ap.parse_args()

    qa = pd.read_parquet(args.qa_in)
    idx = pd.read_parquet(args.corpus_index, columns=["doc_id", "dataset_name"])

    key_to_docs = defaultdict(list)
    docid_to_key = {}
    for doc_id, ds in zip(idx["doc_id"], idx["dataset_name"]):
        if str(ds) != "mpmqa":
            continue
        key = doc_key_mpmqa(doc_id)
        docid_to_key[doc_id] = key
        key_to_docs[key].append(doc_id)

    for key in list(key_to_docs.keys()):
        key_to_docs[key] = sorted(key_to_docs[key])

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
        if not isinstance(doc_id, str) or not doc_id.startswith("mpmqa/"):
            continue
        key = docid_to_key.get(doc_id, "")
        deck = key_to_docs.get(key, [doc_id])
        keep_rows.append(row)
        decks.append(deck)
        deck_keys.append(key)
        deck_sizes.append(len(deck))
        if not key:
            missing_key += 1

    if not keep_rows:
        raise SystemExit("No mpmqa rows found.")

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
        "top_sizes": dict(Counter(deck_sizes).most_common(10)),
    }

    stats_out = Path(args.stats_out)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
