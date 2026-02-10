#!/usr/bin/env python3
"""Build visualmrc decks with relaxed grouping: same domain + nearest date."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DATE_RE = re.compile(r"(\d{4})__([0-1]\d)__([0-3]\d)__")


def doc_key_visualmrc(doc_id: str) -> str:
    # visualmrc/domain/slug02.png -> visualmrc/domain/slug
    base = doc_id.split("/")[-1]
    base = re.sub(r"\.[A-Za-z0-9]+$", "", base)
    base = re.sub(r"\d+$", "", base)
    parts = doc_id.split("/")
    if len(parts) >= 3:
        return "/".join(parts[:-1] + [base])
    return base


def parse_date_from_key(key: str):
    # key example: visualmrc/domain/2016__06__08__slug
    m = DATE_RE.search(key)
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
    except Exception:
        return None


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
        default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_2/qa_gti_filtered_2plus_deck_visualmrc_relaxed/opendocvqa_train.parquet",
    )
    ap.add_argument(
        "--stats-out",
        default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_2/filters/deck_stats/opendocvqa_visualmrc_deck_stats_relaxed.json",
    )
    ap.add_argument("--target-size", type=int, default=20)
    args = ap.parse_args()

    qa = pd.read_parquet(args.qa_in)
    idx = pd.read_parquet(args.corpus_index, columns=["doc_id", "dataset_name"])

    # Build doc_key mapping for visualmrc only
    key_to_docs = defaultdict(list)
    docid_to_key = {}
    key_to_domain = {}
    key_to_date = {}
    for doc_id, ds in zip(idx["doc_id"], idx["dataset_name"]):
        if str(ds) != "visualmrc":
            continue
        key = doc_key_visualmrc(doc_id)
        docid_to_key[doc_id] = key
        key_to_docs[key].append(doc_id)
        parts = key.split("/")
        if len(parts) >= 2:
            key_to_domain[key] = parts[1]
        else:
            key_to_domain[key] = ""
        if key not in key_to_date:
            key_to_date[key] = parse_date_from_key(key)

    for key in list(key_to_docs.keys()):
        key_to_docs[key] = sorted(key_to_docs[key])

    # domain -> list of keys
    domain_to_keys = defaultdict(list)
    for key, dom in key_to_domain.items():
        domain_to_keys[dom].append(key)

    # sort keys by date for each domain (None last)
    for dom in list(domain_to_keys.keys()):
        keys = domain_to_keys[dom]
        keys.sort(key=lambda k: (key_to_date.get(k) is None, key_to_date.get(k) or datetime(1900,1,1).date()))
        domain_to_keys[dom] = keys

    def fill_to_target(base_docs, key, target):
        if len(base_docs) >= target:
            return base_docs
        dom = key_to_domain.get(key, "")
        date = key_to_date.get(key)
        pool = []
        for k in domain_to_keys.get(dom, []):
            if k == key:
                continue
            d = key_to_date.get(k)
            if date and d:
                dist = abs((d - date).days)
            else:
                dist = 10**9
            pool.append((dist, k))
        pool.sort(key=lambda x: x[0])

        seen = set(base_docs)
        filled = list(base_docs)
        for _, k in pool:
            for doc_id in key_to_docs.get(k, []):
                if doc_id in seen:
                    continue
                filled.append(doc_id)
                seen.add(doc_id)
                if len(filled) >= target:
                    return filled
        return filled

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
        if not isinstance(doc_id, str) or not doc_id.startswith("visualmrc/"):
            continue
        key = docid_to_key.get(doc_id, "")
        base = key_to_docs.get(key, [doc_id])
        deck = fill_to_target(base, key, args.target_size)
        keep_rows.append(row)
        decks.append(deck)
        deck_keys.append(key)
        deck_sizes.append(len(deck))
        if not key:
            missing_key += 1

    if not keep_rows:
        raise SystemExit("No visualmrc rows found.")

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
        "lt20": int((sizes < args.target_size).sum()),
        "eq20": int((sizes == args.target_size).sum()),
        "gt20": int((sizes > args.target_size).sum()),
        "missing_key": int(missing_key),
        "target_size": args.target_size,
        "top_sizes": dict(Counter(deck_sizes).most_common(10)),
    }

    stats_out = Path(args.stats_out)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
