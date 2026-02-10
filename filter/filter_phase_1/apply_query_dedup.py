#!/usr/bin/env python3
"""Apply query dedup by merging GTI (post-processing step).

Policy:
- OpenDocVQA, VDR_ibm: merge GTI for similar queries
- SlideVQA: merge only within same deck (deck_name), and ensure merged GTI exists in deck
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_candidates(path: Path, threshold: float) -> list[dict]:
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["score"] >= threshold:
                pairs.append(rec)
    return pairs


def extract_numbers(text: str) -> set[str]:
    text = text or ""
    # Prefer the part after "Query:" if present
    if "Query:" in text:
        text = text.split("Query:", 1)[1]
    return set(re.findall(r"\\b\\d+\\b", text))


def number_mismatch(a: str, b: str) -> bool:
    return extract_numbers(a) != extract_numbers(b)


def build_union(pairs: list[dict]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for rec in pairs:
        union(rec["id_a"], rec["id_b"])

    groups: dict[str, set[str]] = {}
    for x in list(parent.keys()):
        r = find(x)
        groups.setdefault(r, set()).add(x)

    rep = {}
    for _, members in groups.items():
        rep_id = min(members)
        for m in members:
            rep[m] = rep_id
    return rep


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return x
    return [x]


def normalize_gti(gti):
    items = []
    for x in _to_list(gti):
        if isinstance(x, np.ndarray):
            items.extend(x.tolist())
        elif isinstance(x, list):
            items.extend(x)
        else:
            items.append(x)
    # keep as list for parquet
    return [str(x) for x in items]


def merge_gti(a, b):
    if a is None:
        return b
    if b is None:
        return a
    a = normalize_gti(a)
    b = normalize_gti(b)
    return list(dict.fromkeys(a + b))


def apply_generic(df: pd.DataFrame, rep_map: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["rep_id"] = df["id"].map(rep_map).fillna(df["id"])
    kept = []
    for _, group in df.groupby("rep_id"):
        row = group.iloc[0].copy()
        kept.append(row)
    return pd.DataFrame(kept).drop(columns=["rep_id"])


def infer_deck_from_gti(gti, deck_map: dict[str, str]) -> str | None:
    for doc_id in normalize_gti(gti):
        deck = deck_map.get(str(doc_id))
        if deck:
            return deck
    return None


def apply_slidevqa(df: pd.DataFrame, rep_map: dict[str, str], deck_map: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["rep_id"] = df["id"].map(rep_map).fillna(df["id"])
    df["deck_name"] = df["gti"].apply(lambda g: infer_deck_from_gti(g, deck_map))
    df["deck_name"] = df["deck_name"].fillna("__unknown__")

    merged_rows = []
    for (_, deck_name), group in df.groupby(["rep_id", "deck_name"]):
        row = group.iloc[0].copy()
        gti = normalize_gti(row.get("gti"))
        if isinstance(gti, list) and deck_name != "__unknown__":
            gti = [x for x in gti if isinstance(x, str) and deck_name in x]
        row["gti"] = gti
        merged_rows.append(row)
    return pd.DataFrame(merged_rows).drop(columns=["rep_id", "deck_name"], errors="ignore")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--qa-root", default="qa_filtered_RLVR")
    parser.add_argument("--cand-file", default="filters/query_dedup_e5/query_candidates.jsonl")
    parser.add_argument("--threshold", type=float, default=0.995)
    parser.add_argument("--out-dir", default="qa_query_dedup")
    parser.add_argument("--sample-out", default="filters/query_dedup_e5/query_candidates_sample_1000_filtered.csv")
    parser.add_argument("--filtered-out", default="filters/query_dedup_e5/query_candidates_filtered.jsonl")
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    root = Path(args.root)
    qa_root = root / args.qa_root
    out_root = root / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = load_candidates(root / args.cand_file, args.threshold)
    rep_map = build_union(pairs)

    # Build sample view after number-mismatch filtering
    qa = pd.concat(
        [
            pd.read_parquet(qa_root / "OpenDocVQA/opendocvqa_train.parquet", columns=["id", "query"]),
            pd.read_parquet(qa_root / "SlideVQA/slidevqa_train.parquet", columns=["id", "query"]),
            pd.read_parquet(qa_root / "SlideVQA/slidevqa_val.parquet", columns=["id", "query"]),
            pd.read_parquet(qa_root / "SlideVQA/slidevqa_test.parquet", columns=["id", "query"]),
            pd.read_parquet(qa_root / "VDR_ibm/vdr_ibm_train.parquet", columns=["id", "query"]),
        ],
        ignore_index=True,
    )
    qa["id"] = qa["id"].astype(str)
    qmap = dict(zip(qa["id"], qa["query"].astype(str)))

    filtered_pairs = []
    filtered_for_union = []
    for rec in pairs:
        qa = qmap.get(rec["id_a"], "")
        qb = qmap.get(rec["id_b"], "")
        if number_mismatch(qa, qb):
            continue
        filtered_pairs.append(
            {"id_a": rec["id_a"], "query_a": qa, "id_b": rec["id_b"], "query_b": qb, "score": rec["score"]}
        )
        filtered_for_union.append(rec)

    # Rebuild union after number mismatch filtering
    if filtered_for_union:
        rep_map = build_union(filtered_for_union)

    # Write filtered candidates and sample
    if filtered_pairs:
        with (root / args.filtered_out).open("w", encoding="utf-8") as f:
            for rec in filtered_pairs:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        sample = filtered_pairs.copy()
        if len(sample) > args.sample_size:
            import random
            random.seed(17)
            sample = random.sample(sample, args.sample_size)
        pd.DataFrame(sample).to_csv(root / args.sample_out, index=False)

    od = pd.read_parquet(qa_root / "OpenDocVQA/opendocvqa_train.parquet")
    od_out = apply_generic(od, rep_map)
    od_out["gti"] = od_out["gti"].apply(normalize_gti)
    (out_root / "OpenDocVQA").mkdir(parents=True, exist_ok=True)
    od_out.to_parquet(out_root / "OpenDocVQA/opendocvqa_train.parquet", index=False)

    vb = pd.read_parquet(qa_root / "VDR_ibm/vdr_ibm_train.parquet")
    vb_out = apply_generic(vb, rep_map)
    vb_out["gti"] = vb_out["gti"].apply(normalize_gti)
    (out_root / "VDR_ibm").mkdir(parents=True, exist_ok=True)
    vb_out.to_parquet(out_root / "VDR_ibm/vdr_ibm_train.parquet", index=False)

    idx = pd.read_parquet(root / "corpus_index/SlideVQA.parquet", columns=["doc_id", "deck_name"])
    deck_map = dict(zip(idx["doc_id"].astype(str), idx["deck_name"].astype(str)))

    for split in ["train", "val", "test"]:
        sv = pd.read_parquet(qa_root / f"SlideVQA/slidevqa_{split}.parquet")
        sv_out = apply_slidevqa(sv, rep_map, deck_map)
        if "gti" in sv_out.columns:
            sv_out["gti"] = sv_out["gti"].apply(normalize_gti)
        (out_root / "SlideVQA").mkdir(parents=True, exist_ok=True)
        sv_out.to_parquet(out_root / f"SlideVQA/slidevqa_{split}.parquet", index=False)

    print(f"Saved merged QA to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
