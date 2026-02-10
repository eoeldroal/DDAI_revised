#!/usr/bin/env python3
"""Apply doc-independent filtering to all datasets after Qwen runs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


DATASET_QA = {
    "OpenDocVQA": "OpenDocVQA/opendocvqa_train.parquet",
    "VDR_ibm": "VDR_ibm/vdr_ibm_train.parquet",
    "SlideVQA_train": "SlideVQA/slidevqa_train.parquet",
    "SlideVQA_val": "SlideVQA/slidevqa_val.parquet",
    "SlideVQA_test": "SlideVQA/slidevqa_test.parquet",
}


def iter_result_files(results_dir: Path, dataset: str) -> list[Path]:
    return sorted(results_dir.glob(f"{dataset}_*.jsonl"))


def load_doc_independent_counts(
    results_dir: Path,
    dataset: str,
) -> tuple[dict[str, int], dict[str, int]]:
    yes_counts: dict[str, int] = defaultdict(int)
    total_counts: dict[str, int] = defaultdict(int)
    seen_runs: dict[str, set[int]] = defaultdict(set)

    for path in iter_result_files(results_dir, dataset):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if rec.get("dataset") not in (None, dataset):
                    continue
                rid = rec.get("id")
                if rid is None:
                    continue
                rid = str(rid)

                run_idx = rec.get("run_idx", None)
                if run_idx is not None:
                    try:
                        run_idx = int(run_idx)
                    except Exception:
                        run_idx = None

                if run_idx is not None:
                    if run_idx in seen_runs[rid]:
                        continue
                    seen_runs[rid].add(run_idx)

                total_counts[rid] += 1
                if rec.get("doc_independent") == "yes":
                    yes_counts[rid] += 1

    return yes_counts, total_counts


def apply_filter(
    qa_path: Path,
    out_path: Path,
    yes_counts: dict[str, int],
    total_counts: dict[str, int],
    yes_threshold: int,
    min_runs: int,
    require_full_runs: bool,
    removed_report: Path,
) -> dict:
    df = pd.read_parquet(qa_path)
    df["__id"] = df["id"].astype(str)

    def should_remove(rid: str) -> bool:
        yes = yes_counts.get(rid, 0)
        total = total_counts.get(rid, 0)
        if require_full_runs and total < min_runs:
            return False
        return yes >= yes_threshold

    remove_mask = df["__id"].apply(should_remove)
    removed = df[remove_mask].copy()
    kept = df[~remove_mask].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.drop(columns=["__id"], inplace=True)
    kept.to_parquet(out_path, index=False)

    removed_report.parent.mkdir(parents=True, exist_ok=True)
    removed[["id"]].assign(
        yes_count=removed["__id"].map(lambda x: yes_counts.get(x, 0)),
        total_runs=removed["__id"].map(lambda x: total_counts.get(x, 0)),
    ).to_csv(removed_report, index=False)

    return {
        "total": int(len(df)),
        "removed": int(len(removed)),
        "kept": int(len(kept)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_1",
    )
    parser.add_argument("--qa-root", default="qa_query_dedup")
    parser.add_argument(
        "--results-dir",
        default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_1/filters/doc_independent",
    )
    parser.add_argument(
        "--out-root",
        default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_2/qa_doc_independent_filtered",
    )
    parser.add_argument("--yes-threshold", type=int, default=2)
    parser.add_argument("--min-runs", type=int, default=4)
    parser.add_argument("--require-full-runs", action="store_true")
    parser.add_argument("--only-dataset", default="")
    args = parser.parse_args()

    root = Path(args.root) / args.qa_root
    results_dir = Path(args.results_dir)
    out_root = Path(args.out_root)

    summary = {
        "yes_threshold": args.yes_threshold,
        "min_runs": args.min_runs,
        "require_full_runs": args.require_full_runs,
        "datasets": {},
    }

    for dataset, rel_path in DATASET_QA.items():
        if args.only_dataset and dataset != args.only_dataset:
            continue
        qa_path = root / rel_path
        if not qa_path.exists():
            continue
        yes_counts, total_counts = load_doc_independent_counts(results_dir, dataset)
        out_path = out_root / rel_path
        removed_report = out_root / "removed_ids" / f"{dataset}.csv"

        stats = apply_filter(
            qa_path,
            out_path,
            yes_counts,
            total_counts,
            args.yes_threshold,
            args.min_runs,
            args.require_full_runs,
            removed_report,
        )
        stats["results_files"] = [p.name for p in iter_result_files(results_dir, dataset)]
        summary["datasets"][dataset] = stats
        print(f"[{dataset}] total={stats['total']} removed={stats['removed']} kept={stats['kept']}")

    out_root.mkdir(parents=True, exist_ok=True)
    stats_path = out_root / "doc_independent_filter_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"wrote {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
