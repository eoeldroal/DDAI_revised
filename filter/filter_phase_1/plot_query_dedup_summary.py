#!/usr/bin/env python3
"""Visualize query-dedup filtering impact with seaborn."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def count_dir(base: Path) -> dict[str, int]:
    counts = {}
    counts["OpenDocVQA"] = len(
        pd.read_parquet(base / "OpenDocVQA/opendocvqa_train.parquet", columns=["id"])
    )
    counts["VDR_ibm"] = len(
        pd.read_parquet(base / "VDR_ibm/vdr_ibm_train.parquet", columns=["id"])
    )
    for split in ["train", "val", "test"]:
        counts[f"SlideVQA_{split}"] = len(
            pd.read_parquet(base / f"SlideVQA/slidevqa_{split}.parquet", columns=["id"])
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--before", default="qa_filtered_RLVR")
    parser.add_argument("--after", default="qa_query_dedup")
    parser.add_argument("--out-dir", default="filters/query_dedup_e5/viz")
    args = parser.parse_args()

    root = Path(args.root)
    before = root / args.before
    after = root / args.after
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cb = count_dir(before)
    ca = count_dir(after)

    rows = []
    for k, v in cb.items():
        rows.append({"dataset": k, "stage": "before", "count": v})
        rows.append({"dataset": k, "stage": "after", "count": ca.get(k, 0)})
    df = pd.DataFrame(rows)
    removed = pd.DataFrame(
        [
            {"dataset": k, "removed": cb[k] - ca.get(k, 0)}
            for k in cb.keys()
        ]
    )
    removed["before"] = removed["dataset"].map(cb)
    removed["after"] = removed["dataset"].map(ca)
    removed["removed_pct"] = (removed["removed"] / removed["before"]).fillna(0) * 100.0

    sns.set_theme(style="whitegrid", font_scale=1.0)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df, x="dataset", y="count", hue="stage")
    ax.set_title("Query Dedup: QA Counts Before vs After")
    ax.set_xlabel("")
    ax.set_ylabel("QA count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "qa_counts_before_after.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=removed, x="dataset", y="removed", color="#c94f4f")
    ax.set_title("Query Dedup: Removed QA Counts")
    ax.set_xlabel("")
    ax.set_ylabel("Removed count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "qa_removed_counts.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=removed, x="dataset", y="removed_pct", color="#4f77c9")
    ax.set_title("Query Dedup: Removed % by Dataset")
    ax.set_xlabel("")
    ax.set_ylabel("Removed (%)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "qa_removed_percent.png", dpi=200)
    plt.close()

    # Dumbbell plot: before vs after
    plt.figure(figsize=(10, 5))
    order = removed.sort_values("before", ascending=False)["dataset"].tolist()
    ax = plt.gca()
    for _, row in removed.iterrows():
        y = order.index(row["dataset"])
        ax.plot([row["after"], row["before"]], [y, y], color="#999999", linewidth=2)
        ax.scatter(row["after"], y, color="#2b8cbe", s=40, label="after")
        ax.scatter(row["before"], y, color="#de2d26", s=40, label="before")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_title("Query Dedup: Before vs After (Dumbbell)")
    ax.set_xlabel("QA count")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "qa_before_after_dumbbell.png", dpi=200)
    plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
