#!/usr/bin/env python3
"""Seaborn visualizations for RLVR long-answer filtering (v3 criteria)."""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_qa(root: Path) -> pd.DataFrame:
    qa_files = [
        root / "qa/OpenDocVQA/opendocvqa_train.parquet",
        root / "qa/SlideVQA/slidevqa_train.parquet",
        root / "qa/SlideVQA/slidevqa_val.parquet",
        root / "qa/SlideVQA/slidevqa_test.parquet",
        root / "qa/VDR_ibm/vdr_ibm_train.parquet",
    ]
    frames = []
    for p in qa_files:
        df = pd.read_parquet(p, columns=["id", "query", "answer"])
        df["answer"] = df["answer"].fillna("")
        df["source_file"] = p.name
        if "slidevqa" in p.name:
            df["dataset"] = "SlideVQA"
        elif "opendocvqa" in p.name:
            df["dataset"] = "OpenDocVQA"
        else:
            df["dataset"] = "VDR_ibm"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def is_numeric_list(a: str) -> bool:
    num_pat = re.compile(r"^\d+(?:\.\d+)?%?$")
    tokens = [t.strip() for t in re.split(r"[\s,;\[\]\(\)\n]+", a) if t.strip()]
    if not tokens or len(tokens) > 10:
        return False
    return all(num_pat.match(t) for t in tokens)


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    ans = df["answer"].astype(str)
    df["len_chars"] = ans.str.len()
    df["len_words"] = ans.str.split().str.len()
    df["periods"] = ans.str.count(r"\.")
    df["semicolons"] = ans.str.count(";")
    df["has_newline"] = ans.str.contains(r"\n")
    df["numeric_list_like"] = ans.apply(lambda a: is_numeric_list(a.strip()))
    df["flag_v3"] = (
        (~df["numeric_list_like"])
        & (
            (df["len_words"] >= 10)
            | (df["periods"] >= 2)
            | (df["semicolons"] >= 1)
            | (df["has_newline"])
        )
    )
    return df


def main() -> int:
    root = Path("/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_1")
    out_dir = root / "filters/rlvr_long_answer/viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = add_flags(load_qa(root))

    sns.set_theme(style="whitegrid")

    # 1) Total vs flagged counts per dataset
    counts = (
        df.groupby("dataset")["flag_v3"]
        .agg(total="count", flagged="sum")
        .reset_index()
    )
    counts_m = counts.melt("dataset", var_name="type", value_name="count")
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=counts_m, x="dataset", y="count", hue="type")
    ax.set_title("QA Count vs Long-Answer Candidates (v3)")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "qa_counts_vs_flagged.png", dpi=200)
    plt.close()

    # 2) Word count distribution with threshold
    plt.figure(figsize=(7, 4))
    ax = sns.histplot(data=df, x="len_words", hue="dataset", bins=50, element="step", stat="density")
    ax.axvline(10, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 80)
    ax.set_title("Answer Word Count Distribution (v3 threshold=10)")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_dir / "answer_word_count.png", dpi=200)
    plt.close()

    # 3) Period count distribution with threshold
    plt.figure(figsize=(7, 4))
    ax = sns.histplot(data=df, x="periods", hue="dataset", bins=15, element="step", stat="density")
    ax.axvline(2, color="black", linestyle="--", linewidth=1)
    ax.set_title("Answer Period Count Distribution (v3 threshold=2)")
    ax.set_xlabel("Period count")
    ax.set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_dir / "answer_period_count.png", dpi=200)
    plt.close()

    # 4) Flag reasons breakdown
    reason_counts = pd.DataFrame(
        {
            "words>=10": (df["len_words"] >= 10) & (~df["numeric_list_like"]),
            "periods>=2": (df["periods"] >= 2) & (~df["numeric_list_like"]),
            "semicolons>=1": (df["semicolons"] >= 1) & (~df["numeric_list_like"]),
            "has_newline": df["has_newline"] & (~df["numeric_list_like"]),
        }
    ).sum().reset_index()
    reason_counts.columns = ["reason", "count"]
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=reason_counts, x="reason", y="count")
    ax.set_title("Flag Reason Counts (v3)")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "flag_reason_counts.png", dpi=200)
    plt.close()

    print(f"Saved plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
