"""
Data analysis and visualization for KV-cache benchmark results.
Reads `benchmark_results.csv` and `benchmark_summary.csv` produced by test.py,
and generates clearer plots:
  - Per-token time vs sequence length (with/without KV) for each batch
  - Speedup vs sequence length for each batch
  - Peak memory (overall) vs sequence length for each batch (KV on/off)
  - Memory increase ratio vs sequence length for each batch
  - Priming vs Generation memory (stacked bars) for KV on/off

Run:
    python data_analyze.py \
        --results benchmark_results.csv \
        --summary benchmark_summary.csv \
        --out-dir figs
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Matplotlib defaults for readability
plt.style.use("ggplot")


def plot_time_vs_seq(df_results, out_dir):
    for batch in sorted(df_results.batch.unique()):
        dfb = df_results[df_results.batch == batch]
        x = sorted(dfb.seq_len.unique())
        y_no = [dfb[(dfb.seq_len == L) & (dfb.kv_cache == False)].iloc[0].time_ms for L in x]
        y_kv = [dfb[(dfb.seq_len == L) & (dfb.kv_cache == True)].iloc[0].time_ms for L in x]
        plt.figure()
        plt.plot(x, y_no, marker="o", label="No KV")
        plt.plot(x, y_kv, marker="o", label="KV Cache")
        plt.xlabel("Sequence length")
        plt.ylabel("Per-token time (ms)")
        plt.title(f"Per-token time vs seq_len (batch={batch})")
        plt.legend()
        plt.savefig(out_dir / f"time_vs_seq_batch{batch}.png")
        plt.close()


def plot_speedup(df_summary, out_dir):
    for batch in sorted(df_summary.batch.unique()):
        dfs = df_summary[df_summary.batch == batch]
        x = sorted(dfs.seq_len.unique())
        y = [dfs[dfs.seq_len == L].iloc[0].time_speedup for L in x]
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("Sequence length")
        plt.ylabel("Speedup (no_kv / kv)")
        plt.title(f"Time speedup vs seq_len (batch={batch})")
        plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        plt.savefig(out_dir / f"speedup_vs_seq_batch{batch}.png")
        plt.close()


def plot_mem_overall(df_summary, out_dir):
    for batch in sorted(df_summary.batch.unique()):
        dfs = df_summary[df_summary.batch == batch]
        x = sorted(dfs.seq_len.unique())
        y_no = [dfs[dfs.seq_len == L].iloc[0].mem_no_kv_overall for L in x]
        y_kv = [dfs[dfs.seq_len == L].iloc[0].mem_kv_overall for L in x]
        plt.figure()
        plt.plot(x, y_no, marker="o", label="No KV")
        plt.plot(x, y_kv, marker="o", label="KV Cache")
        plt.xlabel("Sequence length")
        plt.ylabel("Peak memory (MB)")
        plt.title(f"Overall peak memory vs seq_len (batch={batch})")
        plt.legend()
        plt.savefig(out_dir / f"mem_overall_vs_seq_batch{batch}.png")
        plt.close()


def plot_mem_increase_ratio(df_summary, out_dir):
    for batch in sorted(df_summary.batch.unique()):
        dfs = df_summary[df_summary.batch == batch]
        x = sorted(dfs.seq_len.unique())
        y = [dfs[dfs.seq_len == L].iloc[0].mem_increase_ratio for L in x]
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("Sequence length")
        plt.ylabel("Memory increase ratio")
        plt.title(f"Memory increase ratio vs seq_len (batch={batch})")
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        plt.savefig(out_dir / f"mem_ratio_vs_seq_batch{batch}.png")
        plt.close()


def plot_mem_breakdown(df_summary, out_dir):
    """
    Stacked bars showing priming vs generation memory for KV on/off.
    """
    for batch in sorted(df_summary.batch.unique()):
        dfs = df_summary[df_summary.batch == batch].sort_values("seq_len")
        x_labels = [str(L) for L in dfs.seq_len.unique()]
        width = 0.35

        no_kv_prim = dfs["mem_no_kv_priming"]
        no_kv_gen = dfs["mem_no_kv_gen"]
        kv_prim = dfs["mem_kv_priming"]
        kv_gen = dfs["mem_kv_gen"]

        x = range(len(x_labels))
        plt.figure()
        # No KV
        plt.bar([i - width / 2 for i in x], no_kv_prim, width, label="No KV priming")
        plt.bar(
            [i - width / 2 for i in x],
            no_kv_gen,
            width,
            bottom=no_kv_prim,
            label="No KV gen",
        )
        # KV
        plt.bar([i + width / 2 for i in x], kv_prim, width, label="KV priming")
        plt.bar(
            [i + width / 2 for i in x],
            kv_gen,
            width,
            bottom=kv_prim,
            label="KV gen",
        )
        plt.xticks(list(x), x_labels)
        plt.xlabel("Sequence length")
        plt.ylabel("Memory (MB)")
        plt.title(f"Memory breakdown (priming/gen) batch={batch}")
        plt.legend()
        plt.savefig(out_dir / f"mem_breakdown_batch{batch}.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/benchmark_results.csv")
    parser.add_argument("--summary", type=str, default="results/benchmark_summary.csv")
    parser.add_argument("--out-dir", type=str, default="results/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.read_csv(args.results)
    df_summary = pd.read_csv(args.summary)

    plot_time_vs_seq(df_results, out_dir)
    plot_speedup(df_summary, out_dir)
    plot_mem_overall(df_summary, out_dir)
    plot_mem_increase_ratio(df_summary, out_dir)
    plot_mem_breakdown(df_summary, out_dir)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
