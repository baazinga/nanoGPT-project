"""
KV Cache Time-Memory Tradeoff Analysis
---------------------------------------
分析 KV cache 的时间加速与内存开销的权衡关系。

指标计算:
1. Time Speedup: time_no_kv / time_kv
2. Memory Overhead: mem_kv_overall - mem_no_kv_overall (MB)
3. Memory Overhead Ratio: (mem_kv - mem_no_kv) / mem_no_kv
4. Efficiency: speedup / abs(mem_overhead) 或 speedup / abs(mem_overhead_ratio)
5. Time Saved per MB: (time_no_kv - time_kv) * gen_steps / mem_overhead

可视化:
- 散点图: speedup vs memory overhead (按 batch/seq_len 着色)
- 效率图: efficiency vs sequence length
- 热力图: tradeoff matrix (seq_len × batch)
- Pareto frontier: 最优权衡点

运行:
    python mem.py --summary benchmark_summary.csv --out-dir figs
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("ggplot")
sns.set_palette("husl")


def load_and_compute_tradeoff(df_summary):
    """计算 tradeoff 相关指标"""
    df = df_summary.copy()

    # 基础指标
    df["mem_overhead_MB"] = df["mem_kv_overall"] - df["mem_no_kv_overall"]
    df["time_saved_ms"] = (df["time_no_kv"] - df["time_kv"]) * df["gen_steps"]

    # 效率指标
    # 1. Speedup per MB overhead (越大越好)
    df["efficiency_speedup_per_MB"] = np.where(
        df["mem_overhead_MB"] != 0,
        df["time_speedup"] / abs(df["mem_overhead_MB"]),
        np.inf
    )

    # 2. Time saved per MB overhead (越大越好)
    df["time_saved_per_MB"] = np.where(
        df["mem_overhead_MB"] != 0,
        df["time_saved_ms"] / abs(df["mem_overhead_MB"]),
        np.inf
    )

    # 3. 使用 ratio 的效率 (处理负值情况)
    df["efficiency_speedup_per_ratio"] = np.where(
        abs(df["mem_increase_ratio"]) > 1e-6,
        df["time_speedup"] / abs(df["mem_increase_ratio"]),
        np.inf
    )

    # 4. 综合得分: speedup * (1 - normalized_mem_overhead)
    # 归一化内存开销 (相对于最大开销)
    max_mem_overhead = df["mem_overhead_MB"].abs().max()
    if max_mem_overhead > 0:
        df["normalized_mem_overhead"] = df["mem_overhead_MB"].abs() / max_mem_overhead
    else:
        df["normalized_mem_overhead"] = 0

    # 综合得分: 速度提升 - 归一化内存开销 (越大越好)
    df["tradeoff_score"] = df["time_speedup"] - df["normalized_mem_overhead"]

    return df


def plot_speedup_vs_memory(df, out_dir):
    """散点图: speedup vs memory overhead"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 按 batch 着色
    ax1 = axes[0]
    for batch in sorted(df.batch.unique()):
        dfb = df[df.batch == batch]
        ax1.scatter(
            dfb["mem_overhead_MB"],
            dfb["time_speedup"],
            label=f"batch={batch}",
            s=100,
            alpha=0.7,
        )
        # 标注序列长度
        for _, row in dfb.iterrows():
            ax1.annotate(
                f"L={row.seq_len}",
                (row["mem_overhead_MB"], row["time_speedup"]),
                fontsize=8,
                alpha=0.6,
            )

    ax1.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="No speedup")
    ax1.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Memory Overhead (MB)")
    ax1.set_ylabel("Time Speedup")
    ax1.set_title("Speedup vs Memory Overhead (by batch)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图: 按 seq_len 着色
    ax2 = axes[1]
    for seq_len in sorted(df.seq_len.unique()):
        dfs = df[df.seq_len == seq_len]
        ax2.scatter(
            dfs["mem_overhead_MB"],
            dfs["time_speedup"],
            label=f"seq_len={seq_len}",
            s=100,
            alpha=0.7,
        )
        # 标注 batch
        for _, row in dfs.iterrows():
            ax2.annotate(
                f"B={row.batch}",
                (row["mem_overhead_MB"], row["time_speedup"]),
                fontsize=8,
                alpha=0.6,
            )

    ax2.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="No speedup")
    ax2.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Memory Overhead (MB)")
    ax2.set_ylabel("Time Speedup")
    ax2.set_title("Speedup vs Memory Overhead (by seq_len)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "tradeoff_speedup_vs_memory.png", dpi=150)
    plt.close()
    print(f"[saved] tradeoff_speedup_vs_memory.png")


def plot_efficiency_metrics(df, out_dir):
    """效率指标随序列长度变化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for batch in sorted(df.batch.unique()):
        dfb = df[df.batch == batch]
        x = sorted(dfb.seq_len.unique())

        # 左上: efficiency (speedup per MB)
        axes[0, 0].plot(
            x,
            [dfb[dfb.seq_len == L].iloc[0]["efficiency_speedup_per_MB"] for L in x],
            marker="o",
            label=f"batch={batch}",
        )
        axes[0, 0].set_xlabel("Sequence length")
        axes[0, 0].set_ylabel("Efficiency (speedup / MB overhead)")
        axes[0, 0].set_title("Efficiency: Speedup per MB Overhead")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 右上: time saved per MB
        axes[0, 1].plot(
            x,
            [dfb[dfb.seq_len == L].iloc[0]["time_saved_per_MB"] for L in x],
            marker="o",
            label=f"batch={batch}",
        )
        axes[0, 1].set_xlabel("Sequence length")
        axes[0, 1].set_ylabel("Time Saved per MB (ms/MB)")
        axes[0, 1].set_title("Time Saved per MB Overhead")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 左下: tradeoff score
        axes[1, 0].plot(
            x,
            [dfb[dfb.seq_len == L].iloc[0]["tradeoff_score"] for L in x],
            marker="o",
            label=f"batch={batch}",
        )
        axes[1, 0].set_xlabel("Sequence length")
        axes[1, 0].set_ylabel("Tradeoff Score")
        axes[1, 0].set_title("Tradeoff Score (speedup - normalized_mem_overhead)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 右下: speedup vs normalized mem overhead
        axes[1, 1].scatter(
            [dfb[dfb.seq_len == L].iloc[0]["normalized_mem_overhead"] for L in x],
            [dfb[dfb.seq_len == L].iloc[0]["time_speedup"] for L in x],
            marker="o",
            s=100,
            label=f"batch={batch}",
        )
        axes[1, 1].set_xlabel("Normalized Memory Overhead")
        axes[1, 1].set_ylabel("Time Speedup")
        axes[1, 1].set_title("Speedup vs Normalized Memory Overhead")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "tradeoff_efficiency_metrics.png", dpi=150)
    plt.close()
    print(f"[saved] tradeoff_efficiency_metrics.png")


def plot_heatmap_tradeoff(df, out_dir):
    """热力图: 不同配置下的 tradeoff"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 准备数据: seq_len × batch 矩阵
    seq_lens = sorted(df.seq_len.unique())
    batches = sorted(df.batch.unique())

    # 1. Speedup 热力图
    speedup_matrix = np.zeros((len(seq_lens), len(batches)))
    for i, seq_len in enumerate(seq_lens):
        for j, batch in enumerate(batches):
            row = df[(df.seq_len == seq_len) & (df.batch == batch)].iloc[0]
            speedup_matrix[i, j] = row["time_speedup"]

    sns.heatmap(
        speedup_matrix,
        xticklabels=batches,
        yticklabels=seq_lens,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=axes[0, 0],
        cbar_kws={"label": "Speedup"},
    )
    axes[0, 0].set_xlabel("Batch size")
    axes[0, 0].set_ylabel("Sequence length")
    axes[0, 0].set_title("Time Speedup Heatmap")

    # 2. Memory overhead 热力图
    mem_matrix = np.zeros((len(seq_lens), len(batches)))
    for i, seq_len in enumerate(seq_lens):
        for j, batch in enumerate(batches):
            row = df[(df.seq_len == seq_len) & (df.batch == batch)].iloc[0]
            mem_matrix[i, j] = row["mem_overhead_MB"]

    sns.heatmap(
        mem_matrix,
        xticklabels=batches,
        yticklabels=seq_lens,
        annot=True,
        fmt=".1f",
        cmap="RdYlBu_r",
        center=0,
        ax=axes[0, 1],
        cbar_kws={"label": "Memory Overhead (MB)"},
    )
    axes[0, 1].set_xlabel("Batch size")
    axes[0, 1].set_ylabel("Sequence length")
    axes[0, 1].set_title("Memory Overhead Heatmap")

    # 3. Efficiency 热力图
    eff_matrix = np.zeros((len(seq_lens), len(batches)))
    for i, seq_len in enumerate(seq_lens):
        for j, batch in enumerate(batches):
            row = df[(df.seq_len == seq_len) & (df.batch == batch)].iloc[0]
            eff_matrix[i, j] = row["efficiency_speedup_per_MB"]
            if np.isinf(eff_matrix[i, j]):
                eff_matrix[i, j] = np.nan

    sns.heatmap(
        eff_matrix,
        xticklabels=batches,
        yticklabels=seq_lens,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        ax=axes[1, 0],
        cbar_kws={"label": "Efficiency"},
    )
    axes[1, 0].set_xlabel("Batch size")
    axes[1, 0].set_ylabel("Sequence length")
    axes[1, 0].set_title("Efficiency (Speedup per MB) Heatmap")

    # 4. Tradeoff score 热力图
    score_matrix = np.zeros((len(seq_lens), len(batches)))
    for i, seq_len in enumerate(seq_lens):
        for j, batch in enumerate(batches):
            row = df[(df.seq_len == seq_len) & (df.batch == batch)].iloc[0]
            score_matrix[i, j] = row["tradeoff_score"]

    sns.heatmap(
        score_matrix,
        xticklabels=batches,
        yticklabels=seq_lens,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=axes[1, 1],
        cbar_kws={"label": "Tradeoff Score"},
    )
    axes[1, 1].set_xlabel("Batch size")
    axes[1, 1].set_ylabel("Sequence length")
    axes[1, 1].set_title("Tradeoff Score Heatmap")

    plt.tight_layout()
    plt.savefig(out_dir / "tradeoff_heatmap.png", dpi=150)
    plt.close()
    print(f"[saved] tradeoff_heatmap.png")


def plot_pareto_frontier(df, out_dir):
    """Pareto frontier: 最优权衡点"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 对于每个 batch，找 Pareto 最优点
    for batch in sorted(df.batch.unique()):
        dfb = df[df.batch == batch].copy()

        # 计算 Pareto 前沿 (speedup 最大化，mem_overhead 最小化)
        # 但注意 mem_overhead 是负值，所以我们要最大化 speedup，最小化 abs(mem_overhead)
        dfb["mem_overhead_abs"] = dfb["mem_overhead_MB"].abs()

        # 简单 Pareto: 对于每个点，检查是否有其他点 speedup 更高且 mem_overhead 更小
        pareto_mask = np.ones(len(dfb), dtype=bool)
        for i in range(len(dfb)):
            for j in range(len(dfb)):
                if i != j:
                    # 如果 j 的 speedup >= i 的 speedup 且 mem_overhead <= i 的 mem_overhead
                    if (dfb.iloc[j]["time_speedup"] >= dfb.iloc[i]["time_speedup"] and
                        dfb.iloc[j]["mem_overhead_abs"] <= dfb.iloc[i]["mem_overhead_abs"]):
                        # 且至少一个严格更好
                        if (dfb.iloc[j]["time_speedup"] > dfb.iloc[i]["time_speedup"] or
                            dfb.iloc[j]["mem_overhead_abs"] < dfb.iloc[i]["mem_overhead_abs"]):
                            pareto_mask[i] = False
                            break

        pareto_points = dfb[pareto_mask]
        non_pareto = dfb[~pareto_mask]

        # 绘制所有点
        ax.scatter(
            non_pareto["mem_overhead_MB"],
            non_pareto["time_speedup"],
            alpha=0.3,
            s=50,
            label=f"batch={batch} (non-Pareto)",
        )

        # 绘制 Pareto 前沿点
        if len(pareto_points) > 0:
            sorted_pareto = pareto_points.sort_values("mem_overhead_MB")
            ax.scatter(
                sorted_pareto["mem_overhead_MB"],
                sorted_pareto["time_speedup"],
                s=150,
                marker="*",
                label=f"batch={batch} (Pareto)",
                edgecolors="k",
                linewidths=1.5,
            )
            # 连线显示前沿
            ax.plot(
                sorted_pareto["mem_overhead_MB"],
                sorted_pareto["time_speedup"],
                linestyle="--",
                alpha=0.5,
            )

            # 标注序列长度
            for _, row in sorted_pareto.iterrows():
                ax.annotate(
                    f"L={row.seq_len}",
                    (row["mem_overhead_MB"], row["time_speedup"]),
                    fontsize=9,
                    fontweight="bold",
                )

    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="No speedup")
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Memory Overhead (MB)")
    ax.set_ylabel("Time Speedup")
    ax.set_title("Pareto Frontier: Optimal Time-Memory Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "tradeoff_pareto_frontier.png", dpi=150)
    plt.close()
    print(f"[saved] tradeoff_pareto_frontier.png")


def save_tradeoff_table(df, out_dir):
    """保存 tradeoff 分析表格"""
    # 选择关键列
    cols = [
        "seq_len",
        "batch",
        "time_speedup",
        "mem_overhead_MB",
        "mem_increase_ratio",
        "efficiency_speedup_per_MB",
        "time_saved_per_MB",
        "tradeoff_score",
    ]
    df_out = df[cols].copy()
    df_out = df_out.sort_values(["batch", "seq_len"])
    df_out.to_csv(out_dir / "tradeoff_analysis.csv", index=False)
    print(f"[saved] tradeoff_analysis.csv")

    # 打印摘要
    print("\n" + "=" * 80)
    print("Tradeoff Analysis Summary")
    print("=" * 80)
    print(f"\nBest speedup: {df['time_speedup'].max():.2f}x")
    print(f"  Config: {df.loc[df['time_speedup'].idxmax(), ['seq_len', 'batch']].to_dict()}")
    print(f"\nBest efficiency (speedup per MB): {df['efficiency_speedup_per_MB'].max():.2f}")
    print(f"  Config: {df.loc[df['efficiency_speedup_per_MB'].idxmax(), ['seq_len', 'batch']].to_dict()}")
    print(f"\nBest tradeoff score: {df['tradeoff_score'].max():.2f}")
    print(f"  Config: {df.loc[df['tradeoff_score'].idxmax(), ['seq_len', 'batch']].to_dict()}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="KV Cache Time-Memory Tradeoff Analysis")
    parser.add_argument(
        "--summary",
        type=str,
        default="results/benchmark_summary.csv",
        help="Path to benchmark_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/figures",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    # 加载数据
    df_summary = pd.read_csv(args.summary)
    print(f"[loaded] {args.summary}: {len(df_summary)} rows")

    # 计算 tradeoff 指标
    df_tradeoff = load_and_compute_tradeoff(df_summary)

    # 创建输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # 生成可视化
    plot_speedup_vs_memory(df_tradeoff, out_dir)
    plot_efficiency_metrics(df_tradeoff, out_dir)
    plot_heatmap_tradeoff(df_tradeoff, out_dir)
    plot_pareto_frontier(df_tradeoff, out_dir)

    # 保存分析表格
    save_tradeoff_table(df_tradeoff, out_dir)

    print(f"\n[completed] All tradeoff analysis saved to {out_dir}/")


if __name__ == "__main__":
    main()
