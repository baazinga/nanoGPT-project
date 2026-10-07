"""
KV-Cache Benchmark
------------------
对同一个 Transformer 解码器 (本工程的 `BigramLanguageModel`，4 层、隐藏维度 256、4 个头)
在 "使用 KV Cache / 不使用 KV Cache" 两种模式下进行推理性能对比。

实验设置（符合老师要求）:
- 序列长度: 128, 512, 1024 （随机生成序列）
- 批量大小: 1, 16
- KV Cache: 开 / 关
- 每组组合重复 10 次，记录平均 per-token 推理时间 (ms) 和峰值显存 (MB)
- 衍生指标：时间加速比 = time_no_kv / time_kv，内存占用增加 = (mem_kv - mem_no_kv) / mem_no_kv

模型设置:
- 使用原始模型（随机初始化），不加载checkpoint（符合论文常见做法）
- 模型架构：4层，隐藏维度256，4个注意力头
- 推理模式：eval()，dropout禁用

运行:
    python benchmark_kv_cache.py --device cuda --vocab-size 5000 --generate-steps 50
生成文件:
    benchmark_results.csv, benchmark_summary.csv, time_vs_seq_batch*.png, mem_vs_seq_kv_batch*.png
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch

from config import Config
from model_with_kvcache import BigramLanguageModel

# 固定实验组合
SEQ_LENGTHS = [128, 512, 1024]
BATCH_SIZES = [1, 16]
REPEATS = 10


def prepare_model(vocab_size: int, device: torch.device) -> BigramLanguageModel:
    """
    创建并初始化原始模型（随机权重），不加载checkpoint。
    符合论文中常见的benchmark做法：使用原始架构进行性能测试。
    """
    Config.vocab_size = vocab_size
    model = BigramLanguageModel(vocab_size).to(device)
    model.eval()  # 设置为推理模式，禁用dropout
    print(f"[Model] Created fresh model (random init): vocab_size={vocab_size}, "
          f"n_layer={Config.n_layer}, n_embd={Config.n_embd}, n_head={Config.n_head}")
    return model


def get_peak_memory(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


@torch.no_grad()
def benchmark_one(
    model: BigramLanguageModel,
    vocab_size: int,
    seq_len: int,
    batch: int,
    use_cache: bool,
    device: torch.device,
    gen_steps: int,
) -> tuple[float, float, float, float]:
    """
    单次benchmark：测量per-token时间和峰值显存。
    包含warmup步骤以消除冷启动影响（符合论文常见做法）。
    """
    # 重新清空 KV cache，防止跨实验污染
    model.reset_cache()

    # 构造随机输入
    idx = torch.randint(0, vocab_size, (batch, seq_len), device=device)

    # Warmup: 预热GPU，消除冷启动影响（论文中常见做法）
    warmup_steps = 3
    for _ in range(warmup_steps):
        model.reset_cache()
        warm_logits, _ = model(idx, use_cache=use_cache)
        warm_token = warm_logits[:, -1].argmax(dim=-1, keepdim=True)
        warm_input = warm_token if use_cache else torch.cat([idx, warm_token], dim=1)
        _ = model(warm_input, use_cache=use_cache)
    if device.type == "cuda":
        torch.cuda.synchronize()
    # 清空 warmup 期间的缓存，避免 priming 时上下文累积导致超过 block_size
    model.reset_cache()

    # 显存峰值：分阶段统计 priming / 生成，并取最大值
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Prime both paths with the same prompt outside the generation timer. Use
    # the priming logit to produce a seed token, avoiding a duplicated prompt
    # token in the cached path.
    priming_logits, _ = model(idx, use_cache=use_cache)
    seed_token = priming_logits[:, -1].argmax(dim=-1, keepdim=True)
    idx = torch.cat([idx, seed_token], dim=1)
    idx_cond = seed_token if use_cache else idx

    # priming 阶段峰值
    priming_peak = get_peak_memory(device)

    # 生成阶段单独统计峰值
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # 计时只覆盖生成阶段
    start = time.perf_counter()

    for _ in range(gen_steps):
        logits, _ = model(idx_cond, use_cache=use_cache)
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_tok], dim=1)
        idx_cond = idx[:, -1:] if use_cache else idx

    if device.type == "cuda":
        torch.cuda.synchronize()

    time_ms = (time.perf_counter() - start) * 1000 / gen_steps
    gen_peak = get_peak_memory(device)
    overall_peak = max(priming_peak, gen_peak)
    return time_ms, priming_peak, gen_peak, overall_peak


def run_all(model: BigramLanguageModel, vocab_size: int, device: torch.device, gen_steps: int):
    results: List[dict] = []
    for seq_len in SEQ_LENGTHS:
        for batch in BATCH_SIZES:
            for use_cache in (False, True):
                times: List[float] = []
                prim_peaks: List[float] = []
                gen_peaks: List[float] = []
                overall_peaks: List[float] = []
                print(f"\n--- seq={seq_len}, batch={batch}, kv_cache={use_cache} ---")
                for _ in range(REPEATS):
                    t, m_prim, m_gen, m_overall = benchmark_one(
                        model=model,
                        vocab_size=vocab_size,
                        seq_len=seq_len,
                        batch=batch,
                        use_cache=use_cache,
                        device=device,
                        gen_steps=gen_steps,
                    )
                    times.append(t)
                    prim_peaks.append(m_prim)
                    gen_peaks.append(m_gen)
                    overall_peaks.append(m_overall)

                results.append(
                    {
                        "seq_len": seq_len,
                        "batch": batch,
                        "kv_cache": use_cache,
                        "time_ms": sum(times) / len(times),
                        "mem_priming_MB": sum(prim_peaks) / len(prim_peaks),
                        "mem_gen_MB": sum(gen_peaks) / len(gen_peaks),
                        "mem_overall_MB": sum(overall_peaks) / len(overall_peaks),
                    }
                )
    return results


def analyze_and_plot(results, gen_steps: int, out_dir: str = "results"):
    out_dir = Path(out_dir)
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "benchmark_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    # 汇总：速度提升 / 内存增加
    summary_rows = []
    for seq_len in SEQ_LENGTHS:
        for batch in BATCH_SIZES:
            no = df[(df.seq_len == seq_len) & (df.batch == batch) & (df.kv_cache == False)].iloc[0]
            yes = df[(df.seq_len == seq_len) & (df.batch == batch) & (df.kv_cache == True)].iloc[0]
            summary_rows.append(
                {
                    "seq_len": seq_len,
                    "batch": batch,
                    "gen_steps": gen_steps,
                    "time_no_kv": no.time_ms,
                    "time_kv": yes.time_ms,
                    "time_speedup": no.time_ms / yes.time_ms,
                    "mem_no_kv_overall": no.mem_overall_MB,
                    "mem_kv_overall": yes.mem_overall_MB,
                    "mem_increase_ratio": (yes.mem_overall_MB - no.mem_overall_MB) / max(no.mem_overall_MB, 1e-6),
                    "mem_no_kv_priming": no.mem_priming_MB,
                    "mem_kv_priming": yes.mem_priming_MB,
                    "mem_no_kv_gen": no.mem_gen_MB,
                    "mem_kv_gen": yes.mem_gen_MB,
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

    # 时间随序列长度
    for batch in BATCH_SIZES:
        dfb = df[df.batch == batch]
        plt.figure()
        x = SEQ_LENGTHS
        y0 = [dfb[(dfb.seq_len == L) & (dfb.kv_cache == False)].iloc[0].time_ms for L in x]
        y1 = [dfb[(dfb.seq_len == L) & (dfb.kv_cache == True)].iloc[0].time_ms for L in x]
        plt.plot(x, y0, marker="o", label="No KV Cache")
        plt.plot(x, y1, marker="o", label="With KV Cache")
        plt.xlabel("Sequence length")
        plt.ylabel("Per-token time (ms)")
        plt.title(f"Per-token inference time (batch={batch})")
        plt.grid(True)
        plt.legend()
        plt.savefig(figure_dir / f"time_vs_seq_batch{batch}.png")
        plt.close()

    # KV cache 内存随序列长度（overall_peak）
    for batch in BATCH_SIZES:
        dfb = df[(df.batch == batch) & (df.kv_cache == True)]
        plt.figure()
        y = [dfb[dfb.seq_len == L].iloc[0].mem_overall_MB for L in SEQ_LENGTHS]
        plt.plot(SEQ_LENGTHS, y, marker="o")
        plt.xlabel("Sequence length")
        plt.ylabel("Peak memory (MB)")
        plt.title(f"KV Cache peak memory (overall, batch={batch})")
        plt.grid(True)
        plt.savefig(figure_dir / f"mem_vs_seq_kv_batch{batch}.png")
        plt.close()

    print("[saved] all plots")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="cuda | cpu")
    parser.add_argument("--vocab-size", type=int, default=5000, help="synthetic vocab size")
    parser.add_argument("--generate-steps", type=int, default=50, help="per trial generated tokens")
    parser.add_argument("--out-dir", type=str, default="results", help="results directory")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("KV-Cache Benchmark (Raw Model, Random Init)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Generate steps per trial: {args.generate_steps}")
    print(f"Sequence lengths: {SEQ_LENGTHS}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Repeats per config: {REPEATS}")
    print("=" * 60)

    # 创建原始模型（随机初始化，不加载checkpoint）
    model = prepare_model(vocab_size=args.vocab_size, device=device)

    # 运行所有实验
    results = run_all(model, vocab_size=args.vocab_size, device=device, gen_steps=args.generate_steps)

    # 分析和可视化
    analyze_and_plot(results, gen_steps=args.generate_steps, out_dir=args.out_dir)

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
