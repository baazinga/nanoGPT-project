"""
GPU 信息检查脚本
用于查看当前使用的 GPU 型号、算力等信息
"""

import torch

def print_gpu_info():
    """打印 GPU 详细信息"""
    print("=" * 60)
    print("GPU Information")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，使用 CPU")
        return

    print(f"✅ CUDA 可用")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print()

    for i in range(torch.cuda.device_count()):
        print(f"--- GPU {i} ---")
        props = torch.cuda.get_device_properties(i)

        print(f"设备名称: {props.name}")
        print(f"显存大小: {props.total_memory / 1024**3:.2f} GB")
        print(f"计算能力: {props.major}.{props.minor}")
        print(f"多处理器数量: {props.multi_processor_count}")

        # 显存使用情况
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"已分配显存: {allocated:.2f} GB")
            print(f"已保留显存: {reserved:.2f} GB")

        print()

    # 当前使用的设备
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"当前使用设备: GPU {current_device}")
        print(f"设备名称: {torch.cuda.get_device_name(current_device)}")

    print("=" * 60)

    # 计算能力说明
    print("\n计算能力说明:")
    print("- 计算能力 (Compute Capability) 格式: major.minor")
    print("- 例如: 7.5 = Turing (RTX 20系列), 8.0 = Ampere (RTX 30系列)")
    print("- 8.6 = Ampere (RTX 30系列部分型号), 8.9 = Ada (RTX 40系列)")
    print("- 更高的计算能力通常意味着更好的性能")


if __name__ == "__main__":
    print_gpu_info()
