from contextlib import nullcontext
from datetime import datetime
import time
import pandas as pd
import torch
import argparse
from cs336_basics.model import BasicsTransformerLM
from tqdm import tqdm
import torch.cuda.nvtx as nvtx
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
def benchmark_model(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型配置字典
    model_configs = {
        'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
        'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
        'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
        'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
        '2.7B': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32}
    }
    
    if args.model_size in model_configs:
        config = model_configs[args.model_size]
    else:
        config = {
            'd_model': args.d_model,
            'd_ff': args.d_ff, 
            'num_layers': args.num_layers,
            'num_heads': args.num_heads
        }
    
    # 初始化模型
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=args.context_length,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=10000
    ).to(device)

    optimizer = AdamW(model.parameters())  # ✅ 新增 optimizer

    if args.compile:
        model = torch.compile(model)

    # 生成随机数据
    batch_size = 4
    inputs = torch.randint(0, 10000, (batch_size, args.context_length), device=device)
    targets = torch.randint(0, 10000, (batch_size, args.context_length), device=device)

    # 根据是否使用混合精度选择autocast上下文
    if args.mixed_precision:
        autocast_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        print("Using BF16 mixed precision")
    else:
        autocast_context = nullcontext()
        print("Using FP32 precision")

    # ---------------- 预热 ----------------
    print(f"Running {args.warmup_steps} warm-up steps...")
    for _ in tqdm(range(args.warmup_steps)):
        if args.forward_only:
            with torch.no_grad(), autocast_context:
                _ = model(inputs)
        else:
            with autocast_context:
                logits = model(inputs)
                loss = cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    targets.view(-1)
                )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
    
    # ---------------- 测量阶段 ----------------
    print(f"Running {args.measurement_steps} measurement steps...")

    forward_times, backward_times, optimizer_times, total_times = [], [], [], []

    for _ in tqdm(range(args.measurement_steps)):
        torch.cuda.synchronize()
        start_total = time.time()

        with nvtx.range("complete_training_step"):
            with nvtx.range("forward_pass"):
                start_f = time.time()
                with autocast_context:
                    logits = model(inputs)
                torch.cuda.synchronize()
                forward_times.append((time.time() - start_f) * 1000)

            if not args.forward_only:
                with nvtx.range("loss_computation"):
                    with autocast_context:
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)), 
                            targets.view(-1)
                        )

                with nvtx.range("backward_pass"):
                    start_b = time.time()
                    loss.backward()
                    torch.cuda.synchronize()
                    backward_times.append((time.time() - start_b) * 1000)

                with nvtx.range("optimizer_step"):
                    start_o = time.time()
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.cuda.synchronize()
                    optimizer_times.append((time.time() - start_o) * 1000)
            else:
                backward_times.append(0)
                optimizer_times.append(0)

        torch.cuda.synchronize()
        total_times.append((time.time() - start_total) * 1000)

    # ---------------- 结果统计 ----------------
    results = {
        'forward_avg': sum(forward_times)/len(forward_times),
        'forward_std': torch.std(torch.tensor(forward_times)).item(),
        'backward_avg': sum(backward_times)/len(backward_times),
        'backward_std': torch.std(torch.tensor(backward_times)).item(),
        'optimizer_avg': sum(optimizer_times)/len(optimizer_times),
        'optimizer_std': torch.std(torch.tensor(optimizer_times)).item(),
        'total_avg': sum(total_times)/len(total_times),
        'total_std': torch.std(torch.tensor(total_times)).item()
    }

    print("\n==== Benchmark Summary (ms per step) ====")
    print(f"Forward:   {results['forward_avg']:.2f} ± {results['forward_std']:.2f}")
    if not args.forward_only:
        print(f"Backward:  {results['backward_avg']:.2f} ± {results['backward_std']:.2f}")
        print(f"Optimizer: {results['optimizer_avg']:.2f} ± {results['optimizer_std']:.2f}")
    print(f"Total:     {results['total_avg']:.2f} ± {results['total_std']:.2f}")
    print("=========================================")

    return results


def main():
    parser = argparse.ArgumentParser(description='Transformer Model Benchmarking')

    parser.add_argument('--model_size', type=str, default='small',
                       choices=['small', 'medium', 'large', 'xl', '2.7B'],
                       help='Model size preset')
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--context_length', type=int, default=512)
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--measurement_steps', type=int, default=10)
    parser.add_argument('--forward_only', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true', 
                       help='Use BF16 mixed precision')

    args = parser.parse_args()

    print(f"Benchmarking {args.model_size} model...")
    print(f"Context length: {args.context_length}")
    print(f"Forward only: {args.forward_only}")

    results = benchmark_model(args)


if __name__ == "__main__":
    main()
