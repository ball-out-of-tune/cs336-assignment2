from datetime import datetime
import time
import timeit
import pandas as pd
import torch
import argparse
import argparse
import sys
from pathlib import Path

# 添加 cs336-basics 目录到 sys.path
basics_dir = Path(__file__).parent.parent / 'cs336-basics'
sys.path.insert(0, str(basics_dir))
from model import BasicsTransformerLM
from tqdm import tqdm
import torch.cuda.nvtx as nvtx


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
    
    # 获取指定模型的配置
    if args.model_size in model_configs:
        config = model_configs[args.model_size]
    else:
        # 如果指定了具体参数，使用自定义配置
        config = {
            'd_model': args.d_model,
            'd_ff': args.d_ff, 
            'num_layers': args.num_layers,
            'num_heads': args.num_heads
        }
    
    # 初始化模型
    model = BasicsTransformerLM(
        vocab_size=10000,  # 固定词汇表大小
        context_length=args.context_length,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=10000
    ).to(device)
    
    if args.compile:
        model = torch.compile(model)
    
    # 生成随机数据
    batch_size = 4  # 固定batch size
    inputs = torch.randint(0, 10000, (batch_size, args.context_length), device=device)
    targets = torch.randint(0, 10000, (batch_size, args.context_length), device=device)
    
    # 预热步骤
    print(f"Running {args.warmup_steps} warm-up steps...")
    for _ in tqdm(range(args.warmup_steps)):
        if args.forward_only:
            # 仅前向传播
            with torch.no_grad():
                logits = model(inputs)
        else:
            # 前向+后向传播
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            loss.backward()
            model.zero_grad()
        
        torch.cuda.synchronize()  # 等待GPU完成
    
    # 测量步骤
    print(f"Running {args.measurement_steps} measurement steps...")
    forward_times = []
    backward_times = []
    
    for _ in tqdm(range(args.measurement_steps)):
        if args.forward_only:
            # 仅测量前向传播时间
            start_time = timeit.default_timer()
            with torch.no_grad():
                logits = model(inputs)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            forward_times.append((end_time - start_time) * 1000)  # 转换为毫秒
        else:
            # 测量前向传播时间
            start_forward = timeit.default_timer()
            logits = model(inputs)
            torch.cuda.synchronize()
            end_forward = timeit.default_timer()
            
            # 测量后向传播时间  
            start_backward = timeit.default_timer()
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            loss.backward()
            torch.cuda.synchronize()
            end_backward = timeit.default_timer()
            
            forward_times.append((end_forward - start_forward) * 1000)
            backward_times.append((end_backward - start_backward) * 1000)
            
            model.zero_grad()
    
    # 输出结果
    if args.forward_only:
        avg_forward = sum(forward_times) / len(forward_times)
        std_forward = torch.std(torch.tensor(forward_times)).item()
        print(f"\nForward Pass Results:")
        print(f"Average time: {avg_forward:.2f} ms")
        print(f"Standard deviation: {std_forward:.2f} ms")
        return {'forward_avg': avg_forward, 'forward_std': std_forward}
    else:
        avg_forward = sum(forward_times) / len(forward_times)
        avg_backward = sum(backward_times) / len(backward_times)
        std_forward = torch.std(torch.tensor(forward_times)).item()
        std_backward = torch.std(torch.tensor(backward_times)).item()
        
        print(f"\nBenchmark Results:")
        print(f"Forward Pass - Average: {avg_forward:.2f} ms, Std: {std_forward:.2f} ms")
        print(f"Backward Pass - Average: {avg_backward:.2f} ms, Std: {std_backward:.2f} ms")
        print(f"Total - Average: {avg_forward + avg_backward:.2f} ms")
        
        return {
            'forward_avg': avg_forward, 'forward_std': std_forward,
            'backward_avg': avg_backward, 'backward_std': std_backward
        }

def main():
    parser = argparse.ArgumentParser(description='Transformer Model Benchmarking')
    
    # 模型大小参数
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['small', 'medium', 'large', 'xl', '2.7B'],
                       help='Model size preset')
    
    # 自定义模型参数（如果不用预设）
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=3072, help='Feedforward dimension')  
    parser.add_argument('--num_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    
    # 基准测试参数
    parser.add_argument('--context_length', type=int, default=512, help='Context length')
    parser.add_argument('--warmup_steps', type=int, default=5, help='Number of warm-up steps')
    parser.add_argument('--measurement_steps', type=int, default=10, help='Number of measurement steps')
    parser.add_argument('--forward_only', action='store_true', help='Only benchmark forward pass')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    
    args = parser.parse_args()
    
    print(f"Benchmarking {args.model_size} model...")
    print(f"Context length: {args.context_length}")
    print(f"Warm-up steps: {args.warmup_steps}")
    print(f"Measurement steps: {args.measurement_steps}")
    print(f"Forward only: {args.forward_only}")
    
    results = benchmark_model(args)
    # 保存结果
    df = save_results_to_table(args, results)
    
    # 打印表格格式
    print("\nResults Table:")
    print(df.to_string(index=False))



def save_results_to_table(args, results, filename=None):
    # 创建结果字典
    result_data = {
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'model_size': [args.model_size],
        'context_length': [args.context_length],
        'd_model': [args.d_model],
        'd_ff': [args.d_ff],
        'num_layers': [args.num_layers],
        'num_heads': [args.num_heads],
        'compile': [args.compile],
        'forward_only': [args.forward_only],
    }
    
    # 添加性能结果
    if args.forward_only:
        result_data.update({
            'forward_avg_ms': [results['forward_avg']],
            'forward_std_ms': [results['forward_std']],
            'backward_avg_ms': [None],
            'backward_std_ms': [None],
            'total_avg_ms': [results['forward_avg']]
        })
    else:
        result_data.update({
            'forward_avg_ms': [results['forward_avg']],
            'forward_std_ms': [results['forward_std']],
            'backward_avg_ms': [results['backward_avg']],
            'backward_std_ms': [results['backward_std']],
            'total_avg_ms': [results['forward_avg'] + results['backward_avg']]
        })
    
    # 创建DataFrame
    df = pd.DataFrame(result_data)
    
    # 生成文件名
    if filename is None:
        filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    import os
    # 保存为不同格式
    folder_name = "benchmark_results"
    # csv文件路径
    csv_filename = os.path.join(folder_name, filename)
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    
    # 也可以保存为Excel
    excel_filename = filename.replace('.csv', '.xlsx')
    # Excel 文件路径
    excel_filename = os.path.join(folder_name, excel_filename)
    df.to_excel(excel_filename, index=False)
    print(f"Results also saved to {excel_filename}")
    
    return df    
    
if __name__ == "__main__":
    main()


