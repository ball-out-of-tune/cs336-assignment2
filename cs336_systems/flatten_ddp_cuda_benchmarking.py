import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim
from dataclasses import dataclass
from typing import List, Tuple

def setup(rank, world_size):
    """Initialize distributed training environment"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

@dataclass
class ModelConfig:
    """XL Model configuration from §1.1.2"""
    vocab_size: int = 50257  # GPT-2 vocab size
    d_model: int = 1600      # XL size
    n_layers: int = 48
    n_heads: int = 25
    d_ff: int = 6400         # 4 * d_model
    max_seq_len: int = 1024
    dropout: float = 0.1

class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model, 
            config.n_heads, 
            dropout=config.dropout,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    """Transformer-based language model (XL size)"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
    def forward(self, input_ids):
        B, T = input_ids.shape
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.position_embedding(pos)
        x = token_emb + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

class BenchmarkMetrics:
    """Track timing metrics"""
    def __init__(self):
        self.forward_times = []
        self.backward_times = []
        self.communication_times = []
        self.total_times = []
    
    def add_step(self, forward_time, backward_time, comm_time, total_time):
        self.forward_times.append(forward_time)
        self.backward_times.append(backward_time)
        self.communication_times.append(comm_time)
        self.total_times.append(total_time)
    
    def get_averages(self, warmup_steps=5):
        """Get average metrics excluding warmup"""
        def avg(lst):
            return sum(lst[warmup_steps:]) / len(lst[warmup_steps:]) if len(lst) > warmup_steps else 0
        
        return {
            'avg_forward_time': avg(self.forward_times),
            'avg_backward_time': avg(self.backward_times),
            'avg_communication_time': avg(self.communication_times),
            'avg_total_time': avg(self.total_times),
            'communication_percentage': (avg(self.communication_times) / avg(self.total_times) * 100) 
                                       if avg(self.total_times) > 0 else 0
        }

# 将下面的函数替换或与 run_naive_ddp_benchmark 同级放入脚本中

import torch._utils

def run_flatten_ddp_benchmark(
    rank: int,
    world_size: int,
    config: ModelConfig,
    batch_size=4,
    seq_len=32,
    num_steps=10,
    warmup_steps=5
) -> BenchmarkMetrics:
    setup(rank, world_size)
    torch.manual_seed(42)
    model = LanguageModel(config).cuda(rank)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # broadcast params from rank 0
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # Pre-compute list of parameters that will have grads (order matters for unflatten)
    # We will use the same ordering when flattening/unflattening
    grad_params = [p for p in model.parameters()]

    metrics = BenchmarkMetrics()
    if rank == 0:
        print(f"\n{'='*40}\nFlattened DDP Benchmark starting\n{'='*40}")

    for step in range(num_steps):
        step_start = time.time()

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=rank)
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=rank)

        torch.cuda.synchronize()
        forward_start = time.time()

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, config.vocab_size), targets.view(-1))

        torch.cuda.synchronize()
        forward_time = time.time() - forward_start

        # backward
        backward_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - backward_start

        # ---------------- Communication: flattened all-reduce ----------------
        comm_start = time.time()

        # collect gradients (only tensors with grad)
        grads = []
        grads_param_index = []  # keep index mapping to params (optional)
        for p in grad_params:
            if p.grad is not None:
                # ensure contiguous
                grads.append(p.grad.data)
                grads_param_index.append(p)

        if len(grads) > 0:
            # flatten -> single tensor
            flat = torch._utils._flatten_dense_tensors(grads)
            # all-reduce once
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            # average
            flat.div_(world_size)
            # unflatten back to list of tensors (same shapes as grads)
            new_grads = torch._utils._unflatten_dense_tensors(flat, grads)
            # copy back to param.grad
            for p, new_g in zip(grads_param_index, new_grads):
                p.grad.data.copy_(new_g)

        torch.cuda.synchronize()
        comm_time = time.time() - comm_start
        # --------------------------------------------------------------------

        optimizer.step()
        torch.cuda.synchronize()
        total_time = time.time() - step_start

        metrics.add_step(forward_time, backward_time, comm_time, total_time)

        if rank == 0 and (step < 5 or step % 5 == 0):
            print(f"Step {step:3d} | Total: {total_time*1000:6.1f}ms | "
                  f"Forward: {forward_time*1000:6.1f}ms | "
                  f"Backward: {backward_time*1000:6.1f}ms | "
                  f"Comm: {comm_time*1000:6.1f}ms ({comm_time/total_time*100:5.1f}%)")

    cleanup()
    return metrics


def benchmark_worker(rank, world_size, config, result_queue):
    """Worker function for multiprocessing"""
    try:
        metrics = run_flatten_ddp_benchmark(rank, world_size, config)
        if rank == 0:
            result_queue.put(metrics)
    except Exception as e:
        print(f"Rank {rank} error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main benchmark execution"""
    world_size = 2  # 1 node x 2 GPUs
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires GPUs.")
        return
    
    if torch.cuda.device_count() < world_size:
        print(f"ERROR: Need {world_size} GPUs, but only {torch.cuda.device_count()} available.")
        return
    
    # XL model configuration
    config = ModelConfig()

    #smaller size model
    # config = ModelConfig(
    #     vocab_size=50257,
    #     d_model=768,    # 缩小模型维度
    #     n_layers=12,     # 缩少层数
    #     n_heads=12,      # 少些注意力头
    #     d_ff=3072,       # MLP 隐藏层
    #     max_seq_len=128, # 缩短序列长度
    #     dropout=0.1
    # )
    
    # Run benchmark
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=benchmark_worker, 
            args=(rank, world_size, config, result_queue)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Get results
    if not result_queue.empty():
        metrics = result_queue.get()
        averages = metrics.get_averages(warmup_steps=5)
        
        print(f"\n{'='*60}")
        print(f"Benchmark Results (averaged over {len(metrics.total_times) - 5} steps)")
        print(f"{'='*60}")
        print(f"Average time per training step:     {averages['avg_total_time']*1000:8.2f} ms")
        print(f"  - Forward pass:                    {averages['avg_forward_time']*1000:8.2f} ms")
        print(f"  - Backward pass:                   {averages['avg_backward_time']*1000:8.2f} ms")
        print(f"  - Gradient communication:          {averages['avg_communication_time']*1000:8.2f} ms")
        print(f"\nCommunication overhead:              {averages['communication_percentage']:8.1f} %")
        print(f"{'='*60}\n")
        
        # Additional analysis
        print("Key Observations:")
        print(f"1. Naive DDP communicates {sum(p.numel() for p in LanguageModel(config).parameters()) / 1e6:.1f}M parameters")
        print(f"2. Each parameter is all-reduced individually (high overhead)")
        print(f"3. Communication takes {averages['communication_percentage']:.1f}% of total training time")
        print(f"4. This overhead can be reduced by bucketing gradients")

if __name__ == "__main__":
    main()