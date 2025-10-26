import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as TorchDDP
import time
from typing import List
import os
import torch.cuda.nvtx as nvtx


# Import your DDP implementation
# Assuming the DDP class is in a file called ddp_overlap.py
# from ddp_overlap import DDP, my_get_ddp_individual_parameters


class GPT2Config:
    """GPT2-small configuration"""
    def __init__(self):
        self.vocab_size = 50257
        self.n_positions = 1024
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12
        self.batch_size = 8
        self.seq_length = 512
    # """GPT-2 XL (Extra Large) configuration (~1.5B parameters)"""
    # def __init__(self):
    #     self.vocab_size = 50257        # 与所有 GPT-2 版本相同
    #     self.n_positions = 1024        # 最大上下文长度
    #     self.n_embd = 1600             # 隐藏层维度（embedding size）
    #     self.n_layer = 48              # Transformer 层数
    #     self.n_head = 25               # 注意力头数（注意：1600 / 25 = 64，每个头64维）
    #     # 注意：batch_size 和 seq_length 不是模型结构参数，通常由训练/推理设置决定
    #     # 以下仅为示例，可根据实际需求调整
    #     self.batch_size = 4            # 通常较小，因模型很大
    #     self.seq_length = 1024         # 可使用最大长度，但受显存限制
    # """GPT-2 Medium configuration (~355M parameters)"""
    # def __init__(self):
    #     self.vocab_size = 50257        # 词表大小，所有 GPT-2 版本一致
    #     self.n_positions = 1024        # 最大序列长度
    #     self.n_embd = 1024             # 隐藏层维度（embedding size）
    #     self.n_layer = 24              # Transformer 层数
    #     self.n_head = 16               # 注意力头数（1024 / 16 = 64，每头64维）
    #     # 以下为训练/推理时的示例设置，非模型结构参数
    #     self.batch_size = 8            # 根据硬件调整
    #     self.seq_length = 1024         # 可设为 ≤1024 的任意值

class SimpleGPT2(nn.Module):
    """Simplified GPT2-small model for benchmarking"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(position_ids)
        hidden_states = tok_emb + pos_emb
        
        # Pass through transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states)
        
        # Final layer norm and projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits


class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    
    def forward(self, x):
        batch_size, seq_length, n_embd = x.shape
        
        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v
        
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_length, n_embd)
        y = self.c_proj(y)
        
        return y


class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# ===== Baseline DDP Implementations =====

class DDPPerParameter(nn.Module):
    """DDP that issues all-reduce for each parameter separately (no overlap)"""
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._broadcast_parameters()
    
    def _broadcast_parameters(self):
        if not dist.is_initialized():
            return
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def synchronize_gradients(self):
        """Synchronous all-reduce for each parameter after backward"""
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)


class DDPConcatenated(nn.Module):
    """DDP that concatenates all gradients and issues single all-reduce (no overlap)"""
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._broadcast_parameters()
    
    def _broadcast_parameters(self):
        if not dist.is_initialized():
            return
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def synchronize_gradients(self):
        """Concatenate all gradients and perform single all-reduce"""
        if not dist.is_initialized():
            return
        
        world_size = dist.get_world_size()
        
        # Collect all gradients
        grad_list = []
        params_with_grad = []
        for param in self.module.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.view(-1))
                params_with_grad.append(param)
        
        if len(grad_list) == 0:
            return
        
        # Concatenate all gradients
        flat_grad = torch.cat(grad_list)
        
        # Single all-reduce
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
        flat_grad.div_(world_size)
        
        # Copy back to parameters
        offset = 0
        for param in params_with_grad:
            numel = param.grad.numel()
            param.grad.copy_(flat_grad[offset:offset+numel].view_as(param.grad))
            offset += numel


# ===== Benchmarking Functions =====

def benchmark_ddp(rank, world_size, ddp_type='overlap', num_iterations=20, warmup=5):
    """
    Benchmark a specific DDP implementation
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        ddp_type: Type of DDP ('overlap', 'per_param', 'concatenated')
        num_iterations: Number of iterations to benchmark
        warmup: Number of warmup iterations
    """
    # Setup distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Create model
    config = GPT2Config()
    model = SimpleGPT2(config).to(device)
    
    # Wrap with appropriate DDP
    if ddp_type == 'overlap':
        # Use your overlapping DDP implementation
        from ddp_overlap_individual_parameters import my_get_ddp_individual_parameters
        ddp_model = my_get_ddp_individual_parameters(model)
    elif ddp_type == 'per_param':
        ddp_model = DDPPerParameter(model)
    elif ddp_type == 'concatenated':
        ddp_model = DDPConcatenated(model)
    else:
        raise ValueError(f"Unknown ddp_type: {ddp_type}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Warmup iterations
    for _ in range(warmup):
        input_ids = torch.randint(0, config.vocab_size, 
                                  (config.batch_size, config.seq_length), 
                                  device=device)
        labels = torch.randint(0, config.vocab_size, 
                              (config.batch_size, config.seq_length), 
                              device=device)
        
        optimizer.zero_grad()
        logits = ddp_model(input_ids)
        loss = criterion(logits.view(-1, config.vocab_size), labels.view(-1))
        loss.backward()
        
        # Synchronize gradients based on DDP type
        if ddp_type == 'overlap':
            ddp_model.finish_gradient_synchronization()
        elif ddp_type in ['per_param', 'concatenated']:
            ddp_model.synchronize_gradients()
        
        optimizer.step()
    
    # Synchronize before timing
    torch.cuda.synchronize()
    dist.barrier()
    
    # Benchmark iterations
    start_time = time.time()
    
    for _ in range(num_iterations):
        input_ids = torch.randint(0, config.vocab_size, 
                                  (config.batch_size, config.seq_length), 
                                  device=device)
        labels = torch.randint(0, config.vocab_size, 
                              (config.batch_size, config.seq_length), 
                              device=device)
        
        optimizer.zero_grad()
        nvtx.range_push("forward")
        logits = ddp_model(input_ids)
        nvtx.range_pop()

        nvtx.range_push("backward")
        loss = criterion(logits.view(-1, config.vocab_size), labels.view(-1))
        loss.backward()
        nvtx.range_pop()
        
        nvtx.range_push("gradient_sync")
        # Synchronize gradients based on DDP type
        if ddp_type == 'overlap':
            ddp_model.finish_gradient_synchronization()
        elif ddp_type in ['per_param', 'concatenated']:
            ddp_model.synchronize_gradients()
        nvtx.range_pop()
        
        nvtx.range_push("optimizer_step")
        optimizer.step()
        nvtx.range_pop()
    # Synchronize after timing
    torch.cuda.synchronize()
    dist.barrier()
    
    end_time = time.time()
    
    # Report results from rank 0
    if rank == 0:
        avg_time = (end_time - start_time) / num_iterations
        print(f"\n{'='*60}")
        print(f"DDP Type: {ddp_type}")
        print(f"Average time per iteration: {avg_time*1000:.2f} ms")
        print(f"Total time for {num_iterations} iterations: {end_time - start_time:.2f} s")
        print(f"{'='*60}\n")
    
    dist.destroy_process_group()


def run_all_benchmarks():
    """Run benchmarks for all three DDP implementations"""
    world_size = 2
    
    print("\n" + "="*60)
    print("Starting DDP Benchmarks (GPT2-small, 2 GPUs)")
    print("="*60)
    
    for ddp_type in ['per_param', 'concatenated', 'overlap']:
        print(f"\nBenchmarking {ddp_type}...")
        mp.spawn(benchmark_ddp, 
                args=(world_size, ddp_type, 20, 5),
                nprocs=world_size,
                join=True)


if __name__ == '__main__':
    # Make sure you have 2 GPUs available
    if torch.cuda.device_count() < 2:
        print("Error: This benchmark requires at least 2 GPUs")
        exit(1)
    
    run_all_benchmarks()