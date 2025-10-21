import torch
import torch.nn.functional as F
import time
import pandas as pd
from itertools import product
import torch._functorch.config as ftc
ftc.donated_buffer = False

# === Step 1: 定义基础 Attention 函数 ===
def attention_forward(Q, K, V):
    """Basic attention implementation without multi-head"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output

# === Step 2: 创建 compiled 版本 ===
compiled_attention_forward = torch.compile(attention_forward)

# === Step 3: Benchmark 函数（含 compiled 版本） ===
def benchmark_attention(include_compiled=True):
    batch_size = 8
    d_model_list = [16, 32, 64]
    seq_len_list = [256, 1024]

    results = []

    for d_model, seq_len in product(d_model_list, seq_len_list):
        for version, func in [
            ("uncompiled", attention_forward),
            ("compiled", compiled_attention_forward),
        ] if include_compiled else [("uncompiled", attention_forward)]:

            print(f"Testing {version} | d_model={d_model}, seq_len={seq_len}")
            try:
                Q = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
                K = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
                V = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)

                # warm up
                for _ in range(5):
                    out = func(Q, K, V)
                    loss = out.sum()
                    loss.backward()
                    torch.cuda.synchronize()
                    Q.grad = K.grad = V.grad = None  # 清除梯度

                torch.cuda.empty_cache()

                # forward timing
                forward_times = []
                for _ in range(20):
                    start = time.time()
                    out = func(Q, K, V)
                    torch.cuda.synchronize()
                    forward_times.append(time.time() - start)

                # backward timing
                out = func(Q, K, V)
                loss = out.sum()
                backward_times = []
                for _ in range(20):
                    start = time.time()
                    loss.backward(retain_graph=True)
                    torch.cuda.synchronize()
                    backward_times.append(time.time() - start)

                avg_forward = sum(forward_times) / len(forward_times) * 1000
                avg_backward = sum(backward_times) / len(backward_times) * 1000

                results.append({
                    'version': version,
                    'd_model': d_model,
                    'seq_len': seq_len,
                    'forward_time_ms': avg_forward,
                    'backward_time_ms': avg_backward,
                    'status': 'Success'
                })

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    results.append({
                        'version': version,
                        'd_model': d_model,
                        'seq_len': seq_len,
                        'forward_time_ms': None,
                        'backward_time_ms': None,
                        'status': 'OOM'
                    })
                    print(f"OOM for {version} d_model={d_model}, seq_len={seq_len}")
                    torch.cuda.empty_cache()
                else:
                    raise e

    return pd.DataFrame(results)

# === Step 4: Run benchmark ===
df = benchmark_attention()
print(df)

# === Step 5: 输出对比表 ===
pivot = df.pivot_table(index=["d_model", "seq_len"], columns="version", values=["forward_time_ms", "backward_time_ms"])
print("\n=== Comparison Table (compiled vs uncompiled) ===")
print(pivot)
