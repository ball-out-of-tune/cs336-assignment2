import torch
import torch.nn.functional as F
import time
import pandas as pd
from itertools import product

def attention_forward(Q, K, V):
    """Basic attention implementation without multi-head"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output

def benchmark_attention():
    batch_size = 8
    # d_model_list = [16, 32, 64, 128]
    d_model_list = [16, 32, 64, 128]
    # seq_len_list = [256, 1024, 4096, 8192, 16384]
    seq_len_list = [256, 1024, 4096, 8192]

    
    results = []
    
    for d_model, seq_len in product(d_model_list, seq_len_list):
        print(f"Testing d_model={d_model}, seq_len={seq_len}")
        
        try:
            # Create random inputs
            Q = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
            K = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
            V = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
            
            # Warm up
            for _ in range(10):
                output = attention_forward(Q, K, V)
                loss = output.sum()
                loss.backward()
                torch.cuda.synchronize()
            
            # Clear memory
            torch.cuda.empty_cache()
            
            # Measure forward pass
            forward_times = []
            for _ in range(100):
                start = time.time()
                output = attention_forward(Q, K, V)
                torch.cuda.synchronize()
                forward_times.append(time.time() - start)
            
            # Measure memory before backward
            memory_before_backward = torch.cuda.memory_allocated()
            
            # Create fresh output for backward timing
            output = attention_forward(Q, K, V)
            loss = output.sum()
            
            # Measure backward pass
            backward_times = []
            for _ in range(100):
                start = time.time()
                loss.backward(retain_graph=True)
                torch.cuda.synchronize()
                backward_times.append(time.time() - start)
            
            avg_forward = sum(forward_times) / len(forward_times) * 1000  # convert to ms
            avg_backward = sum(backward_times) / len(backward_times) * 1000  # convert to ms
            
            results.append({
                'd_model': d_model,
                'seq_len': seq_len,
                'forward_time_ms': avg_forward,
                'backward_time_ms': avg_backward,
                'memory_before_backward_MB': memory_before_backward / 1024**2,
                'status': 'Success'
            })
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                results.append({
                    'd_model': d_model,
                    'seq_len': seq_len,
                    'forward_time_ms': None,
                    'backward_time_ms': None,
                    'memory_before_backward_MB': None,
                    'status': 'OOM'
                })
                print(f"OOM for d_model={d_model}, seq_len={seq_len}")
                torch.cuda.empty_cache()
            else:
                raise e
    
    return pd.DataFrame(results)

# Run benchmark
df = benchmark_attention()
print(df)