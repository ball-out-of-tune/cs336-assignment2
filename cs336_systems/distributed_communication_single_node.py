import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def benchmark_allreduce(rank, world_size, backend, tensor_size_mb, device_type):
    setup(rank, world_size, backend)

    # device setup
    device = torch.device(f"{device_type}:{rank}" if device_type == "cuda" else "cpu")
    torch.manual_seed(rank)

    # data tensor
    num_elements = tensor_size_mb * 1024 * 1024 // 4  # float32 = 4 bytes
    data = torch.randn(num_elements, device=device)

    # warm-up (important)
    for _ in range(5):
        dist.all_reduce(data)

    torch.cuda.synchronize(device) if device_type == "cuda" else None
    dist.barrier()

    # benchmark
    start = time.time()
    for _ in range(10):
        dist.all_reduce(data)
    torch.cuda.synchronize(device) if device_type == "cuda" else None
    dist.barrier()
    end = time.time()

    avg_time = (end - start) / 10
    # collect results
    times = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(times, torch.tensor([avg_time], device=device))

    if rank == 0:
        times_cpu = [t.item() for t in times]
        print(f"\n[Backend={backend}, Device={device_type}, Size={tensor_size_mb}MB, World={world_size}]")
        print(f"Avg all-reduce time (per rank): {times_cpu}")
        print(f"Mean time: {sum(times_cpu)/len(times_cpu):.4f} s")

    cleanup()

def run_experiment(backend, device_type, sizes, world_size):
    for size in sizes:
        mp.spawn(
            benchmark_allreduce,
            args=(world_size, backend, size, device_type),
            nprocs=world_size,
            join=True
        )

if __name__ == "__main__":
    sizes = [1, 10, 100]  # 可扩展到 1024 即 1GB
    # Gloo on CPU
    run_experiment("gloo", "cpu", sizes, world_size=2)
    run_experiment("gloo", "cpu", sizes, world_size=4)

    # NCCL on GPU
    run_experiment("nccl", "cuda", sizes, world_size=2)
    run_experiment("nccl", "cuda", sizes, world_size=4)
