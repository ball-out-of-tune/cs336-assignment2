import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        return self.net(x)

def run_single_process(model, data, target, world_size, epochs=10, lr=1e-2):
    """
    模拟 DDP 行为的单进程训练：
    1. 将数据分成 world_size 个分片
    2. 在每个分片上分别计算梯度（梯度会累积）
    3. 对累积的梯度求平均
    4. 执行参数更新
    """
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = data.size(0)
    shard_size = n // world_size

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 模拟每个 rank 的独立计算
        # 每个分片的梯度会累积到同一个参数上
        for rank in range(world_size):
            local_data = data[rank * shard_size:(rank + 1) * shard_size]
            local_target = target[rank * shard_size:(rank + 1) * shard_size]
            
            out = model(local_data)
            loss = loss_fn(out, local_target)
            loss.backward()  # 梯度自动累积
        
        # 模拟 all-reduce：对累积的梯度求平均
        for param in model.parameters():
            if param.grad is not None:
                param.grad /= world_size
        
        optimizer.step()

    return [p.detach().clone() for p in model.parameters()]

def run_naive_ddp(rank, world_size, data, target, epochs=10, lr=1e-2):
    setup(rank, world_size)

    # 每个进程初始化相同的模型
    torch.manual_seed(0) 
    model = ToyModel()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 将 rank 0 的参数广播到所有进程，确保初始模型一致
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # 切分数据：每个进程只处理自己的一部分
    n = data.size(0)
    shard_size = n // world_size
    local_data = data[rank * shard_size:(rank + 1) * shard_size]
    local_target = target[rank * shard_size:(rank + 1) * shard_size]

    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(local_data)
        loss = loss_fn(output, local_target)
        loss.backward()

        # all-reduce 每个参数的梯度并求平均
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size

        optimizer.step()

    # rank 0 返回模型参数用于比较
    final_params = [p.detach().clone() for p in model.parameters()]
    cleanup()
    return final_params

def ddp_worker(rank, world_size, data, target, result_queue):
    ddp_params = run_naive_ddp(rank, world_size, data, target)
    if rank == 0:
        result_queue.put(ddp_params)

def verify_naive_ddp(world_size=2):
    torch.manual_seed(0)
    data = torch.randn(64, 10)
    target = torch.randn(64, 1)

    # 单进程结果（模拟 DDP 的数据分片行为）
    torch.manual_seed(0)  # 确保模型初始化相同
    single_model = ToyModel()
    single_params = run_single_process(single_model, data, target, world_size)

    # 使用 Manager 创建共享队列
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=ddp_worker, args=(rank, world_size, data, target, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 获取结果
    ddp_params = result_queue.get()

    # 比较参数是否一致
    all_match = True
    for i, (p1, p2) in enumerate(zip(single_params, ddp_params)):
        if not torch.allclose(p1, p2, atol=1e-6):
            print(f"❌ Parameter {i} mismatch!")
            print(f"Max difference: {(p1 - p2).abs().max().item():.2e}")
            all_match = False
            # 只打印第一个不匹配的参数
            break
    
    if all_match:
        print("✅ Naive DDP matches single-process training!")
        print(f"Verified with world_size={world_size}, data_size={data.size(0)}")

if __name__ == "__main__":
    verify_naive_ddp(world_size=2)