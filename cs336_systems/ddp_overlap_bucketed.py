import torch
import torch.distributed as dist
from typing import List, Dict, Tuple, Any


class DDPOverlapBucketed(torch.nn.Module):
    """
    DDP wrapper that buckets parameters (reverse order of module.parameters()) and
    overlaps gradient communication with backward by issuing asynchronous all_reduce
    on bucket buffers once all grads in the bucket are ready.
    """

    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        assert dist.is_initialized(), "torch.distributed must be initialized"
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # broadcast parameters so all ranks start the same
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        # Build list of parameters that require gradients and save meta info
        params = [p for p in self.module.parameters() if p.requires_grad]
        # reverse order as recommended
        params = list(reversed(params))

        # Build buckets: each bucket contains a list of (param, numel, offset)
        bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.buckets: List[Dict[str, Any]] = []
        cur_bucket = {"params": [], "numel": 0, "bytes": 0}
        for p in params:
            numel = p.numel()
            bytes_p = numel * p.element_size()
            # if single param bigger than bucket_size_bytes, place it alone
            if cur_bucket["params"] and (cur_bucket["bytes"] + bytes_p > bucket_size_bytes):
                self.buckets.append(cur_bucket)
                cur_bucket = {"params": [], "numel": 0, "bytes": 0}
            cur_bucket["params"].append(p)
            cur_bucket["numel"] += numel
            cur_bucket["bytes"] += bytes_p
        if cur_bucket["params"]:
            self.buckets.append(cur_bucket)

        # For each bucket create:
        # - a flat buffer tensor on the same device as the bucket's params (we'll allocate per-device lazily)
        # - mapping param -> (bucket_id, offset_in_bucket, numel)
        self.param_to_bucket: Dict[int, Tuple[int, int, int]] = {}
        self.bucket_buffers: List[torch.Tensor] = [None] * len(self.buckets)
        self.bucket_devices: List[torch.device] = [None] * len(self.buckets)
        self.bucket_num_params: List[int] = [len(b["params"]) for b in self.buckets]
        self.bucket_ready_counts: List[int] = [0] * len(self.buckets)
        self.bucket_param_order: List[List[torch.nn.Parameter]] = [b["params"] for b in self.buckets]

        for b_idx, b in enumerate(self.buckets):
            offset = 0
            for p in b["params"]:
                self.param_to_bucket[id(p)] = (b_idx, offset, p.numel())
                offset += p.numel()

        # store outstanding async handles: list of (handle, bucket_id)
        self._outstanding_handles: List[Tuple[torch.distributed.Work, int]] = []

        # register hooks for each parameter to be called when grad is ready
        for p in params:
            # register a hook on gradient tensor creation
            # the hook runs during backward when grad for p is produced
            p.register_hook(self._make_param_hook(p))

        # track iteration to reset counters between iterations
        self._iteration = 0
        self._bucket_buffers_initialized = [False] * len(self.buckets)

    def _make_param_hook(self, param: torch.nn.Parameter):
        """
        Return a hook callable capturing param.
        The hook will be called with the grad Tensor when grad for param is ready.
        """

        def hook(grad):
            # Called in autograd when param.grad (the grad) is ready.
            # Copy grad into bucket buffer and possibly trigger bucket all-reduce.
            info = self.param_to_bucket.get(id(param), None)
            if info is None:
                return grad
            b_idx, offset, numel = info

            # allocate bucket buffer lazily on the same device/dtype as grad
            if not self._bucket_buffers_initialized[b_idx]:
                device = grad.device
                dtype = grad.dtype
                total_numel = self.buckets[b_idx]["numel"]
                # create flat buffer
                buf = torch.empty(total_numel, dtype=dtype, device=device)
                self.bucket_buffers[b_idx] = buf
                self.bucket_devices[b_idx] = device
                self._bucket_buffers_initialized[b_idx] = True

            # flatten grad and copy into buffer at offset
            flat = grad.contiguous().view(-1)
            self.bucket_buffers[b_idx].narrow(0, offset, numel).copy_(flat)

            # increment ready count for this bucket
            self.bucket_ready_counts[b_idx] += 1

            # if all params in bucket are ready, issue async all_reduce
            if self.bucket_ready_counts[b_idx] == self.bucket_num_params[b_idx]:
                # kick off async all_reduce on the bucket buffer
                buf = self.bucket_buffers[b_idx]
                # Use in-place all_reduce with async_op
                handle = dist.all_reduce(buf, op=dist.ReduceOp.SUM, async_op=True)
                self._outstanding_handles.append((handle, b_idx))
            return grad

        return hook

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        Wait for outstanding asynchronous bucket all-reduces to finish,
        divide by world size and copy results back to the parameter.grad tensors.
        Must be called after backward and before optimizer.step().
        """
        # wait for each outstanding handle and copy back
        while self._outstanding_handles:
            handle, b_idx = self._outstanding_handles.pop(0)
            # wait for completion
            handle.wait()
            # after completion buffer contains summed grads across ranks
            buf = self.bucket_buffers[b_idx]
            # average
            buf_div = buf.div_(self.world_size)

            # copy back into each param.grad
            offset = 0
            for p in self.bucket_param_order[b_idx]:
                # If p.grad is None (shouldn't happen), create it
                if p.grad is None:
                    p.grad = torch.empty_like(p.data)
                numel = p.numel()
                # view buffer slice then reshape to param shape
                slice_flat = buf_div.narrow(0, offset, numel)
                # reshape and copy
                p.grad.data.copy_(slice_flat.view_as(p.data))
                offset += numel

        # reset ready counts for next iteration and clear any buffers if devices changed
        for i in range(len(self.bucket_ready_counts)):
            self.bucket_ready_counts[i] = 0
        self._iteration += 1

    # optional helper to explicitly zero gradients of wrapped module
    def zero_grad(self):
        self.module.zero_grad()

    # expose named_parameters / parameters to be compatible-ish with torch.nn.Module
    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)

    def named_parameters(self, recurse: bool = True):
        return self.module.named_parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)
