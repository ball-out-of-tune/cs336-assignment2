import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Any, List


class DDP(nn.Module):
    """
    Distributed Data Parallel wrapper that overlaps gradient communication
    with backward pass computation.
    
    This implementation:
    1. Broadcasts initial parameters to all ranks
    2. Registers hooks on each parameter to trigger all-reduce operations
    3. Overlaps gradient communication with backward computation
    """
    
    def __init__(self, module: nn.Module):
        """
        Initialize DDP wrapper around a PyTorch module.
        
        Args:
            module: The PyTorch module to wrap for distributed training
        """
        super().__init__()
        self.module = module
        
        # Store handles for asynchronous operations
        self.grad_handles: List[Any] = []
        
        # Broadcast initial parameters to ensure all ranks start with same weights
        self._broadcast_parameters()
        
        # Register backward hooks for gradient synchronization
        self._register_hooks()
    
    def _broadcast_parameters(self):
        """
        Broadcast parameters and buffers from rank 0 to all other ranks.
        This ensures all processes start with identical model weights.
        """
        if not dist.is_initialized():
            return
        
        # Broadcast all parameters
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        # Broadcast all buffers (e.g., running stats in BatchNorm)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)
    
    def _register_hooks(self):
        """
        Register backward hooks on each parameter to trigger gradient
        all-reduce operations as soon as gradients are computed.
        
        This enables overlapping communication with computation.
        """
        if not dist.is_initialized():
            return
        
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._make_hook(param))
    
    def _make_hook(self, param: nn.Parameter):
        """
        Create a backward hook for a parameter that performs async all-reduce.
        
        Args:
            param: The parameter to create a hook for
            
        Returns:
            Hook function that will be called when gradient is ready
        """
        def hook(grad: torch.Tensor) -> torch.Tensor:
            """
            Hook function called when gradient is computed.
            Launches asynchronous all-reduce to average gradients across ranks.
            """
            if grad is None:
                return grad
            
            reduce_buffer = grad.detach().clone()

            # Perform asynchronous all-reduce to sum gradients across all ranks
            handle = dist.all_reduce(
                reduce_buffer, 
                op=dist.ReduceOp.SUM, 
                async_op=True
            )
            
            # Store handle to wait for completion later
            self.grad_handles.append((handle, param, reduce_buffer))
            
            return grad
        
        return hook
    
    def forward(self, *inputs, **kwargs):
        """
        Forward pass through the wrapped module.
        
        Args:
            *inputs: Positional arguments to pass to module
            **kwargs: Keyword arguments to pass to module
            
        Returns:
            Output from the wrapped module's forward pass
        """
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        """
        Wait for all asynchronous gradient synchronization operations to complete
        and average the gradients.
        
        This should be called after loss.backward() and before optimizer.step()
        to ensure all gradient communications have finished.
        """
        if not dist.is_initialized():
            return
        
        world_size = max(1, dist.get_world_size())
        
        # Wait for all async operations to complete and average gradients
        for handle, param, reduce_buffer in self.grad_handles:
            # Wait for the all-reduce operation to complete
            handle.wait()
            
           # Average (all_reduce summed across ranks)
            reduce_buffer.div_(world_size)

            # Ensure param.grad exists and has same shape/dtype/device, then copy.
            # If param.grad is None (shouldn't be in normal backward), we set it.
            if param.grad is None:
                # create a fresh tensor on same device/dtype
                param.grad = reduce_buffer.clone()
            else:
                # copy averaged content into param.grad in-place
                param.grad.copy_(reduce_buffer)
        
        # Clear handles for next iteration
        self.grad_handles.clear()


# Adapter functions for testing
def my_get_ddp_individual_parameters(module: nn.Module) -> DDP:
    """
    Adapter function to create DDP instance for testing.
    
    Args:
        module: PyTorch module to wrap
        
    Returns:
        DDP wrapper instance
    """
    return DDP(module)


# def ddp_individual_parameters_on_after_backward(ddp_model: DDP):
#     """
#     Optional adapter function called after backward pass.
    
#     This implementation doesn't require any action here since
#     hooks are automatically triggered during backward pass.
    
#     Args:
#         ddp_model: The DDP model instance
#     """
#     # No action needed - hooks are registered during __init__
#     # and triggered automatically during backward pass
#     pass