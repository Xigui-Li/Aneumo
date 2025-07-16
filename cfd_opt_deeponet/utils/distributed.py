"""
Distributed Utilities for DeepONet-CFD Project

This module provides helper functions for initializing and managing
PyTorch's distributed training environment.

Functions:
- init_process_group: Initializes the process group for distributed training using the NCCL backend.
"""
import torch.distributed as dist

def init_process_group():
    """
    Initialize the distributed process group if it is not already initialized.

    Returns:
        tuple: (int, int) - The rank of the current process and the total number of processes.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size()