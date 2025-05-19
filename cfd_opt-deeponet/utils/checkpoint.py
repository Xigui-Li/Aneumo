"""
Checkpoint Utilities for DeepONet-CFD Project

This module provides functions to save and load model and optimizer states during training.
It supports saving checkpoints for models wrapped with PyTorch's Fully Sharded Data Parallel (FSDP).

Functions:
- save_trunk_net: Save the state of the trunk network.
- save_branch_bc_net: Save the state of the branch_bc network.
- save_branch_bp_net: Save the state of the branch_bp network.
- save_checkpoint_opt: Save the state of the optimizer.
- load_checkpoint: Load the state of a model from a checkpoint.
- load_checkpoint_opt: Load the state of an optimizer from a checkpoint.

Dependencies:
- PyTorch
- FSDP (Fully Sharded Data Parallel) for distributed training
- process_state_dict: A utility function to process state dictionaries for compatibility.

"""
import os
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from .process_dict import process_state_dict
# ----------------------------------------
# 保存 checkpoint
def save_trunk_net(epoch, model, optimizer, save_folder, rank=0):
    """Save trunk_net states."""
    if rank == 0:  # Only save in the main process
        checkpoint_path = os.path.join(save_folder, f"checkpoint_{epoch}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model_path = os.path.join(checkpoint_path, f"trunk_net_{epoch}.pth")

        # Save model states (FSDP wrapped model)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True)):
            model_state = model.state_dict()
            torch.save(model_state, model_path)
        print(f"Checkpoint saved at epoch {epoch} in {checkpoint_path}")

def save_branch_bc_net(epoch, model, optimizer, save_folder, rank=0):
    """Save branch_bc_net states."""
    if rank == 0:  # Only save in the main process
        checkpoint_path = os.path.join(save_folder, f"checkpoint_{epoch}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model_path = os.path.join(checkpoint_path, f"branch_bc_net_{epoch}.pth")
        # Save model states (FSDP wrapped model)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True)):
            model_state = model.state_dict()
            torch.save(model_state, model_path)
        print(f"Checkpoint saved at epoch {epoch} in {checkpoint_path}")

def save_branch_bp_net(epoch, model, optimizer, save_folder, rank=0):
    """Save branch_bp_net states."""
    if rank == 0:  # Only save in the main process
        checkpoint_path = os.path.join(save_folder, f"checkpoint_{epoch}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model_path = os.path.join(checkpoint_path, f"branch_bp_net_{epoch}.pth")
        # Save model states (FSDP wrapped model)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True)):
            model_state = model.state_dict()
            torch.save(model_state, model_path)
        print(f"Checkpoint saved at epoch {epoch} in {checkpoint_path}")


def save_checkpoint_opt(epoch, optimizer, save_folder, rank=0):
    """Save optimizer states."""
    if rank == 0:  # Only save in the main process
        checkpoint_path = os.path.join(save_folder, f"checkpoint_{epoch}")
        optimizer_path = os.path.join(checkpoint_path, f"optimizer_{epoch}.pth")
        os.makedirs(optimizer_path, exist_ok=True)
        # Save optimizer state
        torch.save(optimizer.state_dict(), optimizer_path)
        print(f"Checkpoint saved at epoch {epoch} in {checkpoint_path}")

# ----------------------------------------
# 加载 checkpoint
def load_checkpoint(epoch, model, optimizer, save_folder, rank=0):
    """Load model states."""
    checkpoint_path = os.path.join(save_folder, f"checkpoint_{epoch}")
    model_path = os.path.join(checkpoint_path, f"model_{epoch}.pth")
    
    if os.path.exists(model_path):
        try:
            # Load model
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                model_state = torch.load(model_path)
                model_state = process_state_dict(model_state)
            print(f"Loaded model from {model_path}")
            return True, model_state

        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return False, None
    else:
        print(f"No model checkpoint found at {model_path}, starting from scratch.")
        return False, None
def load_checkpoint_opt(epoch, optimizer, save_folder, rank=0):
    """Load optimizer states."""
    checkpoint_path = os.path.join(save_folder, f"checkpoint_{epoch}")
    optimizer_path = os.path.join(checkpoint_path, f"optimizer_{epoch}.pth")
    if os.path.exists(optimizer_path):
        try:
            # Load optimizer
            optimizer_state = torch.load(optimizer_path)
            optimizer.load_state_dict(optimizer_state)
            print(f"Loaded optimizer from {optimizer_path}")

        except Exception as e:
            print(f"Error loading checkpoint at {checkpoint_path}: {str(e)}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")
