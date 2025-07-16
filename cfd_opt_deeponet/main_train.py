"""
DeepONet-based CFD Surrogate Model with FSDP Implementation

This module implements a surrogate model for computational fluid dynamics (CFD) 
using Deep Operator Networks (DeepONet) architecture. The implementation features:
- Fully Sharded Data Parallel (FSDP) training
- Mixed precision training
- Gradient accumulation
- Distributed training support
- Configurable model architecture

"""
import json
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import filters
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import ShardingStrategy

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from tqdm import tqdm
import torch.distributed as dist

from model import NeuralNetworks as nn_net
from loss import loss as loss
from dataloader.Dataset import CFDJobDataset as create_dataset
from dataloader.load_data import load_case_ids
from dataloader.build_dataloader import build_dataloaders
from utils.distributed import init_process_group
from utils.arg_parser import add_argument
from utils import checkpoint 
from utils.logger import setup_logger
from utils.process_dict import process_state_dict

from omegaconf import OmegaConf
from model.builder import build_networks

import torch.cuda.amp as amp 
from torch.distributed.fsdp import MixedPrecision

torch.manual_seed(20231028)

def main(train_data_loader, val_dataloader, save_folder, local_rank, world_size, train_sampler ,args , cfg):
    """
    Main training loop for the DeepONet-based CFD surrogate model with FSDP implementation.
    
    Args:
        train_data_loader (DataLoader): DataLoader for training data
        val_dataloader (DataLoader): DataLoader for validation data
        save_folder (str): Directory path to save checkpoints and logs
        local_rank (int): Local GPU rank for distributed training
        world_size (int): Total number of GPUs for distributed training
        train_sampler (DistributedSampler): Sampler for distributed training
        args (Namespace): Command line arguments
        cfg (OmegaConf): Configuration object containing model and training parameters
    
    The function implements:
        1. Distributed training setup with FSDP
        2. Model initialization (trunk and branch networks)
        3. Mixed precision training
        4. Gradient accumulation
        5. Periodic model checkpointing
        6. Training and validation metrics logging
    """
    # Initialize distributed environment and logger
    logger = setup_logger(save_folder)
    local_rank = local_rank%torch.cuda.device_count()
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # Build networks with FSDP wrapper
    nets, mp_policy, target_dtype = build_networks(cfg.model, cfg.fsdp)
    trunk_net = nets["trunk"].to(device)
    branch_bc_net = nets["branch_bc"].to(device)
    branch_bp_net = nets["branch_bp"].to(device)

    # Initialize loss function with FSDP wrapped models
    loss_data = loss.DataLoss_Bypass_Plus(trunk_net, branch_bc_net, branch_bp_net).to(device)

    # Define optimizer for all network parameters
    optimizer = torch.optim.Adam(
        list(trunk_net.parameters()) +
        list(branch_bc_net.parameters()) +
        list(branch_bp_net.parameters()),
        lr=cfg.train.learning_rate
    )

    # Load the pre-trained model (if it exists)
    judge_trunk_state, trunk_state = checkpoint.load_checkpoint(cfg.train.resume_epoch, trunk_net, optimizer, save_folder, local_rank)
    judge_branch_bc_state, branch_bc_state =checkpoint.load_checkpoint(cfg.train.resume_epoch, branch_bc_net, optimizer, save_folder, local_rank)
    judge_branch_bp_state, branch_bp_state =checkpoint.load_checkpoint(cfg.train.resume_epoch, branch_bp_net, optimizer, save_folder, local_rank)
    if judge_trunk_state:
        trunk_net.load_state_dict(trunk_state)
    if judge_branch_bc_state:
        branch_bc_net.load_state_dict(branch_bc_state)
    if judge_branch_bp_state:
        branch_bp_net.load_state_dict(branch_bp_state)
    
    N_train = len(train_data_loader)
    N_test = len(val_dataloader)
    trunk_net = torch.compile(trunk_net)
    branch_bc_net = torch.compile(branch_bc_net)
    branch_bp_net = torch.compile(branch_bp_net)

    # Training loop
    w_data_u = torch.tensor(1e3,device=device)
    w_data_p = torch.tensor(1e5,device=device)
    s_epoch = time.time()
    for epoch_id in range(cfg.train.resume_epoch, cfg.train.epochs+1):
        trunk_net.train()
        branch_bc_net.train()
        branch_bp_net.train()
        train_sampler.set_epoch(epoch_id)

        loss_pre_bn = torch.tensor(0.0, device=device)
        loss_vel_bn = torch.tensor(0.0, device=device)
        train_p_bn = torch.tensor(0.0, device=device)
        train_u_bn = torch.tensor(0.0, device=device)
        train_v_bn = torch.tensor(0.0, device=device)
        train_w_bn = torch.tensor(0.0, device=device)
        train_MNAE_p_bn = torch.tensor(0.0, device=device)
        train_MNAE_u_bn = torch.tensor(0.0, device=device)
        train_MNAE_v_bn = torch.tensor(0.0, device=device)
        train_MNAE_w_bn = torch.tensor(0.0, device=device)
        train_dp_bn = torch.tensor(0.0, device=device)
 
        test_p_bn = torch.tensor(0.0, device=device)
        test_u_bn = torch.tensor(0.0, device=device)
        test_v_bn = torch.tensor(0.0, device=device)
        test_w_bn = torch.tensor(0.0, device=device)
        test_MNAE_p_bn = torch.tensor(0.0, device=device)
        test_MNAE_u_bn = torch.tensor(0.0, device=device)
        test_MNAE_v_bn = torch.tensor(0.0, device=device)
        test_MNAE_w_bn = torch.tensor(0.0, device=device)
        test_dp_bn = torch.tensor(0.0, device=device)

        train_time_start = time.time()

        total_loss = torch.tensor(0.0, device=device)
        train_start = time.time()

        metrics = torch.zeros(N_train, 11, device=device)  # Number of indicators/metrics
        accum_steps = 4  # Update the gradient every four steps
        
        for i, (x_train, y_train, x_in_train) in tqdm(
            enumerate(train_data_loader),
            total=N_train,
        ):
            # optimizer.zero_grad()
            with amp.autocast(dtype=target_dtype):
                x_train = x_train.squeeze(0).to(device,non_blocking=True)
                y_train = y_train.squeeze(0).to(device, non_blocking=True)
                
                x_in_train = x_in_train.squeeze(0).squeeze(1).to(device, non_blocking=True)

                num_train = 1800 # number of CFD points after sampling
                loss_pre, loss_vel, train_p, train_u, train_v, train_w, train_MNAE_p, train_MNAE_u, train_MNAE_v, train_MNAE_w, train_dp = \
                    loss_data(x_train, y_train, x_in_train)

                loss_train = (w_data_p * loss_pre + w_data_u * loss_vel) / num_train
            loss_train.backward()
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            loss_pre_bn += loss_pre
            loss_vel_bn += loss_vel
            train_p_bn += train_p
            train_u_bn += train_u
            train_v_bn += train_v.sum()
            train_w_bn += train_w.sum()
            train_MNAE_p_bn += train_MNAE_p
            train_MNAE_u_bn += train_MNAE_u
            train_MNAE_v_bn += train_MNAE_v
            train_MNAE_w_bn += train_MNAE_w
            train_dp_bn += train_dp
        train_time_end = time.time()
        test_start = time.time()
        # Validation phase
        with torch.no_grad():
            with amp.autocast(dtype=target_dtype):
                for batch_id, (X_test, Y_test, X_in_test) in enumerate(val_dataloader):
                    X_test = X_test.to(device,non_blocking=True)
                    Y_test = Y_test.to(device,non_blocking=True)

                    X_in_test = X_in_test.squeeze(0).to(device,non_blocking=True)
                    X_in_test = X_in_test.squeeze(1) if len(X_in_test.shape) == 3 else X_in_test
                    _, _, test_p, test_u, test_v, test_w, test_MNAE_p, test_MNAE_u, test_MNAE_v, test_MNAE_w, test_dp = \
                        loss_data(X_test, Y_test, X_in_test)

                    test_p_bn += test_p
                    test_u_bn += test_u.sum()
                    test_v_bn += test_v.sum()
                    test_w_bn += test_w.sum()
                    test_MNAE_p_bn += test_MNAE_p
                    test_MNAE_u_bn += test_MNAE_u
                    test_MNAE_v_bn += test_MNAE_v
                    test_MNAE_w_bn += test_MNAE_w
                    test_dp_bn += test_dp

        test_time_end = time.time()

        # Get the average loss and metrics
        loss_pre_bn /= N_train
        loss_vel_bn /= N_train
        train_p_bn /= N_train
        train_u_bn /= N_train
        train_v_bn /= N_train
        train_w_bn /= N_train
        train_MNAE_p_bn /= N_train
        train_MNAE_u_bn /= N_train
        train_MNAE_v_bn /= N_train
        train_MNAE_w_bn /= N_train
        train_dp_bn /= N_train

        test_p_bn /= N_test
        test_u_bn /= N_test
        test_v_bn /= N_test
        test_w_bn /= N_test
        test_MNAE_p_bn /= N_test
        test_MNAE_u_bn /= N_test
        test_MNAE_v_bn /= N_test
        test_MNAE_w_bn /= N_test
        test_dp_bn /= N_test

        # Log printing
        if dist.get_rank() == 0:   
            logger.info(
                f"Epoch {epoch_id} | Train Loss: pre={loss_pre_bn:.3f}, vel={loss_vel_bn:.3f}, "
                f"L2_P={train_p_bn:.3f}, L2_U={train_u_bn:.3f}, L2_V={train_v_bn:.3f}, L2_W={train_w_bn:.3f}, L2_DP={train_dp_bn:.3f}, "
                f"MNAE_P={train_MNAE_p_bn:.3f}, MNAE_U={train_MNAE_u_bn:.3f}, MNAE_V={train_MNAE_v_bn:.3f}, MNAE_W={train_MNAE_w_bn:.3f}, "
                f"Test Loss: L2_P={test_p_bn:.3f}, L2_U={test_u_bn:.3f}, L2_V={test_v_bn:.3f}, L2_W={test_w_bn:.3f}, DP={test_dp_bn:.3f}, "
                f"MNAE_P={test_MNAE_p_bn:.3f}, MNAE_U={test_MNAE_u_bn:.3f}, MNAE_V={test_MNAE_v_bn:.3f}, MNAE_W={test_MNAE_w_bn:.3f}, "
                f"Train Time: {((train_time_end - train_time_start) / N_train):.3f}s, Test Time: {((test_time_end - train_time_end) / N_test):.3f}s"
            )

        # Save checkpoints periodically
        if epoch_id % 50 == 0 and dist.get_rank() == 0:  # Save only in the main process
            logger.info(f"Saving checkpoint at epoch {epoch_id}")
            checkpoint.save_trunk_net(epoch_id, trunk_net, optimizer, save_folder, local_rank)
            checkpoint.save_branch_bc_net(epoch_id, branch_bc_net, optimizer, save_folder, local_rank)
            checkpoint.save_branch_bp_net(epoch_id, branch_bp_net, optimizer, save_folder, local_rank)
            checkpoint.save_checkpoint_opt(epoch_id, optimizer, save_folder, rank=0)

    e_epoch = time.time()
    logger.info(f"Training completed. Total epoch time: {e_epoch - s_epoch:.2f} s")

if __name__ == '__main__':
    cfg = OmegaConf.load("config/default.yaml") # pay attention to the config path
    args = add_argument()
    cli_cfg = OmegaConf.create(vars(args)) 
    cfg = OmegaConf.merge(cfg, cli_cfg)

    local_rank, world_size = init_process_group()
    print(f"local_rank={local_rank}, world_size = {world_size}")
    print(f"GPU count == {torch.cuda.device_count()}")

    train_loader, val_loader, train_sampler, save_folder = build_dataloaders(cfg, local_rank, world_size)

    main(train_data_loader=train_loader, val_dataloader=val_loader, save_folder=save_folder,local_rank=local_rank, world_size=world_size, train_sampler = train_sampler, args=args, cfg = cfg)
