"""
DataLoader Construction for DeepONet-CFD Project

This module provides a utility function to construct PyTorch DataLoaders for training and validation datasets.
It supports distributed training by using `DistributedSampler` to partition the dataset across multiple GPUs.

Functions:
- build_dataloaders: Constructs and returns DataLoaders for training and validation datasets.

Dependencies:
- PyTorch DataLoader and DistributedSampler for data loading and distributed training.
- OmegaConf for configuration management.
- CFDJobDataset: Custom dataset class for loading CFD simulation data.
- load_case_ids: Utility function to load case IDs from CSV files.
"""
from typing import Tuple
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import DictConfig
from PrepareDataset.Dataset import CFDJobDataset as create_dataset
from PrepareDataset.load_data import load_case_ids
import os
def build_dataloaders(data_cfg: DictConfig,
                      rank: int,
                      world: int) -> Tuple[DataLoader, DataLoader]:
    """
    Construct train/val dataloaders using config.

    ------------
    data:
      num_points: 1800
      train_flow: [m0.001, m0.002, m0.003]
      val_flow:   [m0.0015, m0.0035]
    """
    # Extract dataset and checkpoint paths from the configuration
    data_dir = data_cfg.paths.data_dir
    save_folder = data_cfg.paths.checkpoint_dir
    os.makedirs(save_folder, exist_ok=True)
    # Load training and validation case IDs from CSV files
    train_ids = load_case_ids(data_cfg.paths.train_csv)
    val_ids   = load_case_ids(data_cfg.paths.val_csv)
    # Create training and validation datasets
    train_ds = create_dataset(data_cfg.paths.data_dir,
                              train_ids,
                              data_cfg.data.train_flow,
                              num_sample=data_cfg.data.num_points)
    val_ds   = create_dataset(data_cfg.paths.data_dir,
                              val_ids,
                              data_cfg.data.test_flow,
                              num_sample=data_cfg.data.num_points)
    # Create distributed samplers for training and validation datasets
    train_sampler = DistributedSampler(train_ds,
                                       num_replicas=world,
                                       rank=rank,
                                       shuffle=True)
    val_sampler   = DistributedSampler(val_ds,
                                       num_replicas=world,
                                       rank=rank,
                                       shuffle=False)
    # Create DataLoaders for training and validation datasets
    train_loader = DataLoader(train_ds,
                              batch_size=data_cfg.data.batch_size,
                              sampler=train_sampler,
                              num_workers=data_cfg.data.num_workers,
                              drop_last=True,
                              pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,
                              batch_size=data_cfg.data.batch_size,
                              sampler=val_sampler,
                              num_workers=data_cfg.data.num_workers,
                              drop_last=True,
                              pin_memory=True,
                              persistent_workers=True)
    return train_loader, val_loader, train_sampler, save_folder
