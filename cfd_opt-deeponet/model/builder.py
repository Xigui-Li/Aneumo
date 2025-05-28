"""
models/builder.py
=================
Factory utilities for constructing all neural-network components
used in the CFD surrogate project.

Usage (in main_train.py)
---------------------------
from models.builder import build_networks

nets, mp_policy = build_networks(cfg.model, cfg.fsdp)

# nets["trunk"], nets["branch_bc"], nets["branch_bp"]
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from .NeuralNetworks import Trunk, Branch, Branch_Bypass 


# --------------------------------------------------------------------------- #
def _make_mixed_policy(dtype: str | None) -> MixedPrecision | None:
    """Return MixedPrecision policy given dtype string."""
    if dtype == "fp16":
        return MixedPrecision(torch.float16, torch.float16, torch.float16), torch.float16
    if dtype == "bf16":
        return MixedPrecision(torch.bfloat16, torch.bfloat16, torch.bfloat16), torch.bfloat16
    if dtype == "fp32":
        return None, torch.float32 

# --------------------------------------------------------------------------- #
def _wrap_fsdp(
    module: nn.Module,
    mp_policy: MixedPrecision | None,
    fsdp_cfg: DictConfig,
) -> nn.Module:
    """Optionally wrap a module with FSDP according to cfg."""
    if not fsdp_cfg.enable:
        return module

    auto_wrap = (
        None
        if fsdp_cfg.sharding == "NO_SHARD"
        else size_based_auto_wrap_policy(min_num_params=fsdp_cfg.min_params)
    )

    return FSDP(
        module,
        sharding_strategy=getattr(ShardingStrategy, fsdp_cfg.sharding),
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        use_orig_params=True,
    )


# --------------------------------------------------------------------------- #
def build_networks(
    model_cfg: DictConfig,
    fsdp_cfg: DictConfig | None = None,
    train_cfg: DictConfig | None = None  
) -> Tuple[Dict[str, nn.Module], MixedPrecision | None]:
    """
    Build trunk / branch networks.

    Parameters
    ----------
    model_cfg : DictConfig
        YAML : ``model`` including the dimensions of each subnet.
    fsdp_cfg : DictConfig, optional
        YAML : ``fsdp`` determines whether using FSDP.

    Returns
    -------
    nets : Dict[str, nn.Module]
        ``{"trunk": trunk_net, "branch_bc": branch_bc_net, "branch_bp": bp_net}``
    mp_policy : MixedPrecision | None
        A reusable mixed-precision strategy for training scripts
    """

    trunk = Trunk(
        in_dim=model_cfg.trunk.in_dim,
        out_dim=model_cfg.trunk.out_dim,
        hidden_num=model_cfg.trunk.hidden,
        layer_num=model_cfg.trunk.layers,
    )
    branch_bc = Branch(
        in_dim=model_cfg.branch_bc.in_dim,
        out_dim=model_cfg.branch_bc.out_dim,
        hidden_num=model_cfg.branch_bc.hidden,
        layer_num=model_cfg.branch_bc.layers,
    )
    branch_bp = Branch_Bypass(
        in_dim=model_cfg.branch_bp.in_dim,
        out_dim=model_cfg.branch_bp.out_dim,
    )


    dtype = train_cfg.dtype if train_cfg is not None else "fp32"
    mp_policy, target_dtype = _make_mixed_policy(dtype)

    if fsdp_cfg is not None:
        trunk = _wrap_fsdp(trunk, mp_policy, fsdp_cfg)
        branch_bc = _wrap_fsdp(branch_bc, mp_policy, fsdp_cfg)
        branch_bp = _wrap_fsdp(branch_bp, mp_policy, fsdp_cfg)


    nets: Dict[str, nn.Module] = {
        "trunk": trunk,
        "branch_bc": branch_bc,
        "branch_bp": branch_bp,
    }
    return nets, mp_policy, target_dtype
