"""
Cross-geometry generalization training script (multi-GPU DDP).

Supports models: DeepONet, FNO, UNet, MeshGraphNets.
Trains on 80 cases (deform 5~12) and tests on 20 cases (deform 13~14)
across 10 different base geometries.

Usage (single GPU):
    python train_cross.py --model deeponet --epochs 200
Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 train_cross.py --model deeponet --epochs 200
"""
import os
import sys
import argparse
import time
import json
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))


# ── DDP utilities ──

def setup_ddp():
    """Initialize DDP if launched via torchrun, otherwise single-GPU mode."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, None, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


# ── Case ID configuration ──
# 10 base geometries × 5 deforms each = 50 cases
# Evenly sampled from available deforms per base geometry.
# Deform indices (0-based within available): pick [0, 2, 4, 6, 8] → 5 per geo
BASE_GEOS = {
    '1':   [5, 8, 10, 12, 14],
    '52':  [1078, 1080, 1083, 1085, 1087],
    '120': [2194, 2196, 2198, 2200, 2203],
    '186': [3278, 3280, 3282, 3285, 3287],
    '240': [4289, 4291, 4295, 4297, 4298],
    '291': [5304, 5306, 5308, 5311, 5313],
    '351': [6348, 6350, 6352, 6355, 6357],
    '406': [7360, 7362, 7365, 7367, 7369],
    '469': [8449, 8451, 8453, 8455, 8458],
    '526': [9510, 9512, 9514, 9517, 9519],
}

# Hold-out deform index for geometry split (0-indexed within each geo's 5 deforms)
# Index 2 = middle deform → interpolation test
TEST_DEFORM_INDICES = [2]


def get_train_test_ids(split_mode='geometry'):
    """Get train/test case IDs based on split mode.

    split_mode='geometry': hold out 1 deform per base geo (index 2, middle).
        Train=40 cases, Test=10 cases. All timesteps used for both.
    split_mode='time': all 50 cases used for both train and test.
        Split by timesteps within each case (front 80% train, back 20% test).
    """
    if split_mode == 'geometry':
        train_ids, test_ids = [], []
        for bg, case_ids in BASE_GEOS.items():
            for i, cid in enumerate(case_ids):
                if i in TEST_DEFORM_INDICES:
                    test_ids.append(cid)
                else:
                    train_ids.append(cid)
        return train_ids, test_ids
    else:  # time
        all_ids = []
        for bg, case_ids in BASE_GEOS.items():
            all_ids.extend(case_ids)
        return all_ids, all_ids


def parse_args():
    p = argparse.ArgumentParser(description='Cross-geometry generalization')
    p.add_argument('--model', type=str, required=True,
                   choices=['deeponet', 'fno', 'unet', 'mgn'],
                   help='Model type')
    p.add_argument('--split_mode', type=str, default='geometry',
                   choices=['geometry', 'time'],
                   help='geometry: hold out deforms, all timesteps; '
                        'time: all cases, split timesteps')
    p.add_argument('--h5_dir', type=str,
                   default='./h5_multi_cross')
    p.add_argument('--boundary_cut', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--checkpoint_dir', type=str, default=None)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--gpu', type=int, default=0, help='GPU id (single-GPU mode only)')
    p.add_argument('--save_interval', type=int, default=20)
    p.add_argument('--vis_interval', type=int, default=10)

    # DeepONet specific
    p.add_argument('--input_steps', type=int, default=2, help='History frames for DeepONet')
    p.add_argument('--output_steps', type=int, default=1, help='Prediction frames for DeepONet')
    p.add_argument('--num_wall_samples', type=int, default=2000,
                   help='Wall point samples (DeepONet/Point models)')
    p.add_argument('--history_encoder', type=str, default='light',
                   choices=['light', 'transformer'])
    p.add_argument('--history_embed_dim', type=int, default=128)
    p.add_argument('--trunk_hidden_dim', type=int, default=128)
    p.add_argument('--trunk_num_layers', type=int, default=4)
    p.add_argument('--branch_dim', type=int, default=256)
    p.add_argument('--use_geometry', action='store_true', default=True)
    p.add_argument('--no_geometry', action='store_true')

    # FNO/UNet specific
    p.add_argument('--resolution', type=int, default=96)
    p.add_argument('--hidden_channels', type=int, default=12)
    p.add_argument('--n_modes', type=int, default=20)
    p.add_argument('--n_layers', type=int, default=4)

    # Transolver specific
    p.add_argument('--trans_dim', type=int, default=128)
    p.add_argument('--trans_heads', type=int, default=8)
    p.add_argument('--trans_slices', type=int, default=8)
    p.add_argument('--trans_layers', type=int, default=8)
    p.add_argument('--trans_ffn', type=int, default=256)

    # MGN specific
    p.add_argument('--mgn_hidden', type=int, default=64)
    p.add_argument('--mgn_layers', type=int, default=10,
                   help='Number of message passing layers')
    p.add_argument('--mgn_k', type=int, default=16,
                   help='k for k-NN edge construction')

    args = p.parse_args()
    if args.no_geometry:
        args.use_geometry = False
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f'../checkpoint_cross/{args.model}_{args.split_mode}'
    return args


# ────────────────────── Metrics ──────────────────────

def compute_metrics(pred, target):
    """WSS metrics: mse, relative L2, mae, mnae."""
    diff = pred - target
    mse = (diff ** 2).mean().item()
    l2 = (torch.sqrt((diff ** 2).sum()) / (torch.sqrt((target ** 2).sum()) + 1e-8)).item()
    mae = torch.abs(diff).mean().item()
    mnae = (torch.abs(diff).mean() / (target.max() - target.min() + 1e-8)).item()
    return {'mse': mse, 'l2': l2, 'mae': mae, 'mnae': mnae}


# ────────────────────── DeepONet train/eval ──────────────────────

def build_deeponet(args, device):
    from transient.models import TemporalDeepONetV2
    model = TemporalDeepONetV2(
        num_input_vars=5,       # vel(3)+pres(1)+WSS(1)
        num_output_vars=1,      # WSS only
        history_embed_dim=args.history_embed_dim,
        history_encoder_type=args.history_encoder,
        history_num_layers=3,
        swin_embed_dim=24,
        use_geometry=args.use_geometry,
        trunk_hidden_dim=args.trunk_hidden_dim,
        trunk_num_layers=args.trunk_num_layers,
        use_cross_attention=True,
        branch_dim=args.branch_dim,
    ).to(device)
    return model


def train_deeponet_epoch(model, loader, optimizer, device, args):
    model.train()
    total_loss = 0
    for batch in loader:
        x_hist, y_hist, x_query, y_output, geo, bc, wall_mask = [
            b.to(device) for b in batch
        ]
        y_pred = model(x_hist, y_hist, x_query, geo)  # [B, N, 1]
        loss = F.mse_loss(y_pred.squeeze(-1), y_output.squeeze(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_deeponet(model, loader, device):
    model.eval()
    all_metrics = []
    for batch in loader:
        x_hist, y_hist, x_query, y_output, geo, bc, wall_mask = [
            b.to(device) for b in batch
        ]
        y_pred = model(x_hist, y_hist, x_query, geo)
        for b in range(y_pred.shape[0]):
            m = compute_metrics(y_pred[b].squeeze(-1), y_output[b].squeeze(-1))
            all_metrics.append(m)
    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}


# ────────────────────── Point model train/eval ──────────────────────

def build_point_model(args, device):
    if args.model == 'transolver':
        from baselines.models.transolver import Transolver_WSS
        model = Transolver_WSS(
            in_channels=11, d_model=args.trans_dim, n_heads=args.trans_heads,
            n_slices=args.trans_slices, ffn_dim=args.trans_ffn,
            n_layers=args.trans_layers,
        ).to(device)
    elif args.model == 'pointnet2':
        from baselines.models.pointnet2 import PointNet2_WSS
        model = PointNet2_WSS(in_features=8, nsample=32).to(device)
    return model


def train_point_epoch(model, loader, optimizer, device, args):
    model.train()
    if args.model == 'pointnet2' and hasattr(model, '_precomputed'):
        model._precomputed = False

    total_loss = 0
    for batch in loader:
        feat, target = [b.to(device) for b in batch]
        pred = model(feat).squeeze(-1)
        loss = ((pred - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_point(model, loader, device):
    model.eval()
    all_metrics = []
    for batch in loader:
        feat, target = [b.to(device) for b in batch]
        pred = model(feat).squeeze(-1)
        for b in range(feat.shape[0]):
            m = compute_metrics(pred[b], target[b])
            all_metrics.append(m)
    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}


# ────────────────────── MGN train/eval ──────────────────────

def build_mgn(args, device):
    from baselines.models.meshgraphnet import MeshGraphNet_WSS
    model = MeshGraphNet_WSS(
        node_input_dim=11,  # coords(3)+vel_prev(3)+p_prev(1)+vel_curr(3)+p_curr(1)
        edge_input_dim=4,   # rel_pos(3)+dist(1)
        hidden_dim=args.mgn_hidden,
        num_message_passing=args.mgn_layers,
        output_dim=1,
    ).to(device)
    return model


def train_mgn_epoch(model, loader, optimizer, device, args):
    model.train()
    total_loss = 0
    for batch in loader:
        node_feat, edge_index, edge_attr, target = [b.to(device) for b in batch]
        pred = model(node_feat, edge_index, edge_attr).squeeze(-1)
        loss = F.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_mgn(model, loader, device, num_wall_samples):
    """Evaluate MGN — split merged graph predictions back per sample."""
    model.eval()
    all_metrics = []
    for batch in loader:
        node_feat, edge_index, edge_attr, target = [b.to(device) for b in batch]
        pred = model(node_feat, edge_index, edge_attr).squeeze(-1)
        # Split back into per-sample chunks
        n = num_wall_samples
        num_graphs = node_feat.shape[0] // n
        for i in range(num_graphs):
            p = pred[i * n:(i + 1) * n]
            t = target[i * n:(i + 1) * n]
            m = compute_metrics(p, t)
            all_metrics.append(m)
    if not all_metrics:
        return {'mse': 0, 'l2': 0, 'mae': 0, 'mnae': 0}
    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}


# ────────────────────── Grid model train/eval ──────────────────────

def build_grid_model(args, device):
    if args.model == 'fno':
        from baselines.models.fno import FNO3D_WSS
        model = FNO3D_WSS(
            in_channels=9, hidden_channels=args.hidden_channels,
            n_modes=(args.n_modes,) * 3, n_layers=args.n_layers, use_mlp=True,
        ).to(device)
    else:
        from baselines.models.unet3d import UNet3D_WSS
        model = UNet3D_WSS(
            in_channels=9, base_channels=args.hidden_channels,
            depth=args.n_layers,
        ).to(device)
    return model


def train_grid_epoch(model, loader, optimizer, device, args):
    model.train()
    total_loss = 0
    for batch in loader:
        x, y_grid, y_wall, wmask, n_wall = [b.to(device) for b in batch]
        pred = model(x)
        weight = 1.0 + 99.0 * wmask
        loss = ((pred - y_grid) ** 2 * weight).sum() / (weight.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_grid(model, loader, device, test_ds):
    """Evaluate grid model — extract wall-node WSS from grid predictions."""
    model.eval()
    all_metrics = []

    sample_idx = 0
    for batch in loader:
        x, y_grid, y_wall, wmask, n_wall = [b.to(device) for b in batch]
        pred = model(x)
        for b in range(x.shape[0]):
            if sample_idx < len(test_ds.samples):
                case_idx, t = test_ds.samples[sample_idx]
                vox = test_ds.cases[case_idx]['vox']
                wijk = torch.from_numpy(vox['wall_vijk']).long().to(device)
                pred_wall = pred[b, 0, wijk[:, 0], wijk[:, 1], wijk[:, 2]]
                nw = n_wall[b].item()
                target_wall = y_wall[b, :nw]  # only valid (non-padded) entries
                m = compute_metrics(pred_wall, target_wall)
                all_metrics.append(m)
            sample_idx += 1

    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}


# ────────────────────── Main ──────────────────────

def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    use_ddp = world_size > 1

    if use_ddp:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    train_ids, test_ids = get_train_test_ids(args.split_mode)

    print_rank0(f"{'='*60}")
    print_rank0(f"Cross-Geometry Generalization: {args.model.upper()}")
    print_rank0(f"Split mode: {args.split_mode}")
    if args.split_mode == 'geometry':
        print_rank0(f"Train: {len(train_ids)} cases, Test: {len(test_ids)} cases")
        print_rank0(f"Hold-out: middle deform per base geo (interpolation test)")
    else:
        print_rank0(f"All {len(train_ids)} cases, t=20~80, train=20~60, test=60~80")
    print_rank0(f"Device: {device}, World size: {world_size}")
    print_rank0(f"{'='*60}")

    # ── Build dataset ──
    from baselines.datasets.cross_dataset import (
        MultiCaseDeepONetDataset, MultiCasePointDataset, MultiCaseGridDataset,
        MultiCaseMGNDataset, mgn_collate_fn,
    )

    if args.model == 'deeponet':
        print_rank0("\nLoading DeepONet datasets...")
        train_ds = MultiCaseDeepONetDataset(
            args.h5_dir, train_ids, num_wall_samples=args.num_wall_samples,
            input_steps=args.input_steps, output_steps=args.output_steps,
            mode='train', split_mode=args.split_mode,
            boundary_cut=args.boundary_cut)
        test_ds = MultiCaseDeepONetDataset(
            args.h5_dir, test_ids, num_wall_samples=args.num_wall_samples,
            input_steps=args.input_steps, output_steps=args.output_steps,
            mode='test', split_mode=args.split_mode,
            boundary_cut=args.boundary_cut)
        model = build_deeponet(args, device)
        train_fn = train_deeponet_epoch
        eval_fn = lambda m, l, d: eval_deeponet(m, l, d)

    elif args.model == 'transolver':
        print_rank0(f"\nLoading point cloud datasets...")
        train_ds = MultiCasePointDataset(
            args.h5_dir, train_ids, num_wall_samples=args.num_wall_samples,
            mode='train', split_mode=args.split_mode,
            boundary_cut=args.boundary_cut)
        test_ds = MultiCasePointDataset(
            args.h5_dir, test_ids, num_wall_samples=args.num_wall_samples,
            mode='test', split_mode=args.split_mode,
            boundary_cut=args.boundary_cut)
        model = build_point_model(args, device)
        train_fn = train_point_epoch
        eval_fn = lambda m, l, d: eval_point(m, l, d)

    elif args.model == 'mgn':
        print_rank0(f"\nLoading MGN graph datasets (k={args.mgn_k})...")
        train_ds = MultiCaseMGNDataset(
            args.h5_dir, train_ids, num_wall_samples=args.num_wall_samples,
            k_neighbors=args.mgn_k,
            mode='train', split_mode=args.split_mode,
            boundary_cut=args.boundary_cut)
        test_ds = MultiCaseMGNDataset(
            args.h5_dir, test_ids, num_wall_samples=args.num_wall_samples,
            k_neighbors=args.mgn_k,
            mode='test', split_mode=args.split_mode,
            boundary_cut=args.boundary_cut)
        model = build_mgn(args, device)
        train_fn = train_mgn_epoch
        _nws = args.num_wall_samples
        eval_fn = lambda m, l, d: eval_mgn(m, l, d, _nws)

    elif args.model in ('fno', 'unet'):
        print_rank0(f"\nLoading grid datasets (res={args.resolution})...")
        train_ds = MultiCaseGridDataset(
            args.h5_dir, train_ids, resolution=args.resolution,
            mode='train', split_mode=args.split_mode,
            boundary_cut=args.boundary_cut)
        test_ds = MultiCaseGridDataset(
            args.h5_dir, test_ids, resolution=args.resolution,
            mode='test', split_mode=args.split_mode,
            boundary_cut=args.boundary_cut)
        model = build_grid_model(args, device)
        train_fn = train_grid_epoch
        eval_fn = lambda m, l, d: eval_grid(m, l, d, test_ds)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"\n{args.model.upper()}: {num_params:,} parameters")

    # ── Sync all ranks after data loading before DDP ──
    if use_ddp:
        print_rank0("Waiting for all ranks to finish data loading...")
        dist.barrier()
        print_rank0("All ranks ready, wrapping model with DDP...")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ── DataLoaders with DistributedSampler ──
    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    collate = mgn_collate_fn if args.model == 'mgn' else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4, pin_memory=True, collate_fn=collate)
    # Test loader: no DistributedSampler — eval runs only on rank 0,
    # needs sequential order to match test_ds.samples
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict']
        # Handle DDP state_dict prefix mismatch
        if use_ddp and not any(k.startswith('module.') for k in state_dict):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not use_ddp and any(k.startswith('module.') for k in state_dict):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print_rank0(f"Resumed from epoch {ckpt['epoch']}")

    best_l2 = float('inf')
    ckpt_dir = Path(args.checkpoint_dir)
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()

    writer = SummaryWriter(log_dir=str(ckpt_dir / 'runs')) if is_main_process() else None

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_fn(model, train_loader, optimizer, device, args)
        scheduler.step()

        # ── Evaluation (only on rank 0 to avoid duplicated work) ──
        if is_main_process():
            if writer and (epoch % args.vis_interval == 0 or epoch == 1):
                writer.add_scalar('train/loss', train_loss, epoch)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

            # For eval, use the unwrapped model on rank 0
            eval_model = model.module if use_ddp else model
            avg = eval_fn(eval_model, test_loader, device)
            is_best = avg['l2'] < best_l2
            if is_best:
                best_l2 = avg['l2']

            dt = time.time() - t0
            print(f"Epoch {epoch:4d}/{args.epochs} ({dt:.0f}s) | loss={train_loss:.6f} | "
                  f"L2={avg['l2']:.4f} MSE={avg['mse']:.6f} MAE={avg['mae']:.6f} "
                  f"MNAE={avg['mnae']:.4f} {'*BEST*' if is_best else ''}")

            if writer and (epoch % args.vis_interval == 0 or epoch == 1):
                for k, v in avg.items():
                    writer.add_scalar(f'val/{k}', v, epoch)

            # Save state_dict without 'module.' prefix for portability
            raw_state = model.module.state_dict() if use_ddp else model.state_dict()
            if is_best:
                torch.save({
                    'epoch': epoch, 'model_state_dict': raw_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_l2': best_l2, 'metrics': avg, 'args': vars(args),
                }, ckpt_dir / 'best_model.pt')

            if epoch % args.save_interval == 0:
                torch.save({
                    'epoch': epoch, 'model_state_dict': raw_state,
                    'optimizer_state_dict': optimizer.state_dict(), 'args': vars(args),
                }, ckpt_dir / f'checkpoint_ep{epoch}.pt')

        # Sync all ranks before next epoch
        if use_ddp:
            dist.barrier()

    if writer:
        writer.close()

    # ── Final per-case evaluation (rank 0 only) ──
    if is_main_process():
        print(f"\n{'='*60}")
        print("Per-case test results (best model):")
        print(f"{'='*60}")
        eval_model = model.module if use_ddp else model
        ckpt = torch.load(ckpt_dir / 'best_model.pt', map_location=device, weights_only=False)
        eval_model.load_state_dict(ckpt['model_state_dict'])

        avg_final = eval_fn(eval_model, test_loader, device)
        print(f"Overall: L2={avg_final['l2']:.4f}, MSE={avg_final['mse']:.6f}, "
              f"MAE={avg_final['mae']:.6f}, MNAE={avg_final['mnae']:.4f}")

        results = {
            'model': args.model,
            'best_l2': best_l2,
            'best_epoch': ckpt['epoch'],
            'final_metrics': avg_final,
            'num_params': num_params,
            'train_cases': len(train_ids),
            'test_cases': len(test_ids),
            'args': vars(args),
        }
        with open(ckpt_dir / 'cross_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nDone. Best L2: {best_l2:.4f}")
        print(f"Results: {ckpt_dir / 'cross_results.json'}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
