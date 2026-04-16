"""
Unified training script for baseline models:
  FNO, U-Net, MeshGraphNets.

Task: Given velocity + pressure at t-1 and t, predict WSS magnitude at t+1.
Evaluation: relative L2, MSE, MAE, MNAE on wall nodes (same as DeepONet results).

Aligned with bash_noswin_201.sh training protocol:
  - case_201.h5, boundary_cut=0.1, output_vars=wss
  - lr=3e-4, CosineAnnealing, grad_clip=1.0
  - Periodic checkpointing and TensorBoard logging
"""
import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description='Train baseline models for WSS prediction')
    p.add_argument('--model', type=str, required=True,
                   choices=['fno', 'unet', 'mgn'],
                   help='Model type')
    p.add_argument('--h5_path', type=str, default='case_201.h5')
    p.add_argument('--boundary_cut', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--accum_steps', type=int, default=4,
                   help='Gradient accumulation steps (effective batch = batch_size * accum_steps)')
    p.add_argument('--checkpoint_dir', type=str, default=None)
    p.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    p.add_argument('--gpu', type=int, default=0)

    # Logging & saving (aligned with DeepONet)
    p.add_argument('--save_interval', type=int, default=100)
    p.add_argument('--vis_interval', type=int, default=100,
                   help='TensorBoard logging interval (epochs)')
    # FNO/UNet specific
    p.add_argument('--resolution', type=int, default=48)
    p.add_argument('--hidden_channels', type=int, default=12,
                   help='Hidden channels (FNO=12 ~1.16M, UNet=14 ~1.12M)')
    p.add_argument('--n_modes', type=int, default=12)
    p.add_argument('--n_layers', type=int, default=4)

    # MGN specific
    p.add_argument('--mgn_hidden', type=int, default=96,
                   help='Hidden dim for MeshGraphNet (96 with mp=15 -> ~1.01M)')
    p.add_argument('--mgn_mp_steps', type=int, default=15)
    p.add_argument('--wall_nhop', type=int, default=3,
                   help='N-hop neighborhood around wall for MGN subgraph')

    # Transolver specific
    p.add_argument('--trans_dim', type=int, default=128, help='Transolver d_model')
    p.add_argument('--trans_heads', type=int, default=8, help='Transolver attention heads')
    p.add_argument('--trans_slices', type=int, default=8, help='Transolver physics-aware slices')
    p.add_argument('--trans_layers', type=int, default=8, help='Transolver blocks')
    p.add_argument('--trans_ffn', type=int, default=256, help='Transolver FFN dim')

    return p.parse_args()


def resolve_h5_path(h5_path):
    if not os.path.isabs(h5_path):
        h5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', h5_path)
    return os.path.realpath(h5_path)


def get_device(gpu_id):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def compute_metrics(pred, target):
    """Compute WSS metrics: mse, relative L2, mae, mnae."""
    diff = pred - target
    mse = (diff ** 2).mean().item()
    l2 = (torch.sqrt((diff ** 2).sum()) / (torch.sqrt((target ** 2).sum()) + 1e-8)).item()
    mae = torch.abs(diff).mean().item()
    mnae = (torch.abs(diff).mean() / (target.max() - target.min() + 1e-8)).item()
    return {'mse': mse, 'l2': l2, 'mae': mae, 'mnae': mnae}


def evaluate(model, test_data, device, model_type, wall_vijk=None,
             edge_index=None, edge_attr=None, wall_indices=None):
    """Unified evaluation for all model types."""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        if model_type in ('fno', 'unet'):
            for batch in test_data:
                x, y_grid, y_wall, wmask = [b.to(device) for b in batch]
                pred = model(x)
                for b in range(x.shape[0]):
                    pred_wall = pred[b, 0][wall_vijk[:, 0], wall_vijk[:, 1], wall_vijk[:, 2]]
                    m = compute_metrics(pred_wall, y_wall[b])
                    all_metrics.append(m)
        elif model_type == 'mgn':
            for idx in range(len(test_data)):
                sample = test_data[idx]
                node_feat = sample['node_feat'].to(device)
                target_wall = sample['target_wall'].to(device)
                with torch.amp.autocast('cuda'):
                    pred = model(node_feat, edge_index, edge_attr)
                pred_wall = pred[wall_indices, 0].float()
                m = compute_metrics(pred_wall, target_wall)
                all_metrics.append(m)
        else:  # pointnet2, transolver
            for batch in test_data:
                feat, target = [b.to(device) for b in batch]
                pred = model(feat).squeeze(-1)          # [B, N]
                for b in range(feat.shape[0]):
                    m = compute_metrics(pred[b], target[b])
                    all_metrics.append(m)

    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}


# ============================================================
# Grid-based training (FNO / U-Net)
# ============================================================
def train_grid_model(args, device):
    from baselines.datasets.voxel_dataset import VoxelCFDDataset
    if args.model == 'fno':
        from baselines.models.fno import FNO3D_WSS
    else:
        from baselines.models.unet3d import UNet3D_WSS

    h5_path = resolve_h5_path(args.h5_path)

    print("Loading training data...")
    train_ds = VoxelCFDDataset(h5_path, resolution=args.resolution,
                                boundary_cut=args.boundary_cut, mode='train', cache=True)
    print("Loading test data...")
    test_ds = VoxelCFDDataset(h5_path, resolution=args.resolution,
                               boundary_cut=args.boundary_cut, mode='test', cache=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Model
    if args.model == 'fno':
        model = FNO3D_WSS(in_channels=9, hidden_channels=args.hidden_channels,
                           n_modes=(args.n_modes,) * 3, n_layers=args.n_layers, use_mlp=True).to(device)
    else:
        model = UNet3D_WSS(in_channels=9, base_channels=args.hidden_channels,
                            depth=args.n_layers).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{args.model.upper()} model: {num_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    start_epoch = 1

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    wall_vijk = torch.from_numpy(train_ds.wall_voxel_ijk).long().to(device)

    best_l2 = float('inf')
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / 'runs'))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x, y_grid, y_wall, wmask = [b.to(device) for b in batch]
            pred = model(x)
            weight = 1.0 + 99.0 * wmask
            loss = ((pred - y_grid) ** 2 * weight).sum() / (weight.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # TensorBoard
        if epoch % args.vis_interval == 0 or epoch == 1:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

        # Eval every epoch (aligned with train_v2.py)
        avg = evaluate(model, test_loader, device, args.model, wall_vijk=wall_vijk)
        is_best = avg['l2'] < best_l2
        if is_best:
            best_l2 = avg['l2']

        print(f"Epoch {epoch:4d}/{args.epochs} | loss={train_loss:.6f} | "
              f"L2={avg['l2']:.4f} MSE={avg['mse']:.6f} MAE={avg['mae']:.6f} "
              f"MNAE={avg['mnae']:.4f} {'*BEST*' if is_best else ''}")

        if epoch % args.vis_interval == 0 or epoch == 1:
            writer.add_scalar('val/l2', avg['l2'], epoch)
            writer.add_scalar('val/mse', avg['mse'], epoch)
            writer.add_scalar('val/mae', avg['mae'], epoch)
            writer.add_scalar('val/mnae', avg['mnae'], epoch)

        if is_best:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_l2': best_l2, 'metrics': avg, 'args': vars(args),
            }, ckpt_dir / 'best_model.pt')

        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'args': vars(args),
            }, ckpt_dir / f'checkpoint_ep{epoch}.pt')

    writer.close()
    print(f"\nTraining complete. Best L2: {best_l2:.4f}")
    return best_l2


# ============================================================
# MeshGraphNet training
# ============================================================
def train_mgn(args, device):
    from baselines.datasets.voxel_dataset import GraphCFDDataset
    from baselines.models.meshgraphnet import MeshGraphNet_WSS

    h5_path = resolve_h5_path(args.h5_path)

    print("Loading training data...")
    train_ds = GraphCFDDataset(h5_path, mode='train', boundary_cut=args.boundary_cut,
                                wall_nhop=args.wall_nhop)
    print("Loading test data...")
    test_ds = GraphCFDDataset(h5_path, mode='test', boundary_cut=args.boundary_cut,
                               wall_nhop=args.wall_nhop)

    model = MeshGraphNet_WSS(
        node_input_dim=11, edge_input_dim=4,
        hidden_dim=args.mgn_hidden, num_message_passing=args.mgn_mp_steps, output_dim=1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nMeshGraphNet model: {num_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    sample0 = train_ds[0]
    edge_index = sample0['edge_index'].to(device)
    edge_attr = sample0['edge_attr'].to(device)
    wall_indices = sample0['wall_indices'].to(device)

    best_l2 = float('inf')
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / 'runs'))

    accum_steps = max(1, args.accum_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        indices = np.random.permutation(len(train_ds))
        for step, idx in enumerate(indices):
            sample = train_ds[idx]
            node_feat = sample['node_feat'].to(device)
            target_wall = sample['target_wall'].to(device)

            with torch.amp.autocast('cuda'):
                pred = model(node_feat, edge_index, edge_attr)
                pred_wall = pred[wall_indices, 0]
                loss = ((pred_wall - target_wall) ** 2).mean() / accum_steps

            scaler.scale(loss).backward()
            train_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0 or step == len(train_ds) - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()
        train_loss /= len(train_ds)

        if epoch % args.vis_interval == 0 or epoch == 1:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

        # Eval every epoch (aligned with train_v2.py)
        avg = evaluate(model, test_ds, device, 'mgn',
                      edge_index=edge_index, edge_attr=edge_attr, wall_indices=wall_indices)
        is_best = avg['l2'] < best_l2
        if is_best:
            best_l2 = avg['l2']

        print(f"Epoch {epoch:4d}/{args.epochs} | loss={train_loss:.6f} | "
              f"L2={avg['l2']:.4f} MSE={avg['mse']:.6f} MAE={avg['mae']:.6f} "
              f"MNAE={avg['mnae']:.4f} {'*BEST*' if is_best else ''}")

        if epoch % args.vis_interval == 0 or epoch == 1:
            writer.add_scalar('val/l2', avg['l2'], epoch)
            writer.add_scalar('val/mse', avg['mse'], epoch)
            writer.add_scalar('val/mae', avg['mae'], epoch)
            writer.add_scalar('val/mnae', avg['mnae'], epoch)

        if is_best:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_l2': best_l2, 'metrics': avg, 'args': vars(args),
            }, ckpt_dir / 'best_model.pt')

        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'args': vars(args),
            }, ckpt_dir / f'checkpoint_ep{epoch}.pt')

    writer.close()
    print(f"\nTraining complete. Best L2: {best_l2:.4f}")
    return best_l2


# ============================================================
# Point-cloud training (PointNet++ / Transolver)
# ============================================================
def train_point_model(args, device):
    from baselines.datasets.voxel_dataset import PointCFDDataset

    if args.model == 'pointnet2':
        from baselines.models.pointnet2 import PointNet2_WSS
    else:
        from baselines.models.transolver import Transolver_WSS

    h5_path = resolve_h5_path(args.h5_path)

    print("Loading training data...")
    train_ds = PointCFDDataset(h5_path, mode='train', boundary_cut=args.boundary_cut)
    print("Loading test data...")
    test_ds = PointCFDDataset(h5_path, mode='test', boundary_cut=args.boundary_cut)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Model
    if args.model == 'pointnet2':
        model = PointNet2_WSS(in_features=8, nsample=32).to(device)
    else:
        model = Transolver_WSS(
            in_channels=11, d_model=args.trans_dim, n_heads=args.trans_heads,
            n_slices=args.trans_slices, ffn_dim=args.trans_ffn,
            n_layers=args.trans_layers,
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{args.model.upper()} model: {num_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    best_l2 = float('inf')
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / 'runs'))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0
        for batch in train_loader:
            feat, target = [b.to(device) for b in batch]
            pred = model(feat).squeeze(-1)              # [B, N]
            loss = ((pred - target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        if epoch % args.vis_interval == 0 or epoch == 1:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

        # Eval every epoch
        avg = evaluate(model, test_loader, device, args.model)
        is_best = avg['l2'] < best_l2
        if is_best:
            best_l2 = avg['l2']

        print(f"Epoch {epoch:4d}/{args.epochs} | loss={train_loss:.6f} | "
              f"L2={avg['l2']:.4f} MSE={avg['mse']:.6f} MAE={avg['mae']:.6f} "
              f"MNAE={avg['mnae']:.4f} {'*BEST*' if is_best else ''}")

        if epoch % args.vis_interval == 0 or epoch == 1:
            writer.add_scalar('val/l2', avg['l2'], epoch)
            writer.add_scalar('val/mse', avg['mse'], epoch)
            writer.add_scalar('val/mae', avg['mae'], epoch)
            writer.add_scalar('val/mnae', avg['mnae'], epoch)

        if is_best:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_l2': best_l2, 'metrics': avg, 'args': vars(args),
            }, ckpt_dir / 'best_model.pt')

        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'args': vars(args),
            }, ckpt_dir / f'checkpoint_ep{epoch}.pt')

    writer.close()
    print(f"\nTraining complete. Best L2: {best_l2:.4f}")
    return best_l2


def main():
    args = parse_args()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f'../checkpoint_baselines/{args.model}'

    device = get_device(args.gpu)
    print(f"{'='*50}")
    print(f"Model: {args.model.upper()}")
    print(f"Device: {device}")
    print(f"H5: {args.h5_path}, boundary_cut={args.boundary_cut}")
    print(f"Epochs: {args.epochs}, lr={args.lr}, batch={args.batch_size}, accum={args.accum_steps}")
    print(f"Save: every {args.save_interval} ep, Eval: every epoch, "
          f"TensorBoard: every {args.vis_interval} ep")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"{'='*50}")

    if args.model in ('fno', 'unet'):
        train_grid_model(args, device)
    elif args.model == 'mgn':
        train_mgn(args, device)
    else:
        train_point_model(args, device)


if __name__ == '__main__':
    main()
