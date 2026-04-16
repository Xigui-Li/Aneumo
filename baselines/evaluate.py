"""
Test script for baseline models (FNO, U-Net, MeshGraphNets).

Output format aligned with inference_full_mesh.py:
  - Per-timestep: MSE, L2_full, L2_sampled, MAE, MNAE
  - Summary: Mean ± Std for all metrics
  - Saves summary.txt + results.json
  - Exports VTP files for visualization (optional --vtp_dir)
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description='Test baseline models')
    p.add_argument('--model', type=str, required=True,
                   choices=['fno', 'unet', 'mgn'])
    p.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pt')
    p.add_argument('--h5_path', type=str, default='case_201.h5')
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--vtp_dir', type=str, default=None,
                   help='VTP source dir for mesh export (e.g. .../case_201)')
    p.add_argument('--smooth', type=int, default=10, help='Smooth iterations for VTP export')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def resolve_h5_path(h5_path):
    if not os.path.isabs(h5_path):
        h5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', h5_path)
    return os.path.realpath(h5_path)


def load_h5_wall_info(h5_path, boundary_cut=0.1):
    """Load wall coordinates and WSS data from h5 for VTP export."""
    with h5py.File(h5_path, 'r') as f:
        coords = f['mesh/coords'][:].astype(np.float32)
        node_type = f['mesh/node_type'][:].astype(np.int64)
        wall_indices = f['mesh/wall_indices'][:].astype(np.int64)
        wss = f['fields/wss'][:].astype(np.float32)  # [T, N_wall, 3]
        time_values = f['time_values'][:].astype(np.float64)

    # Boundary cut (same logic as datasets)
    if boundary_cut > 0:
        inlet_mask = node_type == 1
        outlet_mask = node_type == 2
        if np.any(inlet_mask) and np.any(outlet_mask):
            inlet_c = coords[inlet_mask].mean(0)
            outlet_c = coords[outlet_mask].mean(0)
            direction = outlet_c - inlet_c
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            proj = np.dot(coords - inlet_c, direction)
            p_min, p_max = proj.min(), proj.max()
            p_range = p_max - p_min
            cut = p_range * boundary_cut
            valid = (proj >= p_min + cut) & (proj <= p_max - cut)
            wall_valid = valid[wall_indices]
            wall_indices_cut = wall_indices[wall_valid]
        else:
            wall_indices_cut = wall_indices
    else:
        wall_indices_cut = wall_indices

    wall_coords = coords[wall_indices_cut]
    return wall_coords, wall_indices, wall_indices_cut, wss, time_values, coords


def load_original_wall_vtp(vtp_dir, time_value):
    """Load original wall.vtp for a given time value."""
    import pyvista as pv
    time_str = f"{time_value:.2f}"
    vtp_path = Path(vtp_dir) / time_str / f"{time_str}_wall.vtp"

    if not vtp_path.exists():
        vtp_dir_path = Path(vtp_dir)
        time_dirs = [d.name for d in vtp_dir_path.iterdir() if d.is_dir()]
        best_match, best_diff = None, float('inf')
        for td in time_dirs:
            try:
                diff = abs(float(td) - time_value)
                if diff < best_diff:
                    best_diff = diff
                    best_match = td
            except ValueError:
                continue
        if best_match and best_diff < 0.005:
            time_str = best_match
            vtp_path = Path(vtp_dir) / time_str / f"{time_str}_wall.vtp"

    if not vtp_path.exists():
        return None
    return pv.read(str(vtp_path))


def smooth_on_mesh(mesh, values, iterations=10):
    """Smooth point data using mesh topology."""
    from collections import defaultdict
    neighbors = defaultdict(set)
    for i in range(mesh.n_cells):
        cell = mesh.get_cell(i)
        pts = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        for p in pts:
            neighbors[p].update(pts)
    for p in neighbors:
        neighbors[p].discard(p)

    smoothed = values.copy()
    for _ in range(iterations):
        new_vals = smoothed.copy()
        for i in range(len(smoothed)):
            if neighbors[i]:
                new_vals[i] = 0.5 * smoothed[i] + 0.5 * smoothed[list(neighbors[i])].mean()
        smoothed = new_vals
    return smoothed


def export_vtp(mesh, wss_pred_full, output_path, smooth_iterations=10):
    """Export VTP with WSS_predicted, WSS_ground_truth, WSS_predicted_smooth."""
    mesh.point_data['WSS_predicted'] = wss_pred_full.copy()

    if smooth_iterations > 0:
        mesh.point_data['WSS_predicted_smooth'] = smooth_on_mesh(
            mesh, wss_pred_full, iterations=smooth_iterations)

    if 'wallShearStress' in mesh.point_data:
        original_wss = mesh.point_data['wallShearStress']
        if len(original_wss.shape) > 1:
            wss_gt = np.sqrt(np.sum(original_wss ** 2, axis=-1))
        else:
            wss_gt = original_wss
        mesh.point_data['WSS_ground_truth'] = wss_gt
        mesh.point_data['WSS_error'] = np.abs(wss_pred_full - wss_gt)

    mesh.save(str(output_path))


def map_pred_to_full_wall(pred_wall, wall_indices_cut, wall_indices_full, wss_true_full_t):
    """Map cut-boundary predictions back to full wall indices, filling cut region with GT."""
    from scipy.spatial import cKDTree
    # wss_true_full_t: [N_wall_full, 3] WSS vectors for this timestep
    wss_mag_full = np.sqrt(np.sum(wss_true_full_t ** 2, axis=-1))

    full_pred = wss_mag_full.copy()  # start with GT

    # Build mapping: wall_indices_full[i] -> i
    full_idx_map = {int(idx): i for i, idx in enumerate(wall_indices_full)}

    # Fill in predictions for cut indices
    for i, node_idx in enumerate(wall_indices_cut):
        if int(node_idx) in full_idx_map:
            full_pred[full_idx_map[int(node_idx)]] = pred_wall[i]

    return full_pred


@torch.no_grad()
def test_grid_model(args, saved_args, model, device):
    """Test FNO or U-Net on all test timesteps (full wall mesh)."""
    from baselines.datasets.voxel_dataset import VoxelCFDDataset

    h5_path = resolve_h5_path(args.h5_path)
    resolution = saved_args.get('resolution', 48)
    boundary_cut = saved_args.get('boundary_cut', 0.1)

    ds = VoxelCFDDataset(h5_path, resolution=resolution,
                          boundary_cut=boundary_cut, mode='test', cache=True)

    wall_vijk = torch.from_numpy(ds.wall_voxel_ijk).long().to(device)
    num_wall = len(ds.wall_indices)

    results = []
    pred_per_t = {}  # t_idx -> pred_wall array

    for idx in tqdm(range(len(ds)), desc="Testing"):
        t = ds.t_start + idx
        t_next = t + 1

        x, y_grid, y_wall, wmask = ds[idx]
        x = x.unsqueeze(0).to(device)

        pred = model(x)
        pred_wall = pred[0, 0][wall_vijk[:, 0], wall_vijk[:, 1], wall_vijk[:, 2]].cpu().numpy()
        true_wall = y_wall.numpy()

        pred_per_t[t_next] = pred_wall

        error = np.abs(pred_wall - true_wall)
        mse = np.mean(error ** 2)
        l2_full = np.sqrt(np.sum(error ** 2) / (np.sum(true_wall ** 2) + 1e-8))
        mae = np.mean(error)
        wss_range = true_wall.max() - true_wall.min() + 1e-8
        mnae = np.mean(error / wss_range)

        sample_size = min(8000, num_wall)
        sample_idx = np.random.choice(num_wall, sample_size, replace=False)
        sampled_l2 = np.sqrt(np.sum(error[sample_idx] ** 2) / (np.sum(true_wall[sample_idx] ** 2) + 1e-8))

        results.append({
            't_idx': t_next, 'mse': mse, 'l2_full': l2_full,
            'l2_sampled': sampled_l2, 'mae': mae, 'mnae': mnae
        })

    return results, num_wall, pred_per_t


@torch.no_grad()
def test_mgn_model(args, saved_args, model, device):
    """Test MeshGraphNet on all test timesteps (full wall mesh)."""
    from baselines.datasets.voxel_dataset import GraphCFDDataset

    h5_path = resolve_h5_path(args.h5_path)
    boundary_cut = saved_args.get('boundary_cut', 0.1)
    wall_nhop = saved_args.get('wall_nhop', 3)

    ds = GraphCFDDataset(h5_path, mode='test', boundary_cut=boundary_cut, wall_nhop=wall_nhop)

    sample0 = ds[0]
    edge_index = sample0['edge_index'].to(device)
    edge_attr = sample0['edge_attr'].to(device)
    wall_indices_sub = sample0['wall_indices'].to(device)
    num_wall = len(sample0['wall_indices'])

    results = []
    pred_per_t = {}

    for idx in tqdm(range(len(ds)), desc="Testing"):
        t = ds.t_start + idx
        t_next = t + 1

        sample = ds[idx]
        node_feat = sample['node_feat'].to(device)
        target_wall = sample['target_wall'].numpy()

        with torch.amp.autocast('cuda'):
            pred = model(node_feat, edge_index, edge_attr)
        pred_wall = pred[wall_indices_sub, 0].float().cpu().numpy()

        pred_per_t[t_next] = pred_wall

        error = np.abs(pred_wall - target_wall)
        mse = np.mean(error ** 2)
        l2_full = np.sqrt(np.sum(error ** 2) / (np.sum(target_wall ** 2) + 1e-8))
        mae = np.mean(error)
        wss_range = target_wall.max() - target_wall.min() + 1e-8
        mnae = np.mean(error / wss_range)

        sample_size = min(8000, num_wall)
        sample_idx = np.random.choice(num_wall, sample_size, replace=False)
        sampled_l2 = np.sqrt(np.sum(error[sample_idx] ** 2) / (np.sum(target_wall[sample_idx] ** 2) + 1e-8))

        results.append({
            't_idx': t_next, 'mse': mse, 'l2_full': l2_full,
            'l2_sampled': sampled_l2, 'mae': mae, 'mnae': mnae
        })

    return results, num_wall, pred_per_t


@torch.no_grad()
def test_point_model(args, saved_args, model, device):
    """Test PointNet++ or Transolver on all test timesteps."""
    from baselines.datasets.voxel_dataset import PointCFDDataset

    h5_path = resolve_h5_path(args.h5_path)
    boundary_cut = saved_args.get('boundary_cut', 0.1)

    ds = PointCFDDataset(h5_path, mode='test', boundary_cut=boundary_cut)
    num_wall = ds.num_wall

    results = []
    pred_per_t = {}

    for idx in tqdm(range(len(ds)), desc="Testing"):
        t = ds.t_start + idx
        t_next = t + 1

        feat, target = ds[idx]
        feat = feat.unsqueeze(0).to(device)
        target = target.numpy()

        pred = model(feat).squeeze(0).squeeze(-1).cpu().numpy()

        pred_per_t[t_next] = pred

        error = np.abs(pred - target)
        mse = np.mean(error ** 2)
        l2_full = np.sqrt(np.sum(error ** 2) / (np.sum(target ** 2) + 1e-8))
        mae = np.mean(error)
        wss_range = target.max() - target.min() + 1e-8
        mnae = np.mean(error / wss_range)

        sample_size = min(8000, num_wall)
        sample_idx = np.random.choice(num_wall, sample_size, replace=False)
        sampled_l2 = np.sqrt(np.sum(error[sample_idx] ** 2) / (np.sum(target[sample_idx] ** 2) + 1e-8))

        results.append({
            't_idx': t_next, 'mse': mse, 'l2_full': l2_full,
            'l2_sampled': sampled_l2, 'mae': mae, 'mnae': mnae
        })

    return results, num_wall, pred_per_t


def write_summary(results, num_wall, args, saved_args, output_dir):
    """Write summary.txt in the same format as inference_full_mesh.py."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mses = [r['mse'] for r in results]
    l2s = [r['l2_full'] for r in results]
    l2s_sampled = [r['l2_sampled'] for r in results]
    maes = [r['mae'] for r in results]
    mnaes = [r['mnae'] for r in results]

    print(f"\n{'='*50}")
    print("Full Mesh Inference Summary")
    print(f"{'='*50}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Time steps: {results[0]['t_idx']} to {results[-1]['t_idx']}")
    print(f"Wall points: {num_wall}")
    print(f"Valid points: {num_wall}")
    print(f"Cut points (filled with GT): 0")
    print()
    print(f"Mean MSE: {np.mean(mses):.2e} ± {np.std(mses):.2e}")
    print(f"Mean L2 Error (Full Mesh): {np.mean(l2s):.4f} ± {np.std(l2s):.4f}")
    print(f"Mean L2 Error (Sampled): {np.mean(l2s_sampled):.4f} ± {np.std(l2s_sampled):.4f}")
    print(f"Mean MAE: {np.mean(maes):.2e} ± {np.std(maes):.2e}")
    print(f"Mean MNAE: {np.mean(mnaes):.4f} ± {np.std(mnaes):.4f}")

    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Full Mesh Inference Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Time steps: {results[0]['t_idx']} to {results[-1]['t_idx']}\n")
        f.write(f"Wall points: {num_wall}\n")
        f.write(f"Valid points: {num_wall}\n")
        f.write(f"Cut points (filled with GT): 0\n")
        f.write("\n")
        f.write(f"Mean MSE: {np.mean(mses):.2e} ± {np.std(mses):.2e}\n")
        f.write(f"Mean L2 Error (Full Mesh): {np.mean(l2s):.4f} ± {np.std(l2s):.4f}\n")
        f.write(f"Mean L2 Error (Sampled): {np.mean(l2s_sampled):.4f} ± {np.std(l2s_sampled):.4f}\n")
        f.write(f"Mean MAE: {np.mean(maes):.2e} ± {np.std(maes):.2e}\n")
        f.write(f"Mean MNAE: {np.mean(mnaes):.4f} ± {np.std(mnaes):.4f}\n")
        f.write("\nPer-timestep errors:\n")
        for r in results:
            f.write(f"  t={r['t_idx']}: MSE={r['mse']:.2e}, L2_full={r['l2_full']:.4f}, "
                    f"L2_sampled={r['l2_sampled']:.4f}, MAE={r['mae']:.2e}, MNAE={r['mnae']:.4f}\n")

    print(f"\nResults saved to: {summary_path}")

    json_path = output_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'model': args.model,
            'checkpoint': args.checkpoint,
            'num_wall_points': num_wall,
            'mean_mse': float(np.mean(mses)),
            'mean_l2': float(np.mean(l2s)),
            'mean_l2_sampled': float(np.mean(l2s_sampled)),
            'mean_mae': float(np.mean(maes)),
            'mean_mnae': float(np.mean(mnaes)),
            'std_l2': float(np.std(l2s)),
            'per_timestep': results,
        }, f, indent=2, default=str)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get('args', {})
    print(f"Trained for {ckpt.get('epoch', '?')} epochs, best L2={ckpt.get('best_l2', '?')}")

    # Build model
    if args.model == 'fno':
        from baselines.models.fno import FNO3D_WSS
        model = FNO3D_WSS(
            in_channels=9,
            hidden_channels=saved_args.get('hidden_channels', 12),
            n_modes=(saved_args.get('n_modes', 12),) * 3,
            n_layers=saved_args.get('n_layers', 4),
            use_mlp=True,
        ).to(device)
    elif args.model == 'unet':
        from baselines.models.unet3d import UNet3D_WSS
        model = UNet3D_WSS(
            in_channels=9,
            base_channels=saved_args.get('hidden_channels', 14),
            depth=saved_args.get('n_layers', 4),
        ).to(device)
    elif args.model == 'mgn':
        from baselines.models.meshgraphnet import MeshGraphNet_WSS
        model = MeshGraphNet_WSS(
            node_input_dim=11, edge_input_dim=4,
            hidden_dim=saved_args.get('mgn_hidden', 96),
            num_message_passing=saved_args.get('mgn_mp_steps', 15),
            output_dim=1,
        ).to(device)
    elif args.model == 'pointnet2':
        from baselines.models.pointnet2 import PointNet2_WSS
        model = PointNet2_WSS(in_features=8, nsample=32).to(device)
    else:  # transolver
        from baselines.models.transolver import Transolver_WSS
        model = Transolver_WSS(
            in_channels=11,
            d_model=saved_args.get('trans_dim', 128),
            n_heads=saved_args.get('trans_heads', 8),
            n_slices=saved_args.get('trans_slices', 8),
            ffn_dim=saved_args.get('trans_ffn', 256),
            n_layers=saved_args.get('trans_layers', 8),
        ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model.upper()}, {num_params:,} parameters")

    # Output dir: default to results/{model}/ under project root
    if args.output_dir is None:
        project_root = Path(__file__).parent.parent
        args.output_dir = str(project_root / 'results' / args.model)

    # Run test
    if args.model in ('fno', 'unet'):
        results, num_wall, pred_per_t = test_grid_model(args, saved_args, model, device)
    elif args.model == 'mgn':
        results, num_wall, pred_per_t = test_mgn_model(args, saved_args, model, device)
    else:
        results, num_wall, pred_per_t = test_point_model(args, saved_args, model, device)

    write_summary(results, num_wall, args, saved_args, args.output_dir)

    # Export VTP files if vtp_dir is provided
    if args.vtp_dir is not None:
        print(f"\nExporting VTP files...")
        h5_path = resolve_h5_path(args.h5_path)
        boundary_cut = saved_args.get('boundary_cut', 0.1)
        wall_coords, wall_indices_full, wall_indices_cut, wss_all, time_values, coords = \
            load_h5_wall_info(h5_path, boundary_cut)

        output_dir = Path(args.output_dir)
        for t_idx, pred_wall in pred_per_t.items():
            if t_idx >= len(time_values):
                continue
            tv = time_values[t_idx]
            mesh = load_original_wall_vtp(args.vtp_dir, tv)
            if mesh is None:
                print(f"  t={t_idx} (tv={tv:.2f}): VTP not found, skipping")
                continue

            # Map predictions to full wall (fill cut region with GT)
            # H5 wall order matches VTP point order (same as DeepONet export)
            full_pred = map_pred_to_full_wall(
                pred_wall, wall_indices_cut, wall_indices_full, wss_all[t_idx])

            vtp_path = output_dir / f"wss_t{t_idx:04d}.vtp"
            export_vtp(mesh, full_pred, vtp_path, smooth_iterations=args.smooth)
            print(f"  Saved: {vtp_path.name}")

    print("\nDone!")


if __name__ == '__main__':
    main()
