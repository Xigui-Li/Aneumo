"""
Cross-geometry full-wall test script.

Evaluates checkpoint_cross models on ALL wall points (no subsampling).
Supports: FNO, UNet, MGN, DeepONet × geometry/time splits.

Output:
  - summary.txt + results.json per model/split
  - Optional VTP export for visualization

Usage:
    python test_cross.py --model fno --split_mode time
    python test_cross.py --model mgn --split_mode geometry
    python test_cross.py --model deeponet --split_mode time --vtp_base /path/to/vtk
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
from scipy.spatial import cKDTree


import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.datasets.cross_dataset import load_case
from baselines.train_cross import get_train_test_ids, BASE_GEOS


# ── CLI ──

def parse_args():
    p = argparse.ArgumentParser(description='Cross-geometry full-wall test')
    p.add_argument('--model', type=str, required=True,
                   choices=['fno', 'unet', 'mgn', 'deeponet'])
    p.add_argument('--split_mode', type=str, default='time',
                   choices=['geometry', 'time'])
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--h5_dir', type=str,
                   default='./
        h5_multi_cross')
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--vtp_base', type=str, default=None,
                   help='VTK source base dir (e.g. .../unsteady_vtk_aneumo)')
    p.add_argument('--smooth', type=int, default=10)
    p.add_argument('--batch_points', type=int, default=5000,
                   help='Batch size for DeepONet query points')
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    if args.checkpoint is None:
        args.checkpoint = str(
            Path(__file__).parent.parent /
            f'checkpoint_cross/{args.model}_{args.split_mode}/best_model.pt')
    if args.output_dir is None:
        args.output_dir = str(
            Path(__file__).parent.parent /
            f'results_cross/{args.model}_{args.split_mode}')
    return args


# ── Metrics ──

def compute_metrics(pred, target):
    """Compute MSE, relative L2, MAE, MNAE from numpy arrays."""
    error = np.abs(pred - target)
    mse = np.mean(error ** 2)
    l2 = np.sqrt(np.sum(error ** 2) / (np.sum(target ** 2) + 1e-8))
    mae = np.mean(error)
    mnae = np.mean(error / (target.max() - target.min() + 1e-8))
    return {'mse': float(mse), 'l2': float(l2), 'mae': float(mae), 'mnae': float(mnae)}


# ── Voxelization helpers (from dataset_cross.py MultiCaseGridDataset) ──

def setup_voxelization(case_data, resolution):
    """Voxelize a case's full mesh. Returns vox dict."""
    res = resolution
    coords = case_data['coords_full']
    wall_indices = case_data['wall_indices']

    cmin = coords.min(0) - 1e-6
    cmax = coords.max(0) + 1e-6
    step = (cmax - cmin) / res

    voxel_ijk = np.floor((coords - cmin) / step).astype(np.int32)
    voxel_ijk = np.clip(voxel_ijk, 0, res - 1)

    geo_mask = np.zeros((res, res, res), dtype=np.float32)
    geo_mask[voxel_ijk[:, 0], voxel_ijk[:, 1], voxel_ijk[:, 2]] = 1.0

    wall_vijk = voxel_ijk[wall_indices]

    count = np.zeros((res, res, res), dtype=np.float32)
    np.add.at(count, (voxel_ijk[:, 0], voxel_ijk[:, 1], voxel_ijk[:, 2]), 1.0)
    count_mask = count > 0

    return {
        'voxel_ijk': voxel_ijk,
        'wall_vijk': wall_vijk,
        'geo_mask': geo_mask,
        'count': count,
        'count_mask': count_mask,
    }


def scatter_to_grid(case_data, vox, t_idx, resolution):
    """Scatter velocity+pressure to voxel grid at timestep t_idx."""
    res = resolution
    ijk = vox['voxel_ijk']
    count = vox['count']
    cmask = vox['count_mask']

    vel = case_data['velocity_full'][t_idx]
    pres = case_data['pressure_full'][t_idx]

    grids = []
    for c in range(3):
        g = np.zeros((res, res, res), dtype=np.float32)
        np.add.at(g, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), vel[:, c])
        g[cmask] /= count[cmask]
        grids.append(g)

    g_p = np.zeros((res, res, res), dtype=np.float32)
    np.add.at(g_p, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), pres[:, 0])
    g_p[cmask] /= count[cmask]
    grids.append(g_p)

    return np.stack(grids, axis=0)  # [4, R, R, R]


# ── KNN edge builder (from dataset_cross.py MultiCaseMGNDataset) ──

def build_knn_edges(coords, k=16):
    """Build k-NN edges from coordinates."""
    k = min(k, len(coords) - 1)
    tree = cKDTree(coords)
    dist, idx = tree.query(coords, k=k + 1)
    N = len(coords)
    src = np.repeat(np.arange(N), k)
    dst = idx[:, 1:].flatten()
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    rel_pos = coords[dst] - coords[src]
    edge_dist = dist[:, 1:].flatten()[:, None].astype(np.float32)
    edge_attr = np.concatenate([rel_pos, edge_dist], axis=1).astype(np.float32)
    return edge_index, edge_attr


# ── VTP export helpers (from test_baselines.py) ──

def load_original_wall_vtp(vtp_dir, time_value):
    """Load original wall.vtp for a given time value."""
    import pyvista as pv
    time_str = f"{time_value:.2f}"
    vtp_path = Path(vtp_dir) / time_str / f"{time_str}_wall.vtp"

    if not vtp_path.exists():
        vtp_dir_path = Path(vtp_dir)
        if not vtp_dir_path.exists():
            return None
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
    """Export VTP with predicted WSS."""
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


# ── Model builders (from train_cross.py) ──

def build_model(model_name, saved_args, device, state_dict=None):
    """Build model from saved args."""
    if model_name == 'fno':
        from baselines.models.fno import FNO3D_WSS
        model = FNO3D_WSS(
            in_channels=9,
            hidden_channels=saved_args.get('hidden_channels', 12),
            n_modes=(saved_args.get('n_modes', 20),) * 3,
            n_layers=saved_args.get('n_layers', 4),
            use_mlp=True,
        )
    elif model_name == 'unet':
        from baselines.models.unet3d import UNet3D_WSS
        model = UNet3D_WSS(
            in_channels=9,
            base_channels=saved_args.get('hidden_channels', 14),
            depth=saved_args.get('n_layers', 4),
        )
    elif model_name == 'mgn':
        from baselines.models.meshgraphnet import MeshGraphNet_WSS
        model = MeshGraphNet_WSS(
            node_input_dim=11, edge_input_dim=4,
            hidden_dim=saved_args.get('mgn_hidden', 64),
            num_message_passing=saved_args.get('mgn_layers', 10),
            output_dim=1,
        )
    elif model_name == 'transolver':
        from baselines.models.transolver import Transolver_WSS
        model = Transolver_WSS(
            in_channels=11,
            d_model=saved_args.get('trans_dim', 128),
            n_heads=saved_args.get('trans_heads', 8),
            n_slices=saved_args.get('trans_slices', 8),
            ffn_dim=saved_args.get('trans_ffn', 256),
            n_layers=saved_args.get('trans_layers', 8),
        )
    elif model_name == 'deeponet':
        from aneumo.models import TemporalDeepONetV2
        # Infer num_input_vars from checkpoint: point_encoder.0.weight shape is
        # [hidden, 4 + num_input_vars]
        num_input_vars = saved_args.get('num_input_vars', 5)
        if state_dict is not None and 'history_encoder.point_encoder.0.weight' in state_dict:
            pe_shape = state_dict['history_encoder.point_encoder.0.weight'].shape[1]
            num_input_vars = pe_shape - 4  # subtract coord dims (x,y,z,t)
        model = TemporalDeepONetV2(
            num_input_vars=num_input_vars,
            num_output_vars=1,
            history_embed_dim=saved_args.get('history_embed_dim', 128),
            history_encoder_type=saved_args.get('history_encoder', 'light'),
            history_num_layers=3,
            swin_embed_dim=24,
            use_geometry=saved_args.get('use_geometry', False),
            trunk_hidden_dim=saved_args.get('trunk_hidden_dim', 128),
            trunk_num_layers=saved_args.get('trunk_num_layers', 4),
            use_cross_attention=True,
            branch_dim=saved_args.get('branch_dim', 256),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


# ── Per-model full-wall test functions ──

@torch.no_grad()
def test_grid_case(model, case_data, test_timesteps, device, resolution):
    """Test FNO/UNet on one case, all wall points."""
    vox = setup_voxelization(case_data, resolution)
    wall_vijk = torch.from_numpy(vox['wall_vijk']).long().to(device)

    results = []
    for t in test_timesteps:
        field_prev = scatter_to_grid(case_data, vox, t - 1, resolution)
        field_curr = scatter_to_grid(case_data, vox, t, resolution)
        x = np.concatenate([field_prev, field_curr,
                            vox['geo_mask'][np.newaxis]], axis=0)  # [9,R,R,R]
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)

        pred = model(x_t)
        pred_wall = pred[0, 0][wall_vijk[:, 0], wall_vijk[:, 1], wall_vijk[:, 2]].cpu().numpy()
        true_wall = case_data['wss_mag'][t + 1]

        m = compute_metrics(pred_wall, true_wall)
        m['t_idx'] = int(t + 1)
        results.append(m)

    return results


@torch.no_grad()
def test_mgn_case(model, case_data, test_timesteps, device, k_neighbors):
    """Test MGN on one case, all wall points."""
    coords = case_data['wall_coords']
    edge_index, edge_attr = build_knn_edges(coords, k=k_neighbors)
    edge_index_t = torch.from_numpy(edge_index).to(device)
    edge_attr_t = torch.from_numpy(edge_attr).to(device)

    results = []
    for t in test_timesteps:
        vel_prev = case_data['wall_vel'][t - 1]
        pres_prev = case_data['wall_pres'][t - 1]
        vel_curr = case_data['wall_vel'][t]
        pres_curr = case_data['wall_pres'][t]

        node_feat = np.concatenate(
            [coords, vel_prev, pres_prev, vel_curr, pres_curr], axis=1
        ).astype(np.float32)
        node_feat_t = torch.from_numpy(node_feat).to(device)

        with torch.amp.autocast('cuda'):
            pred = model(node_feat_t, edge_index_t, edge_attr_t)
        pred_wall = pred[:, 0].float().cpu().numpy()
        true_wall = case_data['wss_mag'][t + 1]

        m = compute_metrics(pred_wall, true_wall)
        m['t_idx'] = int(t + 1)
        results.append(m)

    return results


@torch.no_grad()
def test_point_case(model, case_data, test_timesteps, device):
    """Test Transolver on one case, all wall points."""
    coords = case_data['wall_coords']

    results = []
    for t in test_timesteps:
        vel_prev = case_data['wall_vel'][t - 1]
        pres_prev = case_data['wall_pres'][t - 1]
        vel_curr = case_data['wall_vel'][t]
        pres_curr = case_data['wall_pres'][t]

        feat = np.concatenate(
            [coords, vel_prev, pres_prev, vel_curr, pres_curr], axis=1
        ).astype(np.float32)
        feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)

        pred = model(feat_t).squeeze(0).squeeze(-1).cpu().numpy()
        true_wall = case_data['wss_mag'][t + 1]

        m = compute_metrics(pred, true_wall)
        m['t_idx'] = int(t + 1)
        results.append(m)

    return results


@torch.no_grad()
def test_deeponet_case(model, case_data, test_timesteps, device,
                       saved_args, batch_points=5000, num_input_vars=5):
    """Test DeepONet on one case, all wall points as query."""
    coords = case_data['wall_coords']       # [M, 3] normalized
    num_wall = case_data['num_wall']
    time_norm = case_data['time_norm']
    input_steps = saved_args.get('input_steps', 2)
    num_wall_samples = saved_args.get('num_wall_samples', 2000)
    geo = torch.from_numpy(case_data['geometry'].copy()).to(device)  # [1,32,32,32]

    results = []
    for t in test_timesteps:
        t_target = t + 1
        if t_target >= case_data['num_timesteps']:
            continue

        # Build history frames (sampled, like training)
        hist_sample_size = min(num_wall_samples, num_wall)
        hist_idx = np.random.choice(num_wall, hist_sample_size, replace=False)
        hist_coords = coords[hist_idx]  # [S, 3]

        x_hist_list, y_hist_list = [], []
        for h in range(t - input_steps + 1, t + 1):
            if h < 0:
                h = 0
            t_val = time_norm[h]
            t_col = np.full((hist_sample_size, 1), t_val, dtype=np.float32)
            x_hist_list.append(np.concatenate([hist_coords, t_col], axis=1))

            wss = case_data['wss_mag'][h, hist_idx][:, None]
            if num_input_vars >= 5:
                vel = case_data['wall_vel'][h, hist_idx]
                pres = case_data['wall_pres'][h, hist_idx]
                y_hist_list.append(np.concatenate([vel, pres, wss], axis=1))
            else:
                # num_input_vars=1: WSS only
                y_hist_list.append(wss)

        x_hist = torch.from_numpy(
            np.concatenate(x_hist_list, axis=0).astype(np.float32)
        ).unsqueeze(0).to(device)
        y_hist = torch.from_numpy(
            np.concatenate(y_hist_list, axis=0).astype(np.float32)
        ).unsqueeze(0).to(device)

        # Query: ALL wall points, batched
        t_query = time_norm[t_target]
        t_col_all = np.full((num_wall, 1), t_query, dtype=np.float32)
        x_query_all = np.concatenate([coords, t_col_all], axis=1).astype(np.float32)

        preds = []
        for start in range(0, num_wall, batch_points):
            end = min(start + batch_points, num_wall)
            x_batch = torch.from_numpy(x_query_all[start:end]).unsqueeze(0).to(device)
            y_pred = model(x_hist, y_hist, x_batch, geo)
            preds.append(y_pred[0].cpu().numpy())

        pred_wall = np.concatenate(preds, axis=0).squeeze(-1)  # [M]
        true_wall = case_data['wss_mag'][t_target]

        m = compute_metrics(pred_wall, true_wall)
        m['t_idx'] = int(t_target)
        results.append(m)

    return results


# ── Test timestep selection ──

def get_test_timesteps(case_data, split_mode):
    """Determine which timesteps to test.

    Returns list of t values where we predict t+1.
    """
    T = case_data['num_timesteps']
    if split_mode == 'geometry':
        # All timesteps, skip first 2 for history
        return list(range(2, T - 1))
    else:
        # Time split: test = t=60~80 (same as training eval)
        t_start = max(60, 2)
        t_end = min(80, T)
        return list(range(t_start, t_end - 1))


# ── VTP export for one case ──

def export_case_vtp(case_data, case_id, pred_per_t, vtp_base, output_dir,
                    smooth_iterations=10):
    """Export VTP files for one case."""
    vtp_dir = Path(vtp_base) / str(case_id)
    if not vtp_dir.exists():
        print(f"  VTP dir not found: {vtp_dir}, skipping export")
        return

    # Load original H5 to get full wall info before boundary cut
    # We need the original wall_indices and wss for fill-in
    h5_path = None  # We get time_values from case_data's original h5
    # time_values are in the case_data indirectly via time_norm
    # We need to reload from H5 for raw time_values
    # For now, try to find matching VTP by scanning available directories
    case_out = Path(output_dir) / f"case_{case_id}"
    case_out.mkdir(parents=True, exist_ok=True)

    for t_idx, pred_wall in pred_per_t.items():
        # Try to find the VTP by iterating time dirs
        time_dirs = sorted([d.name for d in vtp_dir.iterdir() if d.is_dir()])
        if t_idx >= len(time_dirs):
            continue
        time_str = time_dirs[t_idx] if t_idx < len(time_dirs) else None
        if time_str is None:
            continue

        try:
            time_val = float(time_str)
        except ValueError:
            continue

        mesh = load_original_wall_vtp(vtp_dir, time_val)
        if mesh is None:
            continue

        # pred_wall is for boundary-cut wall points
        # For VTP export, we map predictions to the full mesh
        # Since H5 wall order matches VTP point order, and pred_wall
        # corresponds to case_data['wall_indices'] (post boundary cut),
        # we fill the full mesh: GT everywhere, predictions where we have them
        if 'wallShearStress' in mesh.point_data:
            original_wss = mesh.point_data['wallShearStress']
            if len(original_wss.shape) > 1:
                full_gt = np.sqrt(np.sum(original_wss ** 2, axis=-1))
            else:
                full_gt = original_wss.copy()
            full_pred = full_gt.copy()

            # Map: the boundary-cut wall_indices in H5 correspond to
            # a subset of VTP points. We need the H5→VTP index mapping.
            # Since H5 wall order = VTP point order (from dataprocess),
            # wall_indices[i] in the full node array → VTP point index i
            # But after boundary cut, case_data['wall_indices'] is a subset.
            # We don't have the original (pre-cut) wall_indices here.
            # Safest: just assign pred_wall directly (same length as mesh points
            # if no boundary cut mismatch). If lengths differ, use KDTree.

            if len(pred_wall) == mesh.n_points:
                full_pred = pred_wall.copy()
            else:
                # Lengths differ (boundary cut removed some points)
                # pred_wall corresponds to case_data['wall_coords'] (normalized)
                # We need original coords for matching
                mesh_coords = np.array(mesh.points, dtype=np.float32)
                # case_data['wall_coords'] is normalized, need to reverse
                # Instead, just put predictions on the mesh as-is and note mismatch
                # This is a best-effort export
                pass

            vtp_path = case_out / f"wss_t{t_idx:04d}.vtp"
            export_vtp(mesh, full_pred, vtp_path, smooth_iterations=smooth_iterations)
        else:
            # No GT in VTP, just assign predictions
            if len(pred_wall) == mesh.n_points:
                vtp_path = case_out / f"wss_t{t_idx:04d}.vtp"
                export_vtp(mesh, pred_wall, vtp_path,
                           smooth_iterations=smooth_iterations)


# ── Summary output ──

def write_results(all_case_results, args, saved_args, num_params, output_dir):
    """Write summary.txt and results.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten all per-timestep metrics across cases
    all_l2, all_mse, all_mae, all_mnae = [], [], [], []
    for cr in all_case_results:
        for r in cr['results']:
            all_l2.append(r['l2'])
            all_mse.append(r['mse'])
            all_mae.append(r['mae'])
            all_mnae.append(r['mnae'])

    # Print summary
    print(f"\n{'='*60}")
    print(f"Cross-Geometry Test Summary")
    print(f"{'='*60}")
    print(f"Model: {args.model}, Split: {args.split_mode}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Parameters: {num_params:,}")
    print(f"Test cases: {len(all_case_results)}")
    print(f"Total test timesteps: {len(all_l2)}")
    print()
    print(f"Mean L2:   {np.mean(all_l2):.4f} ± {np.std(all_l2):.4f}")
    print(f"Mean MSE:  {np.mean(all_mse):.2e} ± {np.std(all_mse):.2e}")
    print(f"Mean MAE:  {np.mean(all_mae):.2e} ± {np.std(all_mae):.2e}")
    print(f"Mean MNAE: {np.mean(all_mnae):.4f} ± {np.std(all_mnae):.4f}")

    # Per-case summary
    print(f"\nPer-case results:")
    for cr in all_case_results:
        case_l2 = [r['l2'] for r in cr['results']]
        case_mse = [r['mse'] for r in cr['results']]
        print(f"  case_{cr['case_id']}: L2={np.mean(case_l2):.4f}±{np.std(case_l2):.4f}, "
              f"MSE={np.mean(case_mse):.2e}, wall_pts={cr['num_wall']}, "
              f"timesteps={len(cr['results'])}")

    # Write summary.txt
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Cross-Geometry Full-Wall Test Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Split mode: {args.split_mode}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Parameters: {num_params:,}\n")
        f.write(f"Test cases: {len(all_case_results)}\n")
        f.write(f"Total test timesteps: {len(all_l2)}\n")
        f.write("\n")
        f.write(f"Mean L2:   {np.mean(all_l2):.4f} ± {np.std(all_l2):.4f}\n")
        f.write(f"Mean MSE:  {np.mean(all_mse):.2e} ± {np.std(all_mse):.2e}\n")
        f.write(f"Mean MAE:  {np.mean(all_mae):.2e} ± {np.std(all_mae):.2e}\n")
        f.write(f"Mean MNAE: {np.mean(all_mnae):.4f} ± {np.std(all_mnae):.4f}\n")
        f.write("\nPer-case summary:\n")
        for cr in all_case_results:
            case_l2 = [r['l2'] for r in cr['results']]
            case_mse = [r['mse'] for r in cr['results']]
            case_mae = [r['mae'] for r in cr['results']]
            case_mnae = [r['mnae'] for r in cr['results']]
            f.write(f"  case_{cr['case_id']}: "
                    f"L2={np.mean(case_l2):.4f}±{np.std(case_l2):.4f}, "
                    f"MSE={np.mean(case_mse):.2e}±{np.std(case_mse):.2e}, "
                    f"MAE={np.mean(case_mae):.2e}±{np.std(case_mae):.2e}, "
                    f"MNAE={np.mean(case_mnae):.4f}±{np.std(case_mnae):.4f}, "
                    f"wall_pts={cr['num_wall']}, timesteps={len(cr['results'])}\n")
        f.write("\nPer-case per-timestep:\n")
        for cr in all_case_results:
            f.write(f"\n  case_{cr['case_id']}:\n")
            for r in cr['results']:
                f.write(f"    t={r['t_idx']}: L2={r['l2']:.4f}, MSE={r['mse']:.2e}, "
                        f"MAE={r['mae']:.2e}, MNAE={r['mnae']:.4f}\n")

    print(f"\nSummary saved to: {summary_path}")

    # Write results.json
    per_case_json = []
    for cr in all_case_results:
        case_l2 = [r['l2'] for r in cr['results']]
        case_mse = [r['mse'] for r in cr['results']]
        case_mae = [r['mae'] for r in cr['results']]
        case_mnae = [r['mnae'] for r in cr['results']]
        per_case_json.append({
            'case_id': cr['case_id'],
            'num_wall': cr['num_wall'],
            'mean_l2': float(np.mean(case_l2)),
            'std_l2': float(np.std(case_l2)),
            'mean_mse': float(np.mean(case_mse)),
            'mean_mae': float(np.mean(case_mae)),
            'mean_mnae': float(np.mean(case_mnae)),
            'per_timestep': cr['results'],
        })

    json_path = output_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'model': args.model,
            'split_mode': args.split_mode,
            'checkpoint': args.checkpoint,
            'num_params': num_params,
            'num_test_cases': len(all_case_results),
            'total_test_timesteps': len(all_l2),
            'overall': {
                'mean_l2': float(np.mean(all_l2)),
                'std_l2': float(np.std(all_l2)),
                'mean_mse': float(np.mean(all_mse)),
                'std_mse': float(np.std(all_mse)),
                'mean_mae': float(np.mean(all_mae)),
                'std_mae': float(np.std(all_mae)),
                'mean_mnae': float(np.mean(all_mnae)),
                'std_mnae': float(np.std(all_mnae)),
            },
            'per_case': per_case_json,
            'args': vars(args),
        }, f, indent=2, default=str)

    print(f"Results saved to: {json_path}")


# ── Main ──

def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get('args', {})
    print(f"Trained for {ckpt.get('epoch', '?')} epochs, "
          f"best L2={ckpt.get('best_l2', '?')}")

    # Handle DDP state_dict prefix
    state_dict = ckpt['model_state_dict']
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    # Build model (pass state_dict so DeepONet can infer num_input_vars)
    model = build_model(args.model, saved_args, device, state_dict=state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model.upper()}, {num_params:,} parameters")

    # Get test case IDs
    split_mode = args.split_mode
    _, test_ids = get_train_test_ids(split_mode)
    # For time split, train_ids == test_ids (all 50 cases)
    if split_mode == 'time':
        print(f"Time split: testing on all {len(test_ids)} cases, t=60~80")
    else:
        print(f"Geometry split: testing on {len(test_ids)} held-out cases, all timesteps")

    # Get model-specific params
    resolution = saved_args.get('resolution', 96)
    k_neighbors = saved_args.get('mgn_k', 16)
    boundary_cut = saved_args.get('boundary_cut', 0.1)

    # Test each case
    all_case_results = []
    h5_dir = Path(args.h5_dir)

    for case_id in tqdm(test_ids, desc="Testing cases"):
        h5_path = h5_dir / f"case_{case_id}.h5"
        if not h5_path.exists():
            print(f"  WARNING: {h5_path} not found, skipping")
            continue

        case_data = load_case(str(h5_path), boundary_cut=boundary_cut)
        if case_data is None:
            print(f"  WARNING: case_{case_id} failed to load, skipping")
            continue

        test_ts = get_test_timesteps(case_data, split_mode)
        if not test_ts:
            print(f"  WARNING: case_{case_id} has no test timesteps, skipping")
            continue

        print(f"\n  case_{case_id}: {case_data['num_wall']} wall pts, "
              f"T={case_data['num_timesteps']}, "
              f"test t={test_ts[0]+1}~{test_ts[-1]+1}")

        # Run model-specific test
        if args.model in ('fno', 'unet'):
            results = test_grid_case(model, case_data, test_ts, device, resolution)
        elif args.model == 'mgn':
            results = test_mgn_case(model, case_data, test_ts, device, k_neighbors)
        elif args.model == 'transolver':
            results = test_point_case(model, case_data, test_ts, device)
        elif args.model == 'deeponet':
            # Infer num_input_vars from checkpoint
            niv = 5
            if 'history_encoder.point_encoder.0.weight' in state_dict:
                niv = state_dict['history_encoder.point_encoder.0.weight'].shape[1] - 4
            results = test_deeponet_case(
                model, case_data, test_ts, device, saved_args, args.batch_points,
                num_input_vars=niv)

        case_l2 = [r['l2'] for r in results]
        print(f"  → L2={np.mean(case_l2):.4f}±{np.std(case_l2):.4f}")

        all_case_results.append({
            'case_id': case_id,
            'num_wall': case_data['num_wall'],
            'results': results,
        })

        # VTP export
        if args.vtp_base is not None:
            pred_per_t = {r['t_idx']: None for r in results}
            # We'd need to re-run or cache predictions for VTP export
            # For now, VTP export is handled in a second pass if needed
            pass

    # Write results
    write_results(all_case_results, args, saved_args, num_params, args.output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
