"""
Multi-case dataset for cross-geometry generalization experiment.

Supports all model types:
  - DeepONet: temporal windowing with geometry voxelization
  - Transolver / PointNet++: per-timestep point cloud
  - FNO / UNet: voxelized 3D grids

Two split modes (controlled by split_mode parameter):
  - 'geometry': hold out 2 deforms per base geo for test, all timesteps used.
  - 'time': all 100 cases used, split by timesteps (front 80% train, back 20% test).
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path


# ────────────────────── shared case loader ──────────────────────

def load_case(h5_path, boundary_cut=0.1):
    """Load a single H5 case into memory."""
    try:
        with h5py.File(h5_path, 'r') as f:
            coords = f['mesh/coords'][:].astype(np.float32)
            node_type = f['mesh/node_type'][:].astype(np.int64)
            wall_indices = f['mesh/wall_indices'][:].astype(np.int64)
            velocity = f['fields/velocity'][:].astype(np.float32)
            pressure = f['fields/pressure'][:].astype(np.float32)
            wss = f['fields/wss'][:].astype(np.float32)
            time_values = f['time_values'][:].astype(np.float32)
            edges = f['mesh/edges'][:].astype(np.int64) if 'mesh/edges' in f else None
    except Exception as e:
        print(f"ERROR loading {h5_path}: {e}")
        return None

    # Boundary cut
    if boundary_cut > 0:
        wall_indices, wss = _apply_boundary_cut(
            coords, node_type, wall_indices, wss, boundary_cut)

    if len(wall_indices) == 0:
        print(f"WARNING: {h5_path} has no wall points after boundary cut, skipping")
        return None

    # Wall-only data
    wall_coords = coords[wall_indices]
    wall_vel = velocity[:, wall_indices, :]
    wall_pres = pressure[:, wall_indices, :]
    wss_mag = np.sqrt(np.sum(wss.astype(np.float64) ** 2, axis=-1))  # [T, M] float64
    wss_mag = np.clip(wss_mag, 0, np.finfo(np.float32).max).astype(np.float32)

    # ── Detect & truncate diverged timesteps ──
    # CFD simulations sometimes diverge at the end, producing extreme WSS values.
    # Detect: per-timestep max WSS > threshold → that timestep is diverged.
    WSS_DIVERGE_THRESH = 100.0  # reasonable upper bound for physical WSS
    per_t_max = wss_mag.max(axis=1)  # [T]
    valid_mask = per_t_max < WSS_DIVERGE_THRESH
    last_valid = np.where(valid_mask)[0]
    if len(last_valid) == 0:
        print(f"WARNING: {h5_path} ALL timesteps diverged, skipping")
        return None
    # Truncate at the first diverged timestep (keep contiguous valid prefix)
    T_valid = last_valid[-1] + 1  # use all timesteps up to (and including) last valid
    # Actually, find the first diverged index and truncate before it
    diverged = np.where(~valid_mask)[0]
    if len(diverged) > 0:
        T_valid = diverged[0]  # truncate at first bad timestep
        if T_valid < 3:
            print(f"WARNING: {h5_path} diverges at t={T_valid} (need >=3), skipping")
            return None
        print(f"  {Path(h5_path).stem}: truncating at t={T_valid}/{len(per_t_max)} "
              f"(diverged: max_wss={per_t_max[diverged[0]]:.2e})")
        wss_mag = wss_mag[:T_valid]
        wall_vel = wall_vel[:T_valid]
        wall_pres = wall_pres[:T_valid]
        velocity = velocity[:T_valid]
        pressure = pressure[:T_valid]
        time_values = time_values[:T_valid]

    # Skip cases with too few timesteps
    if len(time_values) < 3:
        print(f"WARNING: {h5_path} only {len(time_values)} timesteps, skipping")
        return None

    # Normalize coords per case
    cmin = wall_coords.min(0)
    crange = wall_coords.max(0) - cmin + 1e-8
    wall_coords_norm = ((wall_coords - cmin) / crange).astype(np.float32)

    # Normalize time
    t_min, t_max = time_values.min(), time_values.max()
    time_norm = ((time_values - t_min) / (t_max - t_min + 1e-8)).astype(np.float32)

    # Voxelize geometry (for DeepONet / FNO / UNet)
    geo_res = 32
    voxel_ijk = np.floor(wall_coords_norm * (geo_res - 1)).astype(np.int32)
    voxel_ijk = np.clip(voxel_ijk, 0, geo_res - 1)
    geo_volume = np.zeros((1, geo_res, geo_res, geo_res), dtype=np.float32)
    geo_volume[0, voxel_ijk[:, 0], voxel_ijk[:, 1], voxel_ijk[:, 2]] = 1.0

    return {
        'wall_coords': wall_coords_norm,   # [M, 3]
        'wall_vel': wall_vel,               # [T, M, 3]
        'wall_pres': wall_pres,             # [T, M, 1]
        'wss_mag': wss_mag,                 # [T, M]
        'time_norm': time_norm,             # [T]
        'geometry': geo_volume,             # [1, 32, 32, 32]
        'num_wall': len(wall_indices),
        'num_timesteps': len(wss_mag),
        'name': Path(h5_path).stem,
        # Full mesh data (for FNO/UNet voxelization and MGN edges)
        'coords_full': coords,
        'velocity_full': velocity,
        'pressure_full': pressure,
        'node_type': node_type,
        'wall_indices': wall_indices,
        'edges': edges,
    }


def _apply_boundary_cut(coords, node_type, wall_indices, wss, boundary_cut):
    """Cut wall points near inlet/outlet boundaries.

    Uses distance to nearest inlet/outlet point rather than 1D projection,
    which is more robust for complex geometries (bifurcations, aneurysms).
    """
    inlet_mask = node_type == 1
    outlet_mask = node_type == 2
    if not np.any(inlet_mask) or not np.any(outlet_mask):
        return wall_indices, wss

    # Characteristic length: diameter of the geometry bounding box
    bbox_diag = np.linalg.norm(coords.max(0) - coords.min(0))
    cut_radius = bbox_diag * boundary_cut

    # Compute min distance from each wall point to any inlet/outlet point
    wall_coords = coords[wall_indices]  # [M, 3]
    boundary_coords = coords[inlet_mask | outlet_mask]  # [K, 3]

    # Chunked distance computation to avoid OOM on large meshes
    min_dist = np.full(len(wall_coords), np.inf, dtype=np.float32)
    chunk_size = 2000
    for i in range(0, len(boundary_coords), chunk_size):
        bc_chunk = boundary_coords[i:i + chunk_size]  # [chunk, 3]
        # [M, chunk]
        dist = np.linalg.norm(
            wall_coords[:, None, :] - bc_chunk[None, :, :], axis=-1)
        min_dist = np.minimum(min_dist, dist.min(axis=1))

    wall_valid = min_dist > cut_radius

    if wall_valid.sum() == 0:
        # Fallback: keep all wall points if cut is too aggressive
        return wall_indices, wss

    return wall_indices[wall_valid], wss[:, wall_valid, :]


def _subsample(N, target):
    """Subsample or pad to target size."""
    if N >= target:
        return np.random.choice(N, target, replace=False)
    else:
        idx = np.arange(N)
        pad_idx = np.random.choice(N, target - N, replace=True)
        return np.concatenate([idx, pad_idx])


# ────────────────────── DeepONet dataset ──────────────────────

class MultiCaseDeepONetDataset(Dataset):
    """
    Multi-case DeepONet dataset for WSS prediction.

    Each sample: use input_steps history frames to predict output_steps future WSS.
    Returns the same format as TemporalCFDDataset (WSS-only mode).

    Returns per sample:
        x_hist:   [input_steps * N_s, 4]  — wall point (x,y,z,t) for history
        y_hist:   [input_steps * N_s, 5]  — vel(3)+pres(1)+WSS(1) at history frames
        x_query:  [output_steps * N_s, 4] — wall point (x,y,z,t) for prediction
        y_output: [output_steps * N_s, 1] — WSS ground truth
        geometry: [1, 32, 32, 32]         — voxelized wall geometry
        bc:       [7]                     — boundary condition features
        wall_mask:[output_steps * N_s]    — all ones (wall only)
    """

    def __init__(self, h5_dir, case_ids, num_wall_samples=2000,
                 input_steps=2, output_steps=1,
                 mode='train', split_mode='geometry',
                 boundary_cut=0.1, skip_first=1,
                 time_range=(20, 80), time_train_end=60):
        self.h5_dir = Path(h5_dir)
        self.num_wall_samples = num_wall_samples
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.mode = mode

        self.cases = []
        self.samples = []  # (case_idx, window_start)

        window = input_steps + output_steps

        for cid in sorted(case_ids):
            h5_path = self.h5_dir / f"case_{cid}.h5"
            if not h5_path.exists():
                print(f"WARNING: {h5_path} not found, skipping")
                continue

            case_data = load_case(h5_path, boundary_cut=boundary_cut)
            if case_data is None:
                continue

            case_idx = len(self.cases)
            self.cases.append(case_data)

            T = case_data['num_timesteps']
            if split_mode == 'geometry':
                # All timesteps — split is by case, not time
                t_start = max(skip_first, 0)
                t_end = T
            else:  # time: use t=20~80 only; train=20~60, test=60~80
                t_lo, t_hi = time_range
                t_mid = time_train_end
                if mode == 'train':
                    t_start = max(skip_first, t_lo)
                    t_end = min(t_mid, T)
                else:
                    t_start = max(t_mid, skip_first)
                    t_end = min(t_hi, T)

            for w in range(t_start, t_end - window + 1):
                self.samples.append((case_idx, w))

        print(f"MultiCaseDeepONetDataset [{mode}|{split_mode}]: {len(self.cases)} cases, "
              f"{len(self.samples)} samples, "
              f"in={input_steps}, out={output_steps}, wall_s={num_wall_samples}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_idx, w_start = self.samples[idx]
        case = self.cases[case_idx]
        N_s = self.num_wall_samples

        sub_idx = _subsample(case['num_wall'], N_s)
        coords = case['wall_coords'][sub_idx]  # [N_s, 3]

        in_indices = list(range(w_start, w_start + self.input_steps))
        out_indices = list(range(w_start + self.input_steps,
                                 w_start + self.input_steps + self.output_steps))

        # Build history (input): vel(3) + pres(1) + wss(1) = 5 dims
        x_hist_list, y_hist_list = [], []
        for t in in_indices:
            t_val = case['time_norm'][t]
            t_col = np.full((N_s, 1), t_val, dtype=np.float32)
            x_hist_list.append(np.concatenate([coords, t_col], axis=1))
            vel = case['wall_vel'][t, sub_idx]          # [N_s, 3]
            pres = case['wall_pres'][t, sub_idx]        # [N_s, 1]
            wss = case['wss_mag'][t, sub_idx][:, None]  # [N_s, 1]
            y_hist_list.append(np.concatenate([vel, pres, wss], axis=1))

        # Build query (output)
        x_query_list, y_out_list = [], []
        for t in out_indices:
            t_val = case['time_norm'][t]
            t_col = np.full((N_s, 1), t_val, dtype=np.float32)
            x_query_list.append(np.concatenate([coords, t_col], axis=1))
            y_out_list.append(case['wss_mag'][t, sub_idx][:, None])

        x_hist = np.concatenate(x_hist_list, axis=0).astype(np.float32)
        y_hist = np.concatenate(y_hist_list, axis=0).astype(np.float32)
        x_query = np.concatenate(x_query_list, axis=0).astype(np.float32)
        y_output = np.concatenate(y_out_list, axis=0).astype(np.float32)
        wall_mask = np.ones(len(x_query), dtype=np.float32)

        # Boundary condition features (per-case)
        coord_range = case['wall_coords'].max(0) - case['wall_coords'].min(0)
        bc = np.zeros(7, dtype=np.float32)
        bc[:3] = coord_range
        bc[3] = np.mean(np.linalg.norm(case['wall_vel'][in_indices], axis=-1))
        bc[4] = case['wall_pres'][in_indices].max() - case['wall_pres'][in_indices].min()
        bc[5] = case['time_norm'][-1] - case['time_norm'][0]
        bc[6] = bc[3] * np.prod(coord_range) ** (1/3)

        return (
            torch.from_numpy(x_hist),
            torch.from_numpy(y_hist),
            torch.from_numpy(x_query),
            torch.from_numpy(y_output),
            torch.from_numpy(case['geometry'].copy()),
            torch.from_numpy(bc),
            torch.from_numpy(wall_mask),
        )


# ────────────────────── Point cloud dataset ──────────────────────

class MultiCasePointDataset(Dataset):
    """
    Multi-case wall point cloud dataset for Transolver / PointNet++.
    Each sample = one timestep from one case.

    Returns:
        feat:   [N_s, 11] — coords(3) + vel_prev(3) + p_prev(1) + vel_curr(3) + p_curr(1)
        target: [N_s]     — WSS magnitude at t+1
    """

    def __init__(self, h5_dir, case_ids, num_wall_samples=4096,
                 mode='train', split_mode='geometry',
                 boundary_cut=0.1, skip_first=1,
                 time_range=(20, 80), time_train_end=60):
        self.h5_dir = Path(h5_dir)
        self.num_wall_samples = num_wall_samples
        self.mode = mode

        self.cases = []
        self.samples = []

        for cid in sorted(case_ids):
            h5_path = self.h5_dir / f"case_{cid}.h5"
            if not h5_path.exists():
                print(f"WARNING: {h5_path} not found, skipping")
                continue

            case_data = load_case(h5_path, boundary_cut=boundary_cut)
            if case_data is None:
                continue

            case_idx = len(self.cases)
            self.cases.append(case_data)

            T = case_data['num_timesteps']
            if split_mode == 'geometry':
                t_start = max(skip_first, 2)
                t_end = T
            else:  # time: use t=20~80 only; train=20~60, test=60~80
                t_lo, t_hi = time_range
                t_mid = time_train_end
                if mode == 'train':
                    t_start = max(skip_first, t_lo, 2)
                    t_end = min(t_mid, T)
                else:
                    t_start = max(t_mid, skip_first, 2)
                    t_end = min(t_hi, T)

            for t in range(t_start, t_end - 1):
                self.samples.append((case_idx, t))

        print(f"MultiCasePointDataset [{mode}|{split_mode}]: {len(self.cases)} cases, "
              f"{len(self.samples)} samples, wall_s={num_wall_samples}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_idx, t = self.samples[idx]
        case = self.cases[case_idx]

        sub_idx = _subsample(case['num_wall'], self.num_wall_samples)

        coords = case['wall_coords'][sub_idx]
        vel_prev = case['wall_vel'][t - 1, sub_idx]
        pres_prev = case['wall_pres'][t - 1, sub_idx]
        vel_curr = case['wall_vel'][t, sub_idx]
        pres_curr = case['wall_pres'][t, sub_idx]

        feat = np.concatenate([coords, vel_prev, pres_prev, vel_curr, pres_curr],
                              axis=1).astype(np.float32)
        target = case['wss_mag'][t + 1, sub_idx].astype(np.float32)

        return torch.from_numpy(feat), torch.from_numpy(target)


# ────────────────────── MGN (Graph) dataset ──────────────────────

class MultiCaseMGNDataset(Dataset):
    """
    Multi-case graph dataset for MeshGraphNet.
    Each sample = one timestep from one case, wall nodes as graph with k-NN edges.

    Returns:
        node_feat:  [N_s, 11] — coords(3) + vel_prev(3) + p_prev(1) + vel_curr(3) + p_curr(1)
        edge_index: [2, N_s*k] — k-NN edge indices
        edge_attr:  [N_s*k, 4] — relative position (3) + distance (1)
        target:     [N_s]      — WSS magnitude at t+1
    """

    def __init__(self, h5_dir, case_ids, num_wall_samples=4096, k_neighbors=16,
                 mode='train', split_mode='geometry',
                 boundary_cut=0.1, skip_first=1,
                 time_range=(20, 80), time_train_end=60):
        self.h5_dir = Path(h5_dir)
        self.num_wall_samples = num_wall_samples
        self.k_neighbors = k_neighbors
        self.mode = mode

        self.cases = []
        self.samples = []

        for cid in sorted(case_ids):
            h5_path = self.h5_dir / f"case_{cid}.h5"
            if not h5_path.exists():
                print(f"WARNING: {h5_path} not found, skipping")
                continue

            case_data = load_case(h5_path, boundary_cut=boundary_cut)
            if case_data is None:
                continue

            case_idx = len(self.cases)
            self.cases.append(case_data)

            T = case_data['num_timesteps']
            if split_mode == 'geometry':
                t_start = max(skip_first, 2)
                t_end = T
            else:  # time: use t=20~80 only; train=20~60, test=60~80
                t_lo, t_hi = time_range
                t_mid = time_train_end
                if mode == 'train':
                    t_start = max(skip_first, t_lo, 2)
                    t_end = min(t_mid, T)
                else:
                    t_start = max(t_mid, skip_first, 2)
                    t_end = min(t_hi, T)

            for t in range(t_start, t_end - 1):
                self.samples.append((case_idx, t))

        print(f"MultiCaseMGNDataset [{mode}|{split_mode}]: {len(self.cases)} cases, "
              f"{len(self.samples)} samples, wall_s={num_wall_samples}, k={k_neighbors}")

    def __len__(self):
        return len(self.samples)

    def _build_knn_edges(self, coords):
        """Build k-NN edges from coordinates."""
        from scipy.spatial import cKDTree
        k = min(self.k_neighbors, len(coords) - 1)
        tree = cKDTree(coords)
        dist, idx = tree.query(coords, k=k + 1)  # +1 for self
        N = len(coords)
        src = np.repeat(np.arange(N), k)
        dst = idx[:, 1:].flatten()
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)

        rel_pos = coords[dst] - coords[src]
        edge_dist = dist[:, 1:].flatten()[:, None].astype(np.float32)
        edge_attr = np.concatenate([rel_pos, edge_dist], axis=1).astype(np.float32)

        return edge_index, edge_attr

    def __getitem__(self, idx):
        case_idx, t = self.samples[idx]
        case = self.cases[case_idx]

        sub_idx = _subsample(case['num_wall'], self.num_wall_samples)

        coords = case['wall_coords'][sub_idx]
        vel_prev = case['wall_vel'][t - 1, sub_idx]
        pres_prev = case['wall_pres'][t - 1, sub_idx]
        vel_curr = case['wall_vel'][t, sub_idx]
        pres_curr = case['wall_pres'][t, sub_idx]

        node_feat = np.concatenate([coords, vel_prev, pres_prev, vel_curr, pres_curr],
                                   axis=1).astype(np.float32)
        target = case['wss_mag'][t + 1, sub_idx].astype(np.float32)

        edge_index, edge_attr = self._build_knn_edges(coords)

        return (
            torch.from_numpy(node_feat),
            torch.from_numpy(edge_index),
            torch.from_numpy(edge_attr),
            torch.from_numpy(target),
        )


def mgn_collate_fn(batch):
    """PyG-style batching: merge graphs by offsetting edge indices."""
    node_feats, edge_indices, edge_attrs, targets = [], [], [], []
    offset = 0
    for nf, ei, ea, tgt in batch:
        node_feats.append(nf)
        edge_indices.append(ei + offset)
        edge_attrs.append(ea)
        targets.append(tgt)
        offset += nf.shape[0]
    return (
        torch.cat(node_feats, 0),
        torch.cat(edge_indices, 1),
        torch.cat(edge_attrs, 0),
        torch.cat(targets, 0),
    )


# ────────────────────── Grid (FNO/UNet) dataset ──────────────────────

class MultiCaseGridDataset(Dataset):
    """
    Multi-case voxelized 3D grid dataset for FNO / UNet.
    Each sample = one timestep from one case, voxelized to fixed resolution.

    Returns:
        x:        [9, R, R, R]  — vel(3)+p(1) at t-1, vel(3)+p(1) at t, geo(1)
        y_grid:   [1, R, R, R]  — WSS on grid at t+1
        y_wall:   [M]           — WSS at wall nodes at t+1 (ground truth)
        wall_mask:[1, R, R, R]  — wall voxel mask
    """

    def __init__(self, h5_dir, case_ids, resolution=96,
                 mode='train', split_mode='geometry',
                 boundary_cut=0.1, skip_first=1,
                 time_range=(20, 80), time_train_end=60):
        self.h5_dir = Path(h5_dir)
        self.resolution = resolution
        self.mode = mode

        self.cases = []
        self.samples = []

        for cid in sorted(case_ids):
            h5_path = self.h5_dir / f"case_{cid}.h5"
            if not h5_path.exists():
                print(f"WARNING: {h5_path} not found, skipping")
                continue

            case_data = load_case(h5_path, boundary_cut=boundary_cut)
            if case_data is None:
                continue

            # Pre-voxelize this case
            vox = self._setup_voxelization(case_data)
            case_data['vox'] = vox

            case_idx = len(self.cases)
            self.cases.append(case_data)

            T = case_data['num_timesteps']
            if split_mode == 'geometry':
                t_start = max(skip_first, 2)
                t_end = T
            else:  # time: use t=20~80 only; train=20~60, test=60~80
                t_lo, t_hi = time_range
                t_mid = time_train_end
                if mode == 'train':
                    t_start = max(skip_first, t_lo, 2)
                    t_end = min(t_mid, T)
                else:
                    t_start = max(t_mid, skip_first, 2)
                    t_end = min(t_hi, T)

            for t in range(t_start, t_end - 1):
                self.samples.append((case_idx, t))

        # Max wall points across all cases (for padding y_wall)
        self.max_wall = max(c['num_wall'] for c in self.cases) if self.cases else 0

        print(f"MultiCaseGridDataset [{mode}|{split_mode}]: {len(self.cases)} cases, "
              f"{len(self.samples)} samples, res={resolution}, max_wall={self.max_wall}")

    def _setup_voxelization(self, case_data):
        """Setup voxelization for a case using full mesh."""
        res = self.resolution
        coords = case_data['coords_full']
        wall_indices = case_data['wall_indices']

        cmin = coords.min(0) - 1e-6
        cmax = coords.max(0) + 1e-6
        step = (cmax - cmin) / res

        voxel_ijk = np.floor((coords - cmin) / step).astype(np.int32)
        voxel_ijk = np.clip(voxel_ijk, 0, res - 1)

        # Geometry mask
        geo_mask = np.zeros((res, res, res), dtype=np.float32)
        geo_mask[voxel_ijk[:, 0], voxel_ijk[:, 1], voxel_ijk[:, 2]] = 1.0

        # Wall mask
        wall_mask = np.zeros((res, res, res), dtype=np.float32)
        wall_vijk = voxel_ijk[wall_indices]
        wall_mask[wall_vijk[:, 0], wall_vijk[:, 1], wall_vijk[:, 2]] = 1.0

        # Count grid for averaging
        count = np.zeros((res, res, res), dtype=np.float32)
        np.add.at(count, (voxel_ijk[:, 0], voxel_ijk[:, 1], voxel_ijk[:, 2]), 1.0)
        count_mask = count > 0

        wall_count = np.zeros((res, res, res), dtype=np.float32)
        np.add.at(wall_count, (wall_vijk[:, 0], wall_vijk[:, 1], wall_vijk[:, 2]), 1.0)
        wall_count_mask = wall_count > 0

        return {
            'voxel_ijk': voxel_ijk,
            'wall_vijk': wall_vijk,
            'geo_mask': geo_mask,
            'wall_mask': wall_mask,
            'count': count,
            'count_mask': count_mask,
            'wall_count': wall_count,
            'wall_count_mask': wall_count_mask,
        }

    def _scatter_to_grid(self, case_data, vox, t_idx):
        """Scatter velocity+pressure to voxel grid."""
        res = self.resolution
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

    def _scatter_wss_to_grid(self, case_data, vox, t_idx):
        """Scatter WSS magnitude to wall voxels."""
        res = self.resolution
        wijk = vox['wall_vijk']
        wcount = vox['wall_count']
        wmask = vox['wall_count_mask']

        wss_m = case_data['wss_mag'][t_idx]
        g = np.zeros((res, res, res), dtype=np.float32)
        np.add.at(g, (wijk[:, 0], wijk[:, 1], wijk[:, 2]), wss_m)
        g[wmask] /= wcount[wmask]
        return g[np.newaxis]  # [1, R, R, R]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_idx, t = self.samples[idx]
        case = self.cases[case_idx]
        vox = case['vox']

        field_prev = self._scatter_to_grid(case, vox, t - 1)  # [4, R, R, R]
        field_curr = self._scatter_to_grid(case, vox, t)        # [4, R, R, R]
        wss_next = self._scatter_wss_to_grid(case, vox, t + 1)  # [1, R, R, R]

        x = np.concatenate([field_prev, field_curr,
                            vox['geo_mask'][np.newaxis]], axis=0)  # [9, R, R, R]

        y_wall_raw = case['wss_mag'][t + 1]  # [M]
        # Pad y_wall to max_wall so all samples have the same shape
        y_wall = np.zeros(self.max_wall, dtype=np.float32)
        M = len(y_wall_raw)
        y_wall[:M] = y_wall_raw
        n_wall = np.array(M, dtype=np.int64)  # actual wall count

        return (
            torch.from_numpy(x),
            torch.from_numpy(wss_next),
            torch.from_numpy(y_wall),
            torch.from_numpy(vox['wall_mask'][np.newaxis].copy()),
            torch.tensor(n_wall),
        )


# ────────────────────── test ──────────────────────

if __name__ == '__main__':
    h5_dir = './h5_multi_cross'
    train_ids = [5, 6, 7, 8, 9, 10, 11, 12]
    test_ids = [13, 14]

    if Path(h5_dir).exists() and (Path(h5_dir) / 'case_5.h5').exists():
        print("=== DeepONet dataset ===")
        ds = MultiCaseDeepONetDataset(h5_dir, train_ids[:2], num_wall_samples=500,
                                       input_steps=2, output_steps=1, mode='train')
        if len(ds) > 0:
            sample = ds[0]
            print(f"  x_hist={sample[0].shape}, y_hist={sample[1].shape}")
            print(f"  x_query={sample[2].shape}, y_output={sample[3].shape}")
            print(f"  geometry={sample[4].shape}, bc={sample[5].shape}")

        print("\n=== Point dataset ===")
        ds2 = MultiCasePointDataset(h5_dir, train_ids[:2], num_wall_samples=1000, mode='train')
        if len(ds2) > 0:
            feat, target = ds2[0]
            print(f"  feat={feat.shape}, target={target.shape}")
    else:
        print(f"H5 dir not ready: {h5_dir}")
