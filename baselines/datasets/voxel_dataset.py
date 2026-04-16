"""
Voxelized 3D grid dataset for FNO and U-Net baselines.
Interpolates unstructured CFD mesh data onto a regular 3D grid.

Task: Given velocity+pressure fields at t-1 and t, predict WSS at t+1.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from scipy.interpolate import NearestNDInterpolator
from pathlib import Path


class VoxelCFDDataset(Dataset):
    """
    Voxelize 3D CFD data onto a regular grid for FNO / U-Net.

    Input channels (at each of 2 input timesteps):
        - velocity_x, velocity_y, velocity_z, pressure  -> 4 channels per step
        - geometry mask (static)                         -> 1 channel
    Total input: 2*4 + 1 = 9 channels on [res, res, res] grid

    Output: WSS magnitude on wall nodes (evaluated via interpolation back to mesh).
    """

    def __init__(self,
                 h5_path: str,
                 resolution: int = 64,
                 input_steps: int = 2,
                 mode: str = 'train',
                 train_ratio: float = 0.8,
                 boundary_cut: float = 0.0,
                 skip_first: int = 1,
                 cache: bool = True):
        self.h5_path = h5_path
        self.resolution = resolution
        self.input_steps = input_steps
        self.mode = mode
        self.boundary_cut = boundary_cut
        self.skip_first = skip_first

        self._load_data()
        self._setup_voxelization()
        self._setup_time_split(train_ratio)

        if cache:
            self._build_cache()

        print(f"VoxelCFDDataset [{mode}]: res={resolution}, "
              f"time=[{self.t_start},{self.t_end}), samples={len(self)}, "
              f"wall_nodes={len(self.wall_indices)}")

    def _load_data(self):
        with h5py.File(self.h5_path, 'r') as f:
            self.coords = f['mesh/coords'][:].astype(np.float32)
            self.node_type = f['mesh/node_type'][:].astype(np.int64)
            self.wall_indices = f['mesh/wall_indices'][:].astype(np.int64)
            self.velocity = f['fields/velocity'][:].astype(np.float32)
            self.pressure = f['fields/pressure'][:].astype(np.float32)
            self.wss = f['fields/wss'][:].astype(np.float32)
            self.time_values = f['time_values'][:].astype(np.float32)

        # Apply boundary cut
        if self.boundary_cut > 0:
            self._apply_boundary_cut()

        # Compute WSS magnitude [T, M]
        self.wss_mag = np.sqrt(np.sum(self.wss ** 2, axis=-1))  # [T, M]

    def _apply_boundary_cut(self):
        inlet_mask = self.node_type == 1
        outlet_mask = self.node_type == 2
        if not np.any(inlet_mask) or not np.any(outlet_mask):
            return

        inlet_c = self.coords[inlet_mask].mean(0)
        outlet_c = self.coords[outlet_mask].mean(0)
        direction = outlet_c - inlet_c
        direction /= np.linalg.norm(direction) + 1e-8

        proj = np.dot(self.coords - inlet_c, direction)
        pmin, pmax = proj.min(), proj.max()
        cut = (pmax - pmin) * self.boundary_cut
        valid = (proj >= pmin + cut) & (proj <= pmax - cut)

        self.valid_mask = valid
        # Update wall indices
        wall_valid = valid[self.wall_indices]
        self.wall_indices = self.wall_indices[wall_valid]
        self.wss = self.wss[:, wall_valid, :]

    def _setup_voxelization(self):
        """Build the mapping between mesh nodes and voxel grid."""
        res = self.resolution

        # Compute bounding box with small padding
        if hasattr(self, 'valid_mask'):
            valid_coords = self.coords[self.valid_mask]
        else:
            valid_coords = self.coords
            self.valid_mask = np.ones(len(self.coords), dtype=bool)

        cmin = valid_coords.min(0) - 1e-6
        cmax = valid_coords.max(0) + 1e-6
        self.grid_min = cmin
        self.grid_max = cmax
        self.grid_step = (cmax - cmin) / res

        # Map each mesh node to its nearest voxel
        all_coords = self.coords
        voxel_ijk = np.floor((all_coords - cmin) / self.grid_step).astype(np.int32)
        voxel_ijk = np.clip(voxel_ijk, 0, res - 1)
        self.node_voxel_ijk = voxel_ijk  # [N, 3]

        # Build geometry mask: which voxels contain mesh nodes
        self.geo_mask = np.zeros((res, res, res), dtype=np.float32)
        valid_voxels = voxel_ijk[self.valid_mask]
        self.geo_mask[valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]] = 1.0

        # Build wall mask on grid
        self.wall_mask_grid = np.zeros((res, res, res), dtype=np.float32)
        wall_voxels = voxel_ijk[self.wall_indices]
        self.wall_mask_grid[wall_voxels[:, 0], wall_voxels[:, 1], wall_voxels[:, 2]] = 1.0

        # For each voxel, store the list of mesh node indices that fall in it
        # This is used for scatter (mesh->grid) and gather (grid->mesh)
        from collections import defaultdict
        self.voxel_to_nodes = defaultdict(list)
        valid_indices = np.where(self.valid_mask)[0]
        for idx in valid_indices:
            i, j, k = voxel_ijk[idx]
            self.voxel_to_nodes[(i, j, k)].append(idx)

        # Precompute wall node -> voxel mapping for output extraction
        self.wall_voxel_ijk = voxel_ijk[self.wall_indices]  # [M, 3]

        # Build the grid coordinate for each voxel center (for potential use)
        grid_x = np.linspace(cmin[0] + self.grid_step[0]/2, cmax[0] - self.grid_step[0]/2, res)
        grid_y = np.linspace(cmin[1] + self.grid_step[1]/2, cmax[1] - self.grid_step[1]/2, res)
        grid_z = np.linspace(cmin[2] + self.grid_step[2]/2, cmax[2] - self.grid_step[2]/2, res)
        self.grid_coords = np.stack(np.meshgrid(grid_x, grid_y, grid_z, indexing='ij'), axis=-1)  # [R,R,R,3]

    def _scatter_to_grid(self, field, t_idx):
        """Map field values from mesh nodes to voxel grid by averaging.

        Args:
            field: [T, N, C] array
            t_idx: timestep index

        Returns:
            grid: [C, res, res, res] tensor
        """
        res = self.resolution
        C = field.shape[2]
        grid = np.zeros((C, res, res, res), dtype=np.float32)
        count = np.zeros((1, res, res, res), dtype=np.float32)

        data = field[t_idx]  # [N, C]
        valid_indices = np.where(self.valid_mask)[0]
        vijk = self.node_voxel_ijk[valid_indices]  # [N_valid, 3]
        vals = data[valid_indices]  # [N_valid, C]

        # Use np.add.at for efficient scatter
        for c in range(C):
            np.add.at(grid[c], (vijk[:, 0], vijk[:, 1], vijk[:, 2]), vals[:, c])
        np.add.at(count[0], (vijk[:, 0], vijk[:, 1], vijk[:, 2]), 1.0)

        # Average where count > 0
        mask = count > 0
        for c in range(C):
            grid[c][mask[0]] /= count[0][mask[0]]

        return grid

    def _setup_time_split(self, train_ratio):
        T = len(self.time_values)
        train_end = int(T * train_ratio)

        if self.mode == 'train':
            self.t_start = max(self.skip_first, self.input_steps)
            self.t_end = train_end
        else:
            self.t_start = max(train_end, self.input_steps)
            self.t_end = T

    def __len__(self):
        return max(0, self.t_end - self.t_start - 1)

    def _build_cache(self):
        """Pre-voxelize all timesteps."""
        print("  Caching voxelized fields...")
        T = len(self.time_values)
        res = self.resolution

        # Cache velocity and pressure grids for all timesteps
        self.vel_grids = np.zeros((T, 3, res, res, res), dtype=np.float32)
        self.pres_grids = np.zeros((T, 1, res, res, res), dtype=np.float32)

        valid_indices = np.where(self.valid_mask)[0]
        vijk = self.node_voxel_ijk[valid_indices]

        # Build count grid once
        count = np.zeros((res, res, res), dtype=np.float32)
        np.add.at(count, (vijk[:, 0], vijk[:, 1], vijk[:, 2]), 1.0)
        count_mask = count > 0

        for t in range(T):
            # Velocity
            vel = self.velocity[t, valid_indices]  # [N_valid, 3]
            for c in range(3):
                grid_c = np.zeros((res, res, res), dtype=np.float32)
                np.add.at(grid_c, (vijk[:, 0], vijk[:, 1], vijk[:, 2]), vel[:, c])
                grid_c[count_mask] /= count[count_mask]
                self.vel_grids[t, c] = grid_c

            # Pressure
            pres = self.pressure[t, valid_indices, 0]  # [N_valid]
            grid_p = np.zeros((res, res, res), dtype=np.float32)
            np.add.at(grid_p, (vijk[:, 0], vijk[:, 1], vijk[:, 2]), pres)
            grid_p[count_mask] /= count[count_mask]
            self.pres_grids[t, 0] = grid_p

        # Compute WSS magnitude
        self.wss_mag = np.sqrt(np.sum(self.wss ** 2, axis=-1))  # [T, M]

        # Build target WSS grid
        self.wss_grids = np.zeros((T, 1, res, res, res), dtype=np.float32)
        wall_vijk = self.wall_voxel_ijk
        wall_count = np.zeros((res, res, res), dtype=np.float32)
        np.add.at(wall_count, (wall_vijk[:, 0], wall_vijk[:, 1], wall_vijk[:, 2]), 1.0)
        wall_count_mask = wall_count > 0

        for t in range(T):
            wss_m = self.wss_mag[t]  # [M]
            grid_w = np.zeros((res, res, res), dtype=np.float32)
            np.add.at(grid_w, (wall_vijk[:, 0], wall_vijk[:, 1], wall_vijk[:, 2]), wss_m)
            grid_w[wall_count_mask] /= wall_count[wall_count_mask]
            self.wss_grids[t, 0] = grid_w

        self._cached = True
        size_mb = (self.vel_grids.nbytes + self.pres_grids.nbytes + self.wss_grids.nbytes) / 1e6
        print(f"  Cache size: {size_mb:.1f} MB")

    def __getitem__(self, idx):
        """
        Returns:
            x: [9, res, res, res] - input (vel*3 + pres at t-1, vel*3 + pres at t, geo_mask)
            y_grid: [1, res, res, res] - WSS magnitude on grid at t+1
            y_wall: [M] - WSS magnitude at wall nodes at t+1 (ground truth)
            wall_mask: [1, res, res, res] - wall voxel mask
        """
        t = self.t_start + idx  # current time
        t_prev = t - 1
        t_next = t + 1

        if hasattr(self, '_cached') and self._cached:
            # Use cached grids
            vel_prev = self.vel_grids[t_prev]  # [3, R, R, R]
            pres_prev = self.pres_grids[t_prev]  # [1, R, R, R]
            vel_curr = self.vel_grids[t]
            pres_curr = self.pres_grids[t]
            wss_next = self.wss_grids[t_next]  # [1, R, R, R]
        else:
            vel_prev = self._scatter_to_grid(self.velocity, t_prev)
            pres_prev = self._scatter_to_grid(self.pressure, t_prev)
            vel_curr = self._scatter_to_grid(self.velocity, t)
            pres_curr = self._scatter_to_grid(self.pressure, t)
            wss_next = self.wss_grids[t_next] if hasattr(self, '_cached') else np.zeros((1, self.resolution, self.resolution, self.resolution), dtype=np.float32)

        # Input: [9, R, R, R] = vel_prev(3) + pres_prev(1) + vel_curr(3) + pres_curr(1) + geo(1)
        x = np.concatenate([vel_prev, pres_prev, vel_curr, pres_curr,
                            self.geo_mask[np.newaxis]], axis=0)

        # Wall WSS ground truth at mesh nodes
        y_wall = self.wss_mag[t_next]  # [M]

        return (
            torch.from_numpy(x),
            torch.from_numpy(wss_next),
            torch.from_numpy(y_wall),
            torch.from_numpy(self.wall_mask_grid[np.newaxis].copy()),
        )


class GraphCFDDataset(Dataset):
    """
    Graph-based dataset for MeshGraphNets.

    Uses only wall nodes and wall-wall edges:
        Full graph: ~162K nodes, ~6.4M edges  -> OOM
        Wall-only graph: ~9K nodes, ~110K edges -> fast & fits in GPU

    Node features: coords(3) + vel(3)*2 + pres(1)*2 = 11
    Edge features: displacement(3) + distance(1) = 4
    Target: WSS magnitude at t+1 for wall nodes [M]
    """

    def __init__(self,
                 h5_path: str,
                 mode: str = 'train',
                 train_ratio: float = 0.8,
                 boundary_cut: float = 0.0,
                 skip_first: int = 1,
                 wall_nhop: int = 0):
        self.h5_path = h5_path
        self.mode = mode
        self.boundary_cut = boundary_cut
        self.skip_first = skip_first

        self._load_data()
        self._extract_wall_graph()
        self._build_graph()
        self._setup_time_split(train_ratio)

        print(f"GraphCFDDataset [{mode}]: wall nodes={self.num_wall_nodes}, "
              f"edges={self.edge_index.shape[1]}, "
              f"time=[{self.t_start},{self.t_end}), samples={len(self)}")

    def _load_data(self):
        with h5py.File(self.h5_path, 'r') as f:
            self.coords_full = f['mesh/coords'][:].astype(np.float32)
            self.node_type_full = f['mesh/node_type'][:].astype(np.int64)
            self.wall_indices_full = f['mesh/wall_indices'][:].astype(np.int64)
            self.edges_raw = f['mesh/edges'][:].astype(np.int64)
            self.velocity_full = f['fields/velocity'][:].astype(np.float32)
            self.pressure_full = f['fields/pressure'][:].astype(np.float32)
            self.wss = f['fields/wss'][:].astype(np.float32)
            self.time_values = f['time_values'][:].astype(np.float32)

        self.num_nodes_full = len(self.coords_full)

        # Apply boundary cut on wall indices / wss
        if self.boundary_cut > 0:
            self._apply_boundary_cut()

        self.wss_mag = np.sqrt(np.sum(self.wss ** 2, axis=-1))  # [T, M]

    def _apply_boundary_cut(self):
        """Cut boundary regions - only filter wall nodes, keep full graph for subgraph extraction."""
        inlet_mask = self.node_type_full == 1
        outlet_mask = self.node_type_full == 2
        if not np.any(inlet_mask) or not np.any(outlet_mask):
            return

        inlet_c = self.coords_full[inlet_mask].mean(0)
        outlet_c = self.coords_full[outlet_mask].mean(0)
        direction = outlet_c - inlet_c
        direction /= np.linalg.norm(direction) + 1e-8

        proj = np.dot(self.coords_full - inlet_c, direction)
        pmin, pmax = proj.min(), proj.max()
        cut = (pmax - pmin) * self.boundary_cut

        # Only filter wall nodes that fall in the cut region
        wall_proj = proj[self.wall_indices_full]
        wall_valid = (wall_proj >= pmin + cut) & (wall_proj <= pmax - cut)
        self.wall_indices_full = self.wall_indices_full[wall_valid]
        self.wss = self.wss[:, wall_valid, :]

    def _extract_wall_graph(self):
        """Extract wall-only graph: only wall nodes and edges between them."""
        print("  Extracting wall-only graph...")

        wall_indices = self.wall_indices_full
        self.num_wall_nodes = len(wall_indices)

        # Build old->new index mapping
        old_to_new = np.full(self.num_nodes_full, -1, dtype=np.int64)
        old_to_new[wall_indices] = np.arange(self.num_wall_nodes)

        # Extract wall node data
        self.coords = self.coords_full[wall_indices]
        self.velocity = self.velocity_full[:, wall_indices, :]
        self.pressure = self.pressure_full[:, wall_indices, :]

        # Filter edges: both endpoints must be wall nodes
        wall_set = set(wall_indices.tolist())
        mask = np.array([e[0] in wall_set and e[1] in wall_set for e in self.edges_raw])
        wall_edges = self.edges_raw[mask]
        self.sub_edges = np.stack([old_to_new[wall_edges[:, 0]],
                                    old_to_new[wall_edges[:, 1]]], axis=1)

        # Wall indices in new graph are simply 0..N-1 (all nodes are wall)
        self.sub_wall_indices = np.arange(self.num_wall_nodes, dtype=np.int64)

        # Free full arrays
        del self.coords_full, self.node_type_full, self.velocity_full, self.pressure_full
        del self.edges_raw

        print(f"  Wall graph: {self.num_wall_nodes} nodes, {len(self.sub_edges)} edges")

    def _build_graph(self):
        """Build edge_index and edge_attr for the wall-only graph."""
        edges = self.sub_edges  # [E, 2]
        # Make bidirectional
        edges_bi = np.concatenate([edges, edges[:, ::-1]], axis=0)
        edges_bi = np.unique(edges_bi, axis=0)
        self.edge_index = edges_bi.T  # [2, E_bi]

        # Edge attributes: relative displacement + distance
        src, dst = self.edge_index
        disp = self.coords[dst] - self.coords[src]
        dist = np.linalg.norm(disp, axis=1, keepdims=True)
        self.edge_attr = np.concatenate([disp, dist], axis=1).astype(np.float32)

        # Normalize coordinates
        coord_min = self.coords.min(0)
        coord_range = self.coords.max(0) - coord_min + 1e-8
        self.coords_norm = ((self.coords - coord_min) / coord_range).astype(np.float32)

        # Normalize edge attributes
        disp_std = np.std(disp, axis=0) + 1e-8
        dist_std = np.std(dist) + 1e-8
        self.edge_attr_norm = np.concatenate([disp / disp_std, dist / dist_std],
                                              axis=1).astype(np.float32)

    def _setup_time_split(self, train_ratio):
        T = len(self.time_values)
        train_end = int(T * train_ratio)

        if self.mode == 'train':
            self.t_start = max(self.skip_first, 2)
            self.t_end = train_end
        else:
            self.t_start = max(train_end, 2)
            self.t_end = T

    def __len__(self):
        return max(0, self.t_end - self.t_start - 1)

    def __getitem__(self, idx):
        """Returns wall-only graph data for one timestep prediction."""
        t = self.t_start + idx
        t_prev = t - 1
        t_next = t + 1

        # Node features (all wall): coords(3) + vel_prev(3) + p_prev(1) + vel_curr(3) + p_curr(1) = 11
        node_feat = np.concatenate([
            self.coords_norm,
            self.velocity[t_prev],
            self.pressure[t_prev],
            self.velocity[t],
            self.pressure[t],
        ], axis=1).astype(np.float32)  # [M, 11]

        # WSS target (all nodes are wall)
        wss_wall = np.sqrt(np.sum(self.wss[t_next] ** 2, axis=-1)).astype(np.float32)  # [M]

        return {
            'node_feat': torch.from_numpy(node_feat),
            'edge_index': torch.from_numpy(self.edge_index.copy()).long(),
            'edge_attr': torch.from_numpy(self.edge_attr_norm.copy()),
            'target_wall': torch.from_numpy(wss_wall),
            'wall_indices': torch.from_numpy(self.sub_wall_indices.copy()).long(),
        }


class PointCFDDataset(Dataset):
    """
    Point cloud dataset for wall surface nodes.
    Used by PointNet++ and Transolver — no edges needed.

    Returns (feat, target) tuples that stack naturally with DataLoader:
        feat:   [N, 11]  — coords(3) + vel_prev(3) + p_prev(1) + vel_curr(3) + p_curr(1)
        target: [N]      — WSS magnitude at t+1
    """

    def __init__(self, h5_path, mode='train', train_ratio=0.8,
                 boundary_cut=0.0, skip_first=1):
        self.mode = mode
        self.boundary_cut = boundary_cut

        with h5py.File(h5_path, 'r') as f:
            coords = f['mesh/coords'][:].astype(np.float32)
            node_type = f['mesh/node_type'][:].astype(np.int64)
            self.wall_indices = f['mesh/wall_indices'][:].astype(np.int64)
            velocity = f['fields/velocity'][:].astype(np.float32)
            pressure = f['fields/pressure'][:].astype(np.float32)
            wss = f['fields/wss'][:].astype(np.float32)
            self.time_values = f['time_values'][:].astype(np.float32)

        # Boundary cut on wall indices
        if boundary_cut > 0:
            inlet_mask = node_type == 1
            outlet_mask = node_type == 2
            if np.any(inlet_mask) and np.any(outlet_mask):
                inlet_c = coords[inlet_mask].mean(0)
                outlet_c = coords[outlet_mask].mean(0)
                direction = outlet_c - inlet_c
                direction /= np.linalg.norm(direction) + 1e-8
                proj = np.dot(coords - inlet_c, direction)
                pmin, pmax = proj.min(), proj.max()
                cut = (pmax - pmin) * boundary_cut
                wall_proj = proj[self.wall_indices]
                wall_valid = (wall_proj >= pmin + cut) & (wall_proj <= pmax - cut)
                self.wall_indices = self.wall_indices[wall_valid]
                wss = wss[:, wall_valid, :]

        # Keep only wall node data
        wall_coords = coords[self.wall_indices]
        cmin = wall_coords.min(0)
        crange = wall_coords.max(0) - cmin + 1e-8
        self.coords_norm = ((wall_coords - cmin) / crange).astype(np.float32)

        self.velocity = velocity[:, self.wall_indices, :]
        self.pressure = pressure[:, self.wall_indices, :]
        self.wss_mag = np.sqrt(np.sum(wss ** 2, axis=-1)).astype(np.float32)
        self.num_wall = len(self.wall_indices)

        # Time split
        T = len(self.time_values)
        train_end = int(T * train_ratio)
        if mode == 'train':
            self.t_start = max(skip_first, 2)
            self.t_end = train_end
        else:
            self.t_start = max(train_end, 2)
            self.t_end = T

        print(f"PointCFDDataset [{mode}]: wall_nodes={self.num_wall}, "
              f"time=[{self.t_start},{self.t_end}), samples={len(self)}")

    def __len__(self):
        return max(0, self.t_end - self.t_start - 1)

    def __getitem__(self, idx):
        t = self.t_start + idx
        feat = np.concatenate([
            self.coords_norm,
            self.velocity[t - 1], self.pressure[t - 1],
            self.velocity[t],     self.pressure[t],
        ], axis=1).astype(np.float32)                       # [N, 11]
        target = self.wss_mag[t + 1]                         # [N]
        return torch.from_numpy(feat), torch.from_numpy(target)


if __name__ == '__main__':
    h5 = './data/case.h5'

    print("=== Testing VoxelCFDDataset ===")
    ds = VoxelCFDDataset(h5, resolution=48, boundary_cut=0.1, cache=True)
    x, y_grid, y_wall, wmask = ds[0]
    print(f"  x: {x.shape}, y_grid: {y_grid.shape}, y_wall: {y_wall.shape}, wmask: {wmask.shape}")
    print(f"  x range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  y_wall range: [{y_wall.min():.6f}, {y_wall.max():.6f}]")

    print("\n=== Testing GraphCFDDataset ===")
    ds_g = GraphCFDDataset(h5, boundary_cut=0.1)
    sample = ds_g[0]
    print(f"  node_feat: {sample['node_feat'].shape}")
    print(f"  edge_index: {sample['edge_index'].shape}")
    print(f"  edge_attr: {sample['edge_attr'].shape}")
    print(f"  target_wall: {sample['target_wall'].shape}")
