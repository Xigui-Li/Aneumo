"""
完整网格推理脚本

功能:
1. 在所有壁面点上进行预测（不是采样）
2. 切除区域使用真值填充
3. 导出完整 VTK 文件用于 ParaView

用法:
    python inference_full_mesh.py \
        --checkpoint checkpoint/v2_with_swin/best_model.pt \
        --time_idx 85 \
        --output_dir results/full_mesh
"""
import argparse
import os
import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm

from transient.models import TemporalDeepONetV2
from transient.dataset import parse_output_vars


def add_argument():
    parser = argparse.ArgumentParser(description="Full Mesh Inference")

    # 必需参数
    parser.add_argument("--checkpoint", required=True, type=str, help="模型 checkpoint 路径")

    # 数据参数
    parser.add_argument("--h5_path", default="case_201.h5", type=str)
    parser.add_argument("--nii_path", default="201.nii.gz", type=str)
    parser.add_argument("--vtp_dir", default=None, type=str,
                        help="原始 VTP 文件目录，用于获取正确的网格结构")

    # 推理参数
    parser.add_argument("--time_idx", default=None, type=int,
                        help="要预测的时间步索引，默认预测所有测试时间步")
    parser.add_argument("--batch_points", default=5000, type=int,
                        help="每批处理的点数（防止显存溢出）")
    parser.add_argument("--validate_mode", action="store_true",
                        help="在验证集上测试（而非测试集），用于对比训练时的验证 L2")

    # 输出参数
    parser.add_argument("--output_dir", default="results/full_mesh", type=str)
    parser.add_argument("--smooth", default=0, type=int, help="平滑迭代次数")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def load_geometry(nii_path):
    """加载 NIfTI 几何影像"""
    try:
        import nibabel as nib
        img = nib.load(nii_path)
        geometry = img.get_fdata().astype(np.float32)
    except ImportError:
        import SimpleITK as sitk
        img = sitk.ReadImage(str(nii_path))
        geometry = sitk.GetArrayFromImage(img).astype(np.float32)

    if len(geometry.shape) == 4:
        geometry = geometry[..., 0]

    geo_min, geo_max = geometry.min(), geometry.max()
    if geo_max > geo_min:
        geometry = (geometry - geo_min) / (geo_max - geo_min)

    return geometry[np.newaxis, ...]  # [1, D, H, W]


def compute_boundary_mask(wall_coords, all_coords, all_node_type, boundary_cut):
    """
    计算边界切除掩码

    Args:
        wall_coords: [M, 3] - 壁面点坐标
        all_coords: [N, 3] - 所有点坐标（用于找 inlet/outlet）
        all_node_type: [N] - 所有点的节点类型
        boundary_cut: float - 切除比例

    Returns:
        valid_mask: bool array [M], True = 有效区域（需要预测）
        cut_mask: bool array [M], True = 切除区域（用真值填充）
    """
    M = len(wall_coords)

    if boundary_cut <= 0:
        return np.ones(M, dtype=bool), np.zeros(M, dtype=bool)

    # 使用全部节点找 inlet/outlet 中心
    inlet_mask = all_node_type == 1
    outlet_mask = all_node_type == 2

    if not np.any(inlet_mask) or not np.any(outlet_mask):
        print("  Warning: Could not find inlet/outlet nodes, skipping boundary cut")
        return np.ones(M, dtype=bool), np.zeros(M, dtype=bool)

    inlet_center = all_coords[inlet_mask].mean(axis=0)
    outlet_center = all_coords[outlet_mask].mean(axis=0)

    # 进出口方向
    direction = outlet_center - inlet_center
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # 将所有节点投影到进出口方向上，计算投影范围
    all_relative = all_coords - inlet_center
    all_projections = np.dot(all_relative, direction)
    proj_min, proj_max = all_projections.min(), all_projections.max()
    proj_range = proj_max - proj_min

    # 切除边界
    cut_dist = proj_range * boundary_cut
    valid_min = proj_min + cut_dist
    valid_max = proj_max - cut_dist

    # 对壁面点计算投影
    wall_relative = wall_coords - inlet_center
    wall_projections = np.dot(wall_relative, direction)

    valid_mask = (wall_projections >= valid_min) & (wall_projections <= valid_max)
    cut_mask = ~valid_mask

    return valid_mask, cut_mask


@torch.no_grad()
def predict_full_mesh(model, coords, time_val, geometry, x_hist, y_hist,
                      device, batch_points=5000, coord_norm_params=None):
    """
    在所有点上进行预测

    Args:
        coords: [N, 3] - 所有点坐标
        time_val: float - 归一化时间值
        geometry: [1, D, H, W] - 几何图像
        x_hist: [1, T_in * N_hist, 4] - 历史帧坐标
        y_hist: [1, T_in * N_hist, num_vars] - 历史帧场值
        batch_points: 每批处理的点数
    """
    model.eval()
    N = len(coords)

    # 归一化坐标
    if coord_norm_params:
        coord_min, coord_range = coord_norm_params
        coords_norm = (coords - coord_min) / coord_range
    else:
        coords_norm = coords

    # 添加时间维度
    t_expanded = np.full((N, 1), time_val, dtype=np.float32)
    x_query_full = np.concatenate([coords_norm, t_expanded], axis=1)  # [N, 4]

    # 分批预测
    predictions = []
    num_batches = (N + batch_points - 1) // batch_points

    geo_tensor = torch.tensor(geometry, dtype=torch.float32).unsqueeze(0).to(device)
    x_hist_tensor = x_hist.to(device)
    y_hist_tensor = y_hist.to(device)

    for i in tqdm(range(num_batches), desc="Predicting", leave=False):
        start_idx = i * batch_points
        end_idx = min((i + 1) * batch_points, N)

        x_batch = torch.tensor(x_query_full[start_idx:end_idx], dtype=torch.float32)
        x_batch = x_batch.unsqueeze(0).to(device)  # [1, batch, 4]

        y_pred = model(x_hist_tensor, y_hist_tensor, x_batch, geo_tensor)
        predictions.append(y_pred[0].cpu().numpy())

    return np.concatenate(predictions, axis=0)  # [N, num_outputs]


def load_original_wall_vtp(vtp_dir, time_value):
    """
    加载原始的 wall.vtp 文件

    Args:
        vtp_dir: VTP 文件所在目录（如 case_200/）
        time_value: 时间值（如 4.85）

    Returns:
        mesh: PyVista PolyData 对象
        coords: 点坐标
    """
    import pyvista as pv

    # 构建文件路径，处理浮点数精度问题
    time_str = f"{time_value:.2f}"
    vtp_path = Path(vtp_dir) / time_str / f"{time_str}_wall.vtp"

    # 如果精确路径不存在，尝试查找最接近的时间步
    if not vtp_path.exists():
        # 列出所有时间步目录
        vtp_dir_path = Path(vtp_dir)
        time_dirs = [d.name for d in vtp_dir_path.iterdir() if d.is_dir()]

        # 找最接近的
        best_match = None
        best_diff = float('inf')
        for td in time_dirs:
            try:
                t = float(td)
                diff = abs(t - time_value)
                if diff < best_diff:
                    best_diff = diff
                    best_match = td
            except ValueError:
                continue

        if best_match and best_diff < 0.005:  # 容差 0.005
            time_str = best_match
            vtp_path = vtp_dir_path / time_str / f"{time_str}_wall.vtp"

    if not vtp_path.exists():
        raise FileNotFoundError(f"Wall VTP not found: {vtp_path}")

    mesh = pv.read(str(vtp_path))
    coords = np.array(mesh.points, dtype=np.float32)

    print(f"  Loaded original wall mesh: {vtp_path}")
    print(f"    Points: {mesh.n_points}, Cells: {mesh.n_cells}")

    return mesh, coords


def map_predictions_to_mesh(mesh, h5_wall_coords, wss_pred, wss_true, wss_error,
                            valid_mask=None, node_type=None):
    """
    将预测值映射到原始网格上

    Args:
        mesh: 原始 VTP 网格（会被修改）
        h5_wall_coords: h5 中的壁面点坐标 [M, 3]
        wss_pred: 预测值 [M]
        wss_true: 真值 [M]（来自 H5，可能顺序不对）
        wss_error: 误差 [M]

    Returns:
        mesh: 添加了场值的网格（保留原始 cell 结构）
    """
    from scipy.spatial import cKDTree

    mesh_coords = np.array(mesh.points, dtype=np.float32)

    # 建立 KD 树进行最近邻匹配
    # 用 h5 的坐标建树，查询 mesh 的每个点对应 h5 中的哪个点
    tree = cKDTree(h5_wall_coords)
    distances, indices = tree.query(mesh_coords, k=1)

    # 检查匹配质量
    max_dist = distances.max()
    mean_dist = distances.mean()
    print(f"    Coordinate matching: max_dist={max_dist:.6f}, mean_dist={mean_dist:.6f}")

    if max_dist > 1e-3:
        print(f"    Warning: Large matching distance detected! Check coordinate alignment.")

    # 映射预测值到原始网格
    mesh.point_data['WSS_predicted'] = wss_pred[indices]

    # 直接使用 VTP 原始的 wallShearStress 作为 ground truth（如果存在）
    if 'wallShearStress' in mesh.point_data:
        original_wss = mesh.point_data['wallShearStress']
        if len(original_wss.shape) > 1:
            # 向量，计算 magnitude
            wss_gt = np.sqrt(np.sum(original_wss ** 2, axis=-1))
        else:
            wss_gt = original_wss
        mesh.point_data['WSS_ground_truth'] = wss_gt
        # 重新计算误差
        mapped_pred = wss_pred[indices]
        mesh.point_data['WSS_error'] = np.abs(mapped_pred - wss_gt)
        mesh.point_data['WSS_relative_error'] = np.abs(mapped_pred - wss_gt) / (np.abs(wss_gt) + 1e-8)
    else:
        # 没有原始数据，使用 H5 的（可能顺序不对）
        mesh.point_data['WSS_ground_truth'] = wss_true[indices]
        mesh.point_data['WSS_error'] = wss_error[indices]
        mesh.point_data['WSS_relative_error'] = wss_error[indices] / (np.abs(wss_true[indices]) + 1e-8)

    if valid_mask is not None:
        mesh.point_data['is_predicted'] = valid_mask[indices].astype(np.int32)

    if node_type is not None:
        mesh.point_data['node_type'] = node_type[indices]

    return mesh


def smooth_point_data_on_mesh(mesh, values, iterations=5):
    """基于网格拓扑平滑点数据"""
    from collections import defaultdict

    # 构建邻接表
    neighbors = defaultdict(set)
    for i in range(mesh.n_cells):
        cell = mesh.get_cell(i)
        pts = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        for p in pts:
            neighbors[p].update(pts)

    # 移除自身
    for p in neighbors:
        neighbors[p].discard(p)

    # 迭代平滑
    smoothed = values.copy()
    for _ in range(iterations):
        new_vals = smoothed.copy()
        for i in range(len(smoothed)):
            if neighbors[i]:
                neighbor_vals = smoothed[list(neighbors[i])]
                new_vals[i] = 0.5 * smoothed[i] + 0.5 * neighbor_vals.mean()
        smoothed = new_vals

    return smoothed


def export_vtk_simple(vtp_dir, time_value, wss_pred, output_path, smooth_iterations=0):
    """
    VTK 导出

    H5 和 VTP 顺序完全一致，直接按索引赋值
    """
    import pyvista as pv

    # 加载原始 VTP
    mesh, _ = load_original_wall_vtp(vtp_dir, time_value)

    # 直接赋值
    mesh.point_data['WSS_predicted'] = wss_pred.copy()

    # 平滑（可选）
    if smooth_iterations > 0:
        wss_smoothed = smooth_point_data_on_mesh(mesh, wss_pred, iterations=smooth_iterations)
        mesh.point_data['WSS_predicted_smooth'] = wss_smoothed

    # 真值
    if 'wallShearStress' in mesh.point_data:
        original_wss = mesh.point_data['wallShearStress']
        if len(original_wss.shape) > 1:
            wss_gt = np.sqrt(np.sum(original_wss ** 2, axis=-1))
        else:
            wss_gt = np.array(original_wss)
        mesh.point_data['WSS_ground_truth'] = wss_gt
        mesh.point_data['WSS_error'] = np.abs(wss_pred - wss_gt)

    mesh.save(str(output_path))
    print(f"  Saved: {output_path}")


def export_vtk_with_direct_prediction(model, vtp_dir, time_value, t_norm, geometry,
                                       x_hist, y_hist, device, coord_norm_params,
                                       batch_points, output_path):
    """
    直接用 VTP 坐标进行预测并导出（已弃用，使用 export_vtk_simple）
    """
    import pyvista as pv
    from scipy.spatial import cKDTree

    # 加载原始 VTP
    mesh, vtp_coords = load_original_wall_vtp(vtp_dir, time_value)

    # 用 H5 壁面坐标建立 KD 树，找到 VTP 每个点对应的 H5 索引
    # 这样可以直接使用 H5 上的预测结果

    # 从 VTP 获取真值
    if 'wallShearStress' in mesh.point_data:
        original_wss = mesh.point_data['wallShearStress']
        if len(original_wss.shape) > 1:
            wss_true_vtp = np.sqrt(np.sum(original_wss ** 2, axis=-1))
        else:
            wss_true_vtp = np.array(original_wss)
    else:
        wss_true_vtp = np.zeros(len(vtp_coords))

    # 计算误差
    wss_error_vtp = np.abs(wss_pred_vtp - wss_true_vtp)

    # 存入 mesh
    mesh.point_data['WSS_predicted'] = wss_pred_vtp
    mesh.point_data['WSS_ground_truth'] = wss_true_vtp
    mesh.point_data['WSS_error'] = wss_error_vtp
    mesh.point_data['WSS_relative_error'] = wss_error_vtp / (np.abs(wss_true_vtp) + 1e-8)

    # 保存
    mesh.save(str(output_path))
    print(f"  Saved: {output_path} (points: {mesh.n_points}, cells: {mesh.n_cells})")

    return wss_pred_vtp, wss_true_vtp


def export_vtk(coords, wss_pred, wss_true, wss_error, output_path,
               valid_mask=None, node_type=None, vtp_dir=None, time_value=None):
    """
    导出 VTK 文件（旧方法，有映射问题时使用 export_vtk_with_direct_prediction）
    """
    try:
        import pyvista as pv

        if vtp_dir is not None and time_value is not None:
            try:
                mesh, _ = load_original_wall_vtp(vtp_dir, time_value)
                mesh = map_predictions_to_mesh(
                    mesh, coords, wss_pred, wss_true, wss_error,
                    valid_mask=valid_mask, node_type=node_type
                )
                mesh.save(str(output_path))
                print(f"  Saved: {output_path} (points: {mesh.n_points}, cells: {mesh.n_cells})")
                return
            except Exception as e:
                print(f"  Failed to load original VTP: {e}")
                print(f"  Falling back to point cloud...")

        print(f"  Warning: No vtp_dir provided, output will be point cloud (no mesh structure)")
        mesh = pv.PolyData(coords)
        mesh.point_data['WSS_predicted'] = wss_pred
        mesh.point_data['WSS_ground_truth'] = wss_true
        mesh.point_data['WSS_error'] = wss_error
        mesh.point_data['WSS_relative_error'] = wss_error / (np.abs(wss_true) + 1e-8)
        if valid_mask is not None:
            mesh.point_data['is_predicted'] = valid_mask.astype(np.int32)
        if node_type is not None:
            mesh.point_data['node_type'] = node_type
        mesh.save(str(output_path))
        print(f"  Saved: {output_path} (points: {mesh.n_points}, cells: {mesh.n_cells})")

    except ImportError:
        print("PyVista not installed. Saving as NPZ instead.")
        np.savez(
            str(output_path).replace('.vtp', '.npz'),
            coords=coords, wss_pred=wss_pred, wss_true=wss_true,
            wss_error=wss_error, valid_mask=valid_mask
        )


def main():
    args = add_argument()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 加载 checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get('args', {})

    # ========== 诊断信息：打印训练参数 ==========
    print("\n" + "=" * 50)
    print("Training Parameters (from checkpoint)")
    print("=" * 50)
    print(f"  num_samples (spatial): {saved_args.get('num_samples', 'N/A')}")
    print(f"  num_wall_samples: {saved_args.get('num_wall_samples', 'N/A')}")
    print(f"  input_steps: {saved_args.get('input_steps', 'N/A')}")
    print(f"  output_steps: {saved_args.get('output_steps', 'N/A')}")
    print(f"  train_ratio: {saved_args.get('train_ratio', 'N/A')}")
    print(f"  boundary_cut: {saved_args.get('boundary_cut', 'N/A')}")
    print(f"  output_vars: {saved_args.get('output_vars', 'N/A')}")
    print("=" * 50 + "\n")

    output_vars = parse_output_vars(saved_args.get('output_vars', 'wss'))
    num_outputs = len(output_vars)
    input_steps = saved_args.get('input_steps', 2)
    output_steps = saved_args.get('output_steps', 1)
    boundary_cut = saved_args.get('boundary_cut', 0.0)
    train_ratio = saved_args.get('train_ratio', 0.8)

    print(f"Output variables: {output_vars}")
    print(f"Input steps: {input_steps}, Output steps: {output_steps}")
    print(f"Boundary cut: {boundary_cut}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据（只加载需要的部分）
    print(f"Loading data from {args.h5_path}...")
    h5_file = h5py.File(args.h5_path, 'r')

    print("  Loading mesh...")
    coords = h5_file['mesh/coords'][:].astype(np.float32)
    node_type = h5_file['mesh/node_type'][:].astype(np.int64)
    wall_indices = h5_file['mesh/wall_indices'][:].astype(np.int64)
    time_values = h5_file['time_values'][:].astype(np.float32)

    print("  Loading WSS data...")
    wss_shape = h5_file['fields/wss'].shape
    print(f"    WSS shape: {wss_shape}")
    wss_data = h5_file['fields/wss'][:].astype(np.float32)  # [T, M, 3]

    h5_file.close()
    print("  Done loading.")

    # ========== 应用 boundary_cut（与训练完全一致） ==========
    num_time_steps = len(time_values)
    print(f"Total points: {len(coords)}, Original wall points: {len(wall_indices)}")
    print(f"Time steps: {num_time_steps}")

    # 先保存原始数据（用于 VTK 输出时的映射）
    original_wall_indices = wall_indices.copy()
    original_wss_data = wss_data.copy()
    original_wall_idx_to_local = {idx: i for i, idx in enumerate(original_wall_indices)}

    if boundary_cut > 0:
        # 与训练 dataset_temporal.py 完全一致的处理
        inlet_mask = node_type == 1
        outlet_mask = node_type == 2

        if np.any(inlet_mask) and np.any(outlet_mask):
            inlet_center = coords[inlet_mask].mean(axis=0)
            outlet_center = coords[outlet_mask].mean(axis=0)

            direction = outlet_center - inlet_center
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            relative_coords = coords - inlet_center
            projections = np.dot(relative_coords, direction)

            proj_min, proj_max = projections.min(), projections.max()
            proj_range = proj_max - proj_min

            cut_dist = proj_range * boundary_cut
            valid_min = proj_min + cut_dist
            valid_max = proj_max - cut_dist

            # 创建有效节点掩码
            all_valid_mask = (projections >= valid_min) & (projections <= valid_max)
            valid_node_indices = np.where(all_valid_mask)[0]

            # 过滤壁面索引（与训练一致）
            valid_wall_mask = np.isin(wall_indices, valid_node_indices)
            wall_indices = wall_indices[valid_wall_mask]

            # 过滤 WSS 数据（与训练一致）
            wss_data = wss_data[:, valid_wall_mask, :]

            # 重建映射（与训练一致）
            wall_idx_to_local = {idx: i for i, idx in enumerate(wall_indices)}

            print(f"Boundary cut applied: {len(wall_indices)}/{len(original_wall_indices)} wall points kept")
        else:
            print("Warning: Could not find inlet/outlet, skipping boundary cut")
            valid_node_indices = np.arange(len(coords))
            wall_idx_to_local = {idx: i for i, idx in enumerate(wall_indices)}
    else:
        valid_node_indices = np.arange(len(coords))
        wall_idx_to_local = {idx: i for i, idx in enumerate(wall_indices)}

    # 壁面点坐标（现在只有有效区域的点）
    wall_coords = coords[wall_indices]  # [M_valid, 3]
    num_wall_points = len(wall_indices)

    # 计算归一化参数（与训练一致：基于有效节点）
    valid_coords = coords[valid_node_indices]
    coord_min = valid_coords.min(axis=0)
    coord_max = valid_coords.max(axis=0)
    coord_range = coord_max - coord_min + 1e-8

    time_min = time_values.min()
    time_max = time_values.max()
    time_range = time_max - time_min + 1e-8

    # 简化的掩码（现在所有保留的点都是有效的）
    valid_mask = np.ones(num_wall_points, dtype=bool)
    cut_mask = np.zeros(num_wall_points, dtype=bool)
    wall_node_types = node_type[wall_indices]

    print(f"Valid wall points for prediction: {num_wall_points}")

    # 加载几何
    geometry = load_geometry(args.nii_path)

    # 创建模型
    model = TemporalDeepONetV2(
        num_input_vars=num_outputs,
        num_output_vars=num_outputs,
        history_embed_dim=saved_args.get('history_embed_dim', 128),
        history_encoder_type=saved_args.get('history_encoder', 'light'),
        history_num_layers=saved_args.get('history_num_layers', 3),
        swin_embed_dim=saved_args.get('swin_embed_dim', 24),
        use_geometry=saved_args.get('use_geometry', True),
        trunk_hidden_dim=saved_args.get('trunk_hidden_dim', 128),
        trunk_num_layers=saved_args.get('trunk_num_layers', 4),
        use_cross_attention=saved_args.get('use_cross_attention', True),
        branch_dim=saved_args.get('branch_dim', 256)
    ).to(device)

    model.load_state_dict(checkpoint['model'])
    print(f"Model loaded from epoch {checkpoint['epoch']}")

    # 确定要预测的时间步
    test_start = int(num_time_steps * train_ratio)
    if args.time_idx is not None:
        time_indices = [args.time_idx]
    elif args.validate_mode:
        # 验证模式：使用训练期间的验证数据（最后 20% 的训练时间范围）
        val_start = int(test_start * 0.8)  # 训练数据的后 20%
        val_end = test_start
        time_indices = list(range(val_start + input_steps, val_end))
        print("*** VALIDATION MODE: Testing on validation time steps (same as training validation) ***")
    else:
        # 预测所有测试时间步
        time_indices = list(range(test_start + input_steps, num_time_steps))

    print(f"Will predict {len(time_indices)} time steps: {time_indices[0]} to {time_indices[-1]}")

    # 获取训练时的采样点数（用于对比计算）
    train_num_wall_samples = saved_args.get('num_wall_samples', None)
    train_num_samples = saved_args.get('num_samples', 1000)
    # 训练时如果设置了 num_wall_samples 就用它，否则用 num_samples
    eval_sample_size = train_num_wall_samples if train_num_wall_samples else train_num_samples
    print(f"Will compute sampled L2 with {eval_sample_size} points (same as training)")

    # 收集所有时间步的指标
    all_mse_errors = []
    all_l2_errors = []
    all_mae_errors = []
    all_mnae_errors = []

    # 采样方式的指标（与训练对齐）
    all_sampled_l2_errors = []

    for t_idx in tqdm(time_indices, desc="Processing time steps"):
        # 构建历史帧数据
        hist_indices = list(range(t_idx - input_steps, t_idx))

        # 采样历史点（与训练一致：从有效壁面点采样）
        # 训练时 WSS-only 模式使用 num_wall_samples 或 num_spatial_samples
        train_sample_size = train_num_wall_samples if train_num_wall_samples else train_num_samples
        num_hist_samples = min(train_sample_size, num_wall_points)
        hist_sample_idx = np.random.choice(num_wall_points, num_hist_samples, replace=False)
        hist_wall_indices = wall_indices[hist_sample_idx]
        hist_coords = coords[hist_wall_indices]
        hist_coords_norm = (hist_coords - coord_min) / coord_range

        x_hist_list = []
        y_hist_list = []

        for h_idx in hist_indices:
            t_norm = (time_values[h_idx] - time_min) / time_range
            t_expanded = np.full((num_hist_samples, 1), t_norm, dtype=np.float32)
            x_t = np.concatenate([hist_coords_norm, t_expanded], axis=1)
            x_hist_list.append(x_t)

            # WSS magnitude（使用过滤后的 wss_data 和新的映射）
            wss_mag = np.zeros((num_hist_samples, 1), dtype=np.float32)
            for i, node_idx in enumerate(hist_wall_indices):
                local_idx = wall_idx_to_local[node_idx]
                wss_vec = wss_data[h_idx, local_idx]
                wss_mag[i, 0] = np.sqrt(np.sum(wss_vec ** 2))
            y_hist_list.append(wss_mag)

        x_hist = torch.tensor(np.concatenate(x_hist_list, axis=0), dtype=torch.float32).unsqueeze(0)
        y_hist = torch.tensor(np.concatenate(y_hist_list, axis=0), dtype=torch.float32).unsqueeze(0)

        # 归一化时间
        t_norm = (time_values[t_idx] - time_min) / time_range

        # 在所有壁面点上预测
        wss_pred = predict_full_mesh(
            model, wall_coords, t_norm, geometry,
            x_hist, y_hist, device,
            batch_points=args.batch_points,
            coord_norm_params=(coord_min, coord_range)
        )
        wss_pred = wss_pred[:, 0]  # [M]

        # 真值 WSS magnitude（使用过滤后的 wss_data）
        wss_true = np.sqrt(np.sum(wss_data[t_idx] ** 2, axis=-1))  # [M_valid]

        # 现在所有点都是有效的，不需要填充切除区域
        wss_final = wss_pred.copy()

        # 计算误差（只在有效区域）
        error = np.abs(wss_final - wss_true)
        error_valid = error[valid_mask]
        wss_true_valid = wss_true[valid_mask]

        mse_error = np.mean(error_valid ** 2)
        l2_error = np.sqrt(np.sum(error_valid ** 2) / (np.sum(wss_true_valid ** 2) + 1e-8))
        mae_error = np.mean(error_valid)
        wss_range = wss_true_valid.max() - wss_true_valid.min() + 1e-8
        mnae_error = np.mean(error_valid / wss_range)

        # ========== 采样方式计算L2（与训练对齐） ==========
        # 从有效区域随机采样，模拟训练时的采样
        # 现在所有点都是有效的（已过滤切除区域）
        sample_size = min(eval_sample_size, num_wall_points)
        sample_idx = np.random.choice(num_wall_points, sample_size, replace=False)

        sampled_pred = wss_final[sample_idx]
        sampled_true = wss_true[sample_idx]
        sampled_error = np.abs(sampled_pred - sampled_true)
        sampled_l2 = np.sqrt(np.sum(sampled_error ** 2) / (np.sum(sampled_true ** 2) + 1e-8))

        all_mse_errors.append(mse_error)
        all_l2_errors.append(l2_error)
        all_mae_errors.append(mae_error)
        all_mnae_errors.append(mnae_error)
        all_sampled_l2_errors.append(sampled_l2)

        print(f"  Time {t_idx}: Full L2={l2_error:.4f}, Sampled L2={sampled_l2:.4f}, MSE={mse_error:.6f}, MAE={mae_error:.6f}")

        # 导出 VTK
        vtk_path = output_dir / f"wss_t{t_idx:04d}.vtp"

        if args.vtp_dir is not None:
            # 构建完整的 H5 壁面预测（包括切除区域用真值填充）
            original_wss_true = np.sqrt(np.sum(original_wss_data[t_idx] ** 2, axis=-1))
            full_wss_pred = original_wss_true.copy()  # 初始化为真值

            # 填入有效区域的预测值
            for i, node_idx in enumerate(wall_indices):
                orig_local_idx = original_wall_idx_to_local[node_idx]
                full_wss_pred[orig_local_idx] = wss_final[i]

            try:
                export_vtk_simple(
                    args.vtp_dir, time_values[t_idx],
                    full_wss_pred,
                    vtk_path,
                    smooth_iterations=args.smooth
                )
            except Exception as e:
                print(f"  VTP export failed: {e}")

    # 汇总统计
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Mean MSE: {np.mean(all_mse_errors):.6f} ± {np.std(all_mse_errors):.6f}")
    print(f"Mean L2 Error (Full Mesh): {np.mean(all_l2_errors):.4f} ± {np.std(all_l2_errors):.4f}")
    print(f"Mean L2 Error (Sampled, like training): {np.mean(all_sampled_l2_errors):.4f} ± {np.std(all_sampled_l2_errors):.4f}")
    print(f"Mean MAE: {np.mean(all_mae_errors):.6f} ± {np.std(all_mae_errors):.6f}")
    print(f"Mean MNAE: {np.mean(all_mnae_errors):.4f} ± {np.std(all_mnae_errors):.4f}")

    # 保存汇总
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Full Mesh Inference Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Time steps: {time_indices[0]} to {time_indices[-1]}\n")
        f.write(f"Wall points: {num_wall_points}\n")
        f.write(f"Valid points: {valid_mask.sum()}\n")
        f.write(f"Cut points (filled with GT): {cut_mask.sum()}\n")
        f.write("\n")
        f.write(f"Mean MSE: {np.mean(all_mse_errors):.6f} ± {np.std(all_mse_errors):.6f}\n")
        f.write(f"Mean L2 Error (Full Mesh): {np.mean(all_l2_errors):.4f} ± {np.std(all_l2_errors):.4f}\n")
        f.write(f"Mean L2 Error (Sampled): {np.mean(all_sampled_l2_errors):.4f} ± {np.std(all_sampled_l2_errors):.4f}\n")
        f.write(f"Mean MAE: {np.mean(all_mae_errors):.6f} ± {np.std(all_mae_errors):.6f}\n")
        f.write(f"Mean MNAE: {np.mean(all_mnae_errors):.4f} ± {np.std(all_mnae_errors):.4f}\n")
        f.write("\nPer-timestep errors:\n")
        for t_idx, mse, l2, l2_sampled, mae, mnae in zip(time_indices, all_mse_errors, all_l2_errors, all_sampled_l2_errors, all_mae_errors, all_mnae_errors):
            f.write(f"  t={t_idx}: MSE={mse:.6f}, L2_full={l2:.4f}, L2_sampled={l2_sampled:.4f}, MAE={mae:.6f}, MNAE={mnae:.4f}\n")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
