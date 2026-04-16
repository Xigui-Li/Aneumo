"""
时序 CFD 数据集加载器
支持从 H5 文件加载带时间维度的血流动力学数据
支持时序预测：用前 N 帧预测后 M 帧

新功能:
  1. 自适应输出: 可选择只预测 p / wss / velocity 等
  2. 进出口切除: 移除边界附近的节点减少误差
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from typing import List, Tuple, Optional, Dict
from pathlib import Path


# ============================================================
# 输出变量配置
# ============================================================
OUTPUT_VAR_MAP = {
    'p': ['p'],
    'pressure': ['p'],
    'u': ['u'],
    'v': ['v'],
    'w': ['w'],
    'velocity': ['u', 'v', 'w'],
    'vel': ['u', 'v', 'w'],
    'wss': ['wss'],
    'all': ['p', 'u', 'v', 'w', 'wss'],
    'flow': ['p', 'u', 'v', 'w'],  # 只有流场，不含 WSS
}

# 变量在原始数据中的索引
VAR_INDEX = {
    'p': 0,
    'u': 1,
    'v': 2,
    'w': 3,
    'wss': 4
}


def parse_output_vars(output_vars_str: str) -> List[str]:
    """
    解析输出变量字符串

    Args:
        output_vars_str: 逗号分隔的变量名，如 'p', 'p,wss', 'velocity', 'all'

    Returns:
        list: 变量列表，如 ['p'], ['p', 'wss'], ['u', 'v', 'w']

    Examples:
        >>> parse_output_vars('p')
        ['p']
        >>> parse_output_vars('wss')
        ['wss']
        >>> parse_output_vars('velocity')
        ['u', 'v', 'w']
        >>> parse_output_vars('p,wss')
        ['p', 'wss']
        >>> parse_output_vars('all')
        ['p', 'u', 'v', 'w', 'wss']
    """
    if output_vars_str is None:
        return ['p', 'u', 'v', 'w', 'wss']  # 默认全部

    vars_list = []
    for var in output_vars_str.split(','):
        var = var.strip().lower()
        if var in OUTPUT_VAR_MAP:
            vars_list.extend(OUTPUT_VAR_MAP[var])
        else:
            raise ValueError(f"Unknown output variable: {var}. Valid: {list(OUTPUT_VAR_MAP.keys())}")

    # 去重并保持顺序
    seen = set()
    result = []
    for v in vars_list:
        if v not in seen:
            seen.add(v)
            result.append(v)

    return result


def get_var_indices(output_vars: List[str]) -> List[int]:
    """获取输出变量在完整输出中的索引"""
    return [VAR_INDEX[var] for var in output_vars]


class TemporalCFDDataset(Dataset):
    """
    时序 CFD 数据集 - 时序预测模式
    用前 input_steps 帧预测后 output_steps 帧

    H5 文件结构 (来自 dataprocess.py):
        mesh/
            coords: [N, 3] - 节点坐标
            node_type: [N] - 节点类型 (0=内部, 1=inlet, 2=outlet, 3=wall)
            wall_indices: [M] - 壁面节点索引
        fields/
            velocity: [T, N, 3] - 时序速度场
            pressure: [T, N, 1] - 时序压力场
            wss: [T, M, 3] - 时序壁面剪切应力
        time_values: [T] - 时间值

    NIfTI 文件: 3D 几何影像
    """

    def __init__(self,
                 h5_path: str,
                 nii_path: str,
                 input_steps: int = 10,
                 output_steps: int = 5,
                 stride: int = 1,
                 num_spatial_samples: int = 2000,
                 num_wall_samples: int = None,  # WSS 专用壁面采样数，None 表示使用共享采样
                 normalize_coords: bool = True,
                 bc_dim: int = 7,
                 mode: str = 'train',
                 train_ratio: float = 0.8,
                 # 自定义时间范围参数
                 skip_first: int = 0,
                 train_frames: int = None,
                 test_frames: int = None,
                 # 自适应输出
                 output_vars: str = 'all',
                 # 进出口切除
                 boundary_cut: float = 0.0,
                 # 内存缓存
                 cache_mode: str = 'ram'):
        """
        Args:
            h5_path: H5 数据文件路径
            nii_path: NIfTI 几何影像文件路径
            input_steps: 输入的时间步数（用于预测的历史帧数）
            output_steps: 输出的时间步数（需要预测的未来帧数）
            stride: 滑动窗口步长
            num_spatial_samples: 每个时间步采样的空间点数（用于流场变量 p, u, v, w）
            num_wall_samples: WSS 专用的壁面点采样数。
                - None: 使用共享采样（从 num_spatial_samples 中提取壁面点）
                - 正整数: 独立采样指定数量的壁面点用于 WSS 训练
                - 推荐设置: 当 output_vars 包含 'wss' 时，建议设置此参数以确保足够的壁面点
            normalize_coords: 是否归一化坐标
            bc_dim: 边界条件维度
            mode: 'train' 或 'test'，决定使用哪部分时间步
            train_ratio: 训练集占总时间步的比例

            自定义时间范围（优先级高于 train_ratio）:
            skip_first: 跳过前 N 帧（不使用）
            train_frames: 训练使用的帧数（从 skip_first 之后开始）
            test_frames: 测试使用的帧数（在训练帧之后）

            自适应输出:
            output_vars: 输出变量，可选:
                - 'p': 只预测压力
                - 'wss': 只预测壁面剪切应力
                - 'velocity' / 'vel': 只预测速度 (u, v, w)
                - 'flow': 压力 + 速度 (p, u, v, w)
                - 'all': 全部 (p, u, v, w, wss)
                - 组合如 'p,wss': 压力和 WSS

            进出口切除:
            boundary_cut: 进出口切除比例 (0.0-0.5)
                - 0: 不切除
                - 0.1: 切除两端各 10% 的区域
                - 0.15: 切除两端各 15% 的区域

            内存缓存:
            cache_mode: 缓存模式
                - 'ram': 预加载所有数据到内存 (推荐，消除 I/O 瓶颈)
                - 'none': 不缓存，每次从磁盘读取
        """
        self.h5_path = Path(h5_path)
        self.cache_mode = cache_mode
        self.nii_path = Path(nii_path)
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.stride = stride
        self.num_spatial_samples = num_spatial_samples
        self.num_wall_samples = num_wall_samples  # WSS 专用壁面采样
        self.normalize_coords = normalize_coords
        self.bc_dim = bc_dim
        self.mode = mode
        self.train_ratio = train_ratio
        self.skip_first = skip_first
        self.train_frames = train_frames
        self.test_frames = test_frames
        self.boundary_cut = boundary_cut

        # 解析输出变量
        self.output_vars = parse_output_vars(output_vars)
        self.output_var_indices = get_var_indices(self.output_vars)
        self.num_outputs = len(self.output_vars)
        self.has_wss = 'wss' in self.output_vars

        # 判断是否只预测 WSS（无流场变量）
        self.wss_only = self.has_wss and not any(v in self.output_vars for v in ['p', 'u', 'v', 'w'])
        # 判断是否使用独立壁面采样
        self.use_separate_wall_sampling = (self.num_wall_samples is not None and
                                            self.num_wall_samples > 0 and
                                            self.has_wss)

        # 加载数据
        self._load_data()
        self._apply_boundary_cut()  # 应用进出口切除
        self._load_geometry()

        # 根据 mode 划分时间范围
        self._setup_time_split()

        print(f"\nDataset [{mode}] loaded:")
        print(f"  Output variables: {self.output_vars} ({self.num_outputs} dims)")
        print(f"  Total nodes: {len(self.coords)}")
        print(f"  Valid nodes (after cut): {len(self.valid_node_indices)}")
        print(f"  Wall nodes: {len(self.wall_indices)}")
        print(f"  Time steps: {len(self.time_values)}")
        print(f"  {mode.capitalize()} time range: [{self.time_start}, {self.time_end})")
        print(f"  Input steps: {input_steps}, Output steps: {output_steps}")
        print(f"  Number of samples: {len(self)}")
        if self.boundary_cut > 0:
            print(f"  Boundary cut: {self.boundary_cut*100:.1f}% each end")
        if self.has_wss:
            if self.wss_only:
                print(f"  WSS-only mode: sampling {min(self.num_wall_samples or self.num_spatial_samples, len(self.wall_indices))} wall points")
            elif self.use_separate_wall_sampling:
                print(f"  Separate wall sampling: {self.num_wall_samples} wall points for WSS")
            else:
                wall_ratio = len(self.wall_indices) / len(self.valid_node_indices) * 100
                expected_wall = int(self.num_spatial_samples * wall_ratio / 100)
                print(f"  WSS warning: shared sampling, ~{expected_wall}/{self.num_spatial_samples} wall points ({wall_ratio:.1f}%)")
                print(f"  Tip: Set num_wall_samples for better WSS training")

        # 预缓存所有样本到内存
        if self.cache_mode == 'ram':
            self._precompute_cache()
            print(f"  Cache mode: RAM (all {len(self._cache)} samples preloaded)")

    def _load_data(self):
        """加载 H5 数据"""
        with h5py.File(self.h5_path, 'r') as f:
            # 加载网格
            self.coords = f['mesh/coords'][:].astype(np.float32)  # [N, 3]
            self.node_type = f['mesh/node_type'][:].astype(np.int64)  # [N]
            self.wall_indices = f['mesh/wall_indices'][:].astype(np.int64)  # [M]

            # 加载场数据
            self.velocity = f['fields/velocity'][:].astype(np.float32)  # [T, N, 3]
            self.pressure = f['fields/pressure'][:].astype(np.float32)  # [T, N, 1]
            self.time_values = f['time_values'][:].astype(np.float32)  # [T]

            # 加载 WSS 数据 [T, M, 3]，M 是壁面节点数
            if 'fields/wss' in f:
                self.wss = f['fields/wss'][:].astype(np.float32)  # [T, M, 3]
            else:
                M = len(self.wall_indices)
                T = len(self.time_values)
                self.wss = np.zeros((T, M, 3), dtype=np.float32)

        # 创建壁面索引映射：wall_indices[i] -> i
        self.wall_idx_to_local = {idx: i for i, idx in enumerate(self.wall_indices)}

        # 初始化为所有节点有效
        self.valid_node_indices = np.arange(len(self.coords))

        # 计算归一化参数
        self._update_normalization_params()

    def _update_normalization_params(self):
        """更新归一化参数"""
        valid_coords = self.coords[self.valid_node_indices]
        self.coord_min = valid_coords.min(axis=0)
        self.coord_max = valid_coords.max(axis=0)
        self.coord_range = self.coord_max - self.coord_min + 1e-8

        self.time_min = self.time_values.min()
        self.time_max = self.time_values.max()
        self.time_range = self.time_max - self.time_min + 1e-8

    def _apply_boundary_cut(self):
        """应用进出口切除"""
        if self.boundary_cut <= 0:
            return

        # 获取 inlet 和 outlet 节点
        inlet_mask = self.node_type == 1
        outlet_mask = self.node_type == 2

        if not np.any(inlet_mask) or not np.any(outlet_mask):
            print("  Warning: Could not find inlet/outlet nodes, skipping boundary cut")
            return

        # 计算 inlet 和 outlet 中心
        inlet_center = self.coords[inlet_mask].mean(axis=0)
        outlet_center = self.coords[outlet_mask].mean(axis=0)

        # 进出口方向（inlet -> outlet）
        direction = outlet_center - inlet_center
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # 将所有节点投影到进出口方向上
        # 投影值 = (coord - inlet_center) · direction
        relative_coords = self.coords - inlet_center
        projections = np.dot(relative_coords, direction)

        # 计算投影范围
        proj_min = projections.min()
        proj_max = projections.max()
        proj_range = proj_max - proj_min

        # 切除边界
        cut_dist = proj_range * self.boundary_cut
        valid_min = proj_min + cut_dist
        valid_max = proj_max - cut_dist

        # 创建有效节点掩码
        valid_mask = (projections >= valid_min) & (projections <= valid_max)
        self.valid_node_indices = np.where(valid_mask)[0]

        # 更新壁面节点索引
        # 保留在有效范围内的壁面节点
        valid_wall_mask = np.isin(self.wall_indices, self.valid_node_indices)
        self.wall_indices = self.wall_indices[valid_wall_mask]

        # 更新 WSS 数据
        self.wss = self.wss[:, valid_wall_mask, :]

        # 重建壁面索引映射
        self.wall_idx_to_local = {idx: i for i, idx in enumerate(self.wall_indices)}

        # 更新归一化参数
        self._update_normalization_params()

        print(f"  Boundary cut applied: {len(self.valid_node_indices)}/{len(self.coords)} nodes kept")

    def _load_geometry(self):
        """加载 NIfTI 几何影像"""
        try:
            import nibabel as nib
            img = nib.load(self.nii_path)
            self.geometry = img.get_fdata().astype(np.float32)
        except ImportError:
            try:
                import SimpleITK as sitk
                img = sitk.ReadImage(str(self.nii_path))
                self.geometry = sitk.GetArrayFromImage(img).astype(np.float32)
            except ImportError:
                raise ImportError("需要安装 nibabel 或 SimpleITK 来读取 NIfTI 文件")

        # 确保是 3D
        if len(self.geometry.shape) == 4:
            self.geometry = self.geometry[..., 0]

        # 归一化到 [0, 1]
        geo_min = self.geometry.min()
        geo_max = self.geometry.max()
        if geo_max > geo_min:
            self.geometry = (self.geometry - geo_min) / (geo_max - geo_min)

        # 添加通道维度 [D, H, W] -> [1, D, H, W]
        self.geometry = self.geometry[np.newaxis, ...]

    def _setup_time_split(self):
        """设置训练/测试的时间划分"""
        total_steps = len(self.time_values)
        window_size = self.input_steps + self.output_steps

        # 使用自定义时间范围（如果指定）
        if self.train_frames is not None or self.test_frames is not None:
            available_after_skip = total_steps - self.skip_first

            if self.train_frames is None:
                self.train_frames = available_after_skip - (self.test_frames or 0)
            if self.test_frames is None:
                self.test_frames = available_after_skip - self.train_frames

            if self.skip_first + self.train_frames + self.test_frames > total_steps:
                raise ValueError(
                    f"时间范围超出数据长度: skip_first({self.skip_first}) + "
                    f"train_frames({self.train_frames}) + test_frames({self.test_frames}) "
                    f"> total_steps({total_steps})"
                )

            if self.mode == 'train':
                self.time_start = self.skip_first
                self.time_end = self.skip_first + self.train_frames
            else:
                self.time_start = self.skip_first + self.train_frames
                self.time_end = self.skip_first + self.train_frames + self.test_frames

            print(f"  Custom time range: skip_first={self.skip_first}, "
                  f"train_frames={self.train_frames}, test_frames={self.test_frames}")
        else:
            train_end = int(total_steps * self.train_ratio)

            if self.mode == 'train':
                self.time_start = 0
                self.time_end = train_end
            else:
                self.time_start = train_end
                self.time_end = total_steps

        # 计算有效的样本数
        available_steps = self.time_end - self.time_start
        if available_steps < window_size:
            raise ValueError(
                f"Not enough time steps for {self.mode}: "
                f"available={available_steps}, required={window_size}"
            )

    def __len__(self) -> int:
        """返回可用的滑动窗口数量"""
        window_size = self.input_steps + self.output_steps
        available_steps = self.time_end - self.time_start
        return max(0, (available_steps - window_size) // self.stride + 1)

    def _precompute_cache(self):
        """预计算并缓存所有样本到内存"""
        from tqdm import tqdm
        print(f"  Preloading {len(self)} samples to RAM...")

        self._cache = []
        # 临时禁用缓存模式以使用原始 __getitem__
        original_cache_mode = self.cache_mode
        self.cache_mode = 'none'

        for idx in tqdm(range(len(self)), desc="  Caching", leave=False):
            sample = self._getitem_uncached(idx)
            self._cache.append(sample)

        self.cache_mode = original_cache_mode

        # 估算内存使用
        sample_size = sum(t.numel() * t.element_size() for t in self._cache[0])
        total_mb = sample_size * len(self._cache) / (1024 * 1024)
        print(f"  Cache size: {total_mb:.1f} MB")

    def _build_output_array(self, t_idx: int, spatial_indices: np.ndarray,
                            wall_only: bool = False) -> np.ndarray:
        """构建指定时间步的输出数组（只包含选择的变量）

        Args:
            t_idx: 时间步索引
            spatial_indices: 空间节点索引
            wall_only: 是否只构建壁面点的输出（用于 WSS-only 模式）
        """
        N_s = len(spatial_indices)

        if wall_only or self.wss_only:
            # WSS-only 模式：spatial_indices 直接是壁面节点的全局索引
            # 需要转换为 WSS 数据的局部索引
            wss_mag = np.zeros((N_s, 1), dtype=np.float32)
            for i, node_idx in enumerate(spatial_indices):
                if node_idx in self.wall_idx_to_local:
                    local_idx = self.wall_idx_to_local[node_idx]
                    wss_vec = self.wss[t_idx, local_idx]  # [3]
                    wss_mag[i, 0] = np.sqrt(np.sum(wss_vec ** 2))
            return wss_mag
        else:
            # 获取原始全量数据
            p = self.pressure[t_idx, spatial_indices]  # [N_s, 1]
            uvw = self.velocity[t_idx, spatial_indices]  # [N_s, 3]

            # WSS magnitude: 只有壁面节点有值，其他节点为 0
            wss_mag = np.zeros((N_s, 1), dtype=np.float32)
            if self.has_wss:
                for i, node_idx in enumerate(spatial_indices):
                    if node_idx in self.wall_idx_to_local:
                        local_idx = self.wall_idx_to_local[node_idx]
                        wss_vec = self.wss[t_idx, local_idx]  # [3]
                        wss_mag[i, 0] = np.sqrt(np.sum(wss_vec ** 2))

            # 按选择的变量构建输出
            full_output = np.concatenate([p, uvw, wss_mag], axis=1)  # [N_s, 5]
            selected_output = full_output[:, self.output_var_indices]  # [N_s, num_outputs]

            return selected_output

    def _build_wss_output(self, t_idx: int, wall_indices: np.ndarray) -> np.ndarray:
        """构建壁面点的 WSS 输出（用于独立壁面采样模式）

        Args:
            t_idx: 时间步索引
            wall_indices: 壁面节点的全局索引
        """
        N_w = len(wall_indices)
        wss_mag = np.zeros((N_w, 1), dtype=np.float32)
        for i, node_idx in enumerate(wall_indices):
            local_idx = self.wall_idx_to_local[node_idx]
            wss_vec = self.wss[t_idx, local_idx]  # [3]
            wss_mag[i, 0] = np.sqrt(np.sum(wss_vec ** 2))
        return wss_mag

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        获取一个时序预测样本

        如果启用了 RAM 缓存，直接返回缓存的样本
        """
        if self.cache_mode == 'ram' and hasattr(self, '_cache'):
            return self._cache[idx]
        return self._getitem_uncached(idx)

    def _getitem_uncached(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        获取一个时序预测样本（不使用缓存）

        采样模式:
        1. WSS-only 模式: 只从壁面点采样，只返回 WSS
        2. 独立壁面采样模式: 流场从所有点采样，WSS 从壁面点独立采样
        3. 共享采样模式: 所有变量从相同的点采样（向后兼容）

        Returns:
            x_input: [input_steps * N_samples, 4] - 输入帧的 (x, y, z, t)
            y_input: [input_steps * N_samples, num_outputs] - 输入帧的输出值
            x_output: [output_steps * N_samples, 4] - 输出帧的 (x, y, z, t)
            y_output: [output_steps * N_samples, num_outputs] - 输出帧的真实值
            geometry: [1, D, H, W] - 3D 几何影像
            bc: [bc_dim] - 边界条件
            wall_mask: [output_steps * N_samples] - 壁面掩码

            独立壁面采样模式额外返回:
            x_wall: [output_steps * N_wall, 4] - 壁面点坐标
            y_wall: [output_steps * N_wall, 1] - 壁面点 WSS 真实值
        """
        # 计算时间窗口的起始位置
        window_start = self.time_start + idx * self.stride

        # 输入帧索引
        input_indices = np.arange(window_start, window_start + self.input_steps)
        # 输出帧索引
        output_indices = np.arange(
            window_start + self.input_steps,
            window_start + self.input_steps + self.output_steps
        )

        # ============ WSS-only 模式 ============
        if self.wss_only:
            return self._getitem_wss_only(input_indices, output_indices)

        # ============ 独立壁面采样模式 ============
        if self.use_separate_wall_sampling:
            return self._getitem_separate_wall(input_indices, output_indices)

        # ============ 共享采样模式（向后兼容） ============
        return self._getitem_shared(input_indices, output_indices)

    def _getitem_wss_only(self, input_indices: np.ndarray, output_indices: np.ndarray):
        """WSS-only 模式：只从壁面点采样"""
        # 从壁面节点中采样
        N_wall = len(self.wall_indices)
        sample_size = min(self.num_wall_samples or self.num_spatial_samples, N_wall)
        sample_local_indices = np.random.choice(N_wall, sample_size, replace=False)
        wall_spatial_indices = self.wall_indices[sample_local_indices]

        # 壁面点坐标
        wall_coords = self.coords[wall_spatial_indices]  # [N_w, 3]
        if self.normalize_coords:
            wall_coords = (wall_coords - self.coord_min) / self.coord_range

        N_w = len(wall_spatial_indices)

        # 构建输入数据
        x_input_list = []
        y_input_list = []
        for t_idx in input_indices:
            t_val = self.time_values[t_idx]
            t_normalized = (t_val - self.time_min) / self.time_range

            t_expanded = np.full((N_w, 1), t_normalized, dtype=np.float32)
            xt = np.concatenate([wall_coords, t_expanded], axis=1)
            x_input_list.append(xt)

            yt = self._build_wss_output(t_idx, wall_spatial_indices)
            y_input_list.append(yt)

        # 构建输出数据
        x_output_list = []
        y_output_list = []
        for t_idx in output_indices:
            t_val = self.time_values[t_idx]
            t_normalized = (t_val - self.time_min) / self.time_range

            t_expanded = np.full((N_w, 1), t_normalized, dtype=np.float32)
            xt = np.concatenate([wall_coords, t_expanded], axis=1)
            x_output_list.append(xt)

            yt = self._build_wss_output(t_idx, wall_spatial_indices)
            y_output_list.append(yt)

        x_input = np.concatenate(x_input_list, axis=0).astype(np.float32)
        y_input = np.concatenate(y_input_list, axis=0).astype(np.float32)
        x_output = np.concatenate(x_output_list, axis=0).astype(np.float32)
        y_output = np.concatenate(y_output_list, axis=0).astype(np.float32)

        # 壁面掩码（全为 1）
        wall_mask = np.ones(len(x_output), dtype=np.float32)

        bc = self._extract_boundary_conditions(input_indices)

        return (
            torch.tensor(x_input, dtype=torch.float32),
            torch.tensor(y_input, dtype=torch.float32),
            torch.tensor(x_output, dtype=torch.float32),
            torch.tensor(y_output, dtype=torch.float32),
            torch.tensor(self.geometry.copy(), dtype=torch.float32),
            torch.tensor(bc, dtype=torch.float32),
            torch.tensor(wall_mask, dtype=torch.float32)
        )

    def _getitem_separate_wall(self, input_indices: np.ndarray, output_indices: np.ndarray):
        """独立壁面采样模式：流场和 WSS 分别采样"""
        # === 流场采样（从所有有效节点） ===
        N = len(self.valid_node_indices)
        sample_size = min(self.num_spatial_samples, N)
        sample_local_indices = np.random.choice(N, sample_size, replace=False)
        spatial_indices = self.valid_node_indices[sample_local_indices]

        sampled_coords = self.coords[spatial_indices]
        if self.normalize_coords:
            sampled_coords = (sampled_coords - self.coord_min) / self.coord_range

        N_s = len(spatial_indices)

        # 创建流场点的壁面掩码
        wall_mask_spatial = np.zeros(N_s, dtype=np.float32)
        for i, node_idx in enumerate(spatial_indices):
            if node_idx in self.wall_idx_to_local:
                wall_mask_spatial[i] = 1.0

        # === 壁面点独立采样（用于 WSS） ===
        N_wall = len(self.wall_indices)
        wall_sample_size = min(self.num_wall_samples, N_wall)
        wall_sample_local_indices = np.random.choice(N_wall, wall_sample_size, replace=False)
        wall_spatial_indices = self.wall_indices[wall_sample_local_indices]

        wall_coords = self.coords[wall_spatial_indices]
        if self.normalize_coords:
            wall_coords = (wall_coords - self.coord_min) / self.coord_range

        N_w = len(wall_spatial_indices)

        # === 构建流场输入数据 ===
        x_input_list = []
        y_input_list = []
        for t_idx in input_indices:
            t_val = self.time_values[t_idx]
            t_normalized = (t_val - self.time_min) / self.time_range

            t_expanded = np.full((N_s, 1), t_normalized, dtype=np.float32)
            xt = np.concatenate([sampled_coords, t_expanded], axis=1)
            x_input_list.append(xt)

            yt = self._build_output_array(t_idx, spatial_indices)
            y_input_list.append(yt)

        # === 构建流场输出数据 ===
        x_output_list = []
        y_output_list = []
        wall_mask_list = []
        for t_idx in output_indices:
            t_val = self.time_values[t_idx]
            t_normalized = (t_val - self.time_min) / self.time_range

            t_expanded = np.full((N_s, 1), t_normalized, dtype=np.float32)
            xt = np.concatenate([sampled_coords, t_expanded], axis=1)
            x_output_list.append(xt)

            yt = self._build_output_array(t_idx, spatial_indices)
            y_output_list.append(yt)
            wall_mask_list.append(wall_mask_spatial)

        # === 构建壁面点数据（用于 WSS） ===
        x_wall_list = []
        y_wall_list = []
        for t_idx in output_indices:
            t_val = self.time_values[t_idx]
            t_normalized = (t_val - self.time_min) / self.time_range

            t_expanded = np.full((N_w, 1), t_normalized, dtype=np.float32)
            xt = np.concatenate([wall_coords, t_expanded], axis=1)
            x_wall_list.append(xt)

            yt = self._build_wss_output(t_idx, wall_spatial_indices)
            y_wall_list.append(yt)

        x_input = np.concatenate(x_input_list, axis=0).astype(np.float32)
        y_input = np.concatenate(y_input_list, axis=0).astype(np.float32)
        x_output = np.concatenate(x_output_list, axis=0).astype(np.float32)
        y_output = np.concatenate(y_output_list, axis=0).astype(np.float32)
        wall_mask = np.concatenate(wall_mask_list, axis=0).astype(np.float32)

        x_wall = np.concatenate(x_wall_list, axis=0).astype(np.float32)
        y_wall = np.concatenate(y_wall_list, axis=0).astype(np.float32)

        bc = self._extract_boundary_conditions(input_indices)

        # 返回 9 个值（比共享模式多 2 个）
        return (
            torch.tensor(x_input, dtype=torch.float32),
            torch.tensor(y_input, dtype=torch.float32),
            torch.tensor(x_output, dtype=torch.float32),
            torch.tensor(y_output, dtype=torch.float32),
            torch.tensor(self.geometry.copy(), dtype=torch.float32),
            torch.tensor(bc, dtype=torch.float32),
            torch.tensor(wall_mask, dtype=torch.float32),
            torch.tensor(x_wall, dtype=torch.float32),  # 壁面点坐标
            torch.tensor(y_wall, dtype=torch.float32)   # 壁面点 WSS 真值
        )

    def _getitem_shared(self, input_indices: np.ndarray, output_indices: np.ndarray):
        """共享采样模式（向后兼容）"""
        # 从有效节点中采样
        N = len(self.valid_node_indices)
        sample_size = min(self.num_spatial_samples, N)
        sample_local_indices = np.random.choice(N, sample_size, replace=False)
        spatial_indices = self.valid_node_indices[sample_local_indices]

        # 采样坐标
        sampled_coords = self.coords[spatial_indices]  # [N_s, 3]
        if self.normalize_coords:
            sampled_coords = (sampled_coords - self.coord_min) / self.coord_range

        N_s = len(spatial_indices)

        # 创建壁面掩码
        wall_mask_spatial = np.zeros(N_s, dtype=np.float32)
        for i, node_idx in enumerate(spatial_indices):
            if node_idx in self.wall_idx_to_local:
                wall_mask_spatial[i] = 1.0

        # 构建输入数据
        x_input_list = []
        y_input_list = []
        for t_idx in input_indices:
            t_val = self.time_values[t_idx]
            t_normalized = (t_val - self.time_min) / self.time_range

            t_expanded = np.full((N_s, 1), t_normalized, dtype=np.float32)
            xt = np.concatenate([sampled_coords, t_expanded], axis=1)
            x_input_list.append(xt)

            yt = self._build_output_array(t_idx, spatial_indices)
            y_input_list.append(yt)

        # 构建输出数据
        x_output_list = []
        y_output_list = []
        wall_mask_list = []
        for t_idx in output_indices:
            t_val = self.time_values[t_idx]
            t_normalized = (t_val - self.time_min) / self.time_range

            t_expanded = np.full((N_s, 1), t_normalized, dtype=np.float32)
            xt = np.concatenate([sampled_coords, t_expanded], axis=1)
            x_output_list.append(xt)

            yt = self._build_output_array(t_idx, spatial_indices)
            y_output_list.append(yt)
            wall_mask_list.append(wall_mask_spatial)

        x_input = np.concatenate(x_input_list, axis=0).astype(np.float32)
        y_input = np.concatenate(y_input_list, axis=0).astype(np.float32)
        x_output = np.concatenate(x_output_list, axis=0).astype(np.float32)
        y_output = np.concatenate(y_output_list, axis=0).astype(np.float32)
        wall_mask = np.concatenate(wall_mask_list, axis=0).astype(np.float32)

        # 边界条件
        bc = self._extract_boundary_conditions(input_indices)

        return (
            torch.tensor(x_input, dtype=torch.float32),
            torch.tensor(y_input, dtype=torch.float32),
            torch.tensor(x_output, dtype=torch.float32),
            torch.tensor(y_output, dtype=torch.float32),
            torch.tensor(self.geometry.copy(), dtype=torch.float32),
            torch.tensor(bc, dtype=torch.float32),
            torch.tensor(wall_mask, dtype=torch.float32)
        )

    def _extract_boundary_conditions(self, time_indices: np.ndarray) -> np.ndarray:
        """提取边界条件特征"""
        bc_features = []

        bc_features.extend(self.coord_range.tolist())

        v_selected = self.velocity[time_indices]
        v_mean = np.mean(np.linalg.norm(v_selected, axis=-1))
        bc_features.append(float(v_mean))

        p_selected = self.pressure[time_indices]
        p_range = float(p_selected.max() - p_selected.min())
        bc_features.append(p_range)

        bc_features.append(float(self.time_range))

        flow_rate = v_mean * np.prod(self.coord_range) ** (1/3)
        bc_features.append(float(flow_rate))

        while len(bc_features) < self.bc_dim:
            bc_features.append(0.0)
        bc_features = bc_features[:self.bc_dim]

        return np.array(bc_features, dtype=np.float32)


class SyntheticTemporalDataset(Dataset):
    """
    合成时序数据集 - 用于本地调试
    支持自适应输出
    """

    def __init__(self,
                 num_samples: int = 100,
                 num_spatial_points: int = 2000,
                 input_steps: int = 10,
                 output_steps: int = 5,
                 geometry_size: Tuple[int, int, int] = (32, 32, 32),
                 bc_dim: int = 7,
                 wall_ratio: float = 0.2,
                 output_vars: str = 'all'):
        self.num_samples = num_samples
        self.num_spatial_points = num_spatial_points
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.geometry_size = geometry_size
        self.bc_dim = bc_dim
        self.wall_ratio = wall_ratio

        # 解析输出变量
        self.output_vars = parse_output_vars(output_vars)
        self.output_var_indices = get_var_indices(self.output_vars)
        self.num_outputs = len(self.output_vars)
        self.has_wss = 'wss' in self.output_vars

        print(f"Created synthetic dataset:")
        print(f"  Samples: {num_samples}")
        print(f"  Output variables: {self.output_vars} ({self.num_outputs} dims)")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        np.random.seed(idx)

        N = self.num_spatial_points
        T_in = self.input_steps
        T_out = self.output_steps

        # 生成空间坐标
        theta = np.random.uniform(0, 2 * np.pi, N)
        phi = np.random.uniform(0, np.pi, N)
        r = np.random.uniform(0.1, 1.0, N) ** (1/3)

        coords = np.stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ], axis=1).astype(np.float32)

        coords = (coords + 1) / 2
        wall_mask = (r > 0.8).astype(np.float32)

        total_steps = T_in + T_out
        times = np.linspace(0, 1, total_steps).astype(np.float32)
        omega = 2 * np.pi * (0.5 + 0.5 * np.random.rand())

        def build_output(t, coords, r, wall_mask, phase):
            u = (-coords[:, 1] * np.sin(phase) * (1 - r)).astype(np.float32)
            v = (coords[:, 0] * np.sin(phase) * (1 - r)).astype(np.float32)
            w = np.zeros(N, dtype=np.float32)
            p = ((1 - r) * np.cos(phase)).astype(np.float32)
            vel_mag = np.sqrt(u**2 + v**2 + w**2)
            wss = (wall_mask * vel_mag * 0.1).astype(np.float32)

            # 完整输出
            full_output = np.stack([p, u, v, w, wss], axis=1)  # [N, 5]
            # 选择输出
            return full_output[:, self.output_var_indices]

        # 输入帧
        x_input_list = []
        y_input_list = []
        for t in times[:T_in]:
            phase = omega * t
            t_expanded = np.full((N, 1), t, dtype=np.float32)
            xt = np.concatenate([coords, t_expanded], axis=1)
            yt = build_output(t, coords, r, wall_mask, phase)
            x_input_list.append(xt)
            y_input_list.append(yt)

        # 输出帧
        x_output_list = []
        y_output_list = []
        wall_mask_list = []
        for t in times[T_in:]:
            phase = omega * t
            t_expanded = np.full((N, 1), t, dtype=np.float32)
            xt = np.concatenate([coords, t_expanded], axis=1)
            yt = build_output(t, coords, r, wall_mask, phase)
            x_output_list.append(xt)
            y_output_list.append(yt)
            wall_mask_list.append(wall_mask)

        x_input = np.concatenate(x_input_list, axis=0)
        y_input = np.concatenate(y_input_list, axis=0)
        x_output = np.concatenate(x_output_list, axis=0)
        y_output = np.concatenate(y_output_list, axis=0)
        wall_mask_out = np.concatenate(wall_mask_list, axis=0)

        # 几何
        D, H, W = self.geometry_size
        geometry = np.zeros((1, D, H, W), dtype=np.float32)
        voxel_indices = (coords * np.array([D - 1, H - 1, W - 1])).astype(np.int32)
        voxel_indices = np.clip(voxel_indices, 0, np.array([D - 1, H - 1, W - 1]))
        for i in range(N):
            d, h, w = voxel_indices[i]
            geometry[0, d, h, w] = 1.0

        bc = np.random.randn(self.bc_dim).astype(np.float32)
        bc[-1] = 1.0

        return (
            torch.tensor(x_input, dtype=torch.float32),
            torch.tensor(y_input, dtype=torch.float32),
            torch.tensor(x_output, dtype=torch.float32),
            torch.tensor(y_output, dtype=torch.float32),
            torch.tensor(geometry, dtype=torch.float32),
            torch.tensor(bc, dtype=torch.float32),
            torch.tensor(wall_mask_out, dtype=torch.float32)
        )


def collate_temporal(batch):
    """自定义 collate 函数

    支持两种模式:
    - 7 个返回值: 共享采样模式 / WSS-only 模式
    - 9 个返回值: 独立壁面采样模式（额外返回 x_wall, y_wall）
    """
    # 检查返回值数量
    num_returns = len(batch[0])
    has_separate_wall = (num_returns == 9)

    if has_separate_wall:
        x_in_list, y_in_list, x_out_list, y_out_list, geo_list, bc_list, wall_mask_list, x_wall_list, y_wall_list = zip(*batch)
    else:
        x_in_list, y_in_list, x_out_list, y_out_list, geo_list, bc_list, wall_mask_list = zip(*batch)

    max_in = max(x.shape[0] for x in x_in_list)
    max_out = max(x.shape[0] for x in x_out_list)

    def pad_tensor(tensor_list, max_len):
        padded = []
        for t in tensor_list:
            n = t.shape[0]
            if n < max_len:
                pad_idx = np.random.choice(n, max_len - n, replace=True)
                t = torch.cat([t, t[pad_idx]], dim=0)
            padded.append(t)
        return torch.stack(padded)

    def pad_1d_tensor(tensor_list, max_len):
        padded = []
        for t in tensor_list:
            n = t.shape[0]
            if n < max_len:
                pad_idx = np.random.choice(n, max_len - n, replace=True)
                t = torch.cat([t, t[pad_idx]], dim=0)
            padded.append(t)
        return torch.stack(padded)

    result = (
        pad_tensor(x_in_list, max_in),
        pad_tensor(y_in_list, max_in),
        pad_tensor(x_out_list, max_out),
        pad_tensor(y_out_list, max_out),
        torch.stack(geo_list),
        torch.stack(bc_list),
        pad_1d_tensor(wall_mask_list, max_out)
    )

    if has_separate_wall:
        max_wall = max(x.shape[0] for x in x_wall_list)
        result = result + (
            pad_tensor(x_wall_list, max_wall),
            pad_tensor(y_wall_list, max_wall)
        )

    return result


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Temporal Prediction Dataset")
    print("=" * 60)

    h5_file = Path("kdd.h5")
    nii_file = Path("200.nii.gz")

    if h5_file.exists() and nii_file.exists():
        print("\n--- Testing with different output_vars ---")

        for output_vars in ['p', 'wss', 'velocity', 'p,wss', 'all']:
            print(f"\n>>> output_vars = '{output_vars}'")
            dataset = TemporalCFDDataset(
                h5_path=str(h5_file),
                nii_path=str(nii_file),
                input_steps=5,
                output_steps=2,
                num_spatial_samples=200,
                mode='train',
                skip_first=5,  # 跳过初始帧
                train_frames=50,
                output_vars=output_vars,
                boundary_cut=0.1  # 切除 10%
            )

            sample = dataset[0]
            if len(sample) == 7:
                x_in, y_in, x_out, y_out, geo, bc, wall_mask = sample
                print(f"    y_output shape: {y_out.shape}")
            else:
                x_in, y_in, x_out, y_out, geo, bc, wall_mask, x_wall, y_wall = sample
                print(f"    y_output shape: {y_out.shape}, x_wall: {x_wall.shape}, y_wall: {y_wall.shape}")

        print("\n--- Testing WSS-only mode (壁面点采样) ---")
        dataset = TemporalCFDDataset(
            h5_path=str(h5_file),
            nii_path=str(nii_file),
            input_steps=5,
            output_steps=2,
            num_spatial_samples=200,  # 会自动限制为壁面点数
            mode='train',
            skip_first=5,
            train_frames=50,
            output_vars='wss',
            boundary_cut=0.1
        )
        sample = dataset[0]
        x_in, y_in, x_out, y_out, geo, bc, wall_mask = sample
        print(f"  WSS-only: x_out shape: {x_out.shape}, wall_mask sum: {wall_mask.sum().item()}")

        print("\n--- Testing separate wall sampling (独立壁面采样) ---")
        dataset = TemporalCFDDataset(
            h5_path=str(h5_file),
            nii_path=str(nii_file),
            input_steps=5,
            output_steps=2,
            num_spatial_samples=200,
            num_wall_samples=500,  # 独立采样 500 个壁面点
            mode='train',
            skip_first=5,
            train_frames=50,
            output_vars='all',  # 包含流场和 WSS
            boundary_cut=0.1
        )
        sample = dataset[0]
        x_in, y_in, x_out, y_out, geo, bc, wall_mask, x_wall, y_wall = sample
        print(f"  Flow: x_out shape: {x_out.shape}")
        print(f"  Wall: x_wall shape: {x_wall.shape}, y_wall shape: {y_wall.shape}")

        print("\n--- Testing boundary_cut ---")
        for cut in [0, 0.1, 0.15, 0.2]:
            dataset = TemporalCFDDataset(
                h5_path=str(h5_file),
                nii_path=str(nii_file),
                input_steps=5,
                output_steps=2,
                num_spatial_samples=200,
                mode='train',
                skip_first=5,
                train_frames=50,
                output_vars='p',
                boundary_cut=cut
            )
            print(f"  boundary_cut={cut}: {len(dataset.valid_node_indices)} nodes")

    else:
        print("\nTesting with synthetic data...")
        for output_vars in ['p', 'wss', 'all']:
            print(f"\n>>> output_vars = '{output_vars}'")
            dataset = SyntheticTemporalDataset(
                num_samples=5,
                num_spatial_points=200,
                input_steps=3,
                output_steps=2,
                output_vars=output_vars
            )
            x_in, y_in, x_out, y_out, geo, bc, wall_mask = dataset[0]
            print(f"    y_output shape: {y_out.shape}")

    print("\n" + "=" * 60)
    print("Dataset test PASSED!")
    print("=" * 60)
