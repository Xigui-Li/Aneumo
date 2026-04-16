"""
自动扫描病例目录，将每个病例的瞬态 VTK 序列写入独立的 H5 文件（并行加速版）

数据结构：
./vtk_data/
    case_001/
        case_001/
            4.01/ ... 5.00/
    case_002/
        case_002/
            4.01/ ... 5.00/

输出：
./h5_data/
    case_001.h5
    case_002.h5
    ...

用法：
python DataPreprocess_parallel.py --num-workers 16
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
from tqdm import tqdm

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    import vtk
except ImportError as exc:
    raise ImportError("需要 vtk，请先安装：pip install vtk") from exc


# ============================================================================
# 数据类型定义
# ============================================================================

class NodeType:
    """节点类型编码"""
    INTERNAL = 0
    INFLOW = 1
    OUTFLOW = 2
    WALL = 3


@dataclass
class StepInfo:
    """单个时间步信息"""
    case_dir: Path
    time_value: float
    internal_vtu: Path
    inlet_vtp: Path
    outlet_vtp: Path
    wall_vtp: Path


@dataclass
class ProcessResult:
    """处理结果"""
    case_name: str
    success: bool
    error_msg: Optional[str] = None
    num_timesteps: int = 0
    num_nodes: int = 0
    num_edges: int = 0
    num_isolated: int = 0
    file_size_mb: float = 0.0


# ============================================================================
# Case 扫描和时间步发现
# ============================================================================

def discover_cases(base_dir: Path, 
                  time_start: float = 4.01, 
                  time_end: float = 5.00, 
                  time_step: float = 0.01) -> List[Path]:
    """
    扫描 base_dir，找出所有有效的 case 目录
    
    有效条件：
    1. 包含同名子文件夹（如 all/case_001/case_001/）
    2. 子文件夹内有从 time_start 到 time_end 的所有时间步文件夹
    """
    if not base_dir.exists():
        raise ValueError(f"基础目录不存在: {base_dir}")
    
    # 生成预期的时间步列表
    expected_times = []
    t = time_start
    while t <= time_end + 1e-6:
        expected_times.append(f"{t:.2f}")
        t += time_step
    
    print(f"预期时间步数量: {len(expected_times)}")
    print(f"时间范围: {expected_times[0]} 到 {expected_times[-1]}")
    
    valid_cases = []
    
    for case_dir in sorted(base_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        
        case_name = case_dir.name
        actual_data_dir = case_dir / case_name
        
        if not actual_data_dir.exists() or not actual_data_dir.is_dir():
            print(f"⚠️  跳过 {case_name}: 缺少同名子文件夹")
            continue
        
        # 检查是否包含所有预期的时间步
        existing_times = {sub.name for sub in actual_data_dir.iterdir() if sub.is_dir()}
        missing_times = set(expected_times) - existing_times
        
        if len(missing_times) == 0:
            valid_cases.append(actual_data_dir)
            print(f"✅ 找到有效 case: {case_name} ({len(expected_times)} 个时间步)")
        else:
            print(f"⚠️  跳过 {case_name}: 缺少 {len(missing_times)} 个时间步")
            if len(missing_times) <= 10:
                print(f"    缺失时间步: {sorted(list(missing_times))[:10]}")
    
    print(f"\n总计找到 {len(valid_cases)} 个有效 case")
    return valid_cases


def discover_steps(case_dir: Path) -> List[StepInfo]:
    """发现某个 case 目录下的所有时间步"""
    steps = []
    
    for sub in sorted(case_dir.iterdir()):
        if not sub.is_dir():
            continue
        
        try:
            tval = float(sub.name)
        except ValueError:
            continue
        
        prefix = sub.name
        internal = sub / f"{prefix}_internal.vtu"
        inlet = sub / f"{prefix}_inlet.vtp"
        outlet = sub / f"{prefix}_outlet.vtp"
        wall = sub / f"{prefix}_wall.vtp"
        
        if internal.exists():
            steps.append(
                StepInfo(
                    case_dir=case_dir,
                    time_value=tval,
                    internal_vtu=internal,
                    inlet_vtp=inlet,
                    outlet_vtp=outlet,
                    wall_vtp=wall,
                )
            )
    
    return sorted(steps, key=lambda s: s.time_value)


# ============================================================================
# VTK 数据读取
# ============================================================================

def read_unstructured_grid(path: Path):
    """读取 VTU 文件"""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def read_polydata(path: Path):
    """读取 VTP 文件"""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def get_points(obj) -> np.ndarray:
    """提取 VTK 对象的点坐标"""
    pts = obj.GetPoints()
    if pts is None:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())], dtype=np.float32)


def extract_fields(grid) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    提取节点坐标、压力、速度
    
    Returns:
        coords: [N, 3] 节点坐标
        pressure: [N] 压力
        velocity: [N, 3] 速度
    """
    pts = grid.GetPoints()
    num = pts.GetNumberOfPoints()
    coords = np.array([pts.GetPoint(i) for i in range(num)], dtype=np.float32)
    
    arr_p = grid.GetPointData().GetArray("p")
    arr_u = grid.GetPointData().GetArray("U")
    
    if arr_p is None or arr_u is None:
        raise ValueError("VTU 缺少字段 p 或 U")
    
    pressure = np.array([arr_p.GetValue(i) for i in range(num)], dtype=np.float32)
    
    vel = np.zeros((num, 3), dtype=np.float32)
    for i in range(num):
        vel[i, 0] = arr_u.GetValue(i * 3)
        vel[i, 1] = arr_u.GetValue(i * 3 + 1)
        vel[i, 2] = arr_u.GetValue(i * 3 + 2)
    
    return coords, pressure, vel


def extract_edges(grid) -> np.ndarray:
    """
    从网格拓扑提取边（单向，i < j）
    
    Returns:
        edges: [E, 2] 边连接
    """
    edges = set()
    
    for cell_id in range(grid.GetNumberOfCells()):
        cell = grid.GetCell(cell_id)
        ids = [cell.GetPointId(i) for i in range(cell.GetPointIds().GetNumberOfIds())]
        
        # 单向边：每对节点仅保留一次 (i, j) 其中 i < j
        for i, j in itertools.combinations(ids, 2):
            if i != j:
                edges.add((min(i, j), max(i, j)))
    
    return np.array(sorted(edges), dtype=np.int64)


# ============================================================================
# 边界和节点类型处理
# ============================================================================

def match_boundary(points: np.ndarray, boundary_points: np.ndarray, tol: float) -> np.ndarray:
    """
    匹配边界点到网格节点
    
    Args:
        points: [N, 3] 网格节点坐标
        boundary_points: [M, 3] 边界节点坐标
        tol: 匹配容差
    
    Returns:
        indices: [K] 匹配到的节点索引
    """
    if boundary_points.size == 0:
        return np.array([], dtype=np.int64)
    
    if cKDTree:
        tree = cKDTree(points)
        dist, idx = tree.query(boundary_points, distance_upper_bound=tol)
        valid = dist < np.inf
        return np.unique(idx[valid]).astype(np.int64)
    
    # Fallback: 字典匹配
    lookup = {tuple(np.round(p, 6)): i for i, p in enumerate(points)}
    hits = []
    for bp in boundary_points:
        key = tuple(np.round(bp, 6))
        if key in lookup:
            hits.append(lookup[key])
    
    return np.unique(hits).astype(np.int64)


def build_node_type(coords: np.ndarray, 
                    inlet: Path, 
                    outlet: Path, 
                    wall: Path, 
                    tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建节点类型标签
    
    Returns:
        node_type: [N] 节点类型（0=内部，1=入口，2=出口，3=壁面）
        wall_indices: [M] 壁面节点索引
    """
    node_type = np.full(coords.shape[0], NodeType.INTERNAL, dtype=np.int64)
    wall_indices = np.array([], dtype=np.int64)
    
    for path, code in ((inlet, NodeType.INFLOW), (outlet, NodeType.OUTFLOW), (wall, NodeType.WALL)):
        if path.exists():
            bpts = get_points(read_polydata(path))
            idx = match_boundary(coords, bpts, tol)
            node_type[idx] = code
            
            if code == NodeType.WALL:
                wall_indices = idx
    
    return node_type, wall_indices


def extract_wss(path: Path) -> np.ndarray:
    """
    提取壁面剪切应力
    
    Returns:
        wss: [M, 3] 壁面剪切应力
    """
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32)
    
    data = read_polydata(path)
    arr = None
    
    for name in ("wallShearStress", "WSS", "wss"):
        arr = data.GetPointData().GetArray(name)
        if arr is not None:
            break
    
    if arr is None:
        return np.zeros((0, 3), dtype=np.float32)
    
    pts = data.GetPoints()
    num = pts.GetNumberOfPoints()
    wss = np.zeros((num, 3), dtype=np.float32)
    
    for i in range(num):
        wss[i, 0] = arr.GetValue(i * 3)
        wss[i, 1] = arr.GetValue(i * 3 + 1)
        wss[i, 2] = arr.GetValue(i * 3 + 2)
    
    return wss


# ============================================================================
# 单个 Case 处理（Worker 函数）
# ============================================================================

def process_single_case_worker(args) -> ProcessResult:
    """
    Worker 函数：处理单个 case（用于并行）

    Args:
        args: (case_name, case_path, output_dir, match_tol)

    Returns:
        ProcessResult
    """
    case_name, case_path, output_dir, match_tol = args
    case_path = Path(case_path)
    output_dir = Path(output_dir)
    output_file = output_dir / f"{case_name}.h5"
    
    try:
        # ===== 1. 发现所有时间步 =====
        steps = discover_steps(case_path)
        
        if not steps:
            return ProcessResult(
                case_name=case_name,
                success=False,
                error_msg="未找到任何有效时间步"
            )
        
        # ===== 2. 读取第一个时间步，获取网格拓扑 =====
        first = steps[0]
        grid0 = read_unstructured_grid(first.internal_vtu)
        coords0, p0, v0 = extract_fields(grid0)
        edges0 = extract_edges(grid0)
        node_type, wall_indices = build_node_type(
            coords0, first.inlet_vtp, first.outlet_vtp, first.wall_vtp, match_tol
        )
        
        # ===== 3. 检查网格质量（计算孤立节点数量）=====
        used_nodes = set(edges0.flatten().tolist())
        num_isolated = coords0.shape[0] - len(used_nodes)
        
        # ===== 4. 收集所有时间步的数据 =====
        velocities = []
        pressures = []
        wss_list = []
        time_values = []
        
        for step in steps:
            # 读取网格和场数据
            grid = read_unstructured_grid(step.internal_vtu)
            coords, p, v = extract_fields(grid)
            
            # 检查网格一致性
            if coords.shape[0] != coords0.shape[0]:
                raise ValueError(f"节点数不一致: {step.internal_vtu}")
            
            edges = extract_edges(grid)
            if edges.shape != edges0.shape or not np.array_equal(edges, edges0):
                raise ValueError(f"网格拓扑变化: {step.internal_vtu}")
            
            velocities.append(v)
            pressures.append(p[:, None])
            
            # WSS 映射到 wall_indices
            wss_wall = extract_wss(step.wall_vtp)
            
            if wss_wall.size == 0:
                mapped = np.zeros((len(wall_indices), 3), dtype=np.float32)
            else:
                wall_pts = get_points(read_polydata(step.wall_vtp))
                idx_map = match_boundary(coords0, wall_pts, match_tol)
                
                mapped = np.zeros((len(wall_indices), 3), dtype=np.float32)
                lookup = {idx_map[i]: i for i in range(len(idx_map))}
                
                for k, nid in enumerate(wall_indices):
                    if nid in lookup:
                        mapped[k] = wss_wall[lookup[nid]]
            
            wss_list.append(mapped)
            time_values.append(step.time_value)
        
        # ===== 5. 转换为数组 =====
        velocity_arr = np.stack(velocities, axis=0)  # [T, N, 3]
        pressure_arr = np.stack(pressures, axis=0)   # [T, N, 1]
        wss_arr = np.stack(wss_list, axis=0) if wss_list else np.zeros((0, 0, 3), dtype=np.float32)
        time_values_arr = np.array(time_values, dtype=np.float32)
        
        # ===== 6. 写入 H5 文件 =====
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, "w") as h5f:
            # Mesh 信息
            mesh = h5f.create_group("mesh")
            mesh.create_dataset("coords", data=coords0, compression="gzip")
            mesh.create_dataset("edges", data=edges0, compression="gzip")
            mesh.create_dataset("node_type", data=node_type, compression="gzip")
            mesh.create_dataset("wall_indices", data=wall_indices, compression="gzip")
            
            # Fields 数据
            fields = h5f.create_group("fields")
            fields.create_dataset("velocity", data=velocity_arr, compression="gzip")
            fields.create_dataset("pressure", data=pressure_arr, compression="gzip")
            fields.create_dataset("wss", data=wss_arr, compression="gzip")
            
            # 时间序列
            h5f.create_dataset("time_values", data=time_values_arr)
            
            # 元数据（只保存标量）
            h5f.attrs["case_name"] = case_name
            h5f.attrs["num_timesteps"] = len(steps)
            h5f.attrs["num_nodes"] = coords0.shape[0]
            h5f.attrs["num_edges"] = edges0.shape[0]
            h5f.attrs["num_wall_nodes"] = len(wall_indices)
            h5f.attrs["num_isolated_nodes"] = num_isolated
            h5f.attrs["description"] = (
                "velocity:[T,N,3], pressure:[T,N,1], wss:[T,M,3], "
                "node_type: 0=internal, 1=inlet, 2=outlet, 3=wall"
            )
        
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        
        return ProcessResult(
            case_name=case_name,
            success=True,
            num_timesteps=len(steps),
            num_nodes=coords0.shape[0],
            num_edges=edges0.shape[0],
            num_isolated=num_isolated,
            file_size_mb=file_size_mb
        )
    
    except Exception as e:
        return ProcessResult(
            case_name=case_name,
            success=False,
            error_msg=str(e)
        )


# ============================================================================
# 并行处理主函数
# ============================================================================

def process_cases_parallel(valid_cases: List[Path],
                          output_dir: Path,
                          match_tol: float,
                          num_workers: int,
                          skip_existing: bool = False) -> Dict[str, int]:
    """
    并行处理所有 case
    
    Args:
        valid_cases: 有效 case 路径列表
        output_dir: 输出目录
        match_tol: 边界匹配容差
        num_workers: 工作进程数
        skip_existing: 是否跳过已存在的文件
    
    Returns:
        统计字典 {'success': int, 'skip': int, 'fail': int}
    """
    # 准备任务列表
    tasks = []
    for case_path in valid_cases:
        case_name = case_path.parent.name
        output_file = output_dir / f"{case_name}.h5"

        if skip_existing and output_file.exists():
            print(f"⏭️  跳过已存在: {case_name}")
            continue

        tasks.append((case_name, case_path, output_dir, match_tol))

    if not tasks:
        print("没有需要处理的 case")
        return {'success': 0, 'skip': len(valid_cases), 'fail': 0}

    print(f"\n{'='*70}")
    print(f"开始并行处理 {len(tasks)} 个 case（使用 {num_workers} 个进程）")
    print(f"{'='*70}\n")

    success_count = 0
    fail_count = 0
    skip_count = len(valid_cases) - len(tasks)

    # 使用 ProcessPoolExecutor 进行并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_case = {
            executor.submit(process_single_case_worker, task): task[0]
            for task in tasks
        }
        
        # 使用 tqdm 显示进度
        with tqdm(total=len(tasks), desc="处理进度", unit="case") as pbar:
            for future in as_completed(future_to_case):
                case_name = future_to_case[future]
                
                try:
                    result = future.result()
                    
                    if result.success:
                        success_count += 1
                        pbar.write(
                            f"✅ {result.case_name}: "
                            f"T={result.num_timesteps}, N={result.num_nodes}, "
                            f"E={result.num_edges}, Isolated={result.num_isolated}, "
                            f"Size={result.file_size_mb:.2f}MB"
                        )
                    else:
                        fail_count += 1
                        pbar.write(f"❌ {result.case_name}: {result.error_msg}")
                
                except Exception as e:
                    fail_count += 1
                    pbar.write(f"❌ {case_name}: 处理异常 - {e}")
                
                pbar.update(1)
    
    return {
        'success': success_count,
        'skip': skip_count,
        'fail': fail_count
    }


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="自动扫描并转换血流动力学数据为 H5 格式（并行加速版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir", 
        type=Path, 
        default=Path("./vtk_data"),
        help="输入数据根目录"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("./h5_data"),
        help="输出 H5 文件目录"
    )
    parser.add_argument(
        "--time-start", 
        type=float, 
        default=4.01,
        help="起始时间"
    )
    parser.add_argument(
        "--time-end", 
        type=float, 
        default=5.00,
        help="结束时间"
    )
    parser.add_argument(
        "--time-step", 
        type=float, 
        default=0.01,
        help="时间步长"
    )
    parser.add_argument(
        "--match-tol", 
        type=float, 
        default=1e-6,
        help="边界匹配容差"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="并行进程数（默认: CPU核心数-2）"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已存在的 H5 文件"
    )
    
    args = parser.parse_args()
    
    # 确定工作进程数
    # if args.num_workers is None:
    #     num_workers = max(1, mp.cpu_count() - 2)
    # else:
    #     num_workers = max(1, min(args.num_workers, mp.cpu_count()))
    num_workers = 16
    
    print(f"{'='*70}")
    print(f"血流动力学数据预处理 - H5 转换（并行加速版）")
    print(f"{'='*70}")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"时间范围: {args.time_start:.2f} 到 {args.time_end:.2f} (步长 {args.time_step:.2f})")
    print(f"CPU核心数: {mp.cpu_count()}, 使用进程数: {num_workers}")
    print(f"{'='*70}\n")
    
    # 发现所有有效 case
    valid_cases = discover_cases(
        args.input_dir, 
        args.time_start, 
        args.time_end, 
        args.time_step
    )
    
    if not valid_cases:
        print("❌ 未找到任何有效 case，程序退出")
        return
    
    # 并行处理
    import time
    start_time = time.time()
    
    stats = process_cases_parallel(
        valid_cases,
        args.output_dir,
        args.match_tol,
        num_workers,
        args.skip_existing
    )
    
    elapsed_time = time.time() - start_time
    
    # 最终汇总
    print(f"\n{'='*70}")
    print(f"处理完成")
    print(f"{'='*70}")
    print(f"✅ 成功: {stats['success']}")
    print(f"⏭️  跳过: {stats['skip']}")
    print(f"❌ 失败: {stats['fail']}")
    print(f"⏱️  总耗时: {elapsed_time/60:.2f} 分钟")
    if stats['success'] > 0:
        print(f"⚡ 平均速度: {elapsed_time/stats['success']:.2f} 秒/case")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # 必须在主模块保护下
    mp.set_start_method('spawn', force=True)
    main()