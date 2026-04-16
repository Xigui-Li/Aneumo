"""
时序 DeepONet V2 测试脚本

功能:
1. 加载训练好的模型
2. 在测试集上评估
3. 生成可视化结果
4. 导出 VTK 文件用于 ParaView

用法:
    # 基本测试
    python test_v2.py --checkpoint checkpoint/v2/best_model.pt

    # 指定测试数据
    python test_v2.py --checkpoint checkpoint/v2/best_model.pt --h5_path test_data.h5

    # 导出所有预测结果
    python test_v2.py --checkpoint checkpoint/v2/best_model.pt --export_all
"""
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from transient.models import TemporalDeepONetV2
from transient.dataset import (
    TemporalCFDDataset, collate_temporal, parse_output_vars
)


def add_argument():
    parser = argparse.ArgumentParser(description="Temporal DeepONet V2 Testing")

    # 必需参数
    parser.add_argument("--checkpoint", required=True, type=str, help="模型 checkpoint 路径")

    # 数据参数
    parser.add_argument("--h5_path", default="kdd.h5", type=str)
    parser.add_argument("--nii_path", default="200.nii.gz", type=str)

    # 测试参数
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_samples", default=5000, type=int, help="测试时采样更多点以获得更准确的评估")
    parser.add_argument("--num_wall_samples", default=None, type=int)

    # 输出参数
    parser.add_argument("--output_dir", default=None, type=str, help="输出目录，默认为 checkpoint 所在目录")
    parser.add_argument("--export_all", action="store_true", help="导出所有样本的预测结果")
    parser.add_argument("--export_vtk", action="store_true", help="导出 VTK 文件")
    parser.add_argument("--num_vis", default=5, type=int, help="可视化样本数")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


class TestVisualizer:
    """测试可视化器"""
    def __init__(self, save_dir, output_vars):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.output_vars = output_vars
        self.var_to_idx = {var: i for i, var in enumerate(output_vars)}

    def create_detailed_plot(self, coords, y_pred, y_true, sample_idx, time_idx=None):
        """创建详细的对比图"""
        coords = coords.cpu().numpy() if torch.is_tensor(coords) else coords
        y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true

        n_vars = len(self.output_vars)
        fig, axes = plt.subplots(n_vars, 4, figsize=(16, 4 * n_vars))

        if n_vars == 1:
            axes = axes.reshape(1, -1)

        for i, var in enumerate(self.output_vars):
            idx = self.var_to_idx[var]
            pred = y_pred[:, idx]
            true = y_true[:, idx]
            error = np.abs(pred - true)

            # 计算指标
            l2_error = np.sqrt(np.sum((pred - true) ** 2) / (np.sum(true ** 2) + 1e-8))
            mae = np.mean(error)
            max_error = np.max(error)

            vmin, vmax = true.min(), true.max()

            # 预测值
            sc1 = axes[i, 0].scatter(coords[:, 0], coords[:, 1], c=pred, s=1, cmap='jet',
                                      vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f'{var} Predicted')
            plt.colorbar(sc1, ax=axes[i, 0])

            # 真值
            sc2 = axes[i, 1].scatter(coords[:, 0], coords[:, 1], c=true, s=1, cmap='jet',
                                      vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f'{var} Ground Truth')
            plt.colorbar(sc2, ax=axes[i, 1])

            # 误差
            sc3 = axes[i, 2].scatter(coords[:, 0], coords[:, 1], c=error, s=1, cmap='hot')
            axes[i, 2].set_title(f'{var} Error (L2={l2_error:.4f})')
            plt.colorbar(sc3, ax=axes[i, 2])

            # 散点图
            axes[i, 3].scatter(true, pred, s=1, alpha=0.5)
            axes[i, 3].plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=2)
            axes[i, 3].set_xlabel('Ground Truth')
            axes[i, 3].set_ylabel('Predicted')
            axes[i, 3].set_title(f'{var} (MAE={mae:.4f}, Max={max_error:.4f})')

            for ax in axes[i, :3]:
                ax.set_aspect('equal')

        plt.tight_layout()

        # 保存
        suffix = f"_t{time_idx}" if time_idx is not None else ""
        fig_path = self.save_dir / f"test_sample_{sample_idx}{suffix}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {fig_path}")

    def create_error_histogram(self, all_errors, var_name):
        """创建误差直方图"""
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.hist(all_errors, bins=50, density=True, alpha=0.7, color='blue')
        ax.axvline(np.mean(all_errors), color='red', linestyle='--',
                   label=f'Mean: {np.mean(all_errors):.4f}')
        ax.axvline(np.median(all_errors), color='green', linestyle='--',
                   label=f'Median: {np.median(all_errors):.4f}')

        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Density')
        ax.set_title(f'{var_name} Error Distribution')
        ax.legend()

        fig_path = self.save_dir / f"error_histogram_{var_name}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def export_vtk(self, coords, y_pred, y_true, sample_idx):
        """导出 VTK 文件"""
        try:
            import pyvista as pv

            coords = coords.cpu().numpy() if torch.is_tensor(coords) else coords
            y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
            y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true

            cloud = pv.PolyData(coords)

            for var in self.output_vars:
                idx = self.var_to_idx[var]
                cloud[f'{var}_pred'] = y_pred[:, idx]
                cloud[f'{var}_true'] = y_true[:, idx]
                cloud[f'{var}_error'] = np.abs(y_pred[:, idx] - y_true[:, idx])

            vtk_path = self.save_dir / f"test_sample_{sample_idx}.vtp"
            cloud.save(str(vtk_path))
            print(f"Exported VTK: {vtk_path}")

        except ImportError:
            print("PyVista not installed. Skip VTK export.")


def compute_metrics(y_pred, y_true, output_vars, wall_mask=None):
    """计算评估指标"""
    metrics = {}
    var_to_idx = {var: i for i, var in enumerate(output_vars)}

    for var in output_vars:
        idx = var_to_idx[var]
        pred = y_pred[..., idx]
        true = y_true[..., idx]

        if var == 'p':
            # 相对压力
            pred = pred - pred.mean()
            true = true - true.mean()

        if var == 'wss' and wall_mask is not None:
            # WSS 只在壁面计算
            mask = wall_mask > 0.5
            if mask.sum() > 0:
                pred = pred[mask]
                true = true[mask]

        # L2 相对误差
        l2 = np.sqrt(np.sum((pred - true) ** 2) / (np.sum(true ** 2) + 1e-8))
        metrics[f'l2_{var}'] = l2

        # MAE
        mae = np.mean(np.abs(pred - true))
        metrics[f'mae_{var}'] = mae

        # RMSE
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        metrics[f'rmse_{var}'] = rmse

        # Max error
        max_err = np.max(np.abs(pred - true))
        metrics[f'max_{var}'] = max_err

    return metrics


@torch.no_grad()
def test(model, test_loader, device, output_vars, args, visualizer):
    """测试函数"""
    model.eval()

    all_metrics = {f'{m}_{v}': [] for v in output_vars for m in ['l2', 'mae', 'rmse', 'max']}
    all_errors = {var: [] for var in output_vars}

    var_to_idx = {var: i for i, var in enumerate(output_vars)}

    print(f"\nTesting on {len(test_loader)} samples...")

    for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
        # 解包数据
        if len(batch_data) == 9:
            x_in, y_in, x_out, y_out, geo, bc, wall_mask, x_wall, y_wall = batch_data
        else:
            x_in, y_in, x_out, y_out, geo, bc, wall_mask = batch_data
            x_wall, y_wall = None, None

        x_in = x_in.to(device)
        y_in = y_in.to(device)
        x_out = x_out.to(device)
        y_out = y_out.to(device)
        geo = geo.to(device)

        # 预测
        y_pred = model(x_in, y_in, x_out, geo)

        # 转换为 numpy
        y_pred_np = y_pred[0].cpu().numpy()
        y_true_np = y_out[0].cpu().numpy()
        coords_np = x_out[0, :, :3].cpu().numpy()
        wall_mask_np = wall_mask[0].cpu().numpy() if wall_mask is not None else None

        # 计算指标
        metrics = compute_metrics(y_pred_np, y_true_np, output_vars, wall_mask_np)
        for k, v in metrics.items():
            if k in all_metrics:
                all_metrics[k].append(v)

        # 收集误差用于直方图
        for var in output_vars:
            idx = var_to_idx[var]
            error = np.abs(y_pred_np[:, idx] - y_true_np[:, idx])
            all_errors[var].extend(error.tolist())

        # 可视化
        if batch_idx < args.num_vis:
            visualizer.create_detailed_plot(coords_np, y_pred_np, y_true_np, batch_idx)

            if args.export_vtk:
                visualizer.export_vtk(coords_np, y_pred_np, y_true_np, batch_idx)

        # 导出所有结果
        if args.export_all:
            np.savez(
                visualizer.save_dir / f"prediction_{batch_idx}.npz",
                coords=coords_np,
                y_pred=y_pred_np,
                y_true=y_true_np,
                wall_mask=wall_mask_np
            )

    # 汇总指标
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    summary = {}
    for var in output_vars:
        print(f"\n{var.upper()}:")
        for metric in ['l2', 'mae', 'rmse', 'max']:
            key = f'{metric}_{var}'
            values = all_metrics[key]
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary[key] = {'mean': mean_val, 'std': std_val}
            print(f"  {metric.upper():6s}: {mean_val:.6f} +/- {std_val:.6f}")

    # 绘制误差直方图
    for var in output_vars:
        visualizer.create_error_histogram(all_errors[var], var)

    # 保存汇总结果
    summary_path = visualizer.save_dir / "test_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Test Results Summary\n")
        f.write("=" * 60 + "\n\n")
        for var in output_vars:
            f.write(f"{var.upper()}:\n")
            for metric in ['l2', 'mae', 'rmse', 'max']:
                key = f'{metric}_{var}'
                f.write(f"  {metric.upper():6s}: {summary[key]['mean']:.6f} +/- {summary[key]['std']:.6f}\n")
            f.write("\n")

    print(f"\nResults saved to: {visualizer.save_dir}")
    return summary


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

    # 从 checkpoint 恢复参数
    saved_args = checkpoint.get('args', {})
    output_vars = parse_output_vars(saved_args.get('output_vars', 'all'))
    num_outputs = len(output_vars)
    print(f"Output variables: {output_vars}")

    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = checkpoint_path.parent / "test_results"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建数据集
    h5_path = Path(args.h5_path)
    nii_path = Path(args.nii_path)

    if not h5_path.exists() or not nii_path.exists():
        raise FileNotFoundError(f"Data files not found: {h5_path}, {nii_path}")

    # 使用保存的参数
    test_dataset = TemporalCFDDataset(
        h5_path=str(h5_path),
        nii_path=str(nii_path),
        input_steps=saved_args.get('input_steps', 10),
        output_steps=saved_args.get('output_steps', 5),
        stride=saved_args.get('stride', 1),
        num_spatial_samples=args.num_samples,
        num_wall_samples=args.num_wall_samples or saved_args.get('num_wall_samples'),
        mode='test',
        train_ratio=saved_args.get('train_ratio', 0.8),
        output_vars=saved_args.get('output_vars', 'all'),
        boundary_cut=saved_args.get('boundary_cut', 0.0)
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_temporal
    )

    print(f"Test samples: {len(test_dataset)}")

    # 创建模型
    model = TemporalDeepONetV2(
        num_input_vars=num_outputs,
        num_output_vars=num_outputs,
        history_embed_dim=saved_args.get('history_embed_dim', 128),
        history_encoder_type=saved_args.get('history_encoder', 'light'),
        history_num_layers=saved_args.get('history_num_layers', 3),
        swin_embed_dim=saved_args.get('swin_embed_dim', 24),
        trunk_hidden_dim=saved_args.get('trunk_hidden_dim', 128),
        trunk_num_layers=saved_args.get('trunk_num_layers', 4),
        use_cross_attention=saved_args.get('use_cross_attention', True),
        branch_dim=saved_args.get('branch_dim', 256)
    ).to(device)

    # 加载权重
    model.load_state_dict(checkpoint['model'])
    print(f"Model loaded from epoch {checkpoint['epoch']}")

    if 'metrics' in checkpoint:
        print(f"Training metrics: {checkpoint['metrics']}")

    # 可视化器
    visualizer = TestVisualizer(output_dir, output_vars)

    # 运行测试
    summary = test(model, test_loader, device, output_vars, args, visualizer)

    print("\nTest completed!")


if __name__ == '__main__':
    main()
