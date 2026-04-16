"""
改进的时序 DeepONet 训练脚本 (V2)

核心改进:
1. 使用 HistoryEncoder 编码历史帧信息
2. 移除了不必要的 bc，边界条件隐含在历史帧中
3. Trunk 与历史信息交互
4. 完善的 checkpoint 管理和可视化

用法:
    # 训练全部变量
    python train_v2.py --output_vars all --epochs 100

    # 只训练 WSS (使用独立壁面采样)
    python train_v2.py --output_vars wss --num_wall_samples 1000 --epochs 100

    # 使用 Transformer 历史编码器 (更强但更慢)
    python train_v2.py --history_encoder transformer --epochs 100
"""
import argparse
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

from transient.models import TemporalDeepONetV2
from transient.dataset import (
    TemporalCFDDataset, SyntheticTemporalDataset, collate_temporal,
    parse_output_vars
)

torch.manual_seed(42)
np.random.seed(42)

# 优化 CUDA 性能 (保持 FP32 精度)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
    # 禁用 TF32 以保持完整 FP32 精度
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class DataPrefetcher:
    """
    数据预取器 - 在 GPU 上异步预加载下一个 batch
    显著减少 CPU-GPU 数据传输等待时间
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None

    def __iter__(self):
        self.iter = iter(self.loader)
        if self.stream:
            self.preload()
        return self

    def preload(self):
        try:
            self.next_batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = self._to_device(self.next_batch)

    def _to_device(self, batch):
        """将 batch 移到 GPU"""
        return tuple(
            t.to(self.device, non_blocking=True) if torch.is_tensor(t) else t
            for t in batch
        )

    def __next__(self):
        if self.stream:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = self.next_batch
            if batch is None:
                raise StopIteration
            self.preload()
            return batch
        else:
            return self._to_device(next(self.iter))

    def __len__(self):
        return len(self.loader)


def add_argument():
    parser = argparse.ArgumentParser(description="Temporal DeepONet V2 Training")

    # 数据参数
    parser.add_argument("--h5_path", default="kdd.h5", type=str)
    parser.add_argument("--nii_path", default="200.nii.gz", type=str)
    parser.add_argument("--use_synthetic", action="store_true")

    # 时序参数
    parser.add_argument("--input_steps", default=2, type=int, help="输入帧数")
    parser.add_argument("--output_steps", default=1, type=int, help="预测帧数")
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--train_ratio", default=0.8, type=float)

    # 采样参数
    parser.add_argument("--num_samples", default=1000, type=int, help="空间采样点数")
    parser.add_argument("--num_wall_samples", default=None, type=int, help="WSS 壁面采样点数")

    # 输出变量
    parser.add_argument("--output_vars", default="all", type=str,
                        help="输出变量: p / wss / velocity / all")
    parser.add_argument("--boundary_cut", default=0.0, type=float)

    # 模型参数
    parser.add_argument("--history_encoder", default="light", type=str,
                        choices=["light", "transformer"], help="历史编码器类型")
    parser.add_argument("--history_embed_dim", default=128, type=int)
    parser.add_argument("--history_num_layers", default=3, type=int)
    parser.add_argument("--swin_embed_dim", default=24, type=int)
    parser.add_argument("--trunk_hidden_dim", default=128, type=int)
    parser.add_argument("--trunk_num_layers", default=4, type=int)
    parser.add_argument("--branch_dim", default=256, type=int)
    parser.add_argument("--use_cross_attention", action="store_true", default=True)
    parser.add_argument("--no_cross_attention", action="store_true")

    # 消融实验
    parser.add_argument("--use_geometry", action="store_true", default=True,
                        help="使用 Swin 几何编码器 (默认开启)")
    parser.add_argument("--no_geometry", action="store_true",
                        help="[消融] 禁用 Swin 几何编码器")

    # 训练参数
    parser.add_argument("-e", "--epochs", default=10000, type=int)
    parser.add_argument("-b", "--batch_size", default=1, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--accum_steps", default=4, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # Checkpoint 参数
    parser.add_argument("--save_interval", default=100, type=int, help="每隔多少 epoch 保存一次")
    parser.add_argument("--checkpoint_dir", default="checkpoint/v2", type=str)
    parser.add_argument("--resume", default=None, type=str)

    # 数据加载参数
    parser.add_argument("--num_workers", default=4, type=int, help="数据加载线程数")
    parser.add_argument("--prefetch", action="store_true", default=True, help="使用数据预取")
    parser.add_argument("--no_prefetch", action="store_true")
    parser.add_argument("--cache_mode", default="ram", choices=["ram", "none"],
                        help="缓存模式: ram=预加载到内存, none=不缓存")

    # 可视化参数
    parser.add_argument("--vis_interval", default=100, type=int, help="每隔多少 epoch 可视化一次")
    parser.add_argument("--num_vis_samples", default=3, type=int, help="可视化样本数")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.no_cross_attention:
        args.use_cross_attention = False

    if args.no_prefetch:
        args.prefetch = False

    if args.no_geometry:
        args.use_geometry = False

    return args


def setup_logger(save_folder):
    logger = logging.getLogger("DeepONetV2")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)

    file_handler = logging.FileHandler(f"{save_folder}/training.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    return logger


class LossV2(nn.Module):
    """
    V2 损失函数 - 适配 TemporalDeepONetV2
    """
    def __init__(self, output_vars, device='cuda'):
        super(LossV2, self).__init__()
        self.output_vars = output_vars
        self.num_outputs = len(output_vars)
        self.device = device

        # 变量索引
        self.var_to_idx = {var: i for i, var in enumerate(output_vars)}
        self.has_p = 'p' in output_vars
        self.has_u = 'u' in output_vars
        self.has_v = 'v' in output_vars
        self.has_w = 'w' in output_vars
        self.has_wss = 'wss' in output_vars

    def forward(self, y_pred, y_true, wall_mask=None,
                y_pred_wall=None, y_true_wall=None):
        """
        Args:
            y_pred: [B, N, num_outputs] - 预测值
            y_true: [B, N, num_outputs] - 真值
            wall_mask: [B, N] - 壁面掩码
            y_pred_wall: [B, M, 1] - 独立壁面 WSS 预测 (可选)
            y_true_wall: [B, M, 1] - 独立壁面 WSS 真值 (可选)
        """
        losses = {}
        zero = torch.tensor(0.0, device=y_pred.device)

        # ========== 压力损失 ==========
        if self.has_p:
            idx = self.var_to_idx['p']
            p_pred = y_pred[..., idx]
            p_true = y_true[..., idx]

            # 相对压力 (减去均值)
            p_pred_rel = p_pred - p_pred.mean(dim=1, keepdim=True)
            p_true_rel = p_true - p_true.mean(dim=1, keepdim=True)

            losses['mse_p'] = F.mse_loss(p_pred_rel, p_true_rel, reduction='sum')
            losses['l2_p'] = torch.sqrt(
                torch.sum((p_pred_rel - p_true_rel) ** 2) /
                (torch.sum(p_true_rel ** 2) + 1e-8)
            )

            # 压差
            p_range_pred = p_pred.max(dim=1)[0] - p_pred.min(dim=1)[0]
            p_range_true = p_true.max(dim=1)[0] - p_true.min(dim=1)[0]
            losses['dp_mae'] = F.l1_loss(p_range_pred, p_range_true)
        else:
            losses['mse_p'] = losses['l2_p'] = losses['dp_mae'] = zero

        # ========== 速度损失 ==========
        vel_mse = zero
        for var in ['u', 'v', 'w']:
            if var in self.var_to_idx:
                idx = self.var_to_idx[var]
                v_pred = y_pred[..., idx]
                v_true = y_true[..., idx]

                losses[f'mse_{var}'] = F.mse_loss(v_pred, v_true, reduction='sum')
                losses[f'l2_{var}'] = torch.sqrt(
                    torch.sum((v_pred - v_true) ** 2) /
                    (torch.sum(v_true ** 2) + 1e-8)
                )
                vel_mse = vel_mse + losses[f'mse_{var}']
            else:
                losses[f'mse_{var}'] = losses[f'l2_{var}'] = zero

        losses['mse_vel'] = vel_mse

        # ========== WSS 损失 ==========
        if self.has_wss:
            # 优先使用独立壁面采样
            if y_pred_wall is not None and y_true_wall is not None:
                wss_pred = y_pred_wall.squeeze(-1)
                wss_true = y_true_wall.squeeze(-1)

                losses['mse_wss'] = F.mse_loss(wss_pred, wss_true, reduction='sum')
                losses['l2_wss'] = torch.sqrt(
                    torch.sum((wss_pred - wss_true) ** 2) /
                    (torch.sum(wss_true ** 2) + 1e-8)
                )
            else:
                # 使用 wall_mask
                idx = self.var_to_idx['wss']
                wss_pred = y_pred[..., idx]
                wss_true = y_true[..., idx]

                if wall_mask is not None:
                    wss_diff = (wss_pred - wss_true) * wall_mask
                    losses['mse_wss'] = torch.sum(wss_diff ** 2)
                    losses['l2_wss'] = torch.sqrt(
                        torch.sum(wss_diff ** 2) /
                        (torch.sum((wss_true * wall_mask) ** 2) + 1e-8)
                    )
                else:
                    losses['mse_wss'] = F.mse_loss(wss_pred, wss_true, reduction='sum')
                    losses['l2_wss'] = torch.sqrt(
                        torch.sum((wss_pred - wss_true) ** 2) /
                        (torch.sum(wss_true ** 2) + 1e-8)
                    )
        else:
            losses['mse_wss'] = losses['l2_wss'] = zero

        return losses


class CheckpointManager:
    """Checkpoint 管理器"""
    def __init__(self, save_dir, max_keep=5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.best_metric = float('inf')
        self.checkpoint_list = []

    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, args, is_best=False):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'args': vars(args),
            'best_metric': self.best_metric
        }

        # 保存周期性 checkpoint
        if epoch % args.save_interval == 0:
            path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, path)
            self.checkpoint_list.append(path)
            print(f"Saved checkpoint: {path}")

            # 删除旧的 checkpoint (保留最近 max_keep 个)
            while len(self.checkpoint_list) > self.max_keep:
                old_path = self.checkpoint_list.pop(0)
                if old_path.exists() and 'best' not in str(old_path):
                    old_path.unlink()

        # 保存最新的 checkpoint (用于恢复训练)
        latest_path = self.save_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        # 保存最佳模型 (单独保存)
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path} (metric: {metrics.get('val_l2_total', 0):.4f})")

    def load_checkpoint(self, model, optimizer=None, scheduler=None, path=None):
        """加载 checkpoint"""
        if path is None:
            path = self.save_dir / "checkpoint_latest.pt"

        if not Path(path).exists():
            print(f"No checkpoint found at {path}")
            return 0

        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if scheduler and checkpoint.get('scheduler'):
            scheduler.load_state_dict(checkpoint['scheduler'])

        self.best_metric = checkpoint.get('best_metric', float('inf'))
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']


class Visualizer:
    """3D CFD 可视化器"""
    def __init__(self, save_dir, output_vars):
        self.save_dir = Path(save_dir) / "visualizations"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.output_vars = output_vars
        self.var_to_idx = {var: i for i, var in enumerate(output_vars)}

    def visualize_prediction(self, coords, y_pred, y_true, epoch, sample_idx=0,
                             wall_mask=None, writer=None):
        """
        可视化预测结果

        Args:
            coords: [N, 3] - 空间坐标
            y_pred: [N, num_outputs] - 预测值
            y_true: [N, num_outputs] - 真值
            epoch: 当前 epoch
            sample_idx: 样本索引
            wall_mask: [N] - 壁面掩码
            writer: TensorBoard writer
        """
        coords = coords.cpu().numpy() if torch.is_tensor(coords) else coords
        y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true

        # 为每个变量创建可视化
        for var in self.output_vars:
            idx = self.var_to_idx[var]
            pred = y_pred[:, idx]
            true = y_true[:, idx]
            error = np.abs(pred - true)

            # 创建 2D 切片图 (沿 z 轴取中间切片)
            fig = self._create_slice_plot(coords, pred, true, error, var)

            # 保存图片
            fig_path = self.save_dir / f"epoch_{epoch}_sample_{sample_idx}_{var}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')

            # 写入 TensorBoard
            if writer is not None:
                writer.add_figure(f'Prediction/{var}', fig, epoch)

            plt.close(fig)

        # 创建 3D 散点图 (可选，耗时较长)
        if len(self.output_vars) > 0:
            fig_3d = self._create_3d_scatter(coords, y_pred, y_true, self.output_vars[0])
            fig_3d_path = self.save_dir / f"epoch_{epoch}_sample_{sample_idx}_3d.png"
            fig_3d.savefig(fig_3d_path, dpi=150, bbox_inches='tight')
            if writer is not None:
                writer.add_figure('Prediction/3D', fig_3d, epoch)
            plt.close(fig_3d)

    def _create_slice_plot(self, coords, pred, true, error, var_name):
        """创建 2D 切片图"""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 找到 z 轴中间位置的点
        z_vals = coords[:, 2]
        z_mid = (z_vals.max() + z_vals.min()) / 2
        z_tol = (z_vals.max() - z_vals.min()) * 0.1  # 10% 容差
        slice_mask = np.abs(z_vals - z_mid) < z_tol

        if slice_mask.sum() < 10:
            # 如果中间切片点太少，使用所有点的投影
            slice_mask = np.ones(len(coords), dtype=bool)

        x_slice = coords[slice_mask, 0]
        y_slice = coords[slice_mask, 1]
        pred_slice = pred[slice_mask]
        true_slice = true[slice_mask]
        error_slice = error[slice_mask]

        # 预测值
        vmin, vmax = true_slice.min(), true_slice.max()
        sc1 = axes[0].scatter(x_slice, y_slice, c=pred_slice, s=2, cmap='jet',
                              vmin=vmin, vmax=vmax)
        axes[0].set_title(f'{var_name} Predicted')
        plt.colorbar(sc1, ax=axes[0])

        # 真值
        sc2 = axes[1].scatter(x_slice, y_slice, c=true_slice, s=2, cmap='jet',
                              vmin=vmin, vmax=vmax)
        axes[1].set_title(f'{var_name} Ground Truth')
        plt.colorbar(sc2, ax=axes[1])

        # 误差
        sc3 = axes[2].scatter(x_slice, y_slice, c=error_slice, s=2, cmap='hot')
        axes[2].set_title(f'{var_name} Error')
        plt.colorbar(sc3, ax=axes[2])

        # 散点图对比
        axes[3].scatter(true_slice, pred_slice, s=1, alpha=0.5)
        axes[3].plot([vmin, vmax], [vmin, vmax], 'r--', label='y=x')
        axes[3].set_xlabel('Ground Truth')
        axes[3].set_ylabel('Predicted')
        axes[3].set_title(f'{var_name} Correlation')
        axes[3].legend()

        for ax in axes[:3]:
            ax.set_aspect('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        plt.tight_layout()
        return fig

    def _create_3d_scatter(self, coords, y_pred, y_true, var_name):
        """创建 3D 散点图"""
        fig = plt.figure(figsize=(12, 5))

        idx = self.var_to_idx[var_name]
        pred = y_pred[:, idx]
        true = y_true[:, idx]

        vmin, vmax = true.min(), true.max()

        # 下采样以加速绘图
        n_points = len(coords)
        if n_points > 5000:
            sample_idx = np.random.choice(n_points, 5000, replace=False)
            coords = coords[sample_idx]
            pred = pred[sample_idx]
            true = true[sample_idx]

        # 预测值
        ax1 = fig.add_subplot(121, projection='3d')
        sc1 = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                          c=pred, s=1, cmap='jet', vmin=vmin, vmax=vmax)
        ax1.set_title(f'{var_name} Predicted')
        plt.colorbar(sc1, ax=ax1, shrink=0.5)

        # 真值
        ax2 = fig.add_subplot(122, projection='3d')
        sc2 = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                          c=true, s=1, cmap='jet', vmin=vmin, vmax=vmax)
        ax2.set_title(f'{var_name} Ground Truth')
        plt.colorbar(sc2, ax=ax2, shrink=0.5)

        plt.tight_layout()
        return fig

    def export_vtk(self, coords, y_pred, y_true, epoch, sample_idx=0):
        """
        导出 VTK 文件用于 ParaView 可视化

        需要安装: pip install pyvista
        """
        try:
            import pyvista as pv

            coords = coords.cpu().numpy() if torch.is_tensor(coords) else coords
            y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
            y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true

            # 创建点云
            cloud = pv.PolyData(coords)

            # 添加变量
            for var in self.output_vars:
                idx = self.var_to_idx[var]
                cloud[f'{var}_pred'] = y_pred[:, idx]
                cloud[f'{var}_true'] = y_true[:, idx]
                cloud[f'{var}_error'] = np.abs(y_pred[:, idx] - y_true[:, idx])

            # 保存
            vtk_path = self.save_dir / f"epoch_{epoch}_sample_{sample_idx}.vtp"
            cloud.save(str(vtk_path))
            print(f"Exported VTK: {vtk_path}")

        except ImportError:
            print("PyVista not installed. Skip VTK export. Install with: pip install pyvista")


def train_epoch(model, loss_fn, optimizer, train_loader, device, epoch, args, scaler=None):
    model.train()

    metrics_sum = {
        'mse_p': 0., 'mse_vel': 0., 'mse_wss': 0.,
        'l2_p': 0., 'l2_u': 0., 'l2_v': 0., 'l2_w': 0., 'l2_wss': 0.,
        'dp_mae': 0.
    }
    num_batches = len(train_loader)

    # 损失权重
    w_p, w_vel, w_wss = 1.0, 1.0, 10.0  # WSS 给更高权重

    use_amp = args.fp16 or args.bf16
    amp_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)

    # 使用数据预取器
    if args.prefetch and device.type == 'cuda':
        data_iter = DataPrefetcher(train_loader, device)
    else:
        data_iter = train_loader

    pbar = tqdm(enumerate(data_iter), total=num_batches, desc=f"Epoch {epoch}")
    optimizer.zero_grad()

    for batch_idx, batch_data in pbar:
        # 解包数据 (如果使用预取器，数据已经在 GPU 上)
        if len(batch_data) == 9:
            x_in, y_in, x_out, y_out, geo, bc, wall_mask, x_wall, y_wall = batch_data
            if not args.prefetch:
                x_wall = x_wall.to(device, non_blocking=True)
                y_wall = y_wall.to(device, non_blocking=True)
        else:
            x_in, y_in, x_out, y_out, geo, bc, wall_mask = batch_data
            x_wall, y_wall = None, None

        if not args.prefetch:
            x_in = x_in.to(device, non_blocking=True)
            y_in = y_in.to(device, non_blocking=True)
            x_out = x_out.to(device, non_blocking=True)
            y_out = y_out.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            wall_mask = wall_mask.to(device, non_blocking=True)

        # 前向传播
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                y_pred = model(x_in, y_in, x_out, geo)

                # 独立壁面采样的 WSS 预测
                y_pred_wall = None
                if x_wall is not None:
                    y_pred_wall = model(x_in, y_in, x_wall, geo)
                    # 只取 WSS 维度
                    if loss_fn.has_wss:
                        wss_idx = loss_fn.var_to_idx['wss']
                        y_pred_wall = y_pred_wall[..., wss_idx:wss_idx+1]

                losses = loss_fn(y_pred, y_out, wall_mask,
                                y_pred_wall=y_pred_wall, y_true_wall=y_wall)

                num_points = x_out.shape[1]
                total_loss = (w_p * losses['mse_p'] + w_vel * losses['mse_vel'] +
                             w_wss * losses['mse_wss']) / num_points / args.accum_steps

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            y_pred = model(x_in, y_in, x_out, geo)

            y_pred_wall = None
            if x_wall is not None:
                y_pred_wall = model(x_in, y_in, x_wall, geo)
                if loss_fn.has_wss:
                    wss_idx = loss_fn.var_to_idx['wss']
                    y_pred_wall = y_pred_wall[..., wss_idx:wss_idx+1]

            losses = loss_fn(y_pred, y_out, wall_mask,
                            y_pred_wall=y_pred_wall, y_true_wall=y_wall)

            num_points = x_out.shape[1]
            total_loss = (w_p * losses['mse_p'] + w_vel * losses['mse_vel'] +
                         w_wss * losses['mse_wss']) / num_points / args.accum_steps

            total_loss.backward()

            if (batch_idx + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # 累积指标
        for k, v in losses.items():
            if k in metrics_sum:
                metrics_sum[k] += v.item()

        pbar.set_postfix({
            'loss': f'{total_loss.item() * args.accum_steps:.4f}',
            'l2_p': f'{losses["l2_p"].item():.4f}',
            'l2_wss': f'{losses["l2_wss"].item():.4f}'
        })

    return {k: v / num_batches for k, v in metrics_sum.items()}


@torch.no_grad()
def validate(model, loss_fn, val_loader, device, args=None, return_predictions=False):
    """
    验证函数

    Args:
        return_predictions: 是否返回预测结果用于可视化
    """
    model.eval()

    metrics_sum = {
        'mse_p': 0., 'mse_vel': 0., 'mse_wss': 0.,
        'l2_p': 0., 'l2_u': 0., 'l2_v': 0., 'l2_w': 0., 'l2_wss': 0.,
        'dp_mae': 0.
    }
    num_batches = len(val_loader)

    predictions = [] if return_predictions else None

    for batch_idx, batch_data in enumerate(val_loader):
        if len(batch_data) == 9:
            x_in, y_in, x_out, y_out, geo, bc, wall_mask, x_wall, y_wall = batch_data
            x_wall = x_wall.to(device, non_blocking=True)
            y_wall = y_wall.to(device, non_blocking=True)
        else:
            x_in, y_in, x_out, y_out, geo, bc, wall_mask = batch_data
            x_wall, y_wall = None, None

        x_in = x_in.to(device, non_blocking=True)
        y_in = y_in.to(device, non_blocking=True)
        x_out = x_out.to(device, non_blocking=True)
        y_out = y_out.to(device, non_blocking=True)
        geo = geo.to(device, non_blocking=True)
        wall_mask = wall_mask.to(device, non_blocking=True)

        y_pred = model(x_in, y_in, x_out, geo)

        y_pred_wall = None
        if x_wall is not None:
            y_pred_wall = model(x_in, y_in, x_wall, geo)
            if loss_fn.has_wss:
                wss_idx = loss_fn.var_to_idx['wss']
                y_pred_wall = y_pred_wall[..., wss_idx:wss_idx+1]

        losses = loss_fn(y_pred, y_out, wall_mask,
                        y_pred_wall=y_pred_wall, y_true_wall=y_wall)

        for k, v in losses.items():
            if k in metrics_sum:
                metrics_sum[k] += v.item()

        # 保存预测结果用于可视化
        if return_predictions and batch_idx < (args.num_vis_samples if args else 3):
            predictions.append({
                'coords': x_out[0, :, :3].cpu(),  # 取第一个 batch 的坐标
                'y_pred': y_pred[0].cpu(),
                'y_true': y_out[0].cpu(),
                'wall_mask': wall_mask[0].cpu() if wall_mask is not None else None
            })

    metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    if return_predictions:
        return metrics, predictions
    return metrics


def main():
    args = add_argument()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logger(args.checkpoint_dir)
    logger.info(f"Arguments: {args}")

    # 解析输出变量
    output_vars = parse_output_vars(args.output_vars)
    num_outputs = len(output_vars)
    logger.info(f"Output variables: {output_vars} ({num_outputs} dims)")

    # 创建数据集
    h5_path = Path(args.h5_path)
    nii_path = Path(args.nii_path)

    if args.use_synthetic or not (h5_path.exists() and nii_path.exists()):
        logger.info("Using synthetic dataset")
        train_dataset = SyntheticTemporalDataset(
            num_samples=100,
            num_spatial_points=args.num_samples,
            input_steps=args.input_steps,
            output_steps=args.output_steps,
            output_vars=args.output_vars
        )
        val_dataset = SyntheticTemporalDataset(
            num_samples=20,
            num_spatial_points=args.num_samples,
            input_steps=args.input_steps,
            output_steps=args.output_steps,
            output_vars=args.output_vars
        )
        geometry_shape = (1, 32, 32, 32)
    else:
        logger.info(f"Using real dataset from {h5_path}")
        train_dataset = TemporalCFDDataset(
            h5_path=str(h5_path),
            nii_path=str(nii_path),
            input_steps=args.input_steps,
            output_steps=args.output_steps,
            stride=args.stride,
            num_spatial_samples=args.num_samples,
            num_wall_samples=args.num_wall_samples,
            mode='train',
            train_ratio=args.train_ratio,
            output_vars=args.output_vars,
            boundary_cut=args.boundary_cut,
            cache_mode=args.cache_mode
        )
        val_dataset = TemporalCFDDataset(
            h5_path=str(h5_path),
            nii_path=str(nii_path),
            input_steps=args.input_steps,
            output_steps=args.output_steps,
            stride=args.stride,
            num_spatial_samples=args.num_samples,
            num_wall_samples=args.num_wall_samples,
            mode='test',
            train_ratio=args.train_ratio,
            output_vars=args.output_vars,
            boundary_cut=args.boundary_cut,
            cache_mode=args.cache_mode
        )
        geometry_shape = train_dataset.geometry.shape

    # 如果使用 RAM 缓存，不需要多线程加载
    actual_workers = 0 if args.cache_mode == 'ram' else args.num_workers
    if args.cache_mode == 'ram':
        logger.info("Using RAM cache, setting num_workers=0")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=actual_workers, pin_memory=True, collate_fn=collate_temporal,
        persistent_workers=(actual_workers > 0),
        prefetch_factor=2 if actual_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=actual_workers, pin_memory=True, collate_fn=collate_temporal,
        persistent_workers=(actual_workers > 0),
        prefetch_factor=2 if actual_workers > 0 else None
    )

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # 创建模型
    model = TemporalDeepONetV2(
        num_input_vars=num_outputs,
        num_output_vars=num_outputs,
        history_embed_dim=args.history_embed_dim,
        history_encoder_type=args.history_encoder,
        history_num_layers=args.history_num_layers,
        swin_embed_dim=args.swin_embed_dim,
        use_geometry=args.use_geometry,  # 消融实验
        trunk_hidden_dim=args.trunk_hidden_dim,
        trunk_num_layers=args.trunk_num_layers,
        use_cross_attention=args.use_cross_attention,
        branch_dim=args.branch_dim
    ).to(device)

    if not args.use_geometry:
        logger.info("[Ablation] Geometry encoder (Swin) DISABLED")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # 损失函数
    loss_fn = LossV2(output_vars, device=device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Checkpoint 管理器
    ckpt_manager = CheckpointManager(args.checkpoint_dir, max_keep=5)

    # 可视化器
    visualizer = Visualizer(args.checkpoint_dir, output_vars)

    # 恢复训练
    start_epoch = 1
    if args.resume:
        start_epoch = ckpt_manager.load_checkpoint(
            model, optimizer, scheduler, args.resume
        ) + 1

    # 混合精度
    use_amp = args.fp16 or args.bf16
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == 'cuda') else None

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'tensorboard'))

    # 训练循环
    best_val_l2 = ckpt_manager.best_metric

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(model, loss_fn, optimizer, train_loader,
                                     device, epoch, args, scaler)

        # 是否需要可视化
        need_vis = (epoch % args.vis_interval == 0) or (epoch == args.epochs)
        if need_vis:
            val_metrics, predictions = validate(
                model, loss_fn, val_loader, device, args,
                return_predictions=True
            )
        else:
            val_metrics = validate(
                model, loss_fn, val_loader, device, args,
                return_predictions=False
            )
            predictions = None

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # 记录到 TensorBoard
        writer.add_scalars('L2/Pressure', {
            'train': train_metrics['l2_p'], 'val': val_metrics['l2_p']
        }, epoch)
        writer.add_scalars('L2/WSS', {
            'train': train_metrics['l2_wss'], 'val': val_metrics['l2_wss']
        }, epoch)
        for var in ['u', 'v', 'w']:
            if f'l2_{var}' in train_metrics:
                writer.add_scalars(f'L2/Velocity_{var}', {
                    'train': train_metrics[f'l2_{var}'],
                    'val': val_metrics[f'l2_{var}']
                }, epoch)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('Loss/dp_mae', val_metrics['dp_mae'], epoch)

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train: L2_P={train_metrics['l2_p']:.4f}, L2_WSS={train_metrics['l2_wss']:.4f} | "
            f"Val: L2_P={val_metrics['l2_p']:.4f}, L2_WSS={val_metrics['l2_wss']:.4f} | "
            f"Time={epoch_time:.1f}s"
        )

        # 可视化
        if need_vis and predictions:
            for i, pred_data in enumerate(predictions):
                visualizer.visualize_prediction(
                    pred_data['coords'],
                    pred_data['y_pred'],
                    pred_data['y_true'],
                    epoch,
                    sample_idx=i,
                    wall_mask=pred_data['wall_mask'],
                    writer=writer
                )
                # 导出 VTK (可选)
                if epoch == args.epochs:  # 最后一个 epoch 导出 VTK
                    visualizer.export_vtk(
                        pred_data['coords'],
                        pred_data['y_pred'],
                        pred_data['y_true'],
                        epoch,
                        sample_idx=i
                    )

        # 保存 checkpoint
        val_l2_total = val_metrics['l2_p'] + val_metrics['l2_wss']
        is_best = val_l2_total < best_val_l2
        if is_best:
            best_val_l2 = val_l2_total
            ckpt_manager.best_metric = best_val_l2

        metrics_to_save = {
            **val_metrics,
            'val_l2_total': val_l2_total
        }
        ckpt_manager.save_checkpoint(
            model, optimizer, scheduler, epoch, metrics_to_save, args, is_best
        )

        if is_best:
            logger.info(f"New best model! L2_total={val_l2_total:.4f}")

    writer.close()
    logger.info("Training completed!")
    logger.info(f"Best validation L2: {best_val_l2:.4f}")
    logger.info(f"Best model saved at: {args.checkpoint_dir}/best_model.pt")


if __name__ == '__main__':
    main()
