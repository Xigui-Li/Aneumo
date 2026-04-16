import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(20231028)
np.random.seed(20231028)
torch.cuda.manual_seed(20231028)


################################################################
# Loss Functions for Temporal DeepONet
# 支持自适应输出变量
################################################################


class TemporalDataLoss(nn.Module):
    """
    时序 DeepONet 损失函数
    支持自适应输出变量: p / wss / velocity / 组合

    Args:
        trunk_net, branch_img_net, branch_bc_net, branch_bp_net: 网络组件
        num_outputs: 输出维度数
        output_vars: 输出变量列表，如 ['p'], ['wss'], ['p', 'u', 'v', 'w', 'wss']
    """
    def __init__(self, trunk_net, branch_img_net, branch_bc_net, branch_bp_net,
                 num_outputs=1, output_vars=None):
        super(TemporalDataLoss, self).__init__()
        self.trunk = trunk_net
        self.branch_img = branch_img_net
        self.branch_bc = branch_bc_net
        self.branch_bp = branch_bp_net
        self.num_outputs = num_outputs

        # 默认输出变量
        if output_vars is None:
            output_vars = ['p', 'u', 'v', 'w', 'wss'][:num_outputs]
        self.output_vars = output_vars

        # 创建变量索引映射
        self.var_to_idx = {var: i for i, var in enumerate(output_vars)}

        # 标记哪些变量存在
        self.has_p = 'p' in output_vars
        self.has_u = 'u' in output_vars
        self.has_v = 'v' in output_vars
        self.has_w = 'w' in output_vars
        self.has_wss = 'wss' in output_vars
        self.has_velocity = self.has_u or self.has_v or self.has_w

        print(f"TemporalDataLoss initialized with output_vars: {output_vars}")

    def forward(self, x, y_true, image, bc, wall_mask=None, x_wall=None, y_wall=None):
        """
        x: [B, N, 4] - 空间坐标 + 时间 (x, y, z, t)
        y_true: [B, N, num_outputs] - 真实值
        image: [B, 1, D, H, W] - 3D 几何图像
        bc: [B, bc_dim] - 边界条件
        wall_mask: [B, N] - 壁面掩码 (1=壁面, 0=非壁面)
        x_wall: [B, M, 4] - 独立壁面采样坐标（可选，用于独立壁面采样模式）
        y_wall: [B, M, 1] - 独立壁面采样 WSS 真值（可选）

        Returns:
            tuple: 包含所有损失和度量的元组
        """
        # Branch networks (只计算一次，所有点共享)
        branch_img_pred = self.branch_img(image).unsqueeze(-1)
        branch_bc_pred = self.branch_bc(bc).unsqueeze(-1)
        branch_pred = torch.cat([branch_img_pred, branch_bc_pred], dim=1)
        branch_bp_pred = self.branch_bp(bc[..., -1:])

        # === 流场预测（使用 x） ===
        trunk_outputs = self.trunk(x)

        h_preds = []
        for trunk_out in trunk_outputs:
            h_pred = torch.matmul(trunk_out, branch_pred)
            h_preds.append(h_pred)

        y_pred = torch.cat(h_preds, dim=-1)
        y_pred = y_pred * branch_bp_pred.unsqueeze(1)

        # === 独立壁面点 WSS 预测（如果提供了 x_wall） ===
        wss_pred_wall = None
        if x_wall is not None and y_wall is not None and self.has_wss:
            trunk_outputs_wall = self.trunk(x_wall)

            # 只取 WSS 对应的输出
            wss_idx = self.var_to_idx.get('wss', -1)
            if wss_idx >= 0 and wss_idx < len(trunk_outputs_wall):
                trunk_out_wss = trunk_outputs_wall[wss_idx]
                h_pred_wss = torch.matmul(trunk_out_wss, branch_pred)
                wss_pred_wall = h_pred_wss * branch_bp_pred.unsqueeze(1)[..., wss_idx:wss_idx+1]

        # 计算损失
        return self._compute_losses(y_pred, y_true, wall_mask,
                                    wss_pred_wall=wss_pred_wall, y_wall=y_wall)

    def _compute_losses(self, y_pred, y_true, wall_mask=None,
                        wss_pred_wall=None, y_wall=None):
        """
        计算各种损失和度量
        根据 output_vars 动态计算

        Args:
            y_pred: [B, N, num_outputs] - 流场预测值
            y_true: [B, N, num_outputs] - 流场真值
            wall_mask: [B, N] - 壁面掩码
            wss_pred_wall: [B, M, 1] - 独立壁面点 WSS 预测（可选）
            y_wall: [B, M, 1] - 独立壁面点 WSS 真值（可选）
        """
        losses = {}
        zero = torch.tensor(0.0, device=y_pred.device)

        # =============== 压力损失 (p) ===============
        if self.has_p:
            p_idx = self.var_to_idx['p']
            p_pred = y_pred[..., p_idx]
            p_true = y_true[..., p_idx]

            p_ref_pred = torch.mean(p_pred, dim=1, keepdim=True)
            p_ref_true = torch.mean(p_true, dim=1, keepdim=True)

            p_range_pred = torch.max(p_pred, dim=1)[0] - torch.min(p_pred, dim=1)[0]
            p_range_true = torch.max(p_true, dim=1)[0] - torch.min(p_true, dim=1)[0]

            # MSE
            losses['loss_p'] = torch.sum(torch.square(p_pred - p_ref_pred.squeeze() + p_ref_true.squeeze() - p_true))

            # 相对 L2
            losses['l2_p'] = torch.sqrt(
                torch.sum(torch.square(p_pred - p_ref_pred.squeeze() + p_ref_true.squeeze() - p_true)) /
                (torch.sum(torch.square(p_true - p_ref_true.squeeze())) + 1e-8)
            )

            # MNAE
            losses['mnae_p'] = torch.mean(
                torch.abs(p_pred - p_ref_pred.squeeze() + p_ref_true.squeeze() - p_true) /
                (p_range_true.unsqueeze(1) + 1e-8)
            )

            # 压差
            losses['dp_l2'] = torch.sqrt(
                torch.sum(torch.square(p_range_pred - p_range_true)) /
                (torch.sum(torch.square(p_range_true)) + 1e-8)
            )
            losses['dp_mae'] = torch.mean(torch.abs(p_range_pred - p_range_true))
        else:
            losses['loss_p'] = zero
            losses['l2_p'] = zero
            losses['mnae_p'] = zero
            losses['dp_l2'] = zero
            losses['dp_mae'] = zero

        # =============== 速度损失 (u, v, w) ===============
        if self.has_velocity:
            vel_loss = zero
            for var in ['u', 'v', 'w']:
                if var in self.var_to_idx:
                    idx = self.var_to_idx[var]
                    v_pred = y_pred[..., idx]
                    v_true = y_true[..., idx]

                    v_range = torch.max(v_true, dim=1)[0] - torch.min(v_true, dim=1)[0]

                    losses[f'loss_{var}'] = torch.sum(torch.square(v_pred - v_true))
                    losses[f'l2_{var}'] = torch.sqrt(
                        torch.sum(torch.square(v_pred - v_true)) /
                        (torch.sum(torch.square(v_true)) + 1e-8)
                    )
                    losses[f'mnae_{var}'] = torch.mean(
                        torch.abs(v_pred - v_true) / (v_range.unsqueeze(1) + 1e-8)
                    )
                    vel_loss = vel_loss + losses[f'loss_{var}']
                else:
                    losses[f'loss_{var}'] = zero
                    losses[f'l2_{var}'] = zero
                    losses[f'mnae_{var}'] = zero

            losses['loss_vel'] = vel_loss
        else:
            losses['loss_vel'] = zero
            for var in ['u', 'v', 'w']:
                losses[f'loss_{var}'] = zero
                losses[f'l2_{var}'] = zero
                losses[f'mnae_{var}'] = zero

        # =============== WSS 损失 ===============
        if self.has_wss:
            # 优先使用独立壁面采样数据（如果提供）
            if wss_pred_wall is not None and y_wall is not None:
                # 独立壁面采样模式：直接在壁面点计算 WSS 损失
                wss_pred = wss_pred_wall.squeeze(-1)  # [B, M]
                wss_true = y_wall.squeeze(-1)  # [B, M]

                num_wall_points = wss_true.numel()
                wss_diff = wss_pred - wss_true

                losses['loss_wss'] = torch.sum(torch.square(wss_diff))
                losses['l2_wss'] = torch.sqrt(
                    torch.sum(torch.square(wss_diff)) /
                    (torch.sum(torch.square(wss_true)) + 1e-8)
                )

                wss_range = torch.max(wss_true) - torch.min(wss_true) + 1e-8
                losses['mnae_wss'] = torch.mean(torch.abs(wss_diff) / wss_range)
            else:
                # 共享采样模式：使用 wall_mask 过滤
                wss_idx = self.var_to_idx['wss']
                wss_pred = y_pred[..., wss_idx]
                wss_true = y_true[..., wss_idx]

                if wall_mask is not None:
                    num_wall_points = torch.sum(wall_mask) + 1e-8
                    wss_diff = (wss_pred - wss_true) * wall_mask

                    losses['loss_wss'] = torch.sum(torch.square(wss_diff))
                    losses['l2_wss'] = torch.sqrt(
                        torch.sum(torch.square(wss_diff)) /
                        (torch.sum(torch.square(wss_true * wall_mask)) + 1e-8)
                    )

                    wss_range = torch.max(wss_true * wall_mask) - torch.min(
                        wss_true * wall_mask + (1 - wall_mask) * 1e8
                    )
                    losses['mnae_wss'] = torch.sum(
                        torch.abs(wss_diff)
                    ) / (num_wall_points * (wss_range + 1e-8))
                else:
                    # 没有 wall_mask，在所有非零 WSS 点计算
                    wss_mask = (torch.abs(wss_true) > 1e-8).float()
                    num_wss_points = torch.sum(wss_mask) + 1e-8
                    wss_diff = (wss_pred - wss_true) * wss_mask

                    losses['loss_wss'] = torch.sum(torch.square(wss_diff))
                    losses['l2_wss'] = torch.sqrt(
                        torch.sum(torch.square(wss_diff)) /
                        (torch.sum(torch.square(wss_true * wss_mask)) + 1e-8)
                    )

                    wss_range = torch.max(wss_true * wss_mask) - torch.min(
                        wss_true * wss_mask + (1 - wss_mask) * 1e8
                    )
                    losses['mnae_wss'] = torch.sum(
                        torch.abs(wss_diff)
                    ) / (num_wss_points * (wss_range + 1e-8))
        else:
            losses['loss_wss'] = zero
            losses['l2_wss'] = zero
            losses['mnae_wss'] = zero

        # 返回元组格式以保持兼容性
        # (loss_p, loss_vel, loss_wss, l2_p, l2_u, l2_v, l2_w, l2_wss,
        #  mnae_p, mnae_u, mnae_v, mnae_w, mnae_wss, dp_l2, dp_mnae, dp_mse, dp_mae)
        return (
            losses['loss_p'],
            losses['loss_vel'],
            losses['loss_wss'],
            losses['l2_p'],
            losses['l2_u'],
            losses['l2_v'],
            losses['l2_w'],
            losses['l2_wss'],
            losses['mnae_p'],
            losses['mnae_u'],
            losses['mnae_v'],
            losses['mnae_w'],
            losses['mnae_wss'],
            losses['dp_l2'],
            losses.get('dp_mnae', losses['dp_l2']),  # 使用 dp_l2 替代
            losses.get('dp_mse', losses['dp_mae']),  # 使用 dp_mae 替代
            losses['dp_mae']
        )


class SimpleLoss(nn.Module):
    """
    简化版损失函数 - 只计算基本的 MSE 和相对 L2
    用于单变量预测场景
    """
    def __init__(self, trunk_net, branch_img_net, branch_bc_net, branch_bp_net,
                 num_outputs=1, output_vars=None):
        super(SimpleLoss, self).__init__()
        self.trunk = trunk_net
        self.branch_img = branch_img_net
        self.branch_bc = branch_bc_net
        self.branch_bp = branch_bp_net
        self.num_outputs = num_outputs
        self.output_vars = output_vars or ['p']
        self.has_wss = 'wss' in self.output_vars

    def forward(self, x, y_true, image, bc, wall_mask=None):
        # Trunk network
        trunk_outputs = self.trunk(x)

        # Branch networks
        branch_img_pred = self.branch_img(image).unsqueeze(-1)
        branch_bc_pred = self.branch_bc(bc).unsqueeze(-1)
        branch_pred = torch.cat([branch_img_pred, branch_bc_pred], dim=1)

        # DeepONet: trunk * branch
        h_preds = []
        for trunk_out in trunk_outputs:
            h_pred = torch.matmul(trunk_out, branch_pred)
            h_preds.append(h_pred)

        y_pred = torch.cat(h_preds, dim=-1)

        # Bypass scaling
        branch_bp_pred = self.branch_bp(bc[..., -1:])
        y_pred = y_pred * branch_bp_pred.unsqueeze(1)

        # 计算损失
        if self.has_wss and wall_mask is not None:
            # WSS: 只在壁面计算
            mask = wall_mask.unsqueeze(-1)
            diff = (y_pred - y_true) * mask
            mse = torch.sum(torch.square(diff))
            l2 = torch.sqrt(mse / (torch.sum(torch.square(y_true * mask)) + 1e-8))
        else:
            # 普通变量
            mse = torch.sum(torch.square(y_pred - y_true))
            l2 = torch.sqrt(mse / (torch.sum(torch.square(y_true)) + 1e-8))

        return mse, l2

