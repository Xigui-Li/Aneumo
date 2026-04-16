import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

try:
    from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
    from monai.utils import ensure_tuple_rep
except ImportError:
    SwinViT = None
    ensure_tuple_rep = None

torch.set_default_dtype(torch.float32)
torch.manual_seed(20231028)

################################################################################
# Temporal DeepONet Networks
################################################################################


class HistoryEncoder(nn.Module):
    """
    历史帧编码器 - 用 Transformer 编码前N帧的流场信息

    输入: 前N帧在采样点的场值 (p, u, v, w) 或 (p, u, v, w, wss)
    输出: 历史特征向量，用于条件化预测
    """
    def __init__(self,
                 input_dim=4,           # 输入场的维度 (p, u, v, w) 或 5 (含wss)
                 embed_dim=128,         # 嵌入维度
                 num_heads=4,           # 注意力头数
                 num_layers=3,          # Transformer 层数
                 max_points=2000,       # 最大采样点数
                 dropout=0.1):
        super(HistoryEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.input_dim = input_dim

        # 输入投影: (x, y, z, t, p, u, v, w) -> embed_dim
        # 坐标 4 + 场值 input_dim
        self.input_proj = nn.Linear(4 + input_dim, embed_dim)

        # 可学习的聚合 token (类似 CLS token)
        self.aggregate_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x_hist, y_hist):
        """
        Args:
            x_hist: [B, T_in * N_s, 4] - 历史帧的坐标 (x, y, z, t)
            y_hist: [B, T_in * N_s, input_dim] - 历史帧的场值 (p, u, v, w, ...)

        Returns:
            history_embedding: [B, embed_dim] - 历史特征向量
        """
        B = x_hist.shape[0]

        # 拼接坐标和场值
        combined = torch.cat([x_hist, y_hist], dim=-1)  # [B, T*N, 4+input_dim]

        # 投影到嵌入空间
        tokens = self.input_proj(combined)  # [B, T*N, embed_dim]

        # 添加聚合 token
        agg_token = self.aggregate_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        tokens = torch.cat([agg_token, tokens], dim=1)  # [B, 1+T*N, embed_dim]

        # Transformer 编码
        encoded = self.transformer(tokens)  # [B, 1+T*N, embed_dim]

        # 取聚合 token 的输出
        history_embedding = encoded[:, 0, :]  # [B, embed_dim]

        # 输出投影
        history_embedding = self.output_proj(history_embedding)

        return history_embedding


class HistoryEncoderLight(nn.Module):
    """
    轻量级历史帧编码器 - 使用简单的 MLP + Pooling
    比 Transformer 版本更快，适合大规模数据
    """
    def __init__(self,
                 input_dim=4,           # 输入场的维度
                 embed_dim=128,         # 输出嵌入维度
                 hidden_dim=256,        # 隐藏层维度
                 num_layers=3):
        super(HistoryEncoderLight, self).__init__()

        self.embed_dim = embed_dim

        # 点级别的特征提取 (类似 PointNet)
        layers = []
        in_dim = 4 + input_dim  # 坐标 + 场值
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else embed_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            ])
            in_dim = out_dim
        self.point_encoder = nn.Sequential(*layers)

        # 时间聚合 (简单的注意力池化)
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1)
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x_hist, y_hist):
        """
        Args:
            x_hist: [B, T_in * N_s, 4] - 历史帧的坐标
            y_hist: [B, T_in * N_s, input_dim] - 历史帧的场值

        Returns:
            history_embedding: [B, embed_dim]
        """
        # 拼接
        combined = torch.cat([x_hist, y_hist], dim=-1)  # [B, T*N, 4+input_dim]

        # 点级别编码
        point_features = self.point_encoder(combined)  # [B, T*N, embed_dim]

        # 注意力权重
        attn_weights = self.attention(point_features)  # [B, T*N, 1]
        attn_weights = F.softmax(attn_weights, dim=1)

        # 加权聚合
        history_embedding = torch.sum(point_features * attn_weights, dim=1)  # [B, embed_dim]

        # 输出投影
        history_embedding = self.output_proj(history_embedding)

        return history_embedding


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块
    让 Trunk 的查询点能够关注历史信息
    """
    def __init__(self, query_dim, context_dim, num_heads=4, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(context_dim, query_dim)
        self.v_proj = nn.Linear(context_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, context):
        """
        Args:
            query: [B, N, query_dim] - 查询点特征 (来自 Trunk)
            context: [B, M, context_dim] - 上下文特征 (历史信息)

        Returns:
            output: [B, N, query_dim] - 融合后的特征
        """
        B, N, D = query.shape

        # 归一化
        query_norm = self.norm1(query)

        # 投影
        Q = self.q_proj(query_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 输出
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)

        # 残差连接
        query = query + self.dropout(out)

        # FFN
        query = query + self.ffn(self.norm2(query))

        return query


class TrunkTemporalV2(nn.Module):
    """
    改进的 Trunk 网络 - 支持与历史信息的交互

    输入: (x, y, z, t) 查询坐标
    可选: 与历史特征进行交叉注意力
    输出: 多个输出头的特征
    """
    def __init__(self,
                 in_dim=4,
                 out_dim=256,
                 hidden_dim=128,
                 num_layers=4,
                 num_outputs=5,
                 use_cross_attention=True,
                 context_dim=128):
        super(TrunkTemporalV2, self).__init__()

        self.out_dim = out_dim
        self.num_outputs = num_outputs
        self.use_cross_attention = use_cross_attention

        # 坐标编码 (傅里叶特征)
        self.coord_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 主干网络
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ))

        # 交叉注意力层 (可选)
        if use_cross_attention:
            self.cross_attn = CrossAttentionFusion(
                query_dim=hidden_dim,
                context_dim=context_dim,
                num_heads=4
            )

        # 输出头
        self.output_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, context=None):
        """
        Args:
            x: [B, N, 4] - 查询坐标 (x, y, z, t)
            context: [B, M, context_dim] - 历史上下文 (可选)

        Returns:
            outputs: list of num_outputs tensors, each [B, N, out_dim//num_outputs]
        """
        # 坐标编码
        h = self.coord_encoder(x)

        # 主干网络
        for layer in self.layers:
            h = h + layer(h)  # 残差连接

        # 交叉注意力
        if self.use_cross_attention and context is not None:
            h = self.cross_attn(h, context)

        # 输出
        out = self.output_head(h)
        out = F.gelu(out)

        # 分割成多个输出
        chunk_size = self.out_dim // self.num_outputs
        outputs = []
        for i in range(self.num_outputs):
            outputs.append(out[..., i*chunk_size:(i+1)*chunk_size])

        return outputs


class TemporalDeepONetV2(nn.Module):
    """
    改进的时序 DeepONet - 用于时序预测任务

    核心改进:
    1. 添加 HistoryEncoder 编码历史帧信息
    2. Branch 融合几何特征和历史特征
    3. Trunk 可以与历史信息交互
    4. 简化 BC (边界条件已隐含在历史帧中)

    消融实验:
    - use_geometry=True: 使用 Swin 编码几何图像 (默认)
    - use_geometry=False: 不使用几何编码，只用历史信息
    """
    def __init__(self,
                 # 输入维度
                 num_input_vars=5,       # 输入场变量数 (p, u, v, w, wss)
                 num_output_vars=5,      # 输出场变量数
                 # 历史编码器
                 history_embed_dim=128,
                 history_encoder_type='light',  # 'transformer' or 'light'
                 history_num_layers=3,
                 # 几何编码器
                 swin_embed_dim=24,
                 use_geometry=True,      # 消融: 是否使用几何编码器
                 # Trunk 网络
                 trunk_hidden_dim=128,
                 trunk_num_layers=4,
                 use_cross_attention=True,
                 # Branch 维度
                 branch_dim=256):
        super(TemporalDeepONetV2, self).__init__()

        self.num_output_vars = num_output_vars
        self.branch_dim = branch_dim
        self.use_geometry = use_geometry

        # ========== 历史编码器 ==========
        if history_encoder_type == 'transformer':
            self.history_encoder = HistoryEncoder(
                input_dim=num_input_vars,
                embed_dim=history_embed_dim,
                num_layers=history_num_layers
            )
        else:
            self.history_encoder = HistoryEncoderLight(
                input_dim=num_input_vars,
                embed_dim=history_embed_dim,
                num_layers=history_num_layers
            )

        # ========== 几何编码器 (Swin) - 可选 ==========
        if use_geometry:
            self.geometry_encoder = Swin_Final(embed_size=swin_embed_dim)

            # 动态计算 Swin 输出维度
            with torch.no_grad():
                dummy_geo = torch.zeros(1, 1, 32, 32, 32)
                swin_out_dim = self.geometry_encoder(dummy_geo).shape[1]
            print(f"Swin output dim: {swin_out_dim}")

            # Branch 融合: 历史特征 + 几何特征
            fusion_input_dim = history_embed_dim + swin_out_dim
            print(f"Branch fusion input dim: {fusion_input_dim} (history={history_embed_dim} + swin={swin_out_dim})")
        else:
            self.geometry_encoder = None
            # Branch 融合: 只有历史特征
            fusion_input_dim = history_embed_dim
            print(f"[Ablation] No geometry encoder, Branch input dim: {fusion_input_dim} (history only)")

        # ========== Branch 融合 ==========
        self.branch_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Linear(branch_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU()
        )

        # ========== Trunk 网络 ==========
        trunk_out_dim = num_output_vars * branch_dim
        self.trunk = TrunkTemporalV2(
            in_dim=4,
            out_dim=trunk_out_dim,
            hidden_dim=trunk_hidden_dim,
            num_layers=trunk_num_layers,
            num_outputs=num_output_vars,
            use_cross_attention=use_cross_attention,
            context_dim=history_embed_dim
        )

        # ========== 输出缩放 (简化版 bypass) ==========
        self.output_scale = nn.Sequential(
            nn.Linear(history_embed_dim, num_output_vars),
            nn.Softplus()  # 确保正数缩放
        )

    def forward(self, x_hist, y_hist, x_query, geometry):
        """
        Args:
            x_hist: [B, T_in * N_s, 4] - 历史帧坐标 (x, y, z, t)
            y_hist: [B, T_in * N_s, num_input_vars] - 历史帧场值
            x_query: [B, T_out * N_s, 4] - 查询坐标 (x, y, z, t_future)
            geometry: [B, 1, D, H, W] - 3D 几何图像 (如果 use_geometry=False 则忽略)

        Returns:
            y_pred: [B, T_out * N_s, num_output_vars] - 预测场值
        """
        B = x_hist.shape[0]

        # 1. 编码历史信息
        history_embed = self.history_encoder(x_hist, y_hist)  # [B, history_embed_dim]

        # 2. 编码几何信息 (可选)
        if self.use_geometry and self.geometry_encoder is not None:
            geometry_embed = self.geometry_encoder(geometry)  # [B, swin_out_dim]
            # 3. Branch 融合: 历史 + 几何
            branch_input = torch.cat([history_embed, geometry_embed], dim=-1)
        else:
            # 3. Branch 融合: 只有历史
            branch_input = history_embed

        branch_embed = self.branch_fusion(branch_input)  # [B, branch_dim]
        branch_embed = branch_embed.unsqueeze(-1)  # [B, branch_dim, 1]

        # 4. Trunk 处理查询点 (可选: 与历史特征交互)
        # 扩展历史嵌入为上下文
        history_context = history_embed.unsqueeze(1)  # [B, 1, history_embed_dim]
        trunk_outputs = self.trunk(x_query, context=history_context)  # list of [B, N, branch_dim]

        # 5. DeepONet: Trunk · Branch
        predictions = []
        for trunk_out in trunk_outputs:
            pred = torch.matmul(trunk_out, branch_embed)  # [B, N, 1]
            predictions.append(pred)

        y_pred = torch.cat(predictions, dim=-1)  # [B, N, num_output_vars]

        # 6. 输出缩放
        scale = self.output_scale(history_embed)  # [B, num_output_vars]
        y_pred = y_pred * scale.unsqueeze(1)

        return y_pred


class Trunk(nn.Module):
    """
    Trunk network for DeepONet
    处理空间坐标输入，输出特征向量
    """
    def __init__(self, in_dim=3, out_dim=96, hidden_num=24, layer_num=4):
        super(Trunk, self).__init__()

        # input layer
        self.IN = nn.Linear(in_dim, hidden_num)

        # hidden layers
        self.hiddens = nn.ModuleList([
            nn.Linear(hidden_num, hidden_num) for _ in range(layer_num)
        ])

        # output layer
        self.OUT = nn.Linear(hidden_num, out_dim)

    def forward(self, x):
        u = self.IN(x)
        u = F.gelu(u)

        for layer in self.hiddens:
            u = layer(u)
            u = F.gelu(u)

        u = self.OUT(u)
        u = F.gelu(u)

        c = u.shape[-1]

        u1 = u[..., 0:c//4]
        u2 = u[..., c//4:2*c//4]
        u3 = u[..., 2*c//4:3*c//4]
        u4 = u[..., 3*c//4:4*c//4]

        return u1, u2, u3, u4


class TrunkTemporal(nn.Module):
    """
    Temporal Trunk network for DeepONet
    处理空间坐标 + 时间输入，输出特征向量
    输入: [B, N, 4] 其中最后一维是 (x, y, z, t)
    输出: 5 个 [B, N, out_dim//5] 的张量 (p, u, v, w, wss)
    """
    def __init__(self, in_dim=4, out_dim=96, hidden_num=64, layer_num=4, num_outputs=5):
        super(TrunkTemporal, self).__init__()

        self.out_dim = out_dim
        self.num_outputs = num_outputs  # 5 for (p, u, v, w, wss)

        # input layer - 直接处理 (x, y, z, t)
        self.IN = nn.Linear(in_dim, hidden_num)

        # hidden layers
        self.hiddens = nn.ModuleList([
            nn.Linear(hidden_num, hidden_num) for _ in range(layer_num)
        ])

        # output layer - 输出 num_outputs 倍的 branch 维度
        self.OUT = nn.Linear(hidden_num, out_dim)

    def forward(self, x):
        """
        x: [B, N, 4] - (x, y, z, t)
        Returns: num_outputs 个 [B, N, out_dim//num_outputs] 的张量
        """
        u = self.IN(x)
        u = F.gelu(u)

        for layer in self.hiddens:
            u = layer(u)
            u = F.gelu(u)

        u = self.OUT(u)
        u = F.gelu(u)

        c = u.shape[-1]
        chunk_size = c // self.num_outputs

        outputs = []
        for i in range(self.num_outputs):
            outputs.append(u[..., i*chunk_size:(i+1)*chunk_size])

        return outputs  # 7 个张量: p, u, v, w, wss_x, wss_y, wss_z


class Branch(nn.Module):
    """
    Branch network for processing boundary conditions
    """
    def __init__(self, in_dim=6, out_dim=96, hidden_num=24, layer_num=4):
        super(Branch, self).__init__()

        self.IN = nn.Linear(in_dim, hidden_num)
        self.hiddens = nn.ModuleList([
            nn.Linear(hidden_num, hidden_num) for _ in range(layer_num)
        ])
        self.OUT = nn.Linear(hidden_num, out_dim)

    def forward(self, x):
        u = self.IN(x)
        u = F.gelu(u)

        for layer in self.hiddens:
            u = layer(u)
            u = F.gelu(u)

        u = self.OUT(u)
        u = F.gelu(u)

        return u


class Branch_Bypass(nn.Module):
    """
    Bypass branch for scaling output
    out_dim=5 for (p, u, v, w, wss)
    """
    def __init__(self, in_dim=1, out_dim=5):
        super(Branch_Bypass, self).__init__()
        self.FC = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        u = self.FC(x)
        return u


class TemporalEncoder(nn.Module):
    """
    时间序列编码器 - 使用 Transformer 编码时间依赖
    """
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2, max_time_steps=100):
        super(TemporalEncoder, self).__init__()

        self.embed_dim = embed_dim

        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, max_time_steps, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, num_timesteps):
        """
        x: [B, T, embed_dim]
        """
        T = x.shape[1]
        x = x + self.pos_embedding[:, :T, :]
        x = self.transformer(x)
        return x


class Swin_Final(nn.Module):
    """
    SwinTransformer for 3D geometry feature extraction
    """
    def __init__(self, embed_size=24):
        super(Swin_Final, self).__init__()
        if SwinViT is None or ensure_tuple_rep is None:
            raise ImportError("monai is required for Swin geometry encoder. "
                              "Install with: pip install monai")
        patch_size = ensure_tuple_rep(2, 3)
        window_size = ensure_tuple_rep(7, 3)  # 使用较小的 window_size 以适应较小的输入
        self.swinViT = SwinViT(
            in_chans=1,
            embed_dim=embed_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3,
            use_v2=False
        )
        self.avg_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in)
        out = hidden_states_out[-1]
        out = self.avg_pooling(out)

        b = out.shape[0]
        out = F.normalize(out.reshape(b, -1), p=1, dim=1)

        return out


class Swin_Temporal(nn.Module):
    """
    SwinTransformer for 4D (3D + time) feature extraction
    处理时序的3D数据
    """
    def __init__(self, embed_size=24, num_timesteps=10):
        super(Swin_Temporal, self).__init__()

        # 3D SwinViT for each timestep
        patch_size = ensure_tuple_rep(2, 3)
        window_size = ensure_tuple_rep(7, 3)
        self.swinViT = SwinViT(
            in_chans=1,
            embed_dim=embed_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3,
            use_v2=False
        )
        self.avg_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 计算特征维度
        # SwinViT 的输出维度是 embed_size * 2^(num_depths-1)
        feature_dim = embed_size * 8  # depths 有 4 层

        # 时间聚合层
        self.temporal_encoder = TemporalEncoder(
            embed_dim=feature_dim,
            num_heads=4,
            num_layers=2,
            max_time_steps=num_timesteps
        )

        # 输出投影
        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x_in, return_temporal=False):
        """
        x_in: [B, T, 1, D, H, W] or [B, 1, D, H, W]
        return_temporal: 如果为 True，返回每个时间步的特征
        """
        if len(x_in.shape) == 5:
            # 单时间步输入 [B, 1, D, H, W]
            hidden_states_out = self.swinViT(x_in)
            out = hidden_states_out[-1]
            out = self.avg_pooling(out)
            b = out.shape[0]
            out = F.normalize(out.reshape(b, -1), p=1, dim=1)
            return out

        # 多时间步输入 [B, T, 1, D, H, W]
        B, T, C, D, H, W = x_in.shape

        # 重塑为 [B*T, C, D, H, W]
        x_reshaped = x_in.view(B * T, C, D, H, W)

        # 通过 SwinViT
        hidden_states_out = self.swinViT(x_reshaped)
        out = hidden_states_out[-1]
        out = self.avg_pooling(out)  # [B*T, feat_dim, 1, 1, 1]

        feat_dim = out.shape[1]
        out = out.view(B, T, feat_dim)  # [B, T, feat_dim]

        # 时间编码
        temporal_out = self.temporal_encoder(out, T)  # [B, T, feat_dim]

        if return_temporal:
            return temporal_out  # [B, T, feat_dim]

        # 聚合所有时间步
        out = temporal_out.mean(dim=1)  # [B, feat_dim]
        out = self.output_proj(out)
        out = F.normalize(out, p=1, dim=1)

        return out


class DeepONetTemporal(nn.Module):
    """
    完整的时序 DeepONet 模型
    结合所有网络组件
    """
    def __init__(self,
                 trunk_in_dim=4,  # (x, y, z, t)
                 trunk_hidden=64,
                 trunk_layers=4,
                 bc_in_dim=7,
                 bc_hidden=64,
                 swin_embed=24,
                 num_timesteps=100):
        super(DeepONetTemporal, self).__init__()

        # 计算维度
        swin_out_dim = swin_embed * 8  # SwinViT 输出维度
        bc_out_dim = 64
        branch_total_dim = swin_out_dim + bc_out_dim
        trunk_out_dim = 4 * branch_total_dim  # 4 个输出通道

        # 网络组件
        self.trunk = TrunkTemporal(
            in_dim=trunk_in_dim,
            out_dim=trunk_out_dim,
            hidden_num=trunk_hidden,
            layer_num=trunk_layers
        )

        self.branch_img = Swin_Final(embed_size=swin_embed)

        self.branch_bc = Branch(
            in_dim=bc_in_dim,
            out_dim=bc_out_dim,
            hidden_num=bc_hidden,
            layer_num=4
        )

        self.branch_bp = Branch_Bypass(in_dim=1, out_dim=4)

    def forward(self, x, image, bc):
        """
        x: [B, N, 4] - 空间坐标 + 时间 (x, y, z, t)
        image: [B, 1, D, H, W] - 3D 几何图像
        bc: [B, bc_dim] - 边界条件

        Returns:
            y_pred: [B, N, 4] - (p, u, v, w)
        """
        # Trunk: 处理空间-时间坐标
        trunk_pred1, trunk_pred2, trunk_pred3, trunk_pred4 = self.trunk(x)  # 4*[B, N, Nm]

        # Branch: 处理图像特征
        branch_img_pred = self.branch_img(image).unsqueeze(-1)  # [B, Nimg, 1]

        # Branch: 处理边界条件
        branch_bc_pred = self.branch_bc(bc).unsqueeze(-1)  # [B, Nbc, 1]

        # 合并 branch 特征
        branch_pred = torch.cat([branch_img_pred, branch_bc_pred], dim=1)  # [B, Nm, 1]

        # DeepONet: trunk * branch
        h_pred1 = torch.matmul(trunk_pred1, branch_pred)  # [B, N, 1]
        h_pred2 = torch.matmul(trunk_pred2, branch_pred)
        h_pred3 = torch.matmul(trunk_pred3, branch_pred)
        h_pred4 = torch.matmul(trunk_pred4, branch_pred)

        y_pred = torch.cat([h_pred1, h_pred2, h_pred3, h_pred4], dim=-1)  # [B, N, 4]

        # Bypass scaling
        branch_bp_pred = self.branch_bp(bc[..., -1:])  # [B, 4]
        y_pred = y_pred * branch_bp_pred.unsqueeze(1)  # [B, N, 4]

        return y_pred


