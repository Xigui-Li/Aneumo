"""
MeshGraphNet for CFD WSS prediction.

Encoder-Processor-Decoder architecture operating directly on the mesh graph.
Adapted from DeepMind's MeshGraphNets (ICLR 2021) for 3D hemodynamic data.

Uses gradient checkpointing to handle large graphs (>100K nodes, >6M edges).
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add


class MLP(nn.Module):
    """Simple MLP with LayerNorm."""
    def __init__(self, in_dim, out_dim, hidden_dim=None, num_layers=2):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        layers.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EdgeBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = MLP(3 * hidden_dim, hidden_dim)

    def forward(self, node_feat, edge_feat, edge_index):
        src, dst = edge_index
        edge_input = torch.cat([node_feat[src], node_feat[dst], edge_feat], dim=-1)
        return edge_feat + self.mlp(edge_input)


class NodeBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = MLP(2 * hidden_dim, hidden_dim)

    def forward(self, node_feat, edge_feat, edge_index, num_nodes):
        _, dst = edge_index
        agg = scatter_add(edge_feat, dst, dim=0, dim_size=num_nodes)
        node_input = torch.cat([node_feat, agg], dim=-1)
        return node_feat + self.mlp(node_input)


class ProcessorBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_block = EdgeBlock(hidden_dim)
        self.node_block = NodeBlock(hidden_dim)

    def forward(self, node_feat, edge_feat, edge_index, num_nodes):
        edge_feat = self.edge_block(node_feat, edge_feat, edge_index)
        node_feat = self.node_block(node_feat, edge_feat, edge_index, num_nodes)
        return node_feat, edge_feat


class MeshGraphNet_WSS(nn.Module):
    """
    MeshGraphNet for WSS prediction with gradient checkpointing.
    """

    def __init__(self,
                 node_input_dim: int = 15,
                 edge_input_dim: int = 4,
                 hidden_dim: int = 64,
                 num_message_passing: int = 10,
                 output_dim: int = 1,
                 use_checkpoint: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_checkpoint = use_checkpoint

        # Encoder
        self.node_encoder = MLP(node_input_dim, hidden_dim)
        self.edge_encoder = MLP(edge_input_dim, hidden_dim)

        # Processor
        self.processors = nn.ModuleList([
            ProcessorBlock(hidden_dim) for _ in range(num_message_passing)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _run_processor(self, proc, h_node, h_edge, edge_index, num_nodes):
        """Wrapper for gradient checkpointing."""
        return proc(h_node, h_edge, edge_index, num_nodes)

    def forward(self, node_feat, edge_index, edge_attr):
        num_nodes = node_feat.shape[0]

        h_node = self.node_encoder(node_feat)
        h_edge = self.edge_encoder(edge_attr)

        for proc in self.processors:
            if self.use_checkpoint and self.training:
                h_node, h_edge = checkpoint(
                    self._run_processor, proc, h_node, h_edge,
                    edge_index, num_nodes, use_reentrant=False
                )
            else:
                h_node, h_edge = proc(h_node, h_edge, edge_index, num_nodes)

        return self.decoder(h_node)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = MeshGraphNet_WSS(node_input_dim=15, edge_input_dim=4, hidden_dim=64, num_message_passing=10)
    print(f"MeshGraphNet_WSS parameters: {count_parameters(model):,}")
