# scalable_gnn_ae.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, TopKPooling, global_mean_pool
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import add_self_loops
import torch_scatter
from typing import List, Tuple
from contextlib import nullcontext
from ns_GNN_KF import (
    GaussianNormalizer,
    createGraphData,
    dataLoader,
    dataNormalizer,
    geometryObject,
)

# ====== Hyperparameters (adjust to your GPU) ======
HIDDEN = 64
LATENT = 32
EDGE_HIDDEN = 64
NUM_ENCODER_BLOCKS = 3  # pooling depth; 3 => ~1/8 nodes if ratio=0.5 each
POOL_RATIO = 0.5  # keep top 50% nodes at each stage
DROP = 0.1
LR = 2e-3
EPOCHS = 300
BATCH_SIZE_NODES = 4096  # NeighborLoader target nodes per batch
NEIGHBORS = [20, 15, 10]  # fanouts per hop
GRAD_CLIP = 1.0
USE_AMP = True  # mixed precision
USE_COMPILE = False  # torch.compile for fused kernels
SCHEDULER_STEP = 50  # epochs per step of lr scheduler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ====== Edge-aware sparse message-passing layer ======
class EdgeGNNLayer(MessagePassing):
    def __init__(
        self, in_ch, out_ch, edge_dim, hidden=EDGE_HIDDEN, dropout=DROP, aggr="mean"
    ):
        super().__init__(aggr=aggr, node_dim=0)
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * in_ch + edge_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_ch),
        )
        self.mlp_upd = nn.Sequential(
            nn.Linear(in_ch + out_ch, hidden), nn.GELU(), nn.Linear(hidden, out_ch)
        )
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x, edge_index, edge_attr):
        x = self.dropout(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)  # shape: [N, out_ch]
        h = torch.cat([x, out], dim=-1)
        h = self.mlp_upd(h)
        h = self.bn(h)
        return F.gelu(h)

    def message(self, x_i, x_j, edge_attr):
        # x_i: dst features, x_j: src features
        m_in = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp_msg(m_in)

    # ====== Encoder using TopK pooling ======


class Encoder(nn.Module):

    def __init__(self, in_ch, out_ch, edge_dim, ratio=POOL_RATIO):
        super().__init__()
        self.gnn = EdgeGNNLayer(in_ch, out_ch, edge_dim)
        self.score = nn.Linear(out_ch, 1)
        self.pool = TopKPooling(out_ch, ratio=ratio)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        s = self.score(x).squeeze(-1)
        x, edge_index, edge_attr, batch, perm, _ = self.pool(
            x, edge_index, batch=batch, attn=s, edge_attr=edge_attr
        )
        return (x, edge_index, edge_attr, batch), perm


# ====== Decoder using unpooling ======
class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, edge_dim):
        super().__init__()
        self.gnn = EdgeGNNLayer(in_ch, out_ch, edge_dim)

    def forward(self, x_coarse, perm, N_prev, edge_index_prev, edge_attr_prev):
        # unpool by scattering back to previous nodes
        x_full = x_coarse.new_zeros((N_prev, x_coarse.size(-1)))
        x_full[perm] = x_coarse
        x_full = self.gnn(x_full, edge_index_prev, edge_attr_prev)
        return x_full


class GraphAutoEncoder(nn.Module):
    def __init__(
        self,
        in_ch,
        edge_dim,
        out_ch=3,
        hidden=HIDDEN,
        latent=LATENT,
        depth=NUM_ENCODER_BLOCKS,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # encoder hierarchy
        encoderlist = []
        channels = hidden
        for _ in range(depth):
            encoderlist.append(Encoder(channels, channels, edge_dim=edge_dim))
        self.enccoder = nn.ModuleList(encoderlist)

        # bottleneck
        self.bottleneck = EdgeGNNLayer(channels, latent, edge_dim)

        # decoder hierarchy
        decoderlist = []
        # latent to hidden projection
        self.latent_up = nn.Linear(latent, channels)
        for _ in range(depth):
            decoderlist.append(Decoder(channels, channels, edge_dim))
        self.decoder = nn.ModuleList(decoderlist)
        self.head = nn.Linear(channels, out_ch)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            getattr(data, "batch", None),
        )
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        x = self.in_proj(x)
        encoder_states = []
        for b in self.encoder:
            N_prev = x.size(0)
            (x, edge_index, edge_attr, batch), perm = b(x, edge_index, edge_attr, batch)
            encoder_states.append(
                (perm, N_prev, edge_index, edge_attr)
            )  # store for unpooling

        x = self.bottleneck(x, edge_index, edge_attr)
        x = F.gelu(self.latent_up(x))

        # decoder path (reverse order of encoder states)
        for b, (perm, N_prev, edge_index_prev, edge_attr_prev) in zip(
            reversed(self.decoder), reversed(encoder_states)
        ):
            x = b(x, perm, N_prev, edge_index_prev, edge_attr_prev)

        y = self.head(x)
        return y


def GraphLoader(graph_data, batch_size_nodes=BATCH_SIZE_NODES, neighbors=NEIGHBORS):
    loader = NeighborLoader(
        graph_data,
        num_neighbors=neighbors,
        batch_size=batch_size_nodes,
        input_nodes=None,  # all nodes
        shuffle=True,
    )
    return loader


def train(model, loader, epochs=EPOCHS, lr=LR, grad_clip=GRAD_CLIP, device="cuda"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=SCHEDULER_STEP, gamma=0.9
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    # optional compiler
    if USE_COMPILE and hasattr(torch, "compile"):
        print("Using torch.compile for optimized training")
        model = torch.compile(model)

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0
        n = 0
        for batch in loader:
            batch = batch.to(device)

            opt.zero_grad()
            amp_ctx = (
                torch.autocast(device_type=device, enabled=USE_AMP)
                if device.startswith("cuda")
                else nullcontext()
            )
            with amp_ctx:
                out = model(batch)
                target = batch.y
                loss = F.mse_loss(out, target)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            total += loss.item() * batch.num_graphs
            n += batch.num_graphs
        scheduler.step()
        print(f"Epoch {epoch:03d}, Loss: {total / n:.6f}")
    return model
