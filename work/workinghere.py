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
            nn.Linear(in_ch + out_ch, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_ch),
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


# ====== Encoder block: GNN -> TopK pool ======
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, edge_dim, ratio=POOL_RATIO):
        super().__init__()
        self.gnn = EdgeGNNLayer(in_ch, out_ch, edge_dim)
        self.score = nn.Linear(out_ch, 1)  # learn node scores for TopK
        self.pool = TopKPooling(out_ch, ratio=ratio)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        s = self.score(x).squeeze(-1)
        x, edge_index, edge_attr, batch, perm, _ = self.pool(
            x, edge_index, batch=batch, attn=s, edge_attr=edge_attr
        )
        # perm keeps indices of kept nodes: needed for unpool
        return (x, edge_index, edge_attr, batch), perm


# ====== Decoder block: GNN + Unpool (scatter back) ======
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, edge_dim):
        super().__init__()
        self.gnn = EdgeGNNLayer(in_ch, out_ch, edge_dim)

    def forward(self, x_coarse, perm, N_prev, edge_index_prev, edge_attr_prev):
        # unpool: scatter zeros to size N_prev and place x_coarse at indices 'perm'
        x_full = x_coarse.new_zeros((N_prev, x_coarse.size(-1)))
        x_full[perm] = x_coarse
        # refine on previous (denser) graph connectivity
        x_full = self.gnn(x_full, edge_index_prev, edge_attr_prev)
        return x_full


# ====== Full UNet-style Autoencoder ======
class SparseUNetAutoEncoder(nn.Module):
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

        # Encoder hierarchy
        enc = []
        ch = hidden
        for _ in range(depth):
            enc.append(EncoderBlock(ch, ch, edge_dim))
        self.encoder = nn.ModuleList(enc)

        # Latent bottleneck on pooled graph
        self.bottleneck = EdgeGNNLayer(ch, latent, edge_dim)

        # Decoder hierarchy (mirror)
        dec = []
        # we will project latent->hidden for decoding at first step
        self.latent_up = nn.Linear(latent, ch)
        for _ in range(depth):
            dec.append(DecoderBlock(ch, ch, edge_dim))
        self.decoder = nn.ModuleList(dec)

        self.head = nn.Linear(ch, out_ch)

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

        # Encoder path: store states for unpooling
        enc_states = []
        for blk in self.encoder:
            N_prev = x.size(0)
            (x, edge_index, edge_attr, batch), perm = blk(
                x, edge_index, edge_attr, batch
            )
            enc_states.append(
                (perm, N_prev, edge_index, edge_attr)
            )  # keep graph just before pooling

        # Latent on coarsest graph
        x = self.bottleneck(x, edge_index, edge_attr)
        x = F.gelu(self.latent_up(x))

        # Decoder path (reverse)
        for blk, (perm, N_prev, edge_index_prev, edge_attr_prev) in zip(
            reversed(self.decoder), reversed(enc_states)
        ):
            x = blk(x, perm, N_prev, edge_index_prev, edge_attr_prev)

        y = self.head(x)  # nodewise reconstruction
        return y


# ====== Training utilities ======
def build_loader(
    big_graph_data, batch_size_nodes=BATCH_SIZE_NODES, num_neighbors=NEIGHBORS
):
    """
    Mini-batch neighbor sampling from a single huge graph.
    Each batch picks 'batch_size_nodes' seed nodes and expands k hops with capped neighbors.
    """
    # add self loops if your operators benefit from it (optional)
    # big_graph_data.edge_index, _ = add_self_loops(big_graph_data.edge_index, num_nodes=big_graph_data.num_nodes)
    loader = NeighborLoader(
        big_graph_data,
        input_nodes=None,  # all nodes can be picked as seeds
        num_neighbors=num_neighbors,  # per layer fanouts
        batch_size=batch_size_nodes,
        shuffle=True,
    )
    return loader


def train(
    model,
    loader,
    epochs=EPOCHS,
    lr=LR,
    use_amp=USE_AMP,
    grad_clip=GRAD_CLIP,
    device="cuda",
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # optional compile (PyTorch 2.1+)
    if USE_COMPILE:
        model = torch.compile(model)

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            # IMPORTANT: NeighborLoader returns a *subgraph* with a mapping;
            # its 'y' is aligned to 'batch.n_id' (the nodes present in the subgraph).
            # We predict for all nodes in the subgraph and compute MSE on those with targets.
            has_target = getattr(batch, "y", None) is not None
            if not has_target:
                continue

            opt.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
                if device.startswith("cuda")
                else nullcontext()
            )
            with amp_ctx:
                pred = model(batch)  # [num_subgraph_nodes, out_dim]
                target = batch.y  # align shapes: [num_subgraph_nodes, out_dim]
                loss = F.mse_loss(pred, target)

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            total += loss.item() * batch.num_nodes
            n += batch.num_nodes

        sch.step()
        print(f"Epoch {ep:03d} | Loss {total/max(n,1):.6f}")

    return model


# ====== Entry point (wire your data) ======
def main():

    data = createGraphData()
    in_ch = data.x.size(-1)
    edge_dim = data.edge_attr.size(-1)
    out_ch = data.y.size(-1) if hasattr(data, "y") else 3

    model = SparseUNetAutoEncoder(in_ch, edge_dim, out_ch=out_ch)
    loader = build_loader(data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained = train(model, loader, device=device)

    torch.save(trained.state_dict(), "model/scalable_gnn_ae.pt")
    print("✅ Saved to model/scalable_gnn_ae.pt")


if __name__ == "__main__":
    main()
