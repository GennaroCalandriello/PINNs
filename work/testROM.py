# diffpool_gnn_ae.py
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor
from torch_geometric.nn import GraphNorm

# If you have these utilities, import them; otherwise replace createGraphData()
from ns_GNN_cav2 import createGraphData, dataLoader, dataNormalizer, geometryObject

# =========================
# Settings / Hyperparameters
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DiffPool AE] Using device: {device}")

HIDDEN = 150
LATENT = 30
EDGE_HIDDEN = 50
DROP = 0.0
LR = 1e-3
EPOCHS = 500
BATCH_SIZE_NODES = 8000
NEIGHBORS = [50, 40, 30]
GRAD_CLIP = 1.0
USE_AMP = True
USE_COMPILE = False
SCHEDULER_STEP = 200
SELF_LOOP = True
MODEL_PATH = "model/gnn_ae_diffpool.pth"
LOSS_PATH = "model/loss_gnn_ae_diffpool.txt"

# DiffPool hierarchy (number of clusters per pooling level)
CLUSTERS_PER_LEVEL: List[int] = [800]


# =========================
# Utilities
# =========================
def build_sparse_adj(edge_index: torch.Tensor, num_nodes: int) -> SparseTensor:
    row, col = edge_index
    return SparseTensor(
        row=row, col=col, sparse_sizes=(num_nodes, num_nodes)
    ).coalesce()


def diffpool_aux_losses(A_sparse: SparseTensor, S: torch.Tensor):
    """
    Link prediction and entropy losses.
    A_sparse: [N,N] SparseTensor (row-normalized not required here).
    S:        [N,C] soft assignments (rows ~ stochastic)
    """
    # A_hat ≈ S S^T  → encourage edges to be respected by assignments
    A_dense = A_sparse.to_dense()
    A_hat = S @ S.t()
    link_loss = F.mse_loss(A_dense, A_hat)

    # Encourage confident-but-not-collapsed assignments
    # entropy per row: -sum_j S_ij log S_ij
    S_clamped = S.clamp_min(1e-8)
    entropy = -(S_clamped * S_clamped.log()).sum(dim=1).mean()

    return link_loss, entropy


def sparse_diff_pool(
    x: torch.Tensor, edge_index: torch.Tensor, S: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sparse DiffPool (eqs. (3)-(4) in DiffPool-style formulations).

    Args:
        x: [N, F] node embeddings
        edge_index: [2, E] edges (COO)
        S: [N, C] soft assignment (rows sum ~ 1)

    Returns:
        x_pool: [C, F]
        edge_index_pool: [2, E'] pooled graph edges (thresholded from A_pool)
    """
    N, C = S.size(0), S.size(1)
    A = build_sparse_adj(edge_index, N)  # [N, N]
    AS = A.matmul(S)  # [N, C]
    A_pool = S.transpose(0, 1) @ AS  # [C, C]
    x_pool = S.transpose(0, 1) @ x  # [C, F]

    # Sparsify A_pool to get a new edge_index
    threshold = (A_pool.abs().mean() * 0.1).item()
    A_pool = A_pool * (A_pool > threshold)
    row_idx, col_idx = A_pool.nonzero(as_tuple=True)
    edge_index_pool = torch.stack([row_idx, col_idx], dim=0)
    return x_pool, edge_index_pool


# =========================
# GNN Building Blocks
# =========================
class EdgeGNNLayer(MessagePassing):
    """
    Simple edge-aware message passing:
      m_ij = MLP([x_i, x_j, e_ij])
      x_i' = MLP([x_i, AGGR_j m_ij])
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        edge_dim: Optional[int],
        hidden: int = EDGE_HIDDEN,
        dropout: float = DROP,
        aggr: str = "mean",
    ):
        super().__init__(aggr=aggr, node_dim=0)
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * in_ch + (edge_dim or 0), hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_ch),
        )
        self.mlp_upd = nn.Sequential(
            nn.Linear(in_ch + out_ch, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_ch),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_ch)
        self.use_res = in_ch == out_ch
        self.edge_dim = edge_dim or 0
        self.graphnorm = GraphNorm(out_ch)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.dropout(x)

        # Add self-loops, padding edge_attr with zeros for the added edges.
        if SELF_LOOP:
            edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            if edge_attr is not None:
                num_added = edge_index_sl.size(1) - edge_index.size(1)
                if num_added > 0:
                    pad = edge_attr.new_zeros((num_added, edge_attr.size(1)))
                    edge_attr_sl = torch.cat([edge_attr, pad], dim=0)
                else:
                    edge_attr_sl = edge_attr
            else:
                edge_attr_sl = None
        else:
            edge_index_sl = edge_index
            edge_attr_sl = edge_attr

        out = self.propagate(edge_index_sl, x=x, edge_attr=edge_attr_sl)
        h = torch.cat([x, out], dim=-1)
        h = self.mlp_upd(h)
        h = self.norm(h)
        if self.use_res:
            h = h + x
        h = F.gelu(h)
        h = self.graphnorm(h, batch)

        return h

    def message(self, x_i, x_j, edge_attr):
        # x_i: dst node, x_j: src node features
        if edge_attr is None:
            edge_attr = x_i.new_zeros((x_i.size(0), self.edge_dim))
        m_in = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp_msg(m_in)


class DiffPoolBlock(nn.Module):
    """
    One encoder stage:
      - EdgeGNN to compute node embeddings
      - Soft assignment S = softmax(MLP(x))
      - DiffPool: (x', edge') = DiffPool(x, A, S)
    """

    def __init__(self, ch: int, edge_dim: int, num_clusters: int):
        super().__init__()
        self.gnn = EdgeGNNLayer(ch, ch, edge_dim)
        self.assign = nn.Sequential(
            nn.Linear(ch, ch),
            nn.GELU(),
            nn.LayerNorm(ch),
            nn.Linear(ch, num_clusters),
            nn.GELU(),
            nn.LayerNorm(num_clusters),
        )

    # def forward(self, x, edge_index, edge_attr, batch):
    #     x = self.gnn(x, edge_index, edge_attr, batch)
    #     S = F.softmax(self.assign(x), dim=-1)  # [N, C]

    #     # keep A_prev for aux losses
    #     A_prev = build_sparse_adj(edge_index, x.size(0))

    #     x_pool, edge_index_pool = sparse_diff_pool(x, edge_index, S)

    #     state = {
    #         "S": S,
    #         "prev_edge_index": edge_index,
    #         "prev_edge_attr": edge_attr,
    #         "A_prev": A_prev,  # <— NEW
    #     }
    #     return (x_pool, edge_index_pool, None, batch), state
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr, batch)
        S = F.softmax(self.assign(x), dim=-1)  # [N, C]

        A_prev = build_sparse_adj(edge_index, x.size(0))  # se usi aux losses

        x_pool, edge_index_pool = sparse_diff_pool(x, edge_index, S)
        C = S.size(1)

        # *** QUI: batch per il grafo coarsened (C nodi) ***
        if batch is None or batch.numel() == 0 or batch.unique().numel() == 1:
            batch_coarse = x_pool.new_zeros(C, dtype=torch.long)  # tutto un grafo
        else:
            # caso multi-grafo (se/quando servirà): "mode" del batch per cluster
            # hard cluster
            cluster = S.argmax(dim=1)  # [N]
            # costruiamo per-cluster un assegnamento di batch tramite maggioranza
            # (semplice fallback: prendiamo il batch del primo nodo assegnato)
            batch_coarse = x_pool.new_zeros(C, dtype=torch.long)
            batch_coarse.index_copy_(
                0, cluster, batch
            )  # semplice ma funziona se non mischia grafi

        state = {
            "S": S,
            "prev_edge_index": edge_index,
            "prev_edge_attr": edge_attr,
            "A_prev": A_prev,
        }
        # *** ritorna batch_coarse ***
        return (x_pool, edge_index_pool, None, batch_coarse), state


# =========================
# Autoencoder (DiffPool-only)
# =========================
class GraphAutoEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int,
        edge_dim: int,
        out_ch: int = 3,
        hidden: int = HIDDEN,
        latent: int = LATENT,
        clusters_per_level: Optional[List[int]] = None,
    ):
        super().__init__()
        assert (
            clusters_per_level and len(clusters_per_level) > 0
        ), "Provide a non-empty list of clusters per DiffPool level."

        self.in_proj = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # Encoder hierarchy (DiffPool-only)
        self.encoder = nn.ModuleList(
            [DiffPoolBlock(hidden, edge_dim, C) for C in clusters_per_level]
        )

        # Bottleneck on the coarsest graph
        self.bottleneck = EdgeGNNLayer(hidden, latent, edge_dim)
        self.latent_up = nn.Linear(latent, hidden)

        # Decoder: mirror the depth with EdgeGNN layers
        depth = len(clusters_per_level)
        self.decoder = nn.ModuleList(
            [EdgeGNNLayer(hidden, hidden, edge_dim) for _ in range(depth)]
        )

        # Final prediction head
        self.head = nn.Linear(hidden, out_ch)

    def forward(self, data, return_aux: bool = False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = self.in_proj(x)
        states = []
        prev_idx, prev_attr = edge_index, edge_attr

        enc_batches = []

        for enc in self.encoder:
            enc_batches.append(batch)
            (x, edge_index, edge_attr, batch), state = enc(
                x, edge_index, edge_attr, batch
            )
            states.append(state)
            prev_idx, prev_attr = edge_index, edge_attr

        x = self.bottleneck(x, edge_index, edge_attr, batch)
        x = F.gelu(self.latent_up(x))

        for level, (dec, state) in enumerate(
            zip(reversed(self.decoder), reversed(states))
        ):
            S = state["S"]
            edge_index_prev = state["prev_edge_index"]
            edge_attr_prev = state["prev_edge_attr"]
            batch_prev = enc_batches[
                -1 - level
            ]  # batch del livello "fine" corrispondente

            # unpool
            x = S @ x  # [N_prev, hidden]
            x = dec(x, edge_index_prev, edge_attr_prev, batch_prev)

        out = self.head(x)

        if return_aux:
            aux = [(st["S"], st["A_prev"]) for st in states]
            return out, aux
        return out


# =========================
# Data Loader
# =========================
def GraphLoader(graph_data, batch_size_nodes=BATCH_SIZE_NODES, neighbors=NEIGHBORS):
    return NeighborLoader(
        graph_data,
        num_neighbors=neighbors,
        batch_size=batch_size_nodes,
        input_nodes=None,  # all nodes
        shuffle=True,
    )


# =========================
# Training
# =========================
def train(
    model: nn.Module,
    loader: NeighborLoader,
    epochs: int = EPOCHS,
    lr: float = LR,
    use_amp: bool = USE_AMP,
    grad_clip: float = GRAD_CLIP,
    scheduler_step: int = SCHEDULER_STEP,
):
    from tqdm import trange

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=scheduler_step, gamma=0.9
    )
    scaler = torch.amp.GradScaler(enabled=use_amp, device=device.type)

    model.train()
    losses = []

    loop = trange(EPOCHS, desc="Training", dynamic_ncols=True)
    for ep in loop:
        tot = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)

            # NeighborLoader marks the first "center" nodes
            center = getattr(batch, "batch_size", batch.x.size(0))

            # Infer output channels from target y
            target_dim = batch.y.size(1)

            with torch.amp.autocast(enabled=use_amp, device_type=device.type):
                out = model(batch)  # [N, out_ch]
                pred = out[:center, :target_dim]  # center-node supervision
                tgt = batch.y[:center, :target_dim]

                loss = F.mse_loss(pred, tgt)

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            tot += loss.item()
            n_batches += 1
        loop.set_postfix(
            {
                "loss": f"{loss.item():.6f}",
            }
        )

        scheduler.step()
        avg = tot / max(1, n_batches)
        losses.append(avg)

    return model, losses


# =========================
# Main
# =========================
def main():
    # Load your graph (PyG Data). Must provide x, edge_index, edge_attr, y.
    data = createGraphData()
    in_ch = data.x.size(-1)
    edge_dim = data.edge_attr.size(-1)
    out_ch = data.y.size(-1)

    model = GraphAutoEncoder(
        in_ch=in_ch,
        edge_dim=edge_dim,
        out_ch=out_ch,
        hidden=HIDDEN,
        latent=LATENT,
        clusters_per_level=CLUSTERS_PER_LEVEL,
    )

    if USE_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)  # optional, PyTorch 2.x

    loader = GraphLoader(data, batch_size_nodes=BATCH_SIZE_NODES, neighbors=NEIGHBORS)
    model, loss_hist = train(model, loader)

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    np.savetxt(LOSS_PATH, np.array(loss_hist))
    print(f"Saved model, loss: {MODEL_PATH}, {LOSS_PATH}")


if __name__ == "__main__":
    main()
