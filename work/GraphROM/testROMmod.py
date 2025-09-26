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
from ns_GNN_cav2 import (
    createGraphData,
    dataLoader,
    dataNormalizer,
    geometryObject,
    GaussianNormalizer,
)

# =========================
# Settings / Hyperparameters
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DiffPool AE] Using device: {device}")

HIDDEN = 96  # best con 72 ma errore alto
LATENT = 40  # best con 42 ma errore alto
EDGE_HIDDEN = 96  # best con 52 ma errore alto
DROP = 0.0
LR = 1e-3
EPOCHS = 2000
BATCH_SIZE_NODES = 5000  # best con 16000 ma errore alto

if BATCH_SIZE_NODES is not None:
    ENFORCE_BLOCKWISE = True  # avoid graph mixing in the multigraph case
else:
    ENFORCE_BLOCKWISE = False

NEIGHBORS = [100, 100, 100]  # per NeighborLoader
GRAD_CLIP = None

SCHEDULER_STEP = 200

MODEL_PATH = "model/gnn_ae_diffpool1.pth"
LOSS_PATH = "model/loss_gnn_ae_diffpool1.txt"
MLP_VARIANT = 1  # 1 per simil MeshGraphNet, 0 per MeshCutriStyle
AGGREGATION = "add"  # "add" | "mean" | "max" ---- prima era "mean" ora "add"
nonlinearFn = nn.ReLU()

# BOOL FLAGS
USE_AMP = True
USE_COMPILE = False
RETURN_AUX = False  # return aux losses (link + entropy) from DiffPool
SELF_LOOP = True  # prima era True

# DiffPool hierarchy (number of clusters per pooling level)
CLUSTERS_PER_LEVEL: List[int] = [1400]  # best con [800] ma errore alto


# =========================
# Utilities
# =========================
def build_sparse_adj(edge_index: torch.Tensor, num_nodes: int) -> SparseTensor:
    row, col = edge_index
    return SparseTensor(
        row=row, col=col, sparse_sizes=(num_nodes, num_nodes)
    ).coalesce()


def diffpool_aux_losses(
    A: SparseTensor,
    S: torch.Tensor,
    neg_ratio: float = 1.0,
    drop_self_loops: bool = True,
    eps: float = 1e-8,
):
    """
    Link+Entropy (e opzionale 'balance') senza densificare:
      - Pos: MSE( <S_i, S_j>, A_ij ) sugli edge
      - Neg: MSE( <S_i, S_j>, 0 ) su coppie (i,j) campionate
      - Entropy: media delle entropie riga
      - Balance (facoltativo): spinge l'uso uniforme dei cluster
    """
    # --- edge list ---
    row, col, val = A.coo()
    if drop_self_loops:
        mask = row != col
        row, col = row[mask], col[mask]
        val = None if val is None else val[mask]

    if val is None:
        val = S.new_ones(row.numel())

    # --- positivi (edge reali) ---
    s_i = S.index_select(0, row)
    s_j = S.index_select(0, col)
    pos_score = (s_i * s_j).sum(dim=1)
    pos_loss = F.mse_loss(pos_score, val)

    # --- negativi (non-edge) ---
    num_pos = row.numel()
    num_neg = max(1, int(neg_ratio * num_pos))
    # campiona indici random; va benissimo per batches di NeighborLoader
    i_neg = torch.randint(0, S.size(0), (num_neg,), device=S.device)
    j_neg = torch.randint(0, S.size(0), (num_neg,), device=S.device)
    neg_score = (S[i_neg] * S[j_neg]).sum(dim=1)
    neg_loss = (neg_score**2).mean()  # target=0

    link_loss = 0.5 * (pos_loss + neg_loss)

    # --- entropy per nodo (favorisce assegnamenti confidenti) ---
    Sc = S.clamp_min(eps)
    entropy = -(Sc * Sc.log()).sum(dim=1).mean()

    # --- balance (opzionale ma utile per evitare collasso su 1 cluster) ---
    p = S.mean(dim=0)  # uso medio dei cluster
    balance = ((p - 1.0 / S.size(1)) ** 2).mean()

    return link_loss, entropy, balance


# def sparse_diff_pool(
#     x: torch.Tensor, edge_index: torch.Tensor, S: torch.Tensor
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Sparse DiffPool (eqs. (3)-(4) in DiffPool-style formulations).

#     Args:
#         x: [N, F] node embeddings
#         edge_index: [2, E] edges (COO)
#         S: [N, C] soft assignment (rows sum ~ 1)

#     Returns:
#         x_pool: [C, F]
#         edge_index_pool: [2, E'] pooled graph edges (thresholded from A_pool)
#     """
#     N, C = S.size(0), S.size(1)
#     A = build_sparse_adj(edge_index, N)  # [N, N]
#     AS = A.matmul(S)  # [N, C]
#     A_pool = S.transpose(0, 1) @ AS  # [C, C]
#     x_pool = S.transpose(0, 1) @ x  # [C, F]

#     # Sparsify A_pool to get a new edge_index
#     threshold = (A_pool.abs().mean() * 0.1).item()
#     A_pool = A_pool * (A_pool > threshold)
#     row_idx, col_idx = A_pool.nonzero(as_tuple=True)
#     edge_index_pool = torch.stack([row_idx, col_idx], dim=0)
#     return x_pool, edge_index_pool


def sparse_diff_pool(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    S: torch.Tensor,
    topk_per_row: int = 8,
    keep_self_loops: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DiffPool su grafi sparsi: A_pool = S^T A S, poi sparsifica con top-k per riga.
    - Simmetrizza A_pool.
    - Opzionalmente rimuove self-loops.
    """
    N, C = S.size()
    A = build_sparse_adj(edge_index, N)  # [N,N] SparseTensor
    AS = A.matmul(S)  # [N,C] denso
    A_pool = S.transpose(0, 1) @ AS  # [C,C] denso

    # simmetrizza per sicurezza
    A_pool = 0.5 * (A_pool + A_pool.transpose(0, 1))

    if not keep_self_loops:
        A_pool.fill_diagonal_(0.0)

    # top-k per riga
    if topk_per_row is not None and topk_per_row > 0 and topk_per_row < C:
        vals, idxs = torch.topk(A_pool, k=topk_per_row, dim=1)
        # costruiamo edge_index da righe e idxs
        row_idx = (
            torch.arange(C, device=A_pool.device)
            .unsqueeze(1)
            .expand_as(idxs)
            .reshape(-1)
        )
        col_idx = idxs.reshape(-1)
        mask = vals.reshape(-1) > 0
        row_idx = row_idx[mask]
        col_idx = col_idx[mask]
        # (opzionale) aggiungi simmetrici per sicurezza
        row_idx2 = col_idx
        col_idx2 = row_idx
        row_idx = torch.cat([row_idx, row_idx2], dim=0)
        col_idx = torch.cat([col_idx, col_idx2], dim=0)
        # rimuovi duplicati
        edge_index_pool = torch.stack([row_idx, col_idx], dim=0)
        edge_index_pool = torch.unique(edge_index_pool, dim=1)
    else:
        # fallback: threshold delicato
        thr = float(A_pool.abs().mean() * 0.1)
        row_idx, col_idx = (A_pool > thr).nonzero(as_tuple=True)
        edge_index_pool = torch.stack([row_idx, col_idx], dim=0)

    x_pool = S.transpose(0, 1) @ x  # [C,F]
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
        aggr: str = AGGREGATION,
    ):
        super().__init__(aggr=aggr, node_dim=0)
        if MLP_VARIANT == 0:
            self.mlp_msg = nn.Sequential(
                nn.Linear(2 * in_ch + (edge_dim or 0), hidden),
                nn.LayerNorm(hidden),
                nonlinearFn,
                nn.Dropout(dropout),
                nn.Linear(hidden, out_ch),
            )
            self.mlp_upd = nn.Sequential(
                nn.Linear(in_ch + out_ch, hidden),
                nn.LayerNorm(hidden),
                nonlinearFn,
                nn.Dropout(dropout),
                nn.Linear(hidden, out_ch),
            )
        elif MLP_VARIANT == 1:
            self.mlp_msg = nn.Sequential(
                nn.Linear(2 * in_ch + (edge_dim or 0), hidden),
                nonlinearFn,
                nn.Linear(hidden, hidden),
                nonlinearFn,
                nn.Linear(hidden, hidden),
                nonlinearFn,
                nn.Linear(hidden, out_ch),
                nn.LayerNorm(out_ch),
            )
            self.mlp_upd = nn.Sequential(
                nn.Linear(in_ch + out_ch, hidden),
                nonlinearFn,
                nn.Linear(hidden, hidden),
                nonlinearFn,
                nn.Linear(hidden, hidden),
                nonlinearFn,
                nn.Linear(hidden, out_ch),
                nn.LayerNorm(out_ch),
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
        # h = self.norm(h)
        if self.use_res:
            h = h + x

        # h = self.graphnorm(h, batch)
        h = F.gelu(h)

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

        self.num_clusters = num_clusters
        self.enforce_blockwise = ENFORCE_BLOCKWISE
        self.gnn = EdgeGNNLayer(ch, ch, edge_dim)
        self.assign = nn.Sequential(
            nn.Linear(ch, ch),
            nonlinearFn,
            # nn.LayerNorm(ch),
            nn.Linear(ch, num_clusters),
            nonlinearFn,
            nn.LayerNorm(num_clusters),
        )

    def forward(self, x, edge_index, edge_attr, batch, tau):
        # 1) message passing
        x = self.gnn(x, edge_index, edge_attr, batch)

        # 2) soft assignment (con tau)
        S = F.softmax(self.assign(x) / tau, dim=-1)  # [N, C]

        # 2b) opzionale: blockwise prima del pooling (coerenza encode/decode)
        if self.enforce_blockwise:
            unique_batches = batch.unique()
            G = unique_batches.numel()
            if G > 1:
                if self.num_clusters % G != 0:
                    raise ValueError(
                        f"Cannot enforce block-wise DiffPool with {self.num_clusters} clusters and {G} graphs in the batch."
                    )
                c_per_graph = self.num_clusters // G
                mask = S.new_zeros(S.shape)
                for g_idx, g in enumerate(unique_batches):
                    start = g_idx * c_per_graph
                    end = start + c_per_graph
                    node_mask = batch == g
                    mask[node_mask, start:end] = 1.0
                S = S * mask
                S = S / S.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # 3) salvi A_prev per le aux losses
        A_prev = build_sparse_adj(edge_index, x.size(0))

        # 4) pooling con la S definitiva (quella che salvi e userai in unpool)
        x_pool, edge_index_pool = sparse_diff_pool(x, edge_index, S)
        C = S.size(1)

        # 5) batch del grafo coarsened
        if batch.unique().numel() == 1:
            batch_coarse = x_pool.new_zeros(C, dtype=torch.long)
        else:
            weight = S.transpose(0, 1)  # [C, N]
            denom = weight.sum(dim=1, keepdim=True).clamp_min(1e-12)
            avg_batch = (weight @ batch.float().unsqueeze(1)) / denom  # [C,1]
            batch_coarse = avg_batch.round().squeeze(1).long()

        state = {
            "S": S,  # la S usata per pooling e che userai per unpool
            "prev_edge_index": edge_index,
            "prev_edge_attr": edge_attr,
            "A_prev": A_prev,
        }
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
            nonlinearFn,
            nn.Linear(hidden, hidden),
            nonlinearFn,
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.graphnorm = GraphNorm(hidden)
        # Encoder hierarchy (DiffPool-only)
        self.encoder = nn.ModuleList(
            [DiffPoolBlock(hidden, edge_dim, C) for C in clusters_per_level]
        )

        # Bottleneck on the coarsest graph
        self.bottleneck = EdgeGNNLayer(hidden, latent, edge_dim)
        # self.latent_up = nn.Linear(latent, hidden)
        self.latent_up = nn.Sequential(
            nn.Linear(latent, hidden),
            nonlinearFn,
            nn.Linear(hidden, hidden),
            nonlinearFn,
            nn.Linear(hidden, hidden),
        )

        # Decoder: mirror the depth with EdgeGNN layers
        depth = len(clusters_per_level)
        self.decoder = nn.ModuleList(
            [EdgeGNNLayer(hidden, hidden, edge_dim) for _ in range(depth)]
        )

        # Final prediction head
        # self.head = nn.Linear(hidden, out_ch)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nonlinearFn,
            nn.Linear(hidden, hidden),
            nonlinearFn,
            nn.Linear(hidden, out_ch),
        )
        # self.norm = nn.LayerNorm(out_ch)

    def forward(self, data, tau: float = 1.0, return_aux: bool = False):
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
                x, edge_index, edge_attr, batch, tau
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
            x = self.graphnorm(x, batch_prev)

        out = self.head(x)  # [N, out_ch]
        # out = self.norm(out)  #!!!!!!! WARNING: LayerNorm on output

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
        subgraph_type="bidirectional",
    )


# =========================
def train(
    model: nn.Module,
    loader: NeighborLoader,
    epochs: int = EPOCHS,
    lr: float = LR,
    use_amp: bool = USE_AMP,
    grad_clip: float = GRAD_CLIP,
    scheduler_step: int = SCHEDULER_STEP,  # (non usato qui, rimasto per compatibilità)
):
    from tqdm import trange
    import math

    model.to(device)
    scaler = torch.amp.GradScaler(enabled=use_amp, device=device.type)

    # Usa davvero il parametro lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # >>> COSINE per step (batch-wise): T_max = total_steps
    steps_per_epoch = max(1, len(loader))
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=3e-5
    )

    model.train()
    losses = []

    loop = trange(epochs, desc="Training", dynamic_ncols=True)

    global_step = 0
    for ep in loop:
        tot_loss = 0.0
        tot_link = 0.0
        tot_ent = 0.0
        tot_bal = 0.0
        n_batches = 0

        # TAU = max(
        #     0.50, 1.0 - 0.75 * (ep / epochs)
        # )  # annealing tau (se vuoi: fisso a 1.0)
        TAU = 1.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)

            center = getattr(batch, "batch_size", batch.x.size(0))
            target_dim = batch.y.size(1)

            with torch.amp.autocast(enabled=use_amp, device_type=device.type):
                if RETURN_AUX:
                    out, aux = model(batch, return_aux=True, tau=TAU)
                else:
                    out = model(batch, return_aux=False, tau=TAU)
                pred = out[:center, :target_dim]
                tgt = batch.y[:center, :target_dim]

                # componi la loss
                if RETURN_AUX:
                    loss_link = 0.0
                    loss_ent = 0.0
                    loss_bal = 0.0
                    for S, A_prev in aux:
                        lk, ent, bal = diffpool_aux_losses(A_prev, S)
                        loss_link = loss_link + lk
                        loss_ent = loss_ent + ent
                        loss_bal = loss_bal + bal
                    # pesi delicati sui termini ausiliari
                    w_link, w_ent, w_bal = 1e-3, 1e-4, 1e-3
                    loss_recon = F.mse_loss(pred, tgt)
                    loss = (
                        loss_recon
                        + w_link * loss_link
                        + w_ent * loss_ent
                        + w_bal * loss_bal
                    )
                else:
                    # due canali (u,v)
                    loss = F.mse_loss(pred[:, 0], tgt[:, 0]) + F.mse_loss(
                        pred[:, 1], tgt[:, 1]
                    )

            # AMP: backward scalato, poi unscale per fare clip
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Step ottimizzatore e scaler
            scaler.step(opt)
            scaler.update()

            # >>> Scheduler: SEMPRE una step per batch (senza gating su overflow)
            scheduler.step()
            global_step += 1

            # Accumulo per log epoca
            tot_loss += float(loss.detach())
            if RETURN_AUX:
                tot_link += float(loss_link.detach())
                tot_ent += float(loss_ent.detach())
                tot_bal += float(loss_bal.detach())
            n_batches += 1

        avg = tot_loss / max(1, n_batches)
        losses.append(avg)

        # Log ordinato: loss media epoca + lr corrente + tau
        log_dict = {
            "loss": f"{avg:.8f}",
            "lr": f"{scheduler.get_last_lr()[0]:.5e}",
            "tau": f"{TAU:.3f}",
        }
        if RETURN_AUX and n_batches > 0:
            log_dict.update(
                {
                    "link": f"{(tot_link/n_batches):.5f}",
                    "ent": f"{(tot_ent /n_batches):.5f}",
                    "bal": f"{(tot_bal /n_batches):.5f}",
                }
            )
        loop.set_postfix(log_dict)

    return model, losses


def trainSimple(
    model: nn.Module,
    loader: NeighborLoader,
    epochs: int = EPOCHS,
    lr: float = LR,
    use_amp: bool = USE_AMP,
    grad_clip: float = GRAD_CLIP,
    scheduler_step: int = SCHEDULER_STEP,  # (non usato qui, rimasto per compatibilità)
    num_workers: int = 5,
):
    from tqdm import trange
    import math

    model.to(device)
    scaler = torch.amp.GradScaler(enabled=use_amp, device=device.type)

    # Usa davvero il parametro lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # >>> COSINE per step (batch-wise): T_max = total_steps
    steps_per_epoch = max(1, len(loader))
    print("Len loader:", len(loader))
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=8e-6
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.999)

    model.train()
    losses = []

    loop = trange(epochs, desc="Training", dynamic_ncols=True)

    global_step = 0
    for ep in loop:
        tot_loss = 0.0
        tot_link = 0.0
        tot_ent = 0.0
        tot_bal = 0.0
        n_batches = 0

        # TAU = max(
        #     0.50, 1.0 - 0.75 * (ep / epochs)
        # )  # annealing tau (se vuoi: fisso a 1.0)
        TAU = 1.0
        # temp = 0
        for batch in loader:
            # print(f"Batch {temp}")
            # temp += 1
            batch = batch.to(device)
            opt.zero_grad()

            center = getattr(batch, "batch_size", batch.x.size(0))
            target_dim = batch.y.size(1)

            with torch.amp.autocast(enabled=use_amp, device_type=device.type):
                if RETURN_AUX:
                    out, aux = model(batch, return_aux=True, tau=TAU)
                else:
                    out = model(batch, return_aux=False, tau=TAU)
                pred = out[:center, :target_dim]
                tgt = batch.y[:center, :target_dim]

                # componi la loss
                if RETURN_AUX:
                    loss_link = 0.0
                    loss_ent = 0.0
                    loss_bal = 0.0
                    for S, A_prev in aux:
                        lk, ent, bal = diffpool_aux_losses(A_prev, S)
                        loss_link = loss_link + lk
                        loss_ent = loss_ent + ent
                        loss_bal = loss_bal + bal
                    # pesi delicati sui termini ausiliari
                    w_link, w_ent, w_bal = 1e-3, 1e-4, 1e-3
                    loss_recon = F.mse_loss(pred, tgt)
                    loss = (
                        loss_recon
                        + w_link * loss_link
                        + w_ent * loss_ent
                        + w_bal * loss_bal
                    )
                else:
                    # due canali (u,v)
                    loss = F.mse_loss(pred[:, 0], tgt[:, 0]) + F.mse_loss(
                        pred[:, 1], tgt[:, 1]
                    )
            loss.backward()

            # >>> Scheduler: SEMPRE una step per batch (senza gating su overflow)
            global_step += 1

            # Accumulo per log epoca
            tot_loss += float(loss.detach())
            if RETURN_AUX:
                tot_link += float(loss_link.detach())
                tot_ent += float(loss_ent.detach())
                tot_bal += float(loss_bal.detach())
            n_batches += 1

        avg = tot_loss / max(1, n_batches)
        losses.append(avg)

        opt.step()
        scheduler.step()

        # Log ordinato: loss media epoca + lr corrente + tau
        log_dict = {
            "loss": f"{avg:.8f}",
            "lr": f"{scheduler.get_last_lr()[0]:.5e}",
            "tau": f"{TAU:.3f}",
        }
        if RETURN_AUX and n_batches > 0:
            log_dict.update(
                {
                    "link": f"{(tot_link/n_batches):.5f}",
                    "ent": f"{(tot_ent /n_batches):.5f}",
                    "bal": f"{(tot_bal /n_batches):.5f}",
                }
            )
        loop.set_postfix(log_dict)

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
    model, loss_hist = trainSimple(model, loader)

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    np.savetxt(LOSS_PATH, np.array(loss_hist))
    print(f"Saved model, loss: {MODEL_PATH}, {LOSS_PATH}")


if __name__ == "__main__":
    main()
