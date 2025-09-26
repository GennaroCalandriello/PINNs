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

HIDDEN = 100  # best con 72 ma errore alto
LATENT = 60  # best con 42 ma errore alto
EDGE_HIDDEN = 100  # best con 52 ma errore alto
DROP = 0.0
LR = 1e-3
EPOCHS = 1400
BATCH_SIZE_NODES = 4000  # best con 16000 ma errore alto

if BATCH_SIZE_NODES is not None:
    ENFORCE_BLOCKWISE = True  # avoid graph mixing in the multigraph case
else:
    ENFORCE_BLOCKWISE = False

NEIGHBORS = [80, 80, 80]
GRAD_CLIP = None

SCHEDULER_STEP = 200

MODEL_PATH = "model/gnn_ae_diffpool1.pth"
LOSS_PATH = "model/loss_gnn_ae_diffpool1.txt"
MLP_VARIANT = 1  # 1 per simil MeshGraphNet, 0 per MeshCutriStyle
AGGREGATION = "add"  # "add" | "mean" | "max" ---- prima era "mean" ora "add"

# BOOL FLAGS
USE_AMP = True
USE_COMPILE = False
RETURN_AUX = False  # return aux losses (link + entropy) from DiffPool
SELF_LOOP = True  # prima era True

# DiffPool hierarchy (number of clusters per pooling level)
CLUSTERS_PER_LEVEL: List[int] = [800]  # best con [800] ma errore alto


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
        aggr: str = AGGREGATION,
    ):
        super().__init__(aggr=aggr, node_dim=0)
        if MLP_VARIANT == 0:
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
        elif MLP_VARIANT == 1:
            self.mlp_msg = nn.Sequential(
                nn.Linear(2 * in_ch + (edge_dim or 0), hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_ch),
                nn.LayerNorm(out_ch),
            )
            self.mlp_upd = nn.Sequential(
                nn.Linear(in_ch + out_ch, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
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

        self.num_clusters = num_clusters
        self.enforce_blockwise = ENFORCE_BLOCKWISE
        self.gnn = EdgeGNNLayer(ch, ch, edge_dim)
        self.assign = nn.Sequential(
            nn.Linear(ch, ch),
            nn.GELU(),
            # nn.LayerNorm(ch),
            nn.Linear(ch, num_clusters),
            nn.GELU(),
            nn.LayerNorm(num_clusters),
        )

    #  def forward(self, x, edge_index, edge_attr, batch, tau):
    #     x = self.gnn(x, edge_index, edge_attr, batch)
    #     S = F.softmax(self.assign(x) / tau, dim=-1)  # [N, C]

    #     A_prev = build_sparse_adj(edge_index, x.size(0))  # se usi aux losses

    #     x_pool, edge_index_pool = sparse_diff_pool(x, edge_index, S)
    #     C = S.size(1)

    #     # *** QUI: batch per il grafo coarsened (C nodi) ***
    #     if batch is None or batch.numel() == 0 or batch.unique().numel() == 1:
    #         batch_coarse = x_pool.new_zeros(C, dtype=torch.long)  # tutto un grafo
    #     else:
    #         # caso multi-grafo (se/quando servirà): "mode" del batch per cluster
    #         # hard cluster
    #         cluster = S.argmax(dim=1)  # [N]
    #         # costruiamo per-cluster un assegnamento di batch tramite maggioranza
    #         # (semplice fallback: prendiamo il batch del primo nodo assegnato)
    #         batch_coarse = x_pool.new_zeros(C, dtype=torch.long)
    #         batch_coarse.index_copy_(
    #             0, cluster, batch
    #         )  # semplice ma funziona se non mischia grafi

    #     state = {
    #         "S": S,
    #         "prev_edge_index": edge_index,
    #         "prev_edge_attr": edge_attr,
    #         "A_prev": A_prev,
    #     }
    #     # *** ritorna batch_coarse ***
    #     return (x_pool, edge_index_pool, None, batch_coarse), state

    def forward(self, x, edge_index, edge_attr, batch, tau):

        # 1. Message passing
        x = self.gnn(x, edge_index, edge_attr, batch)

        # 2. Soft assignment with temperature
        S = F.softmax(self.assign(x) / tau, dim=-1)  # [N, C]

        A_prev = build_sparse_adj(edge_index, x.size(0))  # se usi aux losses

        x_pool, edge_index_pool = sparse_diff_pool(x, edge_index, S)
        C = S.size(1)

        # 3.  # 3. Optional: block-diagonalize S so graphs do not mix clusters
        #    Requires that num_clusters is divisible by num_graphs in this mini-batch.
        if self.enforce_blockwise:
            unique_batches = batch.unique()
            G = unique_batches.numel()  # number of graphs in the batch
            if G > 1:
                if self.num_clusters % G != 0:
                    raise ValueError(
                        f"Cannot enforce block-wise DiffPool with {self.num_clusters} clusters and {G} graphs in the batch."
                    )
                c_per_graph = self.num_clusters // G
                # Build mask
                mask = S.new_zeros(S.shape)
                for g_idx, g in enumerate(unique_batches):
                    start = g_idx * c_per_graph
                    end = start + c_per_graph
                    node_mask = batch == g
                    mask[node_mask, start:end] = 1.0
                S = S * mask
                # re-normalize each row (avoid rows becoming all-zero)
                row_sum = S.sum(dim=1, keepdim=True).clamp_min(1e-12)
                S = S / row_sum

        # *** QUI: batch per il grafo coarsened (C nodi) ***
        if batch.unique().numel() == 1:
            batch_coarse = x_pool.new_zeros(C, dtype=torch.long)  # tutto un grafo
        else:
            batch_float = batch.float()
            weight = S.transpose(0, 1)  # [C, N]
            denom = weight.sum(dim=1, keepdim=True) + 1e-12
            avg_batch = (weight @ batch_float.unsqueeze(1)) / denom  # [C,1]
            batch_coarse = avg_batch.round().squeeze(1).long()
            # caso multi-grafo: "mode" del batch per cluster
            # hard cluster
            # cluster = S.argmax(dim=1)  # [N]
            # # costruiamo per-cluster un assegnamento di batch tramite maggioranza
            # # (semplice fallback: prendiamo il batch del primo nodo assegnato)
            # batch_coarse = x_pool.new_zeros(C, dtype=torch.long)
            # batch_coarse.index_copy_(
            #     0, cluster, batch
            # )  # semplice ma funziona se non mischia grafi

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
        # self.latent_up = nn.Linear(latent, hidden)
        self.latent_up = nn.Sequential(nn.Linear(latent, hidden), nn.GELU())

        # Decoder: mirror the depth with EdgeGNN layers
        depth = len(clusters_per_level)
        self.decoder = nn.ModuleList(
            [EdgeGNNLayer(hidden, hidden, edge_dim) for _ in range(depth)]
        )

        # Final prediction head
        # self.head = nn.Linear(hidden, out_ch)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
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

        out = self.head(x)
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
    )


# =========================
# Training
# =========================
# def train(
#     model: nn.Module,
#     loader: NeighborLoader,
#     epochs: int = EPOCHS,
#     lr: float = LR,
#     use_amp: bool = USE_AMP,
#     grad_clip: float = GRAD_CLIP,
#     scheduler_step: int = SCHEDULER_STEP,
# ):
#     from tqdm import trange

#     model.to(device)
#     # opt = torch.optim.Adam(model.parameters(), lr=lr)
#     # scheduler = torch.optim.lr_scheduler.StepLR(
#     #     opt, step_size=scheduler_step, gamma=0.9
#     # )
#     scaler = torch.amp.GradScaler(enabled=use_amp, device=device.type)
#     opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         opt, T_max=epochs, eta_min=3e-5
#     )

#     model.train()
#     losses = []

#     loop = trange(EPOCHS, desc="Training", dynamic_ncols=True)

#     for ep in loop:
#         tot = 0.0
#         n_batches = 0
#         TAU = max(0.60, 1.0 - 0.75 * (ep / (epochs)))  # annealing
#         # TAU = 1.0  # no annealing

#         for batch in loader:
#             batch = batch.to(device)
#             opt.zero_grad(set_to_none=True)

#             # NeighborLoader marks the first "center" nodes
#             center = getattr(batch, "batch_size", batch.x.size(0))

#             # Infer output channels from target y
#             target_dim = batch.y.size(1)

#             with torch.amp.autocast(enabled=use_amp, device_type=device.type):
#                 if RETURN_AUX:
#                     out, aux = model(
#                         batch, return_aux=RETURN_AUX, tau=TAU
#                     )  # [N, out_ch]
#                 else:
#                     out = model(batch, return_aux=False)  # [N, out_ch]
#                 pred = out[:center, :target_dim]  # center-node supervision
#                 u_pred = pred[:, 0]
#                 v_pred = pred[:, 1]
#                 tgt = batch.y[:center, :target_dim]
#                 u_tgt = tgt[:, 0]
#                 v_tgt = tgt[:, 1]

#                 # loss sui link e entropia
#                 if RETURN_AUX:
#                     loss_link, loss_ent = 0.0, 0.0
#                     for S, A_prev in aux:
#                         lk, ent, bal = diffpool_aux_losses(A_prev, S)
#                         loss_link += lk
#                         loss_ent += ent
#                     loss = (
#                         F.mse_loss(pred, tgt)
#                         + 0.001 * loss_link
#                         + 0.001 * loss_ent
#                         + 0.001 * bal
#                     )
#                 else:
#                     loss = F.mse_loss(u_pred, u_tgt) + F.mse_loss(v_pred, v_tgt)

#             scaler.scale(loss).backward()
#             if grad_clip is not None:
#                 scaler.unscale_(opt)
#                 nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

#             scale_before = scaler.get_scale() if use_amp else None
#             scaler.step(opt)
#             scaler.update()
#             if use_amp:
#                 scale_after = scaler.get_scale()
#                 # se c'è stato overflow, optimizer.step() è saltato => NON step del scheduler
#                 if scale_after >= scale_before:
#                     scheduler.step()
#             else:
#                 scheduler.step()

#             tot += loss.item()
#             n_batches += 1
#         if RETURN_AUX:
#             loop.set_postfix(
#                 {
#                     "loss": f"{(tot / max(1, n_batches)):.6f}",
#                     "link": f"{(loss_link.item() / max(1, n_batches)):.6f}",
#                     "ent": f"{(loss_ent.item() / max(1, n_batches)):.6f}",
#                     "tau": f"{TAU:.4f}",
#                 }
#             )
#         else:
#             loop.set_postfix(
#                 {
#                     "loss": f"{loss.item():.6f}",
#                     "tau": f"{TAU:.4f}",
#                 }
#             )

#         avg = tot / max(1, n_batches)
#         losses.append(avg)


#     return model, losses
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

        TAU = max(
            0.50, 1.0 - 0.75 * (ep / epochs)
        )  # annealing tau (se vuoi: fisso a 1.0)

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
