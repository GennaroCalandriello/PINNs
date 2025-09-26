# diffpool_gnn_ae.py
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
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

HIDDEN = 100  # non dipende da hidden né latent, né da edge_hidden
LATENT = 30
EDGE_HIDDEN = 50  # misura
DROP = 0.0
LR = 2e-3
EPOCHS = 400
GRAD_CLIP = None

MODEL_PATH = "model/gnn_ae_diffpool1.pth"
LOSS_PATH = "model/loss_gnn_ae_diffpool1.txt"
MLP_VARIANT = 0  # 1 ~ MeshGraphNet-like MLP stack, 0 ~ lighter
AGGREGATION = "add"  # "add" | "mean" | "max" aggregation deve essere "add" per DiffPool
nonlinearFn = nn.GELU()

# BOOL FLAGS
USE_AMP = False
USE_COMPILE = False
RETURN_AUX = False  # return aux losses (link + entropy) from DiffPool
SELF_LOOP = True

# DiffPool hierarchy (number of clusters per pooling level)
CLUSTERS_PER_LEVEL: List[int] = [800]  # Non c'entra con la qualità dell'output


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
    Link + Entropy + Balance (senza densificare):
      - Pos: MSE( <S_i, S_j>, A_ij ) sugli edge
      - Neg: MSE( <S_i, S_j>, 0 ) su coppie (i,j) random
      - Entropy: media entropie riga di S (confidenza)
      - Balance: uso medio dei cluster ~ uniforme
    """
    row, col, val = A.coo()
    if drop_self_loops:
        mask = row != col
        row, col = row[mask], col[mask]
        val = None if val is None else val[mask]
    if val is None:
        val = S.new_ones(row.numel())

    # positivi
    s_i = S.index_select(0, row)
    s_j = S.index_select(0, col)
    pos_score = (s_i * s_j).sum(dim=1)
    pos_loss = F.mse_loss(pos_score, val)

    # negativi
    num_pos = row.numel()
    num_neg = max(1, int(neg_ratio * num_pos))
    i_neg = torch.randint(0, S.size(0), (num_neg,), device=S.device)
    j_neg = torch.randint(0, S.size(0), (num_neg,), device=S.device)
    neg_score = (S[i_neg] * S[j_neg]).sum(dim=1)
    neg_loss = (neg_score**2).mean()

    link_loss = 0.5 * (pos_loss + neg_loss)

    # entropy
    Sc = S.clamp_min(eps)
    entropy = -(Sc * Sc.log()).sum(dim=1).mean()

    # balance
    p = S.mean(dim=0)
    balance = ((p - 1.0 / S.size(1)) ** 2).mean()

    return link_loss, entropy, balance


def sparse_diff_pool2(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    S: torch.Tensor,
    topk_per_row: int = 8,
    keep_self_loops: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DiffPool su grafi sparsi: A_pool = S^T A S, sparsificato con top-k per riga.
    Assumiamo *singolo grafo* (nessun multigrafo / mini-batch).
    """
    N, C = S.size()
    A = build_sparse_adj(edge_index, N)  # [N,N] SparseTensor
    AS = A.matmul(S)  # [N,C] denso
    A_pool = S.transpose(0, 1) @ AS  # [C,C] denso

    # simmetrizza
    A_pool = 0.5 * (A_pool + A_pool.transpose(0, 1))
    if not keep_self_loops:
        A_pool.fill_diagonal_(0.0)

    # top-k per riga
    if topk_per_row is not None and 0 < topk_per_row < C:
        vals, idxs = torch.topk(A_pool, k=topk_per_row, dim=1)
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
        # aggiungi simmetrici
        edge_index_pool = torch.stack(
            [torch.cat([row_idx, col_idx], 0), torch.cat([col_idx, row_idx], 0)], dim=0
        )
        edge_index_pool = torch.unique(edge_index_pool, dim=1)
    else:
        thr = float(A_pool.abs().mean() * 0.1)
        row_idx, col_idx = (A_pool > thr).nonzero(as_tuple=True)
        edge_index_pool = torch.stack([row_idx, col_idx], dim=0)

    x_pool = S.transpose(0, 1) @ x  # [C,F]
    return x_pool, edge_index_pool


def sparse_diff_pool(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    S: torch.Tensor,
    topk_per_row: int = 8,
    keep_self_loops: bool = False,
    sparsify_mode: str = "mst_topk",  # 'topk' | 'quantile' | 'energy' | 'mst_topk' | 'knn'
    quantile: float = 0.90,
    energy_frac: float = 0.85,
    knn_k: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DiffPool su grafi sparsi: A_pool = S^T A S, poi sparsificazione.
    Assunzione: singolo grafo.

    sparsify_mode:
      - topk: (default) k massimi per riga
      - quantile: tieni pesi >= quantile globale
      - energy: ordina tutti i pesi e tieni quelli che coprono una frazione dell'energia (somma)
      - mst_topk: spanning tree (con pesi) + (k-1) extra per riga
      - knn: kNN sui centroidi (S^T X) in spazio feature (ignora A_pool, ricostruisce)
    """
    N, C = S.size()
    A = build_sparse_adj(edge_index, N)
    AS = A.matmul(S)  # [N,C]
    A_pool = S.transpose(0, 1) @ AS  # [C,C] denso

    A_pool = 0.5 * (A_pool + A_pool.transpose(0, 1))
    if not keep_self_loops:
        A_pool.fill_diagonal_(0.0)

    # Pooled features
    x_pool = S.transpose(0, 1) @ x  # [C,F]

    def make_edge_index(mask_mat: torch.Tensor) -> torch.Tensor:
        row_idx, col_idx = mask_mat.nonzero(as_tuple=True)
        if row_idx.numel() == 0:
            # fallback: connect linear chain
            row_idx = torch.arange(C - 1, device=A_pool.device)
            col_idx = row_idx + 1
        edge_index_pool = torch.stack([row_idx, col_idx], dim=0)
        # forza simmetria
        rev = edge_index_pool.flip(0)
        edge_index_pool = torch.cat([edge_index_pool, rev], dim=1)
        edge_index_pool = torch.unique(edge_index_pool, dim=1)
        return edge_index_pool

    if C <= 2:
        # trivial
        edge_index_pool = (
            torch.tensor([[0, 1], [1, 0]], device=x.device)
            if C == 2
            else torch.zeros((2, 0), dtype=torch.long, device=x.device)
        )
        return x_pool, edge_index_pool

    mode = sparsify_mode.lower()

    if mode == "topk":
        if topk_per_row is not None and 0 < topk_per_row < C:
            vals, idxs = torch.topk(A_pool, k=topk_per_row, dim=1)
            mask = torch.zeros_like(A_pool, dtype=torch.bool)
            mask.scatter_(1, idxs, vals > 0)
            edge_index_pool = make_edge_index(mask)
        else:
            thr = float(A_pool.abs().mean() * 0.1)
            mask = A_pool > thr
            edge_index_pool = make_edge_index(mask)

    elif mode == "quantile":
        flat = A_pool.flatten()
        pos = flat[flat > 0]
        if pos.numel() == 0:
            edge_index_pool = torch.zeros(
                (2, 0), dtype=torch.long, device=A_pool.device
            )
        else:
            kq = max(1, int((1 - quantile) * pos.numel()))
            thresh = torch.topk(pos, k=kq, largest=True).values.min()
            mask = A_pool >= thresh
            edge_index_pool = make_edge_index(mask)

    elif mode == "energy":
        flat = A_pool.flatten()
        pos = flat[flat > 0]
        if pos.numel() == 0:
            edge_index_pool = torch.zeros(
                (2, 0), dtype=torch.long, device=A_pool.device
            )
        else:
            vals_sorted, idx_sorted = torch.sort(pos, descending=True)
            cumsum = torch.cumsum(vals_sorted, dim=0)
            total = cumsum[-1]
            keep_upto = torch.searchsorted(cumsum, energy_frac * total).item() + 1
            thresh = vals_sorted[keep_upto - 1]
            mask = A_pool >= thresh
            edge_index_pool = make_edge_index(mask)

    elif mode == "mst_topk":
        # Build MST using negative weights (we want maximum spanning tree)
        # Dense Prim (OK if C moderate)
        W = A_pool.clone()
        W[W < 0] = 0
        selected = torch.zeros(C, dtype=torch.bool, device=W.device)
        selected[0] = True
        mst_edges = []
        for _ in range(C - 1):
            # edges from selected -> not selected
            mask_from = selected.unsqueeze(1) & (~selected).unsqueeze(0)
            cand = W * mask_from
            val, idx = torch.max(cand.reshape(-1), dim=0)
            if val <= 0:
                break
            r = idx // C
            c = idx % C
            mst_edges.append((r.item(), c.item()))
            selected[c] = True
        mst_mask = torch.zeros_like(W, dtype=torch.bool)
        for r, c in mst_edges:
            mst_mask[r, c] = True
            mst_mask[c, r] = True
        # add local top-k
        k_loc = max(1, topk_per_row) if topk_per_row else 4
        vals, idxs = torch.topk(W, k=min(k_loc, C), dim=1)
        topk_mask = torch.zeros_like(W, dtype=torch.bool)
        topk_mask.scatter_(1, idxs, vals > 0)
        mask = mst_mask | topk_mask
        edge_index_pool = make_edge_index(mask)

    elif mode == "knn":
        # kNN on pooled features
        Xn = F.normalize(x_pool, dim=-1)
        sim = Xn @ Xn.t()
        sim.fill_diagonal_(-1e9)
        k_loc = min(knn_k, C - 1)
        vals, idxs = torch.topk(sim, k=k_loc, dim=1)
        mask = torch.zeros_like(sim, dtype=torch.bool)
        mask.scatter_(1, idxs, vals > -1e8)
        edge_index_pool = make_edge_index(mask)

    else:
        # fallback: simple threshold
        thr = float(A_pool.abs().mean() * 0.1)
        mask = A_pool > thr
        edge_index_pool = make_edge_index(mask)

    return x_pool, edge_index_pool


# =========================
# GNN Building Blocks
# =========================
class EdgeGNNLayer(MessagePassing):
    """
    Edge-aware message passing:
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
        else:  # deeper
            self.mlp_msg = nn.Sequential(
                # nn.LayerNorm(2 * in_ch + (edge_dim or 0)), #NO FA CASINO
                nn.Linear(2 * in_ch + (edge_dim or 0), hidden),
                nonlinearFn,
                nn.Linear(hidden, hidden),
                nonlinearFn,
                nn.Linear(hidden, hidden),
                nonlinearFn,
                nn.Linear(hidden, out_ch),
                # nn.LayerNorm(out_ch), # NO FA CASINO
            )
            self.mlp_upd = nn.Sequential(
                # nn.LayerNorm(in_ch + out_ch), #NO FA CASINO
                nn.Linear(in_ch + out_ch, hidden),
                nonlinearFn,
                nn.Linear(hidden, hidden),
                nonlinearFn,
                nn.Linear(hidden, hidden),
                nonlinearFn,
                nn.Linear(hidden, out_ch),
                # nn.LayerNorm(out_ch), # NO FA CASINO
            )

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(out_ch)
        self.graphnorm = GraphNorm(out_ch)
        self.use_res = in_ch == out_ch
        self.edge_dim = edge_dim or 0

    def forward(self, x, edge_index, edge_attr, batch=None):
        x = self.dropout(x)
        # x = self.layernorm(x)

        # Self-loops (e padding edge_attr se serve)
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
        if self.use_res:
            h = h + x
        # h = self.layernorm(h) # NO FA CASINO
        h = self.graphnorm(h, batch) if batch is not None else h
        h = F.gelu(h)
        # h = self.layernorm(h)
        return h

    def message(self, x_i, x_j, edge_attr):
        if edge_attr is None:
            edge_attr = x_i.new_zeros((x_i.size(0), self.edge_dim))
        m_in = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp_msg(m_in)


class DiffPoolBlock(nn.Module):
    """
    Encoder stage (singolo grafo):
      - EdgeGNN -> node embeddings
      - Soft assignment S = softmax(MLP(x))
      - DiffPool: (x', edge') = DiffPool(x, A, S)
    """

    def __init__(
        self, ch: int, edge_dim: int, num_clusters: int, layer_norm: bool = True
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.gnn = EdgeGNNLayer(ch, ch, edge_dim)
        self.assign = nn.Sequential(
            nn.Linear(ch, ch),
            nonlinearFn,
            nn.Linear(ch, num_clusters),
            nonlinearFn,
            # nn.LayerNorm(num_clusters),
        )
        if layer_norm:
            self.assign.add_module("ln", nn.LayerNorm(num_clusters))
        self.graphnorm = GraphNorm(ch)

    def forward(self, x, edge_index, edge_attr, tau: float):
        # 1) message passing
        x = self.gnn(x, edge_index, edge_attr, batch=None)
        x = self.graphnorm(
            x, torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )  # QUESTO ABBASSA LA LOSS
        # 2) soft assignment
        S = F.softmax(self.assign(x) / tau, dim=-1)  # [N, C]

        # 3) A_prev per aux losses
        A_prev = build_sparse_adj(edge_index, x.size(0))

        # 4) pooling (singolo grafo)
        x_pool, edge_index_pool = sparse_diff_pool(x, edge_index, S)

        state = {
            "S": S,
            "prev_edge_index": edge_index,
            "prev_edge_attr": edge_attr,
            "A_prev": A_prev,
        }
        return (x_pool, edge_index_pool, None), state


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
            nn.LayerNorm(in_ch),
            nn.Linear(in_ch, hidden),
            nonlinearFn,
            nn.Linear(hidden, hidden),
            nonlinearFn,
            nn.Linear(hidden, hidden),
            # nn.LayerNorm(hidden),  # NON SERVE
        )
        self.graphnorm = GraphNorm(hidden)

        # Encoder (DiffPool-only)
        self.encoder = nn.ModuleList(
            [DiffPoolBlock(hidden, edge_dim, C) for C in clusters_per_level]
        )

        # Bottleneck sul grafo più grosso
        self.bottleneck = EdgeGNNLayer(hidden, latent, edge_dim)
        self.latent_up = nn.Sequential(
            # nn.LayerNorm(latent),  # preconditioning (?)
            nn.Linear(latent, hidden),
            nonlinearFn,
            nn.Linear(hidden, hidden),
            nonlinearFn,
            nn.Linear(hidden, hidden),
            # nn.LayerNorm(hidden),  # preconditioning (?)
        )

        # Decoder: uno per livello
        depth = len(clusters_per_level)
        self.decoder = nn.ModuleList(
            [EdgeGNNLayer(hidden, hidden, edge_dim) for _ in range(depth)]
        )

        # Head finale
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nonlinearFn,
            nn.Linear(hidden, hidden),
            nonlinearFn,
            nn.Linear(hidden, out_ch),
            # nn.LayerNorm(out_ch),
        )

    def forward(self, data, tau: float = 1.0, return_aux: bool = False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Singolo grafo
        x = self.in_proj(x)
        # x = self.graphnorm(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        states = []

        # ENCODER
        for enc in self.encoder:
            (x, edge_index, edge_attr), state = enc(x, edge_index, edge_attr, tau)
            states.append(state)

        # BOTTLENECK (sul grafo coarsest)
        x = self.bottleneck(x, edge_index, edge_attr, batch=None)
        x = self.latent_up(x)

        # DECODER + UNPOOL mirrorando gli stati
        for dec, state in zip(reversed(self.decoder), reversed(states)):
            S = state["S"]
            edge_index_prev = state["prev_edge_index"]
            edge_attr_prev = state["prev_edge_attr"]

            # unpool: [N_prev, hidden] = S @ [C, hidden]
            x = S @ x
            # GraphNorm: batch tutto zero (singolo grafo)
            batch_prev = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = dec(x, edge_index_prev, edge_attr_prev, batch_prev)
            x = self.graphnorm(x, batch_prev)

        out = self.head(x)  # [N, out_ch]

        if return_aux:
            aux = [(st["S"], st["A_prev"]) for st in states]
            return out, aux
        return out


# =========================
# Training (full graph, cosine per epoca)
# =========================
def train(
    model: nn.Module,
    data,  # PyG Data completo (x, edge_index, edge_attr, y)
    epochs: int = EPOCHS,
    lr: float = LR,
    use_amp: bool = USE_AMP,
    grad_clip: float = GRAD_CLIP,
):
    from tqdm import trange

    model.to(device)
    data = data.to(device)
    print("alcuni dati:", data.edge_index[:, :100], data.x[:5], data.y[:5])
    scaler = torch.amp.GradScaler(enabled=use_amp, device=device.type)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine per-epoch: T_max = epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=8e-6
    )

    model.train()
    losses = []
    loop = trange(epochs, desc="Training", dynamic_ncols=True)

    for ep in loop:
        opt.zero_grad(set_to_none=True)
        TAU = 1.0
        # TAU = max(0.5, 1.0 - ep / epochs)  # annealing
        target_dim = data.y.size(1)

        with torch.amp.autocast(enabled=use_amp, device_type=device.type):
            if RETURN_AUX:
                out, aux = model(data, return_aux=True, tau=TAU)
            else:
                out = model(data, return_aux=False, tau=TAU)

            pred = out[:, :target_dim]
            tgt = data.y[:, :target_dim]
            lambda_mag = 0.0  # da tarare
            mag_pred = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-12)
            mag_tgt = torch.sqrt(tgt[:, 0] ** 2 + tgt[:, 1] ** 2 + 1e-12)
            loss_mag = F.mse_loss(mag_pred, mag_tgt)
            # print(f"Loss mag: {loss_mag.item():.6f}")
            if RETURN_AUX:
                loss_link = 0.0
                loss_ent = 0.0
                loss_bal = 0.0
                for S, A_prev in aux:
                    lk, ent, bal = diffpool_aux_losses(A_prev, S)
                    loss_link = loss_link + lk
                    loss_ent = loss_ent + ent
                    loss_bal = loss_bal + bal
                w_link, w_ent, w_bal = 1e-3, 1e-4, 1e-3
                loss_recon = F.mse_loss(pred, tgt)
                loss = w_ent * loss_ent + w_bal * loss_bal + loss_recon
            else:
                loss = F.mse_loss(pred, tgt) + lambda_mag * loss_mag

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()

        scheduler.step()

        avg = float(loss.detach())
        losses.append(avg)

        log_dict = {
            "loss": f"{avg:.8f}",
            "lr": f"{scheduler.get_last_lr()[0]:.5e}",
            "tau": f"{TAU:.3f}",
        }
        if RETURN_AUX:
            log_dict.update(
                {
                    "link": f"{float(loss_link):.5f}",
                    "ent": f"{float(loss_ent):.5f}",
                    "bal": f"{float(loss_bal):.5f}",
                }
            )
        loop.set_postfix(log_dict)

    return model, losses


# =========================
# Main
# =========================
def main():
    # Carica il tuo grafo (PyG Data) con x, edge_index, edge_attr, y
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
        model = torch.compile(model)

    model, loss_hist = train(model, data)

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    np.savetxt(LOSS_PATH, np.array(loss_hist))
    print(f"Saved model, loss: {MODEL_PATH}, {LOSS_PATH}")


if __name__ == "__main__":
    main()
