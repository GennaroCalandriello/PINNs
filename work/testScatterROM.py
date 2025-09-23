# scatter_hier_ae.py
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, GraphNorm
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import add_self_loops

from torch_scatter import scatter_add, scatter_mean

# Se hai già questo modulo, ok; altrimenti sostituisci con il tuo dataset
from ns_GNN_cav2 import (
    createGraphData,
)  # -> deve fornire Data con x, y, edge_index, edge_attr (opz), pos

# =========================
# Settings / Hyperparameters
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
print(f"[ScatterHierarchyAE] Using device: {device}")

# Dimensionalità e training
HIDDEN = 96
LATENT = 30
DROP = 0.0
LR = 3e-4
EPOCHS = 1200
BATCH_SIZE_NODES = 12000
NEIGHBORS = [80, 80, 80]
GRAD_CLIP = 1.0
USE_AMP = True

# Gerarchia: numero di nodi coarse al livello 0 (puoi mettere più livelli)
# Se LEVELS_NODES = [1300, 300], avrai 2 livelli: fine->1300->300
LEVELS_NODES: List[int] = [1300]

# Mappa soft? Se SOFT_K=0 = hard nearest; se >0 usa K-NN soft
SOFT_K = 0  # es. 4 per soft-KNN

# Paths
MODEL_PATH = "model/scatter_hier_ae.pth"
LOSS_PATH = "model/loss_scatter_hier_ae.txt"


# =========================
# Utility: riproducibilità
# =========================
def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# LevelMap & scatter ops
# =========================
class LevelMap:
    """
    Mappa fine→coarse per un livello:
      - hard: idx: LongTensor [N]                , w=None
      - soft: idx: LongTensor [N,K], w: Float [N,K] con sum_k w[n,k] = 1
    num_coarse: numero di nodi del livello coarse.
    """

    def __init__(
        self, idx: torch.Tensor, num_coarse: int, w: Optional[torch.Tensor] = None
    ):
        assert idx.dim() in (1, 2), "idx deve essere [N] (hard) o [N,K] (soft)"
        if w is not None:
            assert idx.shape == w.shape, "idx e w devono avere stessa shape [N,K]"
        self.idx = idx
        self.w = w
        self.num_coarse = int(num_coarse)


def _pool_scatter(x_fine: torch.Tensor, m: LevelMap) -> torch.Tensor:
    """
    x_fine: [N, F]  → x_coarse: [C, F]
    """
    if m.idx.dim() == 1:  # hard 1-to-1
        if m.w is None:
            return scatter_mean(x_fine, m.idx, dim=0, dim_size=m.num_coarse)
        else:
            num = scatter_add(
                x_fine * m.w.unsqueeze(-1), m.idx, dim=0, dim_size=m.num_coarse
            )
            den = (
                scatter_add(m.w, m.idx, dim=0, dim_size=m.num_coarse)
                .clamp_min(1e-12)
                .unsqueeze(-1)
            )
            return num / den
    else:  # soft K-NN
        N, F = x_fine.size()
        K = m.idx.size(1)
        idx_flat = m.idx.reshape(-1)  # [N*K]
        w_flat = (
            m.w.reshape(-1)
            if m.w is not None
            else torch.ones_like(
                m.idx, dtype=x_fine.dtype, device=x_fine.device
            ).reshape(-1)
        )
        x_rep = x_fine.repeat_interleave(K, dim=0)  # [N*K, F]
        num = scatter_add(
            x_rep * w_flat.unsqueeze(-1), idx_flat, dim=0, dim_size=m.num_coarse
        )
        den = (
            scatter_add(w_flat, idx_flat, dim=0, dim_size=m.num_coarse)
            .clamp_min(1e-12)
            .unsqueeze(-1)
        )
        return num / den


def _unpool_scatter(x_coarse: torch.Tensor, m: LevelMap) -> torch.Tensor:
    """
    coarse→fine (prolungamento):
    - hard:   x_fine[n] = x_coarse[idx[n]]
    - soft K: x_fine[n] = sum_k w[n,k] * x_coarse[idx[n,k]]
    """
    if m.idx.dim() == 1:
        x_fine = x_coarse.index_select(0, m.idx)
        if m.w is not None:
            x_fine = x_fine * m.w.unsqueeze(-1)
        return x_fine
    else:
        gather = x_coarse.index_select(0, m.idx.reshape(-1))  # [N*K, F]
        Fdim = gather.size(-1)
        gather = gather.view(m.idx.size(0), m.idx.size(1), Fdim)  # [N,K,F]
        w = (
            m.w
            if m.w is not None
            else torch.full_like(gather[..., 0], 1.0 / m.idx.size(1))
        )
        return (w.unsqueeze(-1) * gather).sum(dim=1)  # [N,F]


# =========================
# Costruzione LevelMap da posizioni
# =========================
@torch.no_grad()
def farthest_point_sampling(pos: torch.Tensor, C: int) -> torch.Tensor:
    """
    Selezione grezza "farthest point" su pos (N,D) per avere ancore coarse (C,D).
    Ritorna gli indici negli N punti.
    """
    N = pos.size(0)
    device_ = pos.device
    choice = torch.empty(C, dtype=torch.long, device=device_)
    # inizializza con un punto casuale
    choice[0] = torch.randint(0, N, (1,), device=device_)
    d2 = torch.full((N,), float("inf"), device=device_)
    for i in range(1, C):
        # aggiorna distanze minime al set scelto
        last = pos[choice[i - 1]].unsqueeze(0)  # [1,D]
        d2 = torch.minimum(d2, (pos - last).pow(2).sum(-1))
        choice[i] = d2.argmax()
    return choice  # [C]


@torch.no_grad()
def make_levelmap_hard_nearest(
    pos_fine: torch.Tensor, anchors: torch.Tensor
) -> LevelMap:
    """
    pos_fine: [N,D], anchors: [C,D] (coarse)
    Ritorna LevelMap hard: idx: [N], num_coarse=C
    """
    # Distanze (attenzione ai grandi N,C; per N~O(1e4), C~O(1e3) ok)
    d2 = torch.cdist(pos_fine, anchors, p=2.0)  # [N,C]
    idx = d2.argmin(dim=1).to(torch.long)  # [N]
    return LevelMap(idx=idx, num_coarse=anchors.size(0))


@torch.no_grad()
def make_levelmap_soft_knn(
    pos_fine: torch.Tensor, anchors: torch.Tensor, K: int = 4, eps: float = 1e-9
) -> LevelMap:
    """
    pos_fine: [N,D], anchors: [C,D]
    Soft K-NN: idx: [N,K], w: [N,K] con normalizzazione 1/sum d
    """
    dist = torch.cdist(pos_fine, anchors, p=2.0)  # [N,C]
    dvals, idx = torch.topk(dist, k=min(K, anchors.size(0)), dim=1, largest=False)
    inv = 1.0 / (dvals + eps)
    w = inv / inv.sum(dim=1, keepdim=True)
    return LevelMap(idx=idx.to(torch.long), num_coarse=anchors.size(0), w=w)


def build_hierarchy_from_positions(
    pos_fine: torch.Tensor, levels_nodes: List[int], soft_k: int = 0, seed: int = 42
) -> List[LevelMap]:
    """
    Costruisce una gerarchia di LevelMap: fine -> C0 -> C1 -> ...
    Al livello 0 usa pos_fine; ai livelli successivi usa le posizioni degli "anchors" precedenti.
    """
    torch.manual_seed(seed)
    hierarchy: List[LevelMap] = []
    # livello 0: seleziona ancore su pos_fine
    C0 = levels_nodes[0]
    idx0 = farthest_point_sampling(pos_fine, C0)  # [C0]
    anchors0 = pos_fine.index_select(0, idx0)  # [C0,D]
    if soft_k and soft_k > 0:
        level0 = make_levelmap_soft_knn(pos_fine, anchors0, K=soft_k)
    else:
        level0 = make_levelmap_hard_nearest(pos_fine, anchors0)
    hierarchy.append(level0)

    # livelli successivi: ancore da anchors del livello precedente
    prev_anchors = anchors0
    for i in range(1, len(levels_nodes)):
        Ci = levels_nodes[i]
        idx_i = farthest_point_sampling(
            prev_anchors, Ci
        )  # scelgo ancore dentro gli anchors precedenti
        anchors_i = prev_anchors.index_select(0, idx_i)
        # mappa dai "nodi del livello i-1" (cioè gli anchors) ai "nodi del livello i"
        if soft_k and soft_k > 0:
            level_i = make_levelmap_soft_knn(prev_anchors, anchors_i, K=soft_k)
        else:
            level_i = make_levelmap_hard_nearest(prev_anchors, anchors_i)
        hierarchy.append(level_i)
        prev_anchors = anchors_i
    return hierarchy


# =========================
# MeshGraphNet layer (rifinitura)
# =========================
class EdgeGNNLayer(MessagePassing):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        edge_dim: Optional[int],
        hidden: int = 64,
        dropout: float = 0.0,
        aggr: str = "mean",
    ):
        super().__init__(aggr=aggr, node_dim=0)
        self.edge_dim = edge_dim or 0
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * in_ch + self.edge_dim, hidden),
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
        self.use_res = in_ch == out_ch
        self.norm = nn.LayerNorm(out_ch)
        self.graphnorm = GraphNorm(out_ch)

    def forward(self, x, edge_index, edge_attr, batch):
        if edge_index is None:
            return x
        if edge_attr is None and self.edge_dim > 0:
            edge_attr = x.new_zeros((edge_index.size(1), self.edge_dim))
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if edge_attr is not None:
            num_added = edge_index_sl.size(1) - edge_index.size(1)
            edge_attr = torch.cat(
                [edge_attr, edge_attr.new_zeros((num_added, edge_attr.size(1)))], dim=0
            )
        out = self.propagate(edge_index_sl, x=x, edge_attr=edge_attr)
        h = self.mlp_upd(torch.cat([x, out], dim=-1))
        h = self.norm(h)
        if self.use_res:
            h = h + x
        h = F.gelu(h)
        return self.graphnorm(h, batch)

    def message(self, x_i, x_j, edge_attr):
        if edge_attr is None:
            edge_attr = x_i.new_zeros((x_i.size(0), self.edge_dim))
        return self.mlp_msg(torch.cat([x_i, x_j, edge_attr], dim=-1))


# =========================
# Modello AE gerarchico scatter
# =========================
class ScatterHierarchyAE(nn.Module):
    """
    Encoder (no pesi):
      X0(fine) --pool(idx0)--> X1 --pool(idx1)--> ... --pool(idxL-1)--> XL
      poi FNN:  XL -> Z (LATENT per-nodo sulla coarse più grossa)
    Decoder:
      Z -> FNN -> H_L  --unpool(idxL-1)--> H_{L-1} --GNN--> ... --unpool(idx0)--> H_0 --GNN--> Y
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        edge_dim: int,
        hierarchy: List[LevelMap],
        hidden: int = HIDDEN,
        latent: int = LATENT,
        gnn_refine_layers: int = 1,
    ):
        super().__init__()
        self.hierarchy = hierarchy
        self.L = len(hierarchy)

        # Proiezione iniziale
        self.in_proj = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.LayerNorm(hidden), nn.GELU()
        )

        # Bottleneck (solo MLP per-nodo sulla coarse più grossa)
        self.to_latent = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, latent)
        )
        self.from_latent = nn.Sequential(nn.Linear(latent, hidden), nn.GELU())

        # Refinement GNN dopo ogni unpool
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        EdgeGNNLayer(hidden, hidden, edge_dim)
                        for _ in range(gnn_refine_layers)
                    ]
                )
                for _ in range(self.L)
            ]
        )

        # Testa finale
        self.head = nn.Linear(hidden, out_ch)

    def _subselect_levelmap_for_batch(
        self, m: LevelMap, n_id: Optional[torch.Tensor]
    ) -> LevelMap:
        """
        Con NeighborLoader, data ha .n_id (indici globali dei nodi del subgrafo).
        Sotto-selezioniamo idx/w per il batch corrente.
        """
        if n_id is None:
            return m
        if m.idx.dim() == 1:
            idx_b = m.idx.index_select(0, n_id)
            w_b = None if m.w is None else m.w.index_select(0, n_id)
        else:
            idx_b = m.idx.index_select(0, n_id)
            w_b = None if m.w is None else m.w.index_select(0, n_id)
        return LevelMap(idx_b, m.num_coarse, w_b)

    def forward(
        self,
        data,
        edge_index_levels: Optional[List[torch.Tensor]] = None,
        edge_attr_levels: Optional[List[torch.Tensor]] = None,
    ):
        """
        Se non passi edge_index_levels/edge_attr_levels, usa sempre il grafo fine del batch.
        """
        x, edge_index_fine, edge_attr_fine = (
            data.x,
            data.edge_index,
            getattr(data, "edge_attr", None),
        )
        batch = getattr(data, "batch", None)
        n_id = getattr(data, "n_id", None)

        x = self.in_proj(x)

        # ----- ENCODER: scatter pooling -----
        maps_batch = []
        for m in self.hierarchy:
            mb = self._subselect_levelmap_for_batch(m, n_id)
            maps_batch.append(mb)
            x = _pool_scatter(x, mb)  # [C_l, hidden]

        # Bottleneck
        z = self.to_latent(x)  # [C_L, LATENT]
        x = self.from_latent(z)  # [C_L, hidden]

        # ----- DECODER: unpool + GNN refine -----
        for l in reversed(range(self.L)):
            mb = maps_batch[l]
            # coarse -> fine
            x = _unpool_scatter(x, mb)  # [N_{l}, hidden]

            # grafo per rifinitura di questo livello
            if edge_index_levels is not None and len(edge_index_levels) == self.L:
                ei = edge_index_levels[l]
                ea = (
                    None
                    if (edge_attr_levels is None or len(edge_attr_levels) != self.L)
                    else edge_attr_levels[l]
                )
            else:
                ei, ea = edge_index_fine, edge_attr_fine

            for g in self.refine[l]:
                x = g(x, ei, ea, batch)

        # Testa
        out = self.head(x)
        return out


# =========================
# Data Loader
# =========================
def GraphLoader(graph_data, batch_size_nodes=BATCH_SIZE_NODES, neighbors=NEIGHBORS):
    return NeighborLoader(
        graph_data,
        num_neighbors=neighbors,
        batch_size=batch_size_nodes,
        input_nodes=None,  # tutti i nodi
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
):
    from tqdm import trange

    model.to(device)
    scaler = torch.amp.GradScaler(enabled=use_amp, device=device.type)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    losses = []
    loop = trange(epochs, desc="Training", dynamic_ncols=True)

    for ep in loop:
        tot = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)

            center = getattr(batch, "batch_size", batch.x.size(0))  # center nodes

            with torch.amp.autocast(enabled=use_amp, device_type=device.type):
                out = model(batch)  # [N_b, out_ch]
                tgt = batch.y[:center, : out.size(1)]
                pred = out[:center, : out.size(1)]
                loss = F.mse_loss(pred, tgt)

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            tot += loss.item()
            n_batches += 1

        avg = tot / max(1, n_batches)
        losses.append(avg)
        loop.set_postfix({"loss": f"{avg:.6f}"})

    return model, losses


# =========================
# Main
# =========================
def main():
    set_seed(123)
    os.makedirs("model", exist_ok=True)

    # Carica grafo (deve avere .pos con coordinate nodali!)
    data = createGraphData()
    assert (
        hasattr(data, "pos") and data.pos is not None
    ), "Serve data.pos per costruire la gerarchia scatter."
    data = data.to(device)

    in_ch = data.x.size(-1)
    out_ch = data.y.size(-1)
    edge_dim = (
        0 if (getattr(data, "edge_attr", None) is None) else data.edge_attr.size(-1)
    )

    # ===== Gerarchia: costruzione da posizioni =====
    # Usa farthest-point sampling sugli N punti di data.pos per ottenere gli anchors di ogni livello.
    # Se hai già una mappa globale idx (N)→[0..C-1], puoi sostituire build_hierarchy_from_positions con la tua.
    hierarchy = build_hierarchy_from_positions(
        pos_fine=data.pos, levels_nodes=LEVELS_NODES, soft_k=SOFT_K, seed=123
    )
    # Sposta LevelMap su device
    for m in hierarchy:
        m.idx = m.idx.to(device)
        if m.w is not None:
            m.w = m.w.to(device)

    # ===== Modello =====
    model = ScatterHierarchyAE(
        in_ch=in_ch,
        out_ch=out_ch,
        edge_dim=edge_dim,
        hierarchy=hierarchy,
        hidden=HIDDEN,
        latent=LATENT,
        gnn_refine_layers=1,
    ).to(device)

    # ===== Loader =====
    loader = GraphLoader(data, batch_size_nodes=BATCH_SIZE_NODES, neighbors=NEIGHBORS)

    # ===== Train =====
    model, loss_hist = train(model, loader)

    # ===== Save =====
    torch.save(model.state_dict(), MODEL_PATH)
    np.savetxt(LOSS_PATH, np.array(loss_hist))
    print(f"Saved model, loss: {MODEL_PATH}, {LOSS_PATH}")


if __name__ == "__main__":
    main()
