import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_max
from torch import Tensor
import numpy as np

from ns_GNN_cav import (
    createGraphData,  # your data function
    dataLoader,
    dataNormalizer,
    geometryObject,
    path_data,
)

# ---------------Hyperparameters-----------------
SCHEDULER_STEP = 200
EPOCHS = 3000
HIDDEN_DIM = 64
LATENT_DIM = 50
NUM_LAYERS = 6
USE_EDGE_ATTN = True
USE_NODE_ATTN = True
USE_ST_GUMBEL = True  # kept for diagnostics only
USE_POOL = True
DROPOUT = 0.1
LEAKY_SLOPE = 0.2

TAU = 0.6
POOL_CLUSTERS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
SCHEDULER_TYPE = "StepLR"  # "StepLR" or "CosineAnnealingLR"
SCHEDULER_GAMMA = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------- Utils -----------------
def scatter_softmax(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    max_per_group, _ = scatter_max(src, index, dim_size=dim_size)
    max_per_group = max_per_group.index_select(0, index)
    expo = (src - max_per_group).exp()
    denom = scatter_add(expo, index, dim_size=dim_size).index_select(0, index)
    return expo / (denom + 1e-12)


def one_hot_argmax(logits: Tensor) -> Tensor:
    index = logits.argmax(dim=-1)
    oneH = torch.zeros_like(logits)
    oneH.scatter_(1, index.unsqueeze(-1), 1.0)
    return oneH


def build_geom_edge_attr(pos: Tensor, edge_index: Tensor) -> Tensor:
    """
    Geometry-only edge features (2D/3D ok):
      [rel(x_j - x_i), r=||rel||, dir=rel/r, invr=1/r]
    Returns [E, 2*D + 2]
    """
    src, dst = edge_index
    rel = pos[dst] - pos[src]  # [E, D]
    r = (rel.pow(2).sum(-1, keepdim=True).sqrt()).clamp_min(1e-8)  # [E,1]
    dir_ = rel / r  # [E, D]
    invr = 1.0 / r  # [E,1]
    return torch.cat([rel, r, dir_, invr], dim=-1)  # [E, 2D+2]


def geom_edge_dim_from_pos_dim(D: int) -> int:
    # rel (D) + r(1) + dir(D) + invr(1) = 2D + 2
    return 2 * D + 2


# ----------------- Layers -----------------
class AttentionGNOLayer(MessagePassing):
    """
    Attention message passing over edges using node+edge embeddings.
    Uses node-attention α_ij (softmax over in-edges of i).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        hidden_dim=HIDDEN_DIM,
        use_edge_attn=USE_EDGE_ATTN,
        use_node_attn=USE_NODE_ATTN,
        dropout=DROPOUT,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.use_node_attn = use_node_attn
        self.use_edge_attn = use_edge_attn

        self.lin_node = nn.Linear(in_channels, hidden_dim)
        self.lin_edge = nn.Linear(edge_dim, hidden_dim)

        if use_node_attn:
            self.a_node = nn.Linear(3 * hidden_dim, 1)

        # (If you later want edge-attn inside MP, wire self.a_edge here)
        if use_edge_attn:
            self.a_edge = nn.Linear(3 * hidden_dim, 1)

        self.msg_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels),
        )

        self.upd = nn.Sequential(
            nn.Linear(in_channels + out_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels),
        )

        self.norm_pre = nn.LayerNorm(in_channels)
        self.norm_post = nn.LayerNorm(out_channels)
        self.residual = in_channels == out_channels
        self.leaky = nn.LeakyReLU(LEAKY_SLOPE)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x = self.norm_pre(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.norm_post(out)
        return out + x if self.residual else out

    def message(self, x_i, x_j, edge_attr, index):
        xi = self.lin_node(x_i)
        xj = self.lin_node(x_j)
        ej = self.lin_edge(edge_attr)
        cat = torch.cat([xi, xj, ej], dim=-1)

        if self.use_node_attn:
            e_node = self.leaky(getattr(self, "a_node")(cat)).squeeze(-1)  # [E]
            size_i = int(index.max().item()) + 1
            alpha_ij = scatter_softmax(e_node, index, dim_size=size_i)  # [E]
        else:
            alpha_ij = torch.ones(cat.size(0), device=cat.device)

        m_raw = self.msg_mlp(cat)
        return self.dropout(m_raw * alpha_ij.unsqueeze(-1))

    def update(self, aggr_out, x):
        return self.upd(torch.cat([x, aggr_out], dim=-1))


class GNOLayer(MessagePassing):
    """Non-attentive fallback layer."""

    def __init__(self, in_channels, out_channels, hidden_dim, edge_dim):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_channels),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_channels),
        )
        self.residual = in_channels == out_channels

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out + x if self.residual else out

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


# ---------- Differentiable Edge-Node Attention Pool (SOFT) ----------
class EdgeNodeAttentionPool(nn.Module):
    """
    Fully differentiable pooling:
      - S_soft = softmax(logits)
      - X' = S^T (alpha ⊙ z)
      - A' = (S[src] ⊙ w)^T S[dst]   with global edge attention beta
      - E' (edge attrs) coarsened analogously
    Returns sparse coarsened graph (top-k per row) + S_soft for soft unpool.
    """

    def __init__(
        self,
        in_dim,
        hidden=HIDDEN_DIM,
        C=POOL_CLUSTERS,
        use_st_gumbel=USE_ST_GUMBEL,
        tau=TAU,
        topk=8,
    ):
        super().__init__()
        self.C = C
        self.use_st_gumbel = use_st_gumbel
        self.tau = tau
        self.topk = topk

        self.pool_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, C)
        )
        self.Wm = nn.Linear(in_dim, hidden)
        self.am = nn.Linear(hidden, 1)
        self.leaky = nn.LeakyReLU(LEAKY_SLOPE)

        self.We = nn.Linear(in_dim, hidden)
        self.ae = nn.Linear(2 * hidden, 1)

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_weight: Tensor = None,
    ):
        device = z.device
        N, d = z.size()
        E = edge_index.size(1)
        if edge_weight is None:
            edge_weight = z.new_ones(E)

        # (1) Assignments (soft for all computations)
        logits = self.pool_mlp(z)  # [N, C]
        S_soft = F.softmax(logits, dim=-1)  # [N, C]

        # Optional hard view for diagnostics only
        if self.use_st_gumbel:
            S_st = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        else:
            S_st = one_hot_argmax(S_soft)
        clusters_ids = S_st.argmax(dim=-1)  # [N] (not used in computations)

        # (2) Node gate
        alpha = torch.sigmoid(self.am(self.leaky(self.Wm(z))))  # [N,1]

        # X' = S^T (alpha ⊙ z)
        X_coarse = S_soft.T @ (alpha * z)  # [C, d]

        # (3) Global edge attention β_ij
        src, dst = edge_index
        zi, zj = z[dst], z[src]
        se_ij = self.ae(
            torch.cat([self.leaky(self.We(zi)), self.leaky(self.We(zj))], dim=-1)
        ).squeeze(
            -1
        )  # [E]
        beta = F.softmax(se_ij, dim=0)  # [E]
        w = (edge_weight * beta).unsqueeze(-1)  # [E,1]

        # (4) Coarse adjacency (dense CxC then sparsify)
        S_src, S_dst = S_soft[src], S_soft[dst]  # [E,C], [E,C]
        A_coarse = (S_src * w).T @ S_dst  # [C,C]

        # (5) Coarse edge attributes via weighted projection
        edge_dim = edge_attr.size(1)
        E_num_list = []
        for k in range(edge_dim):
            wk = (w.squeeze(-1) * edge_attr[:, k]).unsqueeze(-1)  # [E,1]
            Ek = (S_src * wk).T @ S_dst  # [C,C]
            E_num_list.append(Ek)
        E_num = torch.stack(E_num_list, dim=-1)  # [C,C,de]
        E_coarse_dense = E_num / (A_coarse.unsqueeze(-1) + 1e-12)  # [C,C,de]

        # (6) Sparsify with row-wise top-k
        C = self.C
        k = min(self.topk, C)
        vals, idxs = torch.topk(A_coarse, k=k, dim=1)  # [C,k]
        rows = torch.arange(C, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
        cols = idxs.reshape(-1)
        pw = vals.reshape(-1)  # [E']
        edge_index_coarse = torch.stack([rows, cols], dim=0)  # [2,E']
        edge_attr_coarse = E_coarse_dense[rows, cols, :]  # [E',de]

        return X_coarse, edge_index_coarse, pw, edge_attr_coarse, clusters_ids, S_soft


# ----------------- Encoder(s) -----------------
class GNNEncoderWithPool(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        edge_dim,
        num_layers,
        pool_clusters=POOL_CLUSTERS,
    ):
        super().__init__()
        self.edge_dim = edge_dim

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.layers_pre = nn.ModuleList(
            [
                AttentionGNOLayer(
                    hidden_dim,
                    hidden_dim,
                    edge_dim,
                    hidden_dim,
                    use_edge_attn=USE_EDGE_ATTN,
                    use_node_attn=USE_NODE_ATTN,
                )
                for _ in range(num_layers // 2)
            ]
        )

        self.pool = EdgeNodeAttentionPool(
            in_dim=hidden_dim,
            hidden=hidden_dim,
            C=pool_clusters,
            use_st_gumbel=USE_ST_GUMBEL,
            tau=TAU,
            topk=8,
        )

        self.layers_post = nn.ModuleList(
            [
                AttentionGNOLayer(
                    hidden_dim,
                    hidden_dim,
                    edge_dim,
                    hidden_dim,
                    use_edge_attn=USE_EDGE_ATTN,
                    use_node_attn=USE_NODE_ATTN,
                )
                for _ in range(num_layers - num_layers // 2)
            ]
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.embedding(x)
        for layer in self.layers_pre:
            x = layer(x, edge_index, edge_attr)

        Xc, Eic, Ewc, Eac, clusters, S_soft = self.pool(
            x, edge_index, edge_attr=edge_attr
        )

        for layer in self.layers_post:
            Xc = layer(Xc, Eic, Eac)

        znodes = self.to_latent(Xc)  # [C, latent_dim]
        zgraph = znodes.mean(dim=0, keepdim=True)  # [1, latent_dim]
        return znodes, zgraph, (Eic, Eac, clusters, S_soft)


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, edge_dim, num_layers):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                GNOLayer(hidden_dim, hidden_dim, hidden_dim, edge_dim)
                for _ in range(num_layers)
            ]
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return self.to_latent(x)


# ----------------- Decoder (geometry-only edges, lazy build) -----------------
class GNNDecoder(nn.Module):
    """
    Builds edge features from geometry (pos), not from data.edge_attr.
    Lazily instantiates AttentionGNOLayers when edge_dim is known.
    """

    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layers = None  # built on first forward when edge_dim is known
        self.head = nn.Linear(hidden_dim, output_dim)

    def _build_layers(self, edge_dim: int):
        self.layers = nn.ModuleList(
            [
                AttentionGNOLayer(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    edge_dim=edge_dim,
                    hidden_dim=self.hidden_dim,
                    use_edge_attn=USE_EDGE_ATTN,
                    use_node_attn=USE_NODE_ATTN,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, z_node, edge_index, pos):
        x = self.pre(z_node)  # [N, hidden]
        dec_edge_attr = build_geom_edge_attr(pos, edge_index)
        if self.layers is None:
            self._build_layers(dec_edge_attr.size(1))
        for layer in self.layers:
            x = layer(x, edge_index, dec_edge_attr)
        return self.head(x)  # [N, output_dim]


# ----------------- Autoencoder -----------------
class GNNAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        output_dim,
        edge_dim,
        num_layers=NUM_LAYERS,
        use_pool=USE_POOL,
        pool_clusters=POOL_CLUSTERS,
    ):
        super().__init__()
        self.use_pool = use_pool

        if use_pool:
            self.encoder = GNNEncoderWithPool(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                edge_dim=edge_dim,
                num_layers=num_layers,
                pool_clusters=pool_clusters,
            )
        else:
            self.encoder = GNNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                edge_dim=edge_dim,
                num_layers=num_layers,
            )

        # Decoder uses geometry-built edges → no edge_dim needed
        self.decoder = GNNDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )

    def forward(self, data):
        enc_out = self.encoder(data)
        if self.use_pool:
            # encoder returns: Zc [C,d], z_graph [1,d], meta=(eic, eac, clusters, S_soft)
            Zc, Zgraph, meta = enc_out
            _, _, _, S = meta  # S_soft [N,C]
            z_node = S @ Zc  # SOFT unpool: [N, latent_dim]
        else:
            z_node = enc_out  # [N, latent_dim]

        # Decoder uses geometry-only edges derived from positions
        return self.decoder(z_node, data.edge_index, data.pos)


# ----------------- Training -----------------
def train(
    data,
    model,
    epochs=EPOCHS,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    scheduler=SCHEDULER_TYPE,
    scheduler_step=SCHEDULER_STEP,
    scheduler_gamma=SCHEDULER_GAMMA,
    use_amp=True,
    include_pressure=False,
    max_grad_norm=1.0,
):
    data = data.to(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    sl = scheduler.lower()
    if sl in {"steplr", "step"}:
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step, gamma=scheduler_gamma
        )
    elif sl in {"cosineannealinglr", "cosine"}:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
    else:
        sched = None

    scaler = torch.amp.GradScaler(enabled=use_amp)
    autocast_ctx = (
        torch.amp.autocast(enabled=use_amp, device_type=device.type)
        if use_amp
        else nullcontext()
    )

    os.makedirs("model", exist_ok=True)
    loss_history = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            pred = model(data)  # [N, 2 or 3]
            u_pred, v_pred = pred[:, 0], pred[:, 1]
            u_true, v_true = data.y[:, 0], data.y[:, 1]
            loss = F.mse_loss(u_pred, u_true) + F.mse_loss(v_pred, v_true)

            if include_pressure and pred.size(1) > 2 and data.y.size(1) > 2:
                p_pred = pred[:, 2]
                p_true = data.y[:, 2]
                loss = loss + F.mse_loss(p_pred, p_true)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if sched is not None:
            sched.step()

        loss_history.append(loss.item())
        if epoch % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs}  Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "model/gnn_autoencoder.pth")
    np.savetxt("model/loss_history.txt", np.array(loss_history))


# ----------------- Run -----------------
if __name__ == "__main__":
    data = createGraphData()
    input_dim = data.x.size(1)
    edge_dim = data.edge_attr.size(1)  # used only in encoder
    output_dim = data.y.size(1)

    model = GNNAutoEncoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        output_dim=output_dim,
        edge_dim=edge_dim,
        num_layers=NUM_LAYERS,
        use_pool=USE_POOL,
        pool_clusters=POOL_CLUSTERS,
    ).to(device)

    # Quick sanity: ensure decoder can build geometry edges
    assert hasattr(data, "pos"), "data.pos required for geometry-based decoder."
    train(data, model)
