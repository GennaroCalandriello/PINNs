import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import coalesce
from torch import Tensor
import numpy as np
from ns_GNN_cav import (
    createGraphData,  # use your data function here
    dataLoader,
    dataNormalizer,
    path_data,
)

SCHEDULER_STEP = 200
EPOCHS = 1000
HIDDEN_DIM = 128
LATENT_DIM = 64
NUM_LAYERS = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Utilities ----------
def scatter_softmax(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    # Numerically stable softmax per group "index"
    # src: [E], index: [E] in [0, dim_size)
    max_per_group, _ = scatter_max(src, index, dim_size=dim_size)
    max_per_group = max_per_group.index_select(0, index)
    exps = (src - max_per_group).exp()
    denom = scatter_add(exps, index, dim_size=dim_size).index_select(0, index)
    return exps / (denom + 1e-12)


def one_hot_argmax(logits: Tensor) -> Tensor:
    # logits: [N, C] -> one-hot hard assignment [N, C]
    idx = logits.argmax(dim=-1)  # [N]
    oh = torch.zeros_like(logits)
    oh.scatter_(1, idx.unsqueeze(1), 1.0)
    return oh


# ---------- Attention GNO layer ----------
class AttnGNOLayer(MessagePassing):
    """
    Graph message passing with additive attention on nodes (α_ij) and a separate
    edge attention (β_ij). Messages are:
        m_ij = α_ij * β_ij * MLP_msg([x_i', x_j', e_ij'])
    and aggregated with 'add'. Residual + LN included.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        hidden_dim=128,
        use_edge_attn=True,
        use_node_attn=True,
        dropout=0.1,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.use_node_attn = use_node_attn
        self.use_edge_attn = use_edge_attn

        # Linear embeddings
        self.lin_x = nn.Linear(in_channels, hidden_dim)
        self.lin_e = nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None
        self.lin_out = nn.Linear(hidden_dim, out_channels)

        # Additive attention params (node α_ij over neighbors of i)
        if use_node_attn:
            self.a_node = nn.Linear(
                3 * hidden_dim if edge_dim > 0 else 2 * hidden_dim, 1
            )
        # Separate edge attention β_ij (also additive)
        if use_edge_attn:
            self.a_edge = nn.Linear(
                3 * hidden_dim if edge_dim > 0 else 2 * hidden_dim, 1
            )

        # Message MLP
        msg_in = 2 * hidden_dim + (hidden_dim if edge_dim > 0 else 0)
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update block
        self.upd = nn.Sequential(
            nn.Linear(in_channels + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels),
        )

        self.norm_pre = nn.LayerNorm(in_channels)
        self.norm_post = nn.LayerNorm(out_channels)
        self.residual = in_channels == out_channels
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None):
        x = self.norm_pre(x)
        return self._forward_impl(x, edge_index, edge_attr)

    def _forward_impl(self, x, edge_index, edge_attr):
        # propagate calls message -> aggregate -> update
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = self.norm_post(out)
        if self.residual:
            out = out + x
        return out

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        xi = self.lin_x(x_i)
        xj = self.lin_x(x_j)
        if self.lin_e is not None and edge_attr is not None:
            ej = self.lin_e(edge_attr)
            cat_for_msg = torch.cat([xi, xj, ej], dim=-1)
            att_cat = cat_for_msg
        else:
            cat_for_msg = torch.cat([xi, xj], dim=-1)
            att_cat = cat_for_msg

        # Node attention α_ij : softmax over neighbors j of the same target i
        if self.use_node_attn:
            e_node = self.leaky(self.a_node(att_cat)).squeeze(-1)  # [E]
            alpha = torch.zeros_like(e_node)
            # index is the target node indices (i). Softmax per i
            alpha = scatter_softmax(e_node, index, dim_size=size_i)
        else:
            alpha = torch.ones((att_cat.size(0),), device=att_cat.device)

        # Edge attention β_ij : softmax over neighbors j of the same target i (or shared group)
        if self.use_edge_attn:
            e_edge = self.leaky(self.a_edge(att_cat)).squeeze(-1)  # [E]
            beta = scatter_softmax(e_edge, index, dim_size=size_i)
        else:
            beta = torch.ones((att_cat.size(0),), device=att_cat.device)

        msg = self.msg_mlp(cat_for_msg)
        msg = msg * alpha.unsqueeze(-1) * beta.unsqueeze(-1)
        return self.dropout(msg)

    def update(self, aggr_out, x):
        upd = self.upd(torch.cat([x, aggr_out], dim=-1))
        return upd


class GNOLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim, edge_dim):
        super(GNOLayer, self).__init__(aggr="mean")
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


# ---------- Edge–Node Attention Pool (hard assignment) ----------
class EdgeNodeAttentionPool(nn.Module):
    """
    ENA Pooling layer:
      - Assignment: Z -> logits -> hard C-way assignment (argmax or Gumbel-ST)
      - Node attention (global scores normalized within each cluster) -> coarsened X'
      - Edge attention (scores for edges normalized within each cluster-pair) -> coarsened A'
    Returns:
      X_coarse [C, d], edge_index_coarse [2, E'], edge_weight_coarse [E'], cluster_ids [N], S_hard [N, C]
    """

    def __init__(self, in_dim, hidden=128, C=64, use_st_gumbel=True, tau=1.0):
        super().__init__()
        self.C = C
        self.use_st = use_st_gumbel
        self.tau = tau

        # Pool/embed head → assignment logits
        self.pool_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, C)
        )

        # Node attention score m_i = a^T Leaky(Wm z_i)
        self.Wm = nn.Linear(in_dim, hidden)
        self.am = nn.Linear(hidden, 1)
        self.leaky = nn.LeakyReLU(0.2)

        # Edge attention score s_ij = ae^T Leaky([We z_i || We z_j])
        self.We = nn.Linear(in_dim, hidden)
        self.ae = nn.Linear(2 * hidden, 1)

    @torch.no_grad()
    def _hard_from_soft(self, S_soft: Tensor) -> Tensor:
        if self.use_st:
            # Straight-through Gumbel-Softmax (hard)
            S_hard = F.gumbel_softmax(S_soft, tau=self.tau, hard=True, dim=-1)
        else:
            S_hard = one_hot_argmax(S_soft)
        return S_hard

    def forward(self, z: Tensor, edge_index: Tensor, edge_weight: Tensor = None):
        """
        z: [N, d] node embeddings
        edge_index: [2, E]
        edge_weight: [E] or None
        """
        N, d = z.size()
        device = z.device
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device)

        # --- (1) Assignment ---
        logits = self.pool_mlp(z)  # [N, C]
        S_soft = F.softmax(logits, dim=-1)
        S_hard = self._hard_from_soft(logits)  # [N, C] one-hot or ST-hard
        cluster_ids = S_hard.argmax(dim=-1)  # [N]

        # --- (2) Node attention within each cluster ---
        m = self.am(self.leaky(self.Wm(z))).squeeze(-1)  # [N] global node scores
        # softmax within each cluster id
        alpha = scatter_softmax(
            m, cluster_ids, dim_size=self.C
        )  # [N] normalized per cluster
        # Coarsen X': sum_i in cluster p alpha_i z_i
        X_coarse = torch.zeros(self.C, d, device=device)
        X_coarse = scatter_add(
            alpha.unsqueeze(-1) * z,
            cluster_ids.unsqueeze(-1).expand(-1, d),
            dim=0,
            dim_size=self.C,
        )

        # --- (3) Edge attention within each cluster pair (p, q) ---
        src, dst = edge_index
        zi, zj = z[src], z[dst]
        se = self.ae(
            torch.cat([self.leaky(self.We(zi)), self.leaky(self.We(zj))], dim=-1)
        ).squeeze(
            -1
        )  # [E]

        p = cluster_ids[src]  # [E]
        q = cluster_ids[dst]  # [E]
        pair_id = p * self.C + q
        num_pairs = self.C * self.C

        beta = scatter_softmax(
            se, pair_id, dim_size=num_pairs
        )  # [E] normalized within cluster-pair
        weighted = beta * edge_weight  # [E]

        # Aggregate into coarse adjacency A'_{pq} = sum_{i in p, j in q} beta_ij * w_ij
        A_flat = scatter_add(weighted, pair_id, dim_size=num_pairs)  # [C*C]
        # Build sparse edge_index_coarse / edge_weight_coarse
        nonzero = A_flat > 0
        pq_idx = nonzero.nonzero(as_tuple=False).squeeze(-1)  # [E']
        pw = A_flat[pq_idx]  # [E']
        p_idx = pq_idx // self.C
        q_idx = pq_idx % self.C
        edge_index_coarse = torch.stack([p_idx, q_idx], dim=0)
        edge_index_coarse, edge_weight_coarse = coalesce(
            edge_index_coarse, pw, m=self.C, n=self.C
        )

        return X_coarse, edge_index_coarse, edge_weight_coarse, cluster_ids, S_hard


# ---------- How to plug it into your model ----------

# 1) Replace GNOLayer with AttnGNOLayer in your encoder/decoder:
# self.layers = nn.ModuleList([
#     AttnGNOLayer(hidden_dim, hidden_dim, edge_dim, hidden_dim, use_edge_attn=True, use_node_attn=True)
#     for _ in range(num_layers)
# ])


# 2) (Optional) Add pooling between blocks (example inside an encoder):
class GNNEncoderWithPool(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, latent_dim, edge_dim, num_layers, pool_clusters=64
    ):
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
                AttnGNOLayer(
                    hidden_dim,
                    hidden_dim,
                    edge_dim,
                    hidden_dim,
                    use_edge_attn=True,
                    use_node_attn=True,
                )
                for _ in range(num_layers // 2)
            ]
        )
        self.pool = EdgeNodeAttentionPool(
            in_dim=hidden_dim,
            hidden=hidden_dim,
            C=pool_clusters,
            use_st_gumbel=True,
            tau=1.0,
        )
        self.layers2 = nn.ModuleList(
            [
                AttnGNOLayer(
                    hidden_dim,
                    hidden_dim,
                    edge_dim=0,
                    hidden_dim=hidden_dim,
                    use_edge_attn=True,
                    use_node_attn=True,
                )
                for _ in range(num_layers - num_layers // 2)
            ]
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # Pool once (coarsen graph)
        Xc, EIc, EWc, clusters, S_hard = self.pool(x, edge_index)
        # After pooling we don’t carry edge attributes; you can re-derive geom features on EIc if needed
        for layer in self.layers2:
            Xc = layer(Xc, EIc, None)
        return self.to_latent(Xc)


# =========================
# Training utilities
# =========================
from contextlib import nullcontext


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, data, uvnorm=None, use_denorm=False, include_p=False):
    model.eval()
    out = model(data)
    # out: [N, 3] -> u,v,(p)
    u_pred, v_pred = out[:, 0], out[:, 1]
    u_true, v_true = data.y[:, 0], data.y[:, 1]

    if use_denorm and (uvnorm is not None):
        u_pred = uvnorm.decode(u_pred, idx=0)
        v_pred = uvnorm.decode(v_pred, idx=1)
        u_true = uvnorm.decode(u_true, idx=0)
        v_true = uvnorm.decode(v_true, idx=1)

    loss = F.mse_loss(u_pred, u_true) + F.mse_loss(v_pred, v_true)

    if include_p and out.size(1) >= 3 and data.y.size(1) >= 3:
        p_pred = out[:, 2]
        p_true = data.y[:, 2]
        if use_denorm and (uvnorm is not None):
            p_pred = uvnorm.decode(p_pred, idx=2)
            p_true = uvnorm.decode(p_true, idx=2)
        loss = loss + F.mse_loss(p_pred, p_true)

    return loss.item()


def train_attention_autoencoder(
    data,
    model,
    *,
    epochs=3000,
    lr=1e-3,
    weight_decay=1e-4,
    scheduler="cosine",  # "cosine" or "step" or None
    scheduler_step=500,
    scheduler_gamma=0.9,
    max_grad_norm=1.0,
    use_amp=True,
    include_p=False,
    uvnorm=None,  # pass the uv GaussianNormalizer if you want denorm metrics
    denorm_metrics=False,  # compute metrics in physical units (slower)
    save_path="model/gnn_autoencoder_attn.pth",
):
    """
    Trains the attention-based GNN autoencoder on a single (possibly large) graph 'data'.
    Minimizes MSE on u and v (optionally p).
    """
    device = next(model.parameters()).device
    data = data.to(device)
    print(f"Model parameters: {count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step, gamma=scheduler_gamma
        )
    else:
        sched = None

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    autocast_ctx = (
        torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else nullcontext
    )

    best_loss = float("inf")
    loss_history = []

    model.train()
    for ep in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            pred = model(data)  # [N, 3]
            u_pred, v_pred = pred[:, 0], pred[:, 1]
            u_true, v_true = data.y[:, 0], data.y[:, 1]
            loss = F.mse_loss(u_pred, u_true) + F.mse_loss(v_pred, v_true)

            if include_p and pred.size(1) >= 3 and data.y.size(1) >= 3:
                p_pred, p_true = pred[:, 2], data.y[:, 2]
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

        # Track best (by (optional) denorm eval, else by training loss)
        if denorm_metrics and (uvnorm is not None):
            eval_loss = evaluate(
                model, data, uvnorm=uvnorm, use_denorm=True, include_p=include_p
            )
            is_best = eval_loss < best_loss
            disp_loss = eval_loss
        else:
            is_best = loss.item() < best_loss
            disp_loss = loss.item()

        if is_best:
            best_loss = disp_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": ep,
                    "best_loss": best_loss,
                    "cfg": {
                        "epochs": epochs,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "scheduler": scheduler,
                        "scheduler_step": scheduler_step,
                        "scheduler_gamma": scheduler_gamma,
                    },
                },
                save_path,
            )

        if ep % 25 == 0 or ep == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            tag = "phys" if (denorm_metrics and uvnorm is not None) else "norm"
            print(
                f"[{ep:5d}/{epochs}] loss({tag})={disp_loss:.6e}  lr={lr_now:.3e}  best={best_loss:.6e}"
            )

    print(
        f"✅ Training complete. Best {('phys' if (denorm_metrics and uvnorm is not None) else 'norm')} loss: {best_loss:.6e}"
    )
    np.save(save_path.replace(".pth", "_loss.npy"), np.array(loss_history))
    return loss_history


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


class GNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, edge_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GNOLayer(
                    latent_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    edge_dim,
                )
                for i in range(num_layers)
            ]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z, edge_index, edge_attr):
        for layer in self.layers:
            z = layer(z, edge_index, edge_attr)
        return self.output_layer(z)


# ==============================
# 3) Toggleable GNNAutoencoder
# ==============================
class GNNAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        output_dim,
        edge_dim,
        num_layers=4,
        use_kalman=False,
        use_pool=False,
        pool_clusters=64,
    ):
        super().__init__()
        self.use_pool = use_pool
        self.use_kalman = use_kalman
        self.kalman_filter = None

        if use_pool:
            self.encoder = GNNEncoderWithPool(
                input_dim,
                hidden_dim,
                latent_dim,
                edge_dim,
                num_layers=num_layers,
                pool_clusters=pool_clusters,
            )
        else:
            self.encoder = GNNEncoder(
                input_dim, hidden_dim, latent_dim, edge_dim, num_layers=num_layers
            )

        self.decoder = GNNDecoder(
            latent_dim, hidden_dim, output_dim, edge_dim, num_layers=num_layers
        )

    def attach_kalman(self, kalman_filter):
        self.kalman_filter = kalman_filter
        self.use_kalman = True

    def forward(self, data):
        z = self.encoder(data)  # [N, latent] even if pooled (we unpool)
        if self.use_kalman and self.kalman_filter is not None:
            z_mean = z.mean(dim=0).detach().cpu().numpy()
            z_filtered = self.kalman_filter.update(z_mean)
            z = (
                torch.tensor(z_filtered, dtype=torch.float32, device=z.device)
                .unsqueeze(0)
                .repeat(z.size(0), 1)
            )
        return self.decoder(z, data.edge_index, data.edge_attr)


if __name__ == "__main__":
    # 1) Build data
    data = createGraphData()  # your function

    # 2) Get normalizers if you want denorm metrics during training
    # 3) Create model (simple version using AttnGNOLayer inside)
    input_dim = data.x.size(1)
    edge_dim = data.edge_attr.size(1)
    output_dim = 3

    model = GNNAutoencoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        output_dim=output_dim,
        edge_dim=edge_dim,
        num_layers=NUM_LAYERS,
        use_kalman=False,  # attach later if you want
    ).to(device)

    # 4) Train
    train_attention_autoencoder(
        data=data,
        model=model,
        epochs=EPOCHS,
        lr=1e-3,
        weight_decay=1e-4,
        scheduler="cosine",  # "step" or None
        scheduler_step=SCHEDULER_STEP,
        scheduler_gamma=0.9,
        max_grad_norm=1.0,
        use_amp=True,
        include_p=False,  # set True to also supervise pressure
        uvnorm=uvnorm,
        denorm_metrics=False,  # set True to track loss in physical units
        save_path="model/gnn_autoencoder_attn.pth",
    )
