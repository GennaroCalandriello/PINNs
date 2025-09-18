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
USE_ST_GUMBEL = True
USE_POOL = False
DROPOUT = 0.1
LEAKY_SLOPE = 0.2

TAU = 0.6
POOL_CLUSTERS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
SCHEDULER_TYPE = "StepLR"  # "StepLR" or "CosineAnnealingLR"
SCHEDULER_GAMMA = 0.9

"""References used for this implementation:
[1] The Edge-Node Attention-based Differentiable Pooling for Graph Neural Networks: https://arxiv.org/pdf/2405.10218v1
[2] A graph convolutional autoencoder approach to model order reduction for parametrized PDE: https://arxiv.org/abs/2305.08573
[3] Non-linear Manifold Reduced-Order Models with Convolutional Autoencoders and Reduced Over-Collocation Method: https://link.springer.com/article/10.1007/s10915-023-02128-2
[4] Attention Is All You Need: https://arxiv.org/abs/1706.03762
[5] PyTorch Geometric documentation
[6] Self-Attention Graph Pooling: https://arxiv.org/abs/1904.08082
[7] MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing: https://arxiv.org/abs/1905.00067
"""

"""Concepts used:
softmax: https://en.wikipedia.org/wiki/Softmax_function"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scatter_softmax(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Some notes:
    softmax(x) = exp(x_i) / sum_j exp(x_j)
           = exp(x_i - max_x) / sum_j exp(x_j - max_x) ->
    This does not change the result of the softmax, because
    softmax is invariant to adding or subtracting
    the same constant from all elements in the group"""

    max_per_group, _ = scatter_max(src, index, dim_size=dim_size)

    max_per_group = max_per_group.index_select(0, index)
    expo = (
        src - max_per_group
    ).exp()  # subtracts the group max from each element and exponentiates the result, just like in the standard softmax formula.
    denom = scatter_add(expo, index, dim_size=dim_size).index_select(0, index)
    return expo / (denom + 1e-12)  # avoid division by zero


def one_hot_argmax(logits: Tensor) -> Tensor:
    """Takes a 2D tensor of logits (shape
    (N,C), where:
    N is the batch size and
    C is the number of classes) and returns a one-hot encoded tensor of the same shape,
    with a 1 at the position of the maximum value of the Softmax S matrix in each row and 0 elsewhere.
    """
    # logits: (N, C) -> one-hot (N, C)

    index = logits.argmax(dim=-1)
    oneH = torch.zeros_like(logits)
    oneH.scatter_(1, index.unsqueeze(-1), 1.0)

    return oneH


class AttentionGNOLayer(MessagePassing):
    """
    Message passing fedele al paper per il blocco intra-grafo:
      - messaggio grezzo:   m_ij^raw = MLP_msg([x_i', x_j', e_ij'])
      - node-attention:     α_ij     = softmax_j( a^T Leaky([x_i', x_j', e_ij']) ) per ogni target i
      - messaggio finale:   m_ij     = α_ij * m_ij^raw
     β_ij è usata SOLO nel pooling per A^{(l+1)}.

        [*] The attention score is computed from the concatenation
    of three embeddings of size hidden_dim: the target node x_i, the neighbor
    node x_j, and the edge features e_ij. Each of these is first projected to
    hidden_dim (by lin_node/lin_edge). Concatenating them yields a vector of
    length 3 * hidden_dim, so the attention scorer must accept 3*hidden_dim inputs.

    This linear layer produces a single logit per edge
    (often written as a_ij = wᵀ [x_i || x_j || e_ij] + b),
    which you later normalize (e.g., softmax) to get attention weights.
    If you did not use edge features, you would typically have 2*hidden_dim
    (for [x_i || x_j]). If you included more components (e.g., relative position),
    the multiplier would change accordingly.
    Both a_node and a_edge produce per-edge scores,
    and both are fed the same triplet of embeddings:
    x_i (target node), x_j (source node), and e_ij (edge).
    Each is first projected to hidden_dim, then concatenated,
    giving 3 * hidden_dim
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
        super().__init__(aggr="add", node_dim=0)  # "add" aggregation.
        self.use_node_attn = use_node_attn
        self.use_edge_attn = use_edge_attn

        # linear embeddings
        self.lin_node = nn.Linear(in_channels, hidden_dim)
        self.lin_edge = nn.Linear(edge_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, out_channels)

        # additive attention parameters (alpha_ij over neighbors of node i):
        if use_node_attn:
            self.a_node = nn.Linear(3 * hidden_dim, 1)  # a_node([x_i, x_j, e_ij]) [*]

        # separate edge attention (beta_ij over all edges):
        if use_edge_attn:
            self.a_edge = nn.Linear(3 * hidden_dim, 1)

        # msg MLP
        msg_in = 3 * hidden_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels),
        )

        # update block
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
        # print("Edge attr in forward of AttentionGNOLayer", edge_attr)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.norm_post(out)
        return out + x if self.residual else out

    def message(self, x_i, x_j, edge_attr, index):
        # x_i has shape [E, in_channels] (target node features)
        # x_j has shape [E, in_channels] (source node features)
        # edge_attr has shape [E, edge_dim] (edge features)
        xi = self.lin_node(x_i)
        xj = self.lin_node(x_j)
        ej = self.lin_edge(edge_attr)
        cat_for_msg = torch.cat([xi, xj, ej], dim=-1)
        size_i = int(index.max().item()) + 1  # number of target nodes

        # node attentions alpha_ij: softmax over neighbors of node i
        if self.use_node_attn:
            e_node = self.leaky(self.a_node(cat_for_msg)).squeeze(-1)  # [E]
            alpha_ij = scatter_softmax(e_node, index, dim_size=size_i)  # [E]
        else:
            alpha_ij = torch.ones((cat_for_msg.size(0),), device=device)  # [E]

        m_raw = self.msg_mlp(cat_for_msg)  # [E, out_channels]
        msg = m_raw * alpha_ij.unsqueeze(-1)  # [E, out_channels]

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


# --------Edge Node Attention-based Pooling Layer (Hard assignment)--------
class EdgeNodeAttentionPool(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden=HIDDEN_DIM,
        C=POOL_CLUSTERS,
        use_st_gumbel=USE_ST_GUMBEL,
        tau=TAU,
    ):
        super().__init__()
        self.C = C
        self.use_st_gumbel = use_st_gumbel
        self.tau = tau

        # Pool embed head -> assignemt logits (?)
        self.pool_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, C)
        )

        # node attention score m_i = a^T * Leaky(Wm z_i)
        self.Wm = nn.Linear(in_dim, hidden)
        self.am = nn.Linear(hidden, 1)
        self.leaky = nn.LeakyReLU(LEAKY_SLOPE)

        # edge-attention score s_ij = ae^T * Leaky([We z_i || We z_j])
        self.We = nn.Linear(in_dim, hidden)
        self.ae = nn.Linear(2 * hidden, 1)

    # @torch.no_grad()
    def hard_from_logits(self, logits: Tensor) -> Tensor:  # logit is
        if self.use_st_gumbel:
            return F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        else:
            return one_hot_argmax(F.softmax(logits, dim=-1))

    # def forward(self, z, edge_index, edge_attr, edge_weight=None):
    #     device, N, d = z.device, *z.size()
    #     E = edge_index.size(1)
    #     if edge_weight is None:
    #         edge_weight = z.new_ones(E)

    #     # ----- (1) Assignments -----
    #     logits = self.pool_mlp(z)  # [N,C]
    #     S_soft = F.softmax(logits, dim=-1)  # [N,C]
    #     S_st = (
    #         F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
    #         if self.use_st_gumbel
    #         else one_hot_argmax(S_soft)
    #     )  # for viz only
    #     clusters_ids = S_st.argmax(dim=-1)  # [N] (diagnostics)

    #     # ----- (2) Node gate (no hard grouping) -----
    #     alpha = torch.sigmoid(self.am(self.leaky(self.Wm(z)))).unsqueeze(-1)  # [N,1]

    #     # X' = S^T (alpha ⊙ z)
    #     X_coarse = S_soft.T @ (alpha * z)  # [C,d]

    #     # ----- (3) Edge attention β_ij (global) -----
    #     src, dst = edge_index
    #     zi, zj = z[dst], z[src]
    #     se_ij = self.ae(
    #         torch.cat([self.leaky(self.We(zi)), self.leaky(self.We(zj))], dim=-1)
    #     ).squeeze(
    #         -1
    #     )  # [E]
    #     beta = F.softmax(se_ij, dim=0)  # [E]
    #     w = (edge_weight * beta).unsqueeze(-1)  # [E,1]

    #     # A' = (S[src] ⊙ w)^T S[dst]
    #     S_src, S_dst = S_soft[src], S_soft[dst]  # [E,C], [E,C]
    #     A_coarse = (S_src * w).T @ S_dst  # [C,C]

    #     # Edge attrs': weighted by w over edges then projected with S
    #     edge_dim = edge_attr.size(1)
    #     E_num = []
    #     for k in range(edge_dim):
    #         wk = (w.squeeze(-1) * edge_attr[:, k]).unsqueeze(-1)  # [E,1]
    #         E_num_k = (S_src * wk).T @ S_dst  # [C,C]
    #         E_num.append(E_num_k)
    #     E_num = torch.stack(E_num, dim=-1)  # [C,C,edge_dim]
    #     E_coarse = E_num / (A_coarse.unsqueeze(-1) + 1e-12)  # [C,C,edge_dim]

    #     # Sparsify with top-k per row for stability
    #     k_top = min(8, S_soft.size(1))
    #     vals, idxs = torch.topk(A_coarse, k=k_top, dim=1)
    #     rows = (
    #         torch.arange(S_soft.size(1), device=device)
    #         .unsqueeze(1)
    #         .expand(-1, k_top)
    #         .reshape(-1)
    #     )
    #     cols = idxs.reshape(-1)
    #     pw = vals.reshape(-1)
    #     edge_attr_coarse = E_coarse[rows, cols, :]  # [E',edge_dim]
    #     edge_index_coarse = torch.stack([rows, cols], dim=0)  # [2,E']

    #     return X_coarse, edge_index_coarse, pw, edge_attr_coarse, clusters_ids, S_soft

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_weight: Tensor = None,
    ):
        """
        z: [N, d] node embeddings
        edge_index: [2, E]
        edge_weight: [E] or None
        """
        assert edge_attr is not None
        N, d = z.size()
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=device)

        # 1) Assignment logits
        logits = self.pool_mlp(z)  # [N, C]
        S_soft = F.softmax(logits, dim=-1)  # [N, C]
        S_hard = self.hard_from_logits(logits)  # [N, C]
        clusters_ids = S_hard.argmax(dim=-1)  # [N]
        edge_dim = edge_attr.size(1)

        # 2) node attention within each cluster
        m_i = self.am(self.leaky(self.Wm(z))).squeeze(-1)
        alpha_i = scatter_softmax(
            m_i, clusters_ids, dim_size=self.C
        )  # [N] normalized within each cluster
        # Coarsen X': sum_(i in p) alpha_i z_i equation (12) in [1]

        X_coarse = scatter_add(
            alpha_i.unsqueeze(-1) * z,
            clusters_ids.unsqueeze(-1).expand(-1, d),
            dim=0,
            dim_size=self.C,
        )  # [C, d]

        # 3) edge attention over all edges equation (13) in [1]
        src, dst = edge_index
        zi, zj = z[dst], z[src]  # target, source node embeddings
        se_ij = self.ae(
            torch.cat([self.leaky(self.We(zi)), self.leaky(self.We(zj))], dim=-1)
        ).squeeze(
            -1
        )  # [E]

        p = clusters_ids[src]
        q = clusters_ids[dst]
        pair_id = p * self.C + q  # cluster pair ids
        num_pairs = self.C * self.C
        beta_pq = scatter_softmax(
            se_ij, pair_id, dim_size=num_pairs
        )  # [C*C] equation (14) in [1]
        weight_beta = edge_weight * beta_pq  # [E]

        # Aggregate into coarse adjacency matrix A'_{pq} = sum_{i in p, j in q} beta_ij * w_ij
        A_flat = scatter_add(weight_beta, pair_id, dim_size=num_pairs)  # [C*C]

        # -----Provo a coarsenare edge attributes per (p, q) media pesata con weight_beta-----
        num_attr = scatter_add(
            edge_attr * weight_beta.unsqueeze(1),
            pair_id.unsqueeze(1).expand(-1, edge_dim),
            dim=0,
            dim_size=num_pairs,
        )  # [C*C, edge_dim]
        denom = A_flat.unsqueeze(-1) + 1e-12  # [C*C, 1]
        edge_attr_flat = num_attr / denom  # [C*C, edge_dim]
        # ---- Estrai solo le coppie (p,q) presenti ----
        nonzero = A_flat > 0
        pq_idx = nonzero.nonzero(as_tuple=False).squeeze(-1)  # [E']
        pw = A_flat[pq_idx]  # [E']
        edge_attr_coarse = edge_attr_flat[pq_idx]  # [E', edge_dim]
        p_idx = pq_idx // self.C
        q_idx = pq_idx % self.C
        edge_index_coarse = torch.stack([p_idx, q_idx], dim=0)  # [2, E']

        return X_coarse, edge_index_coarse, pw, edge_attr_coarse, clusters_ids, S_hard


class GNNEncoderWithPool(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        edge_dim,
        num_layers,
        pool_clusters=POOL_CLUSTERS,
        use_edge_after_pool=True,
    ):
        super().__init__()

        self.edge_dim = edge_dim
        self.use_edge_after_pool = use_edge_after_pool

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
        )

        self.layers2 = nn.ModuleList(
            [
                AttentionGNOLayer(
                    hidden_dim,
                    hidden_dim,
                    edge_dim=self.edge_dim,
                    hidden_dim=hidden_dim,
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
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # Pooling coarsen graph
        Xc, Eic, Ewc, Eac, clusters, S_hard = self.pool(
            x, edge_index, edge_attr=edge_attr
        )
        # after pooling no edge features, ma perché?
        for layer in self.layers2:
            Xc = layer(Xc, Eic, Eac)
        znodes = self.to_latent(Xc)
        zgraph = znodes.mean(dim=0, keepdim=True)
        return znodes, zgraph, (Eic, Eac, clusters, S_hard)


class GNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, edge_dim, num_layers):
        super().__init__()

        self.use_pool = USE_POOL

        self.pre = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        if self.use_pool:
            self.layers = nn.ModuleList(
                [
                    AttentionGNOLayer(
                        hidden_dim,
                        hidden_dim,
                        edge_dim=edge_dim,
                        hidden_dim=hidden_dim,
                        use_edge_attn=USE_EDGE_ATTN,
                        use_node_attn=USE_NODE_ATTN,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
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

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, z_node, edge_index, edge_attr):
        x = self.pre(z_node)  # [N, hidden]
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)  # keep edge_attr!
        return self.head(x)  # [N, output_dim]


# --------GNN Encoder (without pooling)--------
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

        self.decoder = GNNDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
        )

    def forward(self, data):
        enc_out = self.encoder(data)
        if self.use_pool:
            # the encode returns: Zc [C, d], z_graph [1, d], meta = (eic, edge_attr_c, clusters, s_hard)
            Zc, Zgraph, meta = enc_out
            eic, edge_attr_c, clusters, s_hard = meta
            # print(edge_attr_c)
            # Unpooling back to original nodes
            z_node = s_hard @ Zc  # [N, latent_dim]

        else:
            z_node = enc_out  # [N, latent_dim]

        return self.decoder(z_node, data.edge_index, data.edge_attr)
        # return self.decoder(z_node, eic, edge_attr_c)


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

    if scheduler == "StepLR":
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step, gamma=scheduler_gamma
        )
    elif scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
    else:
        sched = None

    scaler = torch.amp.GradScaler(enabled=use_amp, device=device)
    autocast_ctx = (
        torch.amp.autocast(enabled=use_amp, device_type=device.type)
        if use_amp
        else torch.no_grad
    )
    loss_history = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        with autocast_ctx:
            pred = model(data)
            u_pred, v_pred = pred[:, 0], pred[:, 1]
            # print("u_pred", u_pred)
            u_true, v_true = data.y[:, 0], data.y[:, 1]
            # print("u_true", u_true)
            loss = F.mse_loss(u_pred, u_true) + F.mse_loss(v_pred, v_true)

            if include_pressure:
                p_pred = pred[:, 2]
                p_true = data.y[:, 2]
                loss += F.mse_loss(p_pred, p_true)

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
        if (epoch) % 1 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "model/gnn_autoencoder.pth")
    np.savetxt("model/loss_history.txt", np.array(loss_history))


if __name__ == "__main__":

    data = createGraphData()
    input_dim = data.x.size(1)
    edge_dim = data.edge_attr.size(1)
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

    train(data, model)
