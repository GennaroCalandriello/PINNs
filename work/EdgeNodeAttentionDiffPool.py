# scalable_gnn_ae.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor
from torch_scatter import scatter_add, scatter_max
from typing import Optional, List
from ns_GNN_KF import (
    createGraphData,
    dataLoader,
    dataNormalizer,
    geometryObject,
)

"""References used for this implementation:
[1] The Edge-Node Attention-based Differentiable Pooling for Graph Neural Networks: https://arxiv.org/pdf/2405.10218v1
[2] A graph convolutional autoencoder approach to model order reduction for parametrized PDE: https://arxiv.org/abs/2305.08573
[3] Non-linear Manifold Reduced-Order Models with Convolutional Autoencoders and Reduced Over-Collocation Method: https://link.springer.com/article/10.1007/s10915-023-02128-2
[4] Attention Is All You Need: https://arxiv.org/abs/1706.03762
[5] PyTorch Geometric documentation
[6] Self-Attention Graph Pooling: https://arxiv.org/abs/1904.08082
[7] MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing: https://arxiv.org/abs/1905.00067
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====== Hyperparameters (adjust to your GPU) ======
HIDDEN = 150
LATENT = 70
EDGE_HIDDEN = 48
# NUM_ENCODER_BLOCKS = 4  # pooling depth; 3 => ~1/8 nodes if ratio=0.5 each
POOL_RATIO = 0.5  # keep top % nodes at each stage
DROP = 0.1
LR = 1e-3
EPOCHS = 200
BATCH_SIZE_NODES = 4096  # NeighborLoader target nodes per batch
NEIGHBORS = [50, 40, 30]  # fanouts per hop
GRAD_CLIP = 1.0
USE_AMP = True  # mixed precision
USE_COMPILE = False  # torch.compile for fused kernels
SCHEDULER_STEP = 200  # epochs per step of lr scheduler
SELF_LOOP = True
LEAKY = 0.1  # for LeakyReLU in attention
ATTENTION_CHANNELS = 96  # attention channels in EdgeNodeAttentionPooling
POOL_TYPE = (
    "edge_node"  # 'topk', 'diffpool', 'edge_node' (diffpool dà qualche errore ancora!)
)
CLUSTERS_PER_LEVEL = [
    12000,
    8000,
    5000,
    3000,
    2000,
]  # for 'diffpool' and 'edge_node' only
NUM_ENCODER_BLOCKS = len(CLUSTERS_PER_LEVEL)


def build_sparse_adj(edge_index, num_nodes):
    row, col = edge_index
    return SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))


def sparse_diff_pool(x, edge_index, S):
    """See eqs (3) and (4) in the paper [1]"""
    # x: [N, F] node embedding matrix
    N, C = S.size(0), S.size(1)
    A = build_sparse_adj(edge_index, N)  # shape: [N, N] adjacency matrix
    AS = A.matmul(S)  # shape: [N, C]
    A_pool = S.transpose(0, 1) @ AS  # shape: [C, C] #new coarsened adjacency matrix
    x_pool = S.transpose(0, 1) @ x  # shape: [C, F] #new coarsened node features,
    # now build pooled edges (top-k approach)
    threshold = (A_pool.abs().mean() * 0.1).item()
    A_pool = A_pool * (A_pool > threshold)
    row_idx, col_idx = A_pool.nonzero(as_tuple=True)
    edge_index_pool = torch.stack([row_idx, col_idx], dim=0)

    return x_pool, edge_index_pool


class EdgeNodeAttentionPooling(nn.Module):
    def __init__(self, in_ch, att_ch=ATTENTION_CHANNELS, leaky=LEAKY):
        super().__init__()
        self.Wm = nn.Linear(in_ch, att_ch, bias=True)  # for node attention
        self.am = nn.Linear(att_ch, 1, bias=False)

        self.We = nn.Linear(in_ch, att_ch, bias=True)  # for edge attention
        self.ae = nn.Linear(2 * att_ch, 1, bias=False)
        self.leaky = nn.LeakyReLU(leaky)

    def group_softmax(scores, group_ids, n_groups):

        max_g = scatter_max(scores, group_ids, dim=0, dim_size=n_groups)[0]
        scores = scores - max_g[group_ids]
        exp = torch.exp(scores)
        denom = scatter_add(exp, group_ids, dim=0, dim_size=n_groups).clamp_min(1e-9)
        return exp / denom[group_ids]

    def forward(self, z, edge_index, S_soft, edge_weight=None):
        # z: [N, F] node features, edge_index: [2, E] edges, S_soft: [N, C] soft assignment matrix
        # equation 9 in [1]
        N, F = z.size()
        C = S_soft.size(1)  # number of clusters
        src, dst = edge_index

        # hard assignment (one-hot) from S_soft
        cluster = S_soft.argmax(dim=1)  # shape: [N,]

        # node attention alpha within a cluster
        m = self.am(self.leaky(self.Wm(z))).squeeze(
            -1
        )  # shape: [N,] eq. 10 I'm not sure
        alpha = self.group_softmax(m, cluster, C)  # shape: [N,]
        X_pool = scatter_add(
            alpha.unsqueeze(-1) * z, cluster, dim=0, dim_size=C
        )  # shape: [C, F]

        # edge attention beta between cluster pairs
        zi = self.We(z[src])  # shape: [E, att_ch]
        zj = self.We(z[dst])  # shape: [E, att_ch]
        e_raw = self.ae(torch.cat([zi, zj], dim=-1)).squeeze(-1)  # shape: [E,]
        pair_id = (
            cluster[src] * C + cluster[dst]
        )  # shape: [E,] unique id for each cluster pairù
        beta = self.group_softmax(e_raw, pair_id, C * C)  # shape: [E,]
        w = beta if edge_weight is None else beta * edge_weight
        rowC = cluster[src]
        colC = cluster[dst]
        A_pool = SparseTensor(
            row=rowC, col=colC, value=w, sparse_size=(C, C)
        ).coalesce()  # shape: [C, C]
        ei_pool_row, ei_pool_col, ew_pool = A_pool.coo()
        edge_index_pool = torch.stack([ei_pool_row, ei_pool_col], dim=0)
        return X_pool, edge_index_pool, ew_pool, cluster


class EdgeNodePoolBlock(nn.Module):
    def __init__(self, ch, edge_dim, num_clusters):
        super().__init__()
        self.gnn = EdgeGNNLayer(ch, ch, edge_dim)
        self.assign = nn.Sequential(
            nn.Linear(ch, ch),
            nn.LayerNorm(ch),
            nn.GELU(),
            nn.Linear(ch, ch),
            nn.LayerNorm(ch),
            nn.GELU(),
        )
        self.pool = EdgeNodeAttentionPooling(in_ch=ch, att_ch=max(32, ch // 2))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        S = F.softmax(self.assign(x), dim=-1)  # shape: [N, num_clusters]
        x_pool, edge_index_pool, edge_weight_pool, cluster = self.pool(x, edge_index, S)
        state = {
            "type": "edge_node",
            "cluster": cluster,
            "C": S.size(1),
            "prev_edge_index": edge_index,
            "prev_edge_attr": edge_attr,
        }
        return (x_pool, edge_index_pool, None, batch), state


class TopKPoolingBlock(nn.Module):
    def __init__(self, ch, edge_dim, ratio=POOL_RATIO):
        super().__init__()
        self.gnn = EdgeGNNLayer(ch, ch, edge_dim)
        self.score = nn.Linear(ch, 1)
        self.pool = TopKPooling(ch, ratio=ratio)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        s = self.score(x).squeeze(-1)
        x, edge_index, edge_attr, batch, perm, _ = self.pool(
            x, edge_index, batch=batch, attn=s, edge_attr=edge_attr
        )
        state = {
            "type": "topk",
            "perm": perm,
            "prev_edge_index": None,
            "prev_edge_attr": None,
            "N_prev": None,
        }
        return (x, edge_index, edge_attr, batch), state


class DiffPoolBlock(nn.Module):
    def __init__(self, ch, edge_dim, num_clusters):
        super().__init__()
        self.gnn = EdgeGNNLayer(ch, ch, edge_dim)
        self.assign = nn.Sequential(
            nn.Linear(ch, ch), nn.GELU(), nn.Linear(ch, num_clusters)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        S = F.softmax(self.assign(x), dim=-1)
        x_pool, edge_index_pool = sparse_diff_pool(x, edge_index, S)
        state = {
            "type": "diffpool",
            "S": S,
            "prev_edge_index": edge_index,
            "prev_edge_attr": edge_attr,
        }
        return (x_pool, edge_index_pool, None, batch), state


# ====== Edge-aware sparse message-passing layer ======
class EdgeGNNLayer(MessagePassing):
    def __init__(
        self,
        in_ch,
        out_ch,
        edge_dim: Optional[int],
        hidden=EDGE_HIDDEN,
        dropout=DROP,
        aggr="mean",
    ):
        super().__init__(aggr=aggr, node_dim=0)
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * in_ch + edge_dim, hidden),
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
        # self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_ch)
        self.use_res = in_ch == out_ch

    def forward(self, x, edge_index, edge_attr):
        x = self.dropout(x)
        # adding self-loops to the adjacency matrix ma non so cosa sia
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
        out = self.propagate(
            edge_index_sl, x=x, edge_attr=edge_attr_sl
        )  # shape: [N, out_ch]
        h = torch.cat([x, out], dim=-1)
        h = self.mlp_upd(h)
        # h = self.bn(h)
        h = self.norm(h)
        if self.use_res:
            h = h + x
        return F.gelu(h)

    def message(self, x_i, x_j, edge_attr):
        # x_i: dst features, x_j: src features
        edge_attr = x_i.new_zeros((x_i.size(0), 0)) if edge_attr is None else edge_attr
        m_in = torch.cat([x_i, x_j, edge_attr], dim=-1)

        return self.mlp_msg(m_in)

    # ====== Encoder using TopK pooling ======


class GraphAutoEncoder(nn.Module):
    def __init__(
        self,
        in_ch,
        edge_dim,
        out_ch=3,
        hidden=HIDDEN,
        latent=LATENT,
        depth=NUM_ENCODER_BLOCKS,
        pool_type=POOL_TYPE,
        clusters_per_level: Optional[List[int]] = None,
    ):
        super().__init__()
        self.pool_type = pool_type
        self.in_proj = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # encoder hierarchy
        self.encoder = nn.ModuleList()
        if pool_type == "topk":
            for _ in range(depth):
                self.encoder.append(
                    TopKPoolingBlock(hidden, edge_dim, ratio=POOL_RATIO)
                )
        elif pool_type in ("diffpool", "edge_node"):
            assert clusters_per_level is not None and len(clusters_per_level) == depth
            for C in clusters_per_level:
                if pool_type == "diffpool":
                    self.encoder.append(
                        (
                            DiffPoolBlock
                            if pool_type == "diffpool"
                            else EdgeNodePoolBlock
                        )(hidden, edge_dim, C)
                    )
        else:
            raise ValueError(
                f"Unknown pool_type: {pool_type}. Choose from 'topk', 'diffpool', 'edge_node'."
            )
        # bottleneck on coarsest graph
        self.bottleneck = EdgeGNNLayer(hidden, latent, edge_dim)
        self.latent_up = nn.Linear(latent, hidden)

        # decoder hierarchy (reverse order of encoder)
        self.decoder = nn.ModuleList(
            [EdgeGNNLayer(hidden, hidden, edge_dim) for _ in range(depth)]
        )
        self.head = nn.Linear(hidden, out_ch)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = self.in_proj(x)
        states = []
        prev_idx, prev_attr = edge_index, edge_attr

        # ====== Encoder ======
        for enc in self.encoder:
            N_prev = x.size(0)
            (x, edge_index, edge_attr, batch), state = enc(
                x, edge_index, edge_attr, batch
            )
            if state["type"] == "topk":
                state["N_prev"] = N_prev
                state["prev_edge_index"] = prev_idx
                state["prev_edge_attr"] = prev_attr
            else:  # diffpool and edge_node already have prev_edge_index and prev_edge_attr
                pass
            states.append(state)
            prev_idx, prev_attr = edge_index, edge_attr

        # bottleneck/latent space
        x = self.bottleneck(x, edge_index, edge_attr)
        x = F.gelu(self.latent_up(x))

        # ====== Decoder ======
        for dec, state in zip(reversed(self.decoder), reversed(states)):
            if state["type"] == "topk":
                perm, N_prev = state["perm"], state["N_prev"]
                edge_index_prev, edge_attr_prev = (
                    state["prev_edge_index"],
                    state["prev_edge_attr"],
                )
                x_full = x.new_zeros((N_prev, x.size(-1)))
                x_full[perm] = x
                x = dec(x_full, edge_index_prev, edge_attr_prev)
            elif state["type"] == "diffpool":
                S = state["S"]
                edge_index_prev, edge_attr_prev = (
                    state["prev_edge_index"],
                    state["prev_edge_attr"],
                )
                x = S @ x  # unpool by multiplying with assignment matrix soft
                x = dec(x, edge_index_prev, edge_attr_prev)
            else:  # edge_node
                cluster, C = state["cluster"], state["C"]
                edge_index_prev, edge_attr_prev = (
                    state["prev_edge_index"],
                    state["prev_edge_attr"],
                )
                S_H = F.one_hot(cluster, num_classes=C).float()  # hard assignment
                x = S_H @ x
                x = dec(x, edge_index_prev, edge_attr_prev)

        return self.head(x)


def GraphLoader(graph_data, batch_size_nodes=BATCH_SIZE_NODES, neighbors=NEIGHBORS):
    loader = NeighborLoader(
        graph_data,
        num_neighbors=neighbors,
        batch_size=batch_size_nodes,
        input_nodes=None,  # all nodes
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
    scheduler_step=SCHEDULER_STEP,
    # ---- boundary weighting (attivo) ----
    use_boundary_weights=True,
    gamma=4.0,  # rinforzo sui nodi vicino al muro
    delta=0.03,  # spessore fascia (in coordinate normalizzate)
    # ---- boundary condition losses (opzionale) ----
    use_bc_losses=True,
    lambda_np=1.0,  # peso no-penetration
    lambda_ns=0.2,  # peso no-slip
    # ---- dove leggere sdf/n_hat se non sono in batch.sdf / batch.n_hat ----
    sdf_col=4,  # indice colonna sdf in batch.x (se non esiste batch.sdf)
    nx_col=6,
    ny_col=7,  # indici colonne n_hat in batch.x (se non esiste batch.n_hat)
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=scheduler_step, gamma=0.9
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and (device.type == "cuda"))

    model.train()
    loss_list = []

    for ep in range(1, epochs + 1):
        tot = 0.0
        cnt = 0

        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)

            # center nodes: se NeighborLoader, altrimenti prendi tutti
            center = getattr(batch, "batch_size", batch.x.size(0))

            # target dimension (es. u,v o u,v,p)
            target_dim = batch.y.size(1)

            # --- autocast solo su CUDA ---
            ctx = torch.cuda.amp.autocast(enabled=use_amp and (device.type == "cuda"))
            with ctx:
                out = model(batch)  # [N, out_ch]
                pred = out[:center, :target_dim]
                tgt = batch.y[:center, :target_dim]

                # ---- boundary weights (da sdf) ----
                if use_boundary_weights:
                    if hasattr(batch, "sdf"):
                        sdf_c = batch.sdf[:center]  # [center,1]
                    else:
                        sdf_c = batch.x[:center, sdf_col : sdf_col + 1]
                    # peso più alto vicino al muro (sdf≈0+)
                    w_boundary = 1.0 + (gamma - 1.0) * torch.exp(
                        -(sdf_c.clamp_min(0.0) / delta)
                    )
                else:
                    w_boundary = 1.0

                # ---- data loss pesata (MSE) ----
                err = (pred - tgt) ** 2
                loss_data = (w_boundary * err).mean()

                # ---- BC losses (opzionali, solo su u,v) ----
                loss_bc = pred.new_zeros(())
                if use_bc_losses and target_dim >= 2:
                    # normali
                    if hasattr(batch, "n_hat"):
                        n_hat_c = batch.n_hat[:center, :2]  # [center,2]
                    else:
                        n_hat_c = batch.x[
                            :center, nx_col : ny_col + 1
                        ]  # [center,2] nx,ny
                    # tangente 2D
                    t_hat_c = torch.stack([-n_hat_c[:, 1], n_hat_c[:, 0]], dim=1)

                    u_pred = pred[:, :2]  # [center,2]
                    u_n = (u_pred * n_hat_c).sum(dim=1, keepdim=True)
                    u_t = (u_pred * t_hat_c).sum(dim=1, keepdim=True)

                    # fascia vicino al muro
                    if not use_boundary_weights:
                        # se non abbiamo calcolato sdf_c sopra
                        if hasattr(batch, "sdf"):
                            sdf_c = batch.sdf[:center]
                        else:
                            sdf_c = batch.x[:center, sdf_col : sdf_col + 1]
                    mask_bc = (sdf_c < delta).float()

                    loss_np = (mask_bc * (u_n**2)).mean()  # no-penetration
                    loss_ns = (mask_bc * (u_t**2)).mean()  # no-slip (soft)
                    loss_bc = lambda_np * loss_np + lambda_ns * loss_ns

                loss = loss_data + loss_bc

            # ---- backward/step ----
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            tot += loss.item()
            cnt += 1

        scheduler.step()
        avg = tot / max(1, cnt)
        print(f"Epoch {ep:03d} | loss {avg:.6f}")
        loss_list.append(avg)

    return model, loss_list


def main():
    # Load and preprocess data
    data = createGraphData()
    in_ch = data.x.size(-1)
    edge_dim = data.edge_attr.size(-1)
    out_ch = data.y.size(-1)

    model = GraphAutoEncoder(
        in_ch=in_ch,
        edge_dim=edge_dim,
        out_ch=out_ch,
        clusters_per_level=CLUSTERS_PER_LEVEL,
    )
    loader = GraphLoader(data)
    trained_model, loss_fn = train(model, loader)
    torch.save(trained_model.state_dict(), "model/gnn_ae.pth")
    np.savetxt("model/loss_gnn_ae.npy", np.array(loss_fn))
    print("Model saved to model/gnn_ae.pth")


if __name__ == "__main__":
    main()
