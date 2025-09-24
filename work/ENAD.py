# scalable_gnn_ae.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.loader import NeighborLoader

from torch_sparse import SparseTensor
from torch_scatter import scatter_add, scatter_max
from typing import Optional, List

from ns_GNN_cav import (
    createGraphData,  # usa la tua funzione data qui sopra
    dataLoader,
    dataNormalizer,
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

# -------------------- Hyperparameters --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

HIDDEN = 32
LATENT = 30
EDGE_HIDDEN = 24
DROP = 0.1
LR = 1e-3
EPOCHS = 100

BATCH_SIZE_NODES = 768
NEIGHBORS = [5, 4, 3]

GRAD_CLIP = 1.0
USE_AMP = True
USE_COMPILE = False
SCHEDULER_STEP = 200

LEAKY = 0.1
ATTENTION_CHANNELS = 10

# Pooling layout (edge-node only)
CLUSTERS_PER_LEVEL = [2000, 1000]
NUM_ENCODER_BLOCKS = len(CLUSTERS_PER_LEVEL)


# -------------------- Edge-Node Attention Pooling --------------------
class EdgeNodeAttentionPooling(nn.Module):
    def __init__(self, in_ch, att_ch=ATTENTION_CHANNELS, leaky=LEAKY):
        super().__init__()
        # node attention
        self.Wm = nn.Linear(in_ch, att_ch, bias=True)
        self.am = nn.Linear(att_ch, 1, bias=False)
        # edge attention
        self.We = nn.Linear(in_ch, att_ch, bias=True)
        self.ae = nn.Linear(2 * att_ch, 1, bias=False)
        # paper-faithful nonlinearity
        self.leaky = nn.LeakyReLU(leaky)

    @staticmethod
    def group_softmax(scores, group_ids, n_groups):
        max_g = scatter_max(scores, group_ids, dim=0, dim_size=n_groups)[0]
        scores = scores - max_g[group_ids]
        exp = torch.exp(scores)
        denom = scatter_add(exp, group_ids, dim=0, dim_size=n_groups).clamp_min(1e-9)
        return exp / denom[group_ids]

    def forward(self, z, edge_index, S_soft, edge_weight):
        # z: [N,F], edge_index: [2,E], S_soft: [N,C], edge_weight: [E]
        N, F = z.size()
        C = S_soft.size(1)
        src, dst = edge_index

        # hard assignment from soft S
        cluster = S_soft.argmax(dim=1)  # [N]

        # ----- node attention within cluster -----
        m = self.am(self.leaky(self.Wm(z))).squeeze(-1)  # [N]
        alpha = self.group_softmax(m, cluster, C)  # [N]
        X_pool = scatter_add(
            alpha.unsqueeze(-1) * z, cluster, dim=0, dim_size=C
        )  # [C,F]

        # ----- edge attention between cluster pairs -----
        zi = self.We(z[src])  # [E, att_ch]
        zj = self.We(z[dst])  # [E, att_ch]
        e_raw = self.ae(torch.cat([zi, zj], dim=-1)).squeeze(-1)  # [E]
        pair_id = cluster[src] * C + cluster[dst]  # [E]
        beta = self.group_softmax(e_raw, pair_id, C * C)  # [E]

        # multiply attention by original edge weights (paper Eq. 16: E ⊙ A)
        w = beta * edge_weight  # [E]

        rowC = cluster[src]
        colC = cluster[dst]
        A_pool = SparseTensor(
            row=rowC, col=colC, value=w, sparse_sizes=(C, C)
        ).coalesce()
        ei_row, ei_col, ew_pool = A_pool.coo()
        edge_index_pool = torch.stack([ei_row, ei_col], dim=0)  # [2, E_pool]

        return X_pool, edge_index_pool, ew_pool, cluster  # ew_pool: [E_pool]


# -------------------- Edge-aware message passing --------------------
class EdgeGNNLayer(MessagePassing):
    def __init__(
        self,
        in_ch,
        out_ch,
        edge_dim: int,  # fisso = 1 in tutta l'architettura
        hidden=EDGE_HIDDEN,
        dropout=DROP,
        aggr="mean",
    ):
        super().__init__(aggr=aggr, node_dim=0)
        self.edge_dim = int(edge_dim)
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
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_ch)
        self.use_res = in_ch == out_ch

    def forward(self, x, edge_index, edge_attr):
        # x: [N,Fin], edge_index: [2,E], edge_attr: [E, self.edge_dim]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)  # [N, out_ch]
        h = torch.cat([x, out], dim=-1)
        h = self.mlp_upd(h)
        h = self.norm(h)
        if self.use_res:
            h = h + x
        return F.gelu(h)

    def message(self, x_i, x_j, edge_attr):
        # x_i: [E,Fin], x_j: [E,Fin], edge_attr: [E, self.edge_dim]
        m_in = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp_msg(m_in)


# -------------------- Pool block (edge-node only) --------------------
class EdgeNodePoolBlock(nn.Module):
    def __init__(self, ch, num_clusters):
        super().__init__()
        # Tutte le GNN qui usano edge_dim=1 (coerente con proiezione edge)
        self.gnn = EdgeGNNLayer(ch, ch, edge_dim=1)
        self.assign = nn.Sequential(
            nn.Linear(ch, ch),
            nn.LayerNorm(ch),
            nn.GELU(),
            nn.Linear(ch, ch),
            nn.LayerNorm(ch),
            nn.GELU(),
            nn.Linear(ch, num_clusters),  # logits per node → softmax
        )
        self.pool = EdgeNodeAttentionPooling(in_ch=ch, att_ch=max(32, ch // 2))

    def forward(self, x, edge_index, edge_attr_1c, batch):
        # edge_attr_1c: [E,1]
        x = self.gnn(x, edge_index, edge_attr_1c)
        S = F.softmax(self.assign(x), dim=-1)  # [N, C]

        # pooling ha bisogno di vector weights [E]
        w_in = edge_attr_1c.squeeze(-1)  # [E]

        x_pool, edge_index_pool, edge_weight_pool, cluster = self.pool(
            x, edge_index, S, edge_weight=w_in
        )

        state = {
            "type": "edge_node",
            "cluster": cluster,  # [N]
            "C": S.size(1),  # int
            "prev_edge_index": edge_index,
            "prev_edge_attr": edge_attr_1c,  # [E,1] (proiezione iniziale)
        }

        # passa i pesi aggregati come edge_attr del livello successivo
        edge_attr_next = edge_weight_pool.unsqueeze(-1)  # [E_pool,1]
        return (x_pool, edge_index_pool, edge_attr_next, batch), state


# -------------------- Autoencoder (edge-node only) --------------------
class GraphAutoEncoder(nn.Module):
    def __init__(
        self,
        in_ch,
        edge_in_dim,  # dimensione edge feature d'ingresso
        out_ch=3,
        hidden=HIDDEN,
        latent=LATENT,
        clusters_per_level: Optional[List[int]] = None,
    ):
        super().__init__()
        assert clusters_per_level is not None and len(clusters_per_level) > 0
        self.depth = len(clusters_per_level)

        # Proiezione nodi
        self.in_proj = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # Proiezione edge features a 1 canale (CONSISTENTE tra i livelli)
        assert edge_in_dim >= 1, "data.edge_attr deve esistere e avere almeno 1 canale"
        self.edge_proj_in = nn.Linear(edge_in_dim, 1, bias=False)

        # Encoder hierarchy (edge-node only)
        self.encoder = nn.ModuleList(
            [EdgeNodePoolBlock(hidden, C) for C in clusters_per_level]
        )

        # Bottleneck su grafo più grosso (edge_dim=1)
        self.bottleneck = EdgeGNNLayer(hidden, latent, edge_dim=1)
        self.latent_up = nn.Linear(latent, hidden)

        # Decoder hierarchy (reverse order of encoder)
        self.decoder = nn.ModuleList(
            [EdgeGNNLayer(hidden, hidden, edge_dim=1) for _ in range(self.depth)]
        )
        self.head = nn.Linear(hidden, out_ch)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Proiezioni iniziali
        x = self.in_proj(x)  # [N, hidden]
        edge_attr_1c = self.edge_proj_in(edge_attr)  # [E,1]

        states = []

        # ----- Encoder -----
        for enc in self.encoder:
            (x, edge_index, edge_attr_1c, _batch), state = enc(
                x, edge_index, edge_attr_1c, getattr(data, "batch", None)
            )
            states.append(state)

        # ----- Bottleneck -----
        x = self.bottleneck(x, edge_index, edge_attr_1c)
        x = F.gelu(self.latent_up(x))

        # ----- Decoder (unpool hard via cluster labels) -----
        for dec, state in zip(reversed(self.decoder), reversed(states)):
            cluster, C = state["cluster"], state["C"]
            edge_index_prev = state["prev_edge_index"]
            edge_attr_prev_1c = state["prev_edge_attr"]  # [E_prev,1]

            S_H = F.one_hot(cluster, num_classes=C).float()  # [N_prev, C]
            x = S_H @ x  # [N_prev, hidden]
            x = dec(x, edge_index_prev, edge_attr_prev_1c)

        return self.head(x)


# -------------------- Data loader --------------------
def GraphLoader(graph_data, batch_size_nodes=BATCH_SIZE_NODES, neighbors=NEIGHBORS):
    return NeighborLoader(
        graph_data,
        num_neighbors=neighbors,
        batch_size=batch_size_nodes,
        input_nodes=None,
        shuffle=True,
    )


# -------------------- Training (NO boundary stuff) --------------------
def train(
    model,
    loader,
    epochs=EPOCHS,
    lr=LR,
    use_amp=USE_AMP,
    grad_clip=GRAD_CLIP,
    scheduler_step=SCHEDULER_STEP,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=scheduler_step, gamma=0.9
    )
    scaler = torch.amp.GradScaler(enabled=use_amp and (device.type == "cuda"))

    model.train()
    loss_list = []

    for ep in range(1, epochs + 1):
        tot = 0.0
        cnt = 0

        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)

            center = getattr(batch, "batch_size", batch.x.size(0))
            target_dim = batch.y.size(1)

            ctx = torch.amp.autocast(enabled=use_amp, device_type=device.type)
            with ctx:
                out = model(batch)  # [N, out_ch]
                pred = out[:center, :target_dim]
                tgt = batch.y[:center, :target_dim]

                # puro MSE
                loss = F.mse_loss(pred, tgt)

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


# -------------------- Main --------------------
def main():
    data = (
        createGraphData()
    )  # la tua funzione (edge_attr = [dx, dy, dist], shape [E,3])
    in_ch = data.x.size(-1)

    # edge_attr deve esistere e non essere None
    assert (
        hasattr(data, "edge_attr") and data.edge_attr is not None
    ), "data.edge_attr must exist and be non-None"
    edge_in_dim = data.edge_attr.size(-1)

    out_ch = data.y.size(-1)

    model = GraphAutoEncoder(
        in_ch=in_ch,
        edge_in_dim=edge_in_dim,
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
