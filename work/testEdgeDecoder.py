import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.data import Data
from ns_GNN_cav2 import createGraphData, dataLoader, dataNormalizer, geometryObject
import numpy as np
from torch_geometric.nn import MessagePassing

# --- IPERPARAMETRI ---
HIDDEN_CHANNELS = (50, 50, 50)  # can be a tuple/list
C = 200
C_LIST = [512, 256, 64]

BOTTLENECKLATENTSPACE = 50  # dimensione spazio latente globale
DROPOUT = 0.0  # dropout rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = createGraphData().to(device)
IN_DIM = data.x.size(1)
EDGE_DIM = data.edge_attr.size(1)
OUT_DIM = data.y.size(1)
USE_LN = False
EPOCHS = 1400
MODEL_PATH = "model/graph_autoencoder_diffpool_edges.pth"
LOSS_PATH = "model/loss_history_diffpool_edges.npy"
KIND_CONV = "GMMConv"
EDGEATTENTIONHIDDEN = 256

# ==================== Small utils ====================


class EdgeNormalizer(nn.Module):
    def __init__(self, t, eps=1e-8):
        super().__init__()
        m = t.mean(0)
        s = t.std(0).clamp_min(eps)
        self.register_buffer("mean", m)
        self.register_buffer("std", s)

    def encode(self, t):
        return (t - self.mean) / self.std

    def decode(self, t):
        return t * self.std + self.mean


class EdgeAttrDecoder(nn.Module):
    """
    Ricostruisce l'attributo d'arco \hat{e}_{ij} (dimensione edge_dim_out).
    Input: x_i, x_j (embedding nodali dopo unpool) e, opzionalmente, l'edge_attr "grezzo" upscalato.
    """

    def __init__(self, node_dim, edge_dim_out, hidden=128, use_edge_skip=False):
        super().__init__()
        self.use_edge_skip = use_edge_skip
        in_dim = 2 * node_dim + (edge_dim_out if use_edge_skip else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, edge_dim_out),
        )

    def forward(self, x, edge_index, edge_attr_up=None):
        src, dst = edge_index
        feats = [x[src], x[dst]]
        if self.use_edge_skip and edge_attr_up is not None:
            feats.append(edge_attr_up)
        h = torch.cat(feats, dim=-1)
        return self.mlp(h)  # [E, edge_dim_out]


class EdgeMLP(nn.Module):
    def __init__(self, edge_dim, in_ch, out_ch, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden), nn.GELU(), nn.Linear(hidden, in_ch * out_ch)
        )

    def forward(self, e):
        return self.net(e)


class EdgeAttention(nn.Module):  # <<< NEW (typo fix)
    def __init__(self, node_dim, edge_dim, hidden=EDGEATTENTIONHIDDEN):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, z, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        s = self.scorer(torch.cat([z[src], z[dst], edge_attr], dim=-1)).squeeze(
            -1
        )  # [E]
        return s


class NodeAttention(nn.Module):
    def __init__(self, node_dim, hidden=EDGEATTENTIONHIDDEN):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.scorer(z).squeeze(-1)  # [N]


# ===== SCEGLI LA CONV =====
def make_conv(kind, in_ch, out_ch, edge_dim=None, K=5, heads=1):
    if kind == "NNConv":
        weighted = EdgeMLP(edge_dim, in_ch, out_ch, hidden=20)
        return gnn.NNConv(in_ch, out_ch, weighted, aggr="mean")
    if kind == "GMMConv":
        assert edge_dim is not None, "GMMConv richiede edge_dim (pseudo dim)."
        return gnn.GMMConv(in_ch, out_ch, dim=edge_dim, kernel_size=6)
    if kind == "ChebConv":
        return gnn.ChebConv(in_ch, out_ch, K=K)
    if kind == "GCNConv":
        return gnn.GCNConv(in_ch, out_ch)
    if kind == "GATConv":
        return gnn.GATConv(in_ch, out_ch, heads=heads, concat=False)
    raise NotImplementedError


def apply_conv(conv, kind, x, edge_index, edge_attr):
    if kind in ["GMMConv", "NNConv"]:
        return conv(x, edge_index, edge_attr)
    if kind in ["ChebConv", "GCNConv"]:
        return conv(x, edge_index)
    if kind == "GATConv":
        return conv(x, edge_index)
    raise NotImplementedError


# --- util: MLP con opzionale LayerNorm e Dropout ---
def mlp(sizes, act=nn.GELU, last_act=False, dropout=0.1, use_ln=USE_LN):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if use_ln:
            layers += [nn.LayerNorm(sizes[i + 1])]
        if i < len(sizes) - 2 or last_act:
            layers += [act()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
    return nn.Sequential(*layers)


# ==================== GNO Decoder ====================


class GNOLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim, edge_dim, p_drop=0.1):
        super().__init__(aggr="mean")
        self.edge_dim = edge_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, out_channels),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, out_channels),
        )
        self.residual = in_channels == out_channels

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:
            E = edge_index.size(1)
            edge_attr = x.new_zeros((E, self.edge_dim))
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )
        if out is None:
            raise RuntimeError("[GNOLayer] propagate() returned None")
        out = self.update(out, x)
        return out + x if self.residual else out

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.upd_mlp(torch.cat([x, aggr_out], dim=-1))


class GNNDecoder(nn.Module):
    def __init__(
        self, latent_dim, hidden_dim, output_dim, edge_dim, num_layers=3, p_drop=0.1
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.layers = nn.ModuleList(
            [
                GNOLayer(
                    latent_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    edge_dim,
                    p_drop=p_drop,
                )
                for i in range(num_layers)
            ]
        )
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z, edge_index, edge_attr):
        if z is None:
            raise RuntimeError("[GNNDecoder] input z is None")
        for li, layer in enumerate(self.layers):
            z = layer(z, edge_index, edge_attr)
            if z is None:
                raise RuntimeError(f"[GNNDecoder] layer {li} returned None")
        return self.out_head(z)


# ==================== Encoder heads ====================


class EncoderWithDiffPoolHeads(nn.Module):
    def __init__(
        self,
        in_dim,
        edge_dim,
        hidden_channels,
        conv=KIND_CONV,
        skip=True,
        act=F.elu,
        dropout=0.0,
    ):
        super().__init__()
        self.act = act
        self.conv_kind = conv
        self.skip = skip
        self.dropout = dropout

        ch = [in_dim] + hidden_channels
        self.embed_convs = nn.ModuleList(
            [
                make_conv(conv, ch[i], ch[i + 1], edge_dim=edge_dim)
                for i in range(len(ch) - 1)
            ]
        )
        self.assign_convs = nn.ModuleList(
            [
                make_conv(conv, ch[i], ch[i + 1], edge_dim=edge_dim)
                for i in range(len(ch) - 1)
            ]
        )

        self.embed_norms = nn.ModuleList(
            [nn.LayerNorm(ch[i + 1]) for i in range(len(ch) - 1)]
        )
        self.assign_norms = nn.ModuleList(
            [nn.LayerNorm(ch[i + 1]) for i in range(len(ch) - 1)]
        )

    def forward(self, x, edge_index, edge_attr):
        z = x
        for i, conv in enumerate(self.embed_convs):
            z_res = z
            z = apply_conv(conv, self.conv_kind, z, edge_index, edge_attr)
            z = self.embed_norms[i](z)
            z = self.act(z)
            if self.dropout > 0:
                z = F.dropout(z, p=self.dropout, training=self.training)
            if self.skip and z.shape == z_res.shape:
                z = z + z_res

        s = x
        for i, conv in enumerate(self.assign_convs):
            s_res = s
            s = apply_conv(conv, self.conv_kind, s, edge_index, edge_attr)
            s = self.assign_norms[i](s)
            s = self.act(s)
            if self.dropout > 0:
                s = F.dropout(s, p=self.dropout, training=self.training)
            if self.skip and s.shape == s_res.shape:
                s = s + s_res

        return z, s


# ==================== Autoencoder w/ DiffPool ====================


class GraphAutoencoderDiffPool(nn.Module):
    def __init__(
        self,
        in_dim,
        edge_dim,
        hidden_channels=(64, 64),
        assign_hidden=64,
        C=256,
        bottleneck=64,
        out_dim=2,
        conv="GMMConv",
        skip=True,
        dropout=0.0,
        # --- NEW knobs ---
        edge_attn_mode="hard",  # <<< NEW: default mode
        edge_topk=2,  # <<< NEW: default k for topk
        lambda_ent=1e-3,  # <<< NEW: weight for entropy loss
        lambda_link=1e-3,  # <<< NEW: weight for link-pred loss
        use_gumbel=False,  # <<< NEW: optional quasi-hard assignments
        gumbel_tau=1.0,  # <<< NEW: temperature
    ):
        super().__init__()
        self.C = C
        self.edge_attn_mode = edge_attn_mode  # <<< NEW
        self.edge_topk = int(edge_topk)  # <<< NEW
        self.lambda_ent = lambda_ent  # <<< NEW
        self.lambda_link = lambda_link  # <<< NEW
        self.use_gumbel = use_gumbel  # <<< NEW
        self.gumbel_tau = gumbel_tau  # <<< NEW

        self.enc = EncoderWithDiffPoolHeads(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_channels=list(hidden_channels),
            conv=conv,
            skip=skip,
            act=F.elu,
            dropout=dropout,
        )
        H = hidden_channels[-1]

        self.assign_head = nn.Sequential(
            nn.LayerNorm(H), nn.Linear(H, H), nn.GELU(), nn.Linear(H, C)
        )

        self.coarse_proj = mlp([H, H], dropout=dropout, use_ln=USE_LN)
        self.graph_readout = mlp([H, 2 * H, bottleneck], dropout=dropout, use_ln=USE_LN)
        self.edge_attn = EdgeAttention(
            H, edge_dim, hidden=EDGEATTENTIONHIDDEN
        )  # <<< NEW name
        self.node_attn = NodeAttention(H, hidden=EDGEATTENTIONHIDDEN)
        self.edge_decoder = EdgeAttrDecoder(
            node_dim=H,  # deve combaciare con X_up.shape[1]
            edge_dim_out=edge_dim,
            hidden=EDGEATTENTIONHIDDEN,
            use_edge_skip=False,
        )

        # prima: edge_dim=edge_dim
        self.node_decoder = GNNDecoder(
            latent_dim=H,
            hidden_dim=max(96, H),
            output_dim=out_dim,
            edge_dim=edge_dim + 1,  # <<< cambia qui
            num_layers=3,
            p_drop=max(0.1, dropout),
        )

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        N, device = x.size(0), x.device

        # -------- ENCODER --------
        z, s_feat = self.enc(x, edge_index, edge_attr)  # [N,H], [N,H]
        S_logits = self.assign_head(s_feat)  # [N,C]

        # Assignments: softmax standard o Gumbel-Softmax
        if self.use_gumbel:  # <<< NEW (facoltativo)
            S = F.gumbel_softmax(S_logits, tau=self.gumbel_tau, hard=False, dim=-1)
        else:
            S = F.softmax(S_logits, dim=-1)  # [N,C]

        # -------- EDGE ATTENTION (stabile) --------
        s_e = self.edge_attn(z, edge_index, edge_attr)  # [E]
        w = F.softplus(s_e) + 1e-6  # <<< NEW: stabile & positivo
        src, dst = edge_index[0], edge_index[1]

        # -------- Coarsening A_coarse (diagnostico) --------
        mode = self.edge_attn_mode
        k = int(self.edge_topk)

        if mode == "hard":
            p = S[src].argmax(dim=1)  # [E]
            q = S[dst].argmax(dim=1)  # [E]
            pq = p * self.C + q
            A_coarse_flat = torch.zeros(self.C * self.C, device=device)
            A_coarse_flat.index_add_(0, pq, w)
            A_coarse = A_coarse_flat.view(self.C, self.C)  # [C,C]

        elif mode == "topk":
            S_val, S_idx = torch.topk(S, k, dim=1)  # [N,k]
            vps, ips = S_val[src], S_idx[src]  # [E,k]
            vqs, iqs = S_val[dst], S_idx[dst]  # [E,k]
            vp = vps.unsqueeze(2).expand(-1, k, k)
            vq = vqs.unsqueeze(1).expand(-1, k, k)
            weight = (w.view(-1, 1, 1) * vp * vq).reshape(-1)  # [E*k*k]
            p = ips.unsqueeze(2).expand(-1, k, k).reshape(-1)
            q = iqs.unsqueeze(1).expand(-1, k, k).reshape(-1)
            pq = p * self.C + q
            A_coarse_flat = torch.zeros(self.C * self.C, device=device)
            A_coarse_flat.index_add_(0, pq, weight)
            A_coarse = A_coarse_flat.view(self.C, self.C)
            # --- Unpool strutturale: A_up coerente con S ---

        else:
            raise ValueError(f"edge_attn_mode sconosciuta: {mode}")

        # -------- NODE ATTENTION --------
        # --- Unpool strutturale sempre ---
        A_up_dense = S @ A_coarse @ S.t()  # [N,N]
        src, dst = edge_index
        a_up = A_up_dense[src, dst].unsqueeze(-1)  # [E,1]
        m = self.node_attn(z)  # [N]
        num = torch.exp(m).unsqueeze(1) * S  # [N,C]
        den = num.sum(dim=0, keepdim=True) + 1e-16
        alpha = num / den  # [N,C]

        # Pool N→C
        X_coarse = alpha.T @ z  # [C,H]
        Xc = self.coarse_proj(X_coarse.unsqueeze(0)).squeeze(0)  # [C,H]

        # Readout (opz.)
        g = Xc.mean(dim=0, keepdim=True)  # [1,H]
        z_graph = self.graph_readout(g)  # [1,bottleneck]

        # Unpool
        X_up = S @ Xc  # [N,H]

        # Decoder
        # y_hat = self.node_decoder(X_up, edge_index, edge_attr)  # [N, out_dim]
        edge_attr_dec = torch.cat([a_up, edge_attr], dim=1)
        y_hat = self.node_decoder(X_up, edge_index, edge_attr_dec)

        # BASTA passare X_up: sono le embedding nodali dopo unpool
        e_hat = self.edge_decoder(
            X_up, edge_index, edge_attr_up=edge_attr_dec
        )  # [E, EDGE_DIM]

        # # In realtà ti basta passare X_up:
        # e_hat = self.edge_decoder(X_up, edge_index, edge_attr_up=None)  # [E, EDGE_DIM]

        # --- Ritorno anche roba utile per le regolarizzazioni ---
        return (
            y_hat,
            e_hat,
            {
                "S": S,
                "A_coarse": A_coarse,
                "z_graph": z_graph,
                "w": w,
                "edge_index": edge_index,
                "A_up_edge": a_up,
            },
        )

    # --- Semplici regolarizzazioni su S (da chiamare nel training) ---
    def regularization_losses(
        self, aux, neg_ratio: float = 0.0, rng: torch.Generator | None = None
    ):
        """
        Sparse link-pred:
        L_ent  = media entropie di riga (positiva)
        L_link = MSE su archi:  w_hat  ~  <S_i, S_j>
        (opz.) + negative sampling su non-archi

        Args:
        neg_ratio: quante "non-edges" campionare rispetto agli edges (0.0 = nessuna)
        rng: torch.Generator per sampling deterministico (opz.)
        """
        S = aux["S"]  # [N,C]
        edge_index = aux["edge_index"]  # [2,E]
        w = aux["w"]  # [E]
        device = S.device
        eps = 1e-8

        # --- Entropia (come prima) ---
        L_ent = (S * (S.add(eps).log())).sum(dim=1).mean().abs()

        # --- Link pred SOLO sugli archi (sparse) ---
        src, dst = edge_index[0], edge_index[1]  # [E]
        sim_pos = (S[src] * S[dst]).sum(dim=-1)  # [E], in [0,1]

        # normalizza i pesi attn w in [0,1] (robusto)
        w_max = torch.clamp(w.max(), min=eps)
        w_hat = (w / w_max).clamp(0.0, 1.0)  # [E]

        # MSE sugli archi
        L_link_pos = F.mse_loss(sim_pos, w_hat)

        # --- (Opzionale) Negative sampling: non-archi ---
        if neg_ratio > 0.0:
            E = src.numel()
            num_neg = int(E * float(neg_ratio))
            N = S.size(0)
            if rng is None:
                rng = torch.Generator(device=device)
            # campiona coppie casuali finché non sono "diverse" dagli archi (approssimazione)
            i_neg = torch.randint(
                low=0, high=N, size=(num_neg,), generator=rng, device=device
            )
            j_neg = torch.randint(
                low=0, high=N, size=(num_neg,), generator=rng, device=device
            )
            # similarità attesa per non-archi dovrebbe essere bassa (target ~ 0)
            sim_neg = (S[i_neg] * S[j_neg]).sum(dim=-1)  # [num_neg] in [0,1]
            zero = torch.zeros_like(sim_neg)
            L_link_neg = F.mse_loss(sim_neg, zero)
            L_link = 0.5 * (L_link_pos + L_link_neg)
        else:
            L_link = L_link_pos

        return L_ent, L_link


# ----------------- USO -----------------
if __name__ == "__main__":
    from tqdm import trange

    model = GraphAutoencoderDiffPool(
        in_dim=IN_DIM,
        edge_dim=EDGE_DIM,
        out_dim=OUT_DIM,
        hidden_channels=HIDDEN_CHANNELS,
        C=C,
        bottleneck=BOTTLENECKLATENTSPACE,
        conv=KIND_CONV,
        skip=True,
        dropout=DROPOUT,
        edge_attn_mode="topk",  # "hard" o "topk"
        edge_topk=2,
        lambda_ent=1e-3,
        lambda_link=1e-3,
        use_gumbel=True,
        gumbel_tau=0.90,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_hist = []

    loop = trange(EPOCHS, desc="Training", dynamic_ncols=True)
    edge_norm = EdgeNormalizer(data.edge_attr)
    data.edge_attr = edge_norm.encode(data.edge_attr)

    for epoch in loop:
        model.train()
        opt.zero_grad()
        y_hat, e_hat, aux = model(data)
        # 3.1) Node reconstruction

        recon_nodes = F.mse_loss(y_hat, data.y)

        # 3.2) Edge reconstruction (se hai edge_attr target)
        # Se EDGE_DIM è geometrico (dx, dy, dist, etc.), MSE è ok:
        recon_edges = F.mse_loss(e_hat, data.edge_attr)

        # 3.3) Consistency tra e_hat e A_up_edge (facoltativa ma potente)
        # Spinge e_hat ad essere coerente con l’upscaling strutturale
        a_up = aux["A_up_edge"]  # [E,1]
        consistency = F.mse_loss(
            e_hat.norm(p=2, dim=1, keepdim=True), a_up
        )  # esempio semplice

        # 3.4) Regolarizzazioni su S (come già hai)
        L_ent, L_link = model.regularization_losses(aux)

        # 3.5) Loss totale (pesi da tarare)
        loss = recon_nodes + 0.2 * consistency + model.lambda_link * L_link

        loss.backward()
        opt.step()

        # tqdm: show live metrics senza ristampare
        loop.set_postfix(
            {
                "recon_nodes": f"{recon_nodes.item():.6f}",
                "recon_edges": f"{recon_edges.item():.6f}",
                "consistency": f"{consistency.item():.6f}",
                "L_ent": f"{L_ent.item():.6f}",
                "L_link": f"{L_link.item():.6f}",
                "loss": f"{loss.item():.6f}",
            }
        )

        if epoch % 2 == 0:
            loss_hist.append(loss.item())

    torch.save(model.state_dict(), MODEL_PATH)
    np.savetxt(LOSS_PATH, np.array(loss_hist))
