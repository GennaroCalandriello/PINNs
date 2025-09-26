import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import numpy as np
import pickle

"""
4. GNN Autoencoder

- GNN encoder and decoder.
- If use_kalman is True, the latent code is passed through a Kalman filter before decoding.
- Forward pass: input graph data → encoder, compresses node features into latent space → optional Kalman filter if not None → decoder, reconstructs node features.

GNNEncoder:
- Embeds node features with an MLP, applies multiple GNOLayer message-passing blocks, and compresses to the latent dimension.

GNNDecoder:
- Takes latent node codes, applies multiple GNOLayer blocks, and reconstructs the target variables via an output MLP.

GNOLayer (MessagePassing):
- For each edge, computes a message using source/target node features and edge features.
- Aggregates incoming messages for each node (mean).
- Updates each node's feature via another MLP, optionally with a residual connection if input and output dims match.
"""

# hyperpar
num_layers = 6
hidden_dim = 50
latent_dim = 10
epochs = 3000
scheduler_step = 500
assign_dim = 5  # num_clusters for DiffPool
# path_data = "patches/patch_max_uniformCyl.pkl"  # Update with actual path
# path_data = "patches/experiments20k.pkl"
# path_data = "patches/experiments.pkl"
path_data = "patches/cavity20k.pkl"
radius = 3000
DROPOUT = 0.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def fit_norm_over_all(
    results,
    use_volume=True,
    dim=2,  # dim=2: area→h=sqrt(A); dim=3: h=V**(1/3)
    sample_per_t=None,  # es. 5000 per ridurre RAM; None = tutti
    use_log_for_V=False,
    device="cuda",
):
    import numpy as np

    xs, us, vs = [], [], []

    for tup in results:
        # supporta entrambi gli schemi results:
        # (t, centers, U, neighbors, edge_index[, V])
        centers = tup[1]
        U = tup[2]
        xs.append(centers)
        us.append(U)

        if use_volume and len(tup) >= 6:
            V = tup[-1]  # ultima voce se presente
            vs.append(V)

    X = np.concatenate(xs, axis=0)
    Uall = np.concatenate(us, axis=0)

    if sample_per_t is not None and sample_per_t < X.shape[0]:
        idx = np.random.choice(X.shape[0], sample_per_t, replace=False)
        X = X[idx]
        Uall = Uall[idx]
        if use_volume and vs:
            Vall = np.concatenate(vs, axis=0)[idx]
    else:
        if use_volume and vs:
            Vall = np.concatenate(vs, axis=0)

    # Normalizzatori pos e U
    xynorm = GaussianNormalizer(X)
    uvnorm = GaussianNormalizer(Uall)
    if device == "cuda":
        xynorm.cuda()
        uvnorm.cuda()

    # Normalizzatore per H (da V) o per V_log
    hnorm = None
    volnorm = None
    if use_volume and vs:
        if use_log_for_V:
            Vlog = np.log1p(np.clip(Vall, 0.0, None)).reshape(-1, 1)
            volnorm = GaussianNormalizer(Vlog)
            volnorm.cuda() if device == "cuda" else None
        else:
            # H = V^(1/dim)
            if dim == 2:
                H = np.sqrt(np.clip(Vall, 1e-30, None)).reshape(-1, 1)
            else:
                H = np.power(np.clip(Vall, 1e-30, None), 1.0 / dim).reshape(-1, 1)
            hnorm = GaussianNormalizer(H)
            hnorm.cuda() if device == "cuda" else None

    return xynorm, uvnorm, hnorm, volnorm


# ===== 1. Data loader and normalizer =====
def dataLoader(path_data=path_data):

    with open(path_data, "rb") as f:
        data = pickle.load(f)

    results = data["results"]
    idx_cells = data["idx_cells"]
    print("Data loaded successfully.")
    print(f"Number of cells: {len(idx_cells)}, Number of time steps: {len(results)}")

    return results, idx_cells


# ===== 2. GAUSSIAN NORMALIZER =====
class GaussianNormalizer:
    def __init__(self, x, eps=0.0):
        x = torch.as_tensor(x, dtype=torch.float32)
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x, idx=None):
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            if idx is None:
                raise ValueError(
                    "You must specify idx when decoding a single variable."
                )
            return x * (self.std[idx] + self.eps) + self.mean[idx]
        else:
            return x * (self.std + self.eps) + self.mean

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self


def dataNormalizer(results):

    t0, centers, U, neighbours, edge_index, vol = results[0]
    print(centers)
    print(f"Data shape: {U.shape}")
    xynorm = GaussianNormalizer(centers)
    uvnorm = GaussianNormalizer(U)
    volnorm = GaussianNormalizer(vol)
    xynorm.cuda()
    uvnorm.cuda()
    volnorm.cuda()
    print("Data normalized successfully.")

    return t0, centers, U, neighbours, vol, edge_index, xynorm, uvnorm, volnorm


def geometryObject(xy, center, width, height):
    """Rectangle: returns (normalized distance, inside_mask).
    dist_norm = max( dx/(w/2), dy/(h/2) )."""
    cx, cy = center
    dx = (xy[:, 0] - cx).abs()
    dy = (xy[:, 1] - cy).abs()
    half_w = width * 0.5
    half_h = height * 0.5
    inside = (dx <= half_w) & (dy <= half_h)
    dist_norm = torch.maximum(dx / (half_w + 1e-12), dy / (half_h + 1e-12))
    return dist_norm.unsqueeze(1), inside.unsqueeze(1)


# ...existing code...
from torch_geometric.data import Data
import torch


def createGraphData():
    """Crea un PyTorch Geometric Data object dai dati normalizzati."""

    # --- Carica e normalizza ---
    results, idx_cells = dataLoader(path_data)
    t0, centers, U, neighbours, V, edge_index, xynorm, uvnorm, volnorm = dataNormalizer(
        results
    )

    # --- Normalizzazioni (tutti 2D su device) ---
    U_norm = uvnorm.encode(torch.tensor(U, dtype=torch.float32, device=device))  # [N,2]
    centers_norm = xynorm.encode(
        torch.tensor(centers, dtype=torch.float32, device=device)
    )  # [N,2]
    V_norm = volnorm.encode(torch.tensor(V, dtype=torch.float32, device=device)).view(
        -1, 1
    )  # [N,1]  <<< IMPORTANTISSIMO

    sigma_x = xynorm.std[0].item()
    sigma_y = xynorm.std[1].item()
    print(f"Sigma X: {sigma_x}, Sigma Y: {sigma_y}")

    # --- Definizione rettangolo fisico ---
    cx_phys, cy_phys = 1050, 100
    rect_width_phys, rect_height_phys = 100, 200

    # --- Conversione in spazio normalizzato ---
    center_norm = torch.tensor(
        [
            (cx_phys - xynorm.mean[0].item()) / sigma_x,
            (cy_phys - xynorm.mean[1].item()) / sigma_y,
        ],
        dtype=torch.float32,
        device=device,
    )
    rect_width_norm = rect_width_phys / sigma_x
    rect_height_norm = rect_height_phys / sigma_y

    print(f"Rectangle center (norm): {center_norm.tolist()}")
    print(
        f"Rectangle size (norm W x H): {rect_width_norm:.3f} x {rect_height_norm:.3f}"
    )

    # --- Feature geometriche del rettangolo ---
    rect_dist_norm, rect_mask = geometryObject(
        centers_norm,
        (center_norm[0].item(), center_norm[1].item()),
        rect_width_norm,
        rect_height_norm,
    )
    rect_mask = rect_mask.float()  # [N,1]  <<< assicurati sia float
    # rect_dist_norm è già [N,1]

    # --- Features nodali ---
    # Tutti [N,?] con la seconda dimensione presente
    feature_list = [U_norm, rect_dist_norm, rect_mask, V_norm]
    x = torch.cat(feature_list, dim=1)  # OK

    # --- Target ---
    y = U_norm

    # --- pos = coordinate dei nodi ---
    pos = centers_norm

    # --- Costruzione edge_index ---
    edge_src, edge_dst = [], []
    for i, nbs in enumerate(neighbours):
        for j in nbs:
            if i != j:
                edge_src.append(i)
                edge_dst.append(j)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)

    # --- Edge attributes ---
    rel = pos[edge_index[1]] - pos[edge_index[0]]  # [E,2]
    dist = torch.norm(rel, dim=1, keepdim=True)  # [E,1]
    dirn = rel / (dist + 1e-12)  # [E,2]
    edge_attr = torch.cat([rel, dist, dirn], dim=1)  # [E,5]

    # ea_norm = GaussianNormalizer(edge_attr)
    # ea_norm.cuda()
    # edge_attr = ea_norm.encode(edge_attr)

    print("Some edge attributes samples after norm:", edge_attr[:2])

    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y)


def createGraphData2(path_data=path_data, use_log_for_V=True, dim=2, sample_per_t=None):
    """
    Crea un PyTorch Geometric Data object dai dati normalizzati.
    - Normalizzatori (pos, U, H/logV) fittati su T interi.
    - H = V^(1/dim) (default dim=2 -> sqrt(area)); in alternativa log1p(V).
    - Grafo statico dal primo frame (pos, edges).
    """
    # -------- Carica tutta la serie --------
    results, idx_cells = dataLoader(path_data)

    # -------- Fit normalizzatori su T interi --------
    xs, us, vs = [], [], []
    has_volume = False
    for tup in results:
        # supporta schemi results:
        # (t, centers, U, neighbors, edge_index)  oppure
        # (t, centers, U, neighbors, edge_index, V)
        centers = tup[1]
        U = tup[2]
        xs.append(centers)
        us.append(U)
        if len(tup) >= 6:
            has_volume = True
            vs.append(tup[-1])

    X_all = np.concatenate(xs, axis=0)
    U_all = np.concatenate(us, axis=0)

    if sample_per_t is not None and sample_per_t < X_all.shape[0]:
        idx = np.random.choice(X_all.shape[0], sample_per_t, replace=False)
        X_all = X_all[idx]
        U_all = U_all[idx]
        if has_volume:
            V_all = np.concatenate(vs, axis=0)[idx]
    else:
        if has_volume:
            V_all = np.concatenate(vs, axis=0)

    # normalizzatori pos e U su T interi
    xynorm = GaussianNormalizer(X_all)
    xynorm.cuda()
    uvnorm = GaussianNormalizer(U_all)
    uvnorm.cuda()

    # normalizzatore per H (da V) o per logV (se presente)
    hnorm = None
    volnorm = None
    if has_volume:
        if use_log_for_V:
            Vlog_all = np.log1p(np.clip(V_all, 0.0, None)).reshape(-1, 1)
            volnorm = GaussianNormalizer(Vlog_all)
            volnorm.cuda()
        else:
            if dim == 2:
                H_all = np.sqrt(np.clip(V_all, 1e-30, None)).reshape(-1, 1)
            else:
                H_all = np.power(np.clip(V_all, 1e-30, None), 1.0 / dim).reshape(-1, 1)
            hnorm = GaussianNormalizer(H_all)
            hnorm.cuda()

    # -------- Prendi un frame per costruire il grafo statico --------
    tup0 = results[0]
    # (t, centers, U, neighbors, edge_index [, V])
    centers = tup0[1]
    U = tup0[2]
    neighbours = tup0[3]
    # edge_index dal frame (se già fornito) o da neighbours
    if isinstance(tup0[4], np.ndarray) or (torch.is_tensor(tup0[4])):
        edge_index_np = tup0[4]
        # garantisci [2,E]
        if isinstance(edge_index_np, np.ndarray):
            if edge_index_np.shape[0] != 2:
                edge_index_np = edge_index_np.T
        else:
            edge_index_np = edge_index_np.cpu().numpy()
        edge_src = edge_index_np[0].tolist()
        edge_dst = edge_index_np[1].tolist()
    else:
        edge_src, edge_dst = [], []
        for i, nbs in enumerate(neighbours):
            for j in nbs:
                if i != j:
                    edge_src.append(i)
                    edge_dst.append(j)

    # Volume del frame (se presente)
    V = tup0[-1] if (has_volume and len(tup0) >= 6) else None

    # -------- Normalizza pos, U, (H/logV) --------
    centers_t = torch.tensor(centers, dtype=torch.float32, device=device)
    U_t = torch.tensor(U, dtype=torch.float32, device=device)

    centers_norm = xynorm.encode(centers_t)  # [N,2]
    U_norm = uvnorm.encode(U_t)  # [N,2]

    H_norm = None
    if has_volume and V is not None:
        V_t = torch.tensor(V, dtype=torch.float32, device=device)
        if use_log_for_V:
            Vlog = torch.log1p(V_t.clamp_min(0.0)).view(-1, 1)  # [N,1]
            H_norm = volnorm.encode(Vlog)  # [N,1] (logV_norm)
        else:
            H_t = (
                V_t.clamp_min(1e-30).sqrt()
                if dim == 2
                else V_t.clamp_min(1e-30).pow(1.0 / dim)
            ).view(-1, 1)
            H_norm = hnorm.encode(H_t)  # [N,1]

    # -------- Rettangolo fisico -> feature geometriche --------
    sigma_x = xynorm.std[0].item()
    sigma_y = xynorm.std[1].item()
    cx_phys, cy_phys = 1050, 100
    rect_width_phys, rect_height_phys = 100, 200

    center_norm = torch.tensor(
        [
            (cx_phys - xynorm.mean[0].item()) / sigma_x,
            (cy_phys - xynorm.mean[1].item()) / sigma_y,
        ],
        dtype=torch.float32,
        device=device,
    )
    rect_width_norm = rect_width_phys / sigma_x
    rect_height_norm = rect_height_phys / sigma_y

    rect_dist_norm, rect_mask = geometryObject(
        centers_norm,
        (center_norm[0].item(), center_norm[1].item()),
        rect_width_norm,
        rect_height_norm,
    )
    rect_mask = rect_mask.float()  # [N,1]

    # -------- Node features --------
    feats = [U_norm, rect_dist_norm, rect_mask]
    if H_norm is not None:
        feats.append(H_norm)  # [N,1]
    x = torch.cat(
        feats, dim=1
    )  # [N, F] x = [U_norm(2), rect_dist(1), rect_mask(1), (H/logV)(1)]

    # -------- Target --------
    y = U_norm  # autoencoder

    # -------- pos & edges --------
    pos = centers_norm
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)

    # -------- Edge attributes (non normalizzati, come hai richiesto) --------
    rel = pos[edge_index[1]] - pos[edge_index[0]]  # [E,2]
    dist = torch.norm(rel, dim=1, keepdim=True)  # [E,1]
    dirn = rel / (dist + 1e-12)  # [E,2]
    edge_attr = torch.cat([rel, dist, dirn], dim=1)  # [E,5]

    print("Edge attr samples:", edge_attr[:2])

    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y)


if __name__ == "__main__":

    MODEL_PATH = "model/gnn_autoencoder.pth"
    LOSS_PATH = "model/loss_history_gnn_autoencoder.npy"

    import torch
    from torch_scatter import scatter_add

    class GNNAutoencoder(nn.Module):
        def __init__(
            self,
            input_dim,
            hidden_dim,
            latent_dim,
            output_dim,
            edge_dim,
            num_layers=num_layers,
            use_kalman=False,
        ):
            super().__init__()
            self.encoder = GNNEncoder(
                input_dim, hidden_dim, latent_dim, edge_dim, num_layers=num_layers
            )
            self.decoder = GNNDecoder(
                latent_dim, hidden_dim, output_dim, edge_dim, num_layers=num_layers
            )
            self.use_kalman = use_kalman
            self.kalman_filter = None

        def attach_kalman(self, kalman_filter):
            self.kalman_filter = kalman_filter
            self.use_kalman = True

        def forward(self, data):
            z = self.encoder(data)
            if self.use_kalman and self.kalman_filter is not None:
                z_mean = z.mean(dim=0).detach().cpu().numpy()
                z_filtered = self.kalman_filter.update(z_mean)
                z_filtered = torch.tensor(
                    z_filtered, dtype=torch.float32, device=z.device
                )
                z = z_filtered.unsqueeze(0).repeat(z.shape[0], 1)
            return self.decoder(z, data.edge_index, data.edge_attr)

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
                nn.Dropout(DROPOUT),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, z, edge_index, edge_attr):
            for layer in self.layers:
                z = layer(z, edge_index, edge_attr)
            return self.output_layer(z)

    class GNOLayer(MessagePassing):
        def __init__(self, in_channels, out_channels, hidden_dim, edge_dim):
            super(GNOLayer, self).__init__(aggr="mean")
            self.msg_mlp = nn.Sequential(
                nn.Linear(2 * in_channels + edge_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(hidden_dim, out_channels),
            )
            self.update_mlp = nn.Sequential(
                nn.Linear(in_channels + out_channels, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(DROPOUT),
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

    def train(data, latent_dim=latent_dim, epochs=epochs):
        from tqdm import trange

        """Train the GNN autoencoder."""
        print("Starting training...")
        results, _ = dataLoader(path_data)
        t0, centers, U, neighbours, edge_index, xynorm, uvnorm = dataNormalizer(results)

        input_dim = data.x.shape[1]
        edge_dim = data.edge_attr.shape[1]
        output_dim = 3  # u, v, p

        model = GNNAutoencoder(
            input_dim, hidden_dim, latent_dim, output_dim, edge_dim
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step, gamma=0.9
        )

        data = data.to(device)
        loss_history = []
        loop = trange(epochs, desc="Training", dynamic_ncols=True)

        for epoch in loop:
            optimizer.zero_grad()
            pred = model(data)
            u_pred = pred[:, 0]
            v_pred = pred[:, 1]
            p_pred = pred[:, 2]
            # denormalize predictions
            # u_pred = uvnorm.decode(u_pred, idx=0)
            # v_pred = uvnorm.decode(v_pred, idx=1)
            # true values:
            u_true = data.y[:, 0]
            v_true = data.y[:, 1]
            # u_true = uvnorm.decode(u_true, idx=0)
            # v_true = uvnorm.decode(v_true, idx=1)
            # p_true = data.y[:, 2]
            loss = F.mse_loss(u_pred, u_true) + F.mse_loss(v_pred, v_true)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loop.set_postfix(
                {
                    "loss": f"{loss.item():.6f}",
                }
            )

            if epoch % 2 == 0:
                loss_history.append(loss.item())
        print(" ✅ Training complete.")

        torch.save(model.state_dict(), MODEL_PATH)
        np.save(LOSS_PATH, np.array(loss_history))
        print(" ✅ Model and loss history saved.")

    if __name__ == "__main__":
        data = createGraphData()
        train(data, latent_dim=latent_dim, epochs=epochs)
        print("Training finished and model saved.")
