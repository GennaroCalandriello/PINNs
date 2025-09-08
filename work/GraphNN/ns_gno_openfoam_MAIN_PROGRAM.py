import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt


# ===== 1. PARAMETERS =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
hidden_channels = 150  # è qui che gestisco la nonlinearità (?)
output_channels = 3  # u, v, p
epochs = 1000
scheduler_step = 500
os.makedirs("model", exist_ok=True)

num_layers = 8
# data loader
load_data = True  # Se True, carica i dati da file pickle, salvare i file da openPyMeshParallel.py, qui il multiprocessing non funziona bene
patch_size = 40000
# path_pkl = f"patch_{patch_size//1000}k_uniform.pkl"
path_pkl = "patches/patch_max_uniform.pkl"  # file pickle con i dati della patch
# modelSavePath = f"model/gno_modelGELU{patch_size//1000}k.pth"
modelSavePath = "model/gno_modelGELUall.pth"


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


# ===== 3. GNO LAYER & MODEL =====
class GNOLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim, edge_dim):
        super(GNOLayer, self).__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, out_channels),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, out_channels),
        )
        self.residual = in_channels == out_channels

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.residual:
            return out + x
        else:
            return out

    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_mlp(msg_input)

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class GNOModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, edge_dim, num_layers=num_layers
    ):
        super(GNOModel, self).__init__()
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
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return self.decoder(x)


# load from file

if load_data:
    import pickle

    with open(path_pkl, "rb") as f:
        data = pickle.load(f)

    results = data["results"]
    idx_cells = data["idx_cells"]

print(f"Loaded {len(results)} snapshots with patch size {len(idx_cells)} cells.")

print(f"Serie temporale: {len(results)} snapshot - patch size {patch_size}")

# ----- Estrazione dati di normalizzazione dal PRIMO snapshot -----
t0, centers, U, neighbors, edge_index = results[0]
print("Shape of centers:", centers.shape)
print("Shape of U:", U.shape)
print("Shape of neighbors:", len(neighbors), "neighbors per cella")
print("Shape of edge_index:", edge_index.shape)
print("Max and min centers:", centers.max(), centers.min())
print("Some centers:", centers[:10])  # primi 10 centri
print("Max and min U:", U.max(), U.min())
print("Some neighbours:", neighbors[0][:10])  # primi 10 vicini della prima cella
xynorm = GaussianNormalizer(centers)
uvnorm = GaussianNormalizer(U)
xynorm.cuda()
uvnorm.cuda()


def compute_cyl_features(xy, center=(0.0, 0.0), radius=1.0):
    cx, cy = center
    dist = torch.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    inside = dist < radius
    dist_norm = dist / radius
    return dist_norm.unsqueeze(1), inside.unsqueeze(1)


def create_graph_data_mesh(
    centers,
    uv,
    neighbors,
    add_cyl_features=True,
    cyl_center=(0.0, 0.0),
    cyl_radius=0.51,
):
    """
    Crea PyG Data usando la connettività mesh reale (no kNN).
    - centers: [N,2] (torch tensor, già su device)
    - uv: [N,2] (torch tensor, già su device)
    - neighbors: lista di liste di indici batch
    """
    # Assicurati che tutto sia tensor torch su device
    centers = torch.as_tensor(centers, dtype=torch.float32, device=device)
    uv = torch.as_tensor(uv, dtype=torch.float32, device=device)

    feature_list = [centers, uv]
    if add_cyl_features:
        dist_norm, inside = compute_cyl_features(
            centers, center=cyl_center, radius=cyl_radius
        )
        feature_list.append(dist_norm)
        feature_list.append(inside)
    # torch.cat invece di np.concatenate!
    x = torch.cat(feature_list, dim=1)
    y = uv  # (già tensor)
    # Edge index
    edge_src, edge_dst = [], []
    for i, nbs in enumerate(neighbors):
        for j in nbs:
            edge_src.append(i)
            edge_dst.append(j)
    edge_src = torch.tensor(edge_src, dtype=torch.long, device=device)
    edge_dst = torch.tensor(edge_dst, dtype=torch.long, device=device)
    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    rel_pos = centers[edge_dst] - centers[edge_src]  # [num_edges, 2]
    dist = torch.norm(rel_pos, dim=1, keepdim=True)  # [num_edges, 1]
    edge_attr = torch.cat([rel_pos, dist], dim=1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ===== 6. TRAINING: PATCH @ t0 =====
centers_norm = xynorm.encode(
    torch.tensor(centers, dtype=torch.float32, device=xynorm.mean.device)
)
print("centers_norm", centers_norm)
print("Max and min centers_norm:", centers_norm.max(), centers_norm.min())
U_norm = uvnorm.encode(torch.tensor(U, dtype=torch.float32, device=uvnorm.mean.device))
# U_norm = uvnorm.encode(U)

# Parametri del dominio (usa quelli che vuoi)
rad = 3000
sigma_x = xynorm.std[0].item()
sigma_y = xynorm.std[1].item()
print("sigma_y:", sigma_y)
print("sigma_x:", sigma_x)
rad_scaled = rad / sigma_x
print(f"Scaled radius: {rad_scaled:.2f} (original {rad})")

graph_data = create_graph_data_mesh(
    centers_norm,
    U_norm,
    neighbors,
    add_cyl_features=True,
    cyl_center=(0.0, 0.0),
    cyl_radius=rad_scaled,
)


def train(data, epochs=epochs):
    input_dim = data.x.shape[1]
    edge_dim = data.edge_attr.shape[1]
    model = GNOModel(input_dim, hidden_channels, output_channels, edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=0.9
    )
    data = data.to(device)
    loss_hist = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        data.x.requires_grad_(True)
        pred = model(data)
        u_pred = uvnorm.decode(pred[:, 0], idx=0)
        v_pred = uvnorm.decode(pred[:, 1], idx=1)
        u_true = uvnorm.decode(data.y[:, 0], idx=0)
        v_true = uvnorm.decode(data.y[:, 1], idx=1)
        loss_data = F.mse_loss(u_pred, u_true) + F.mse_loss(v_pred, v_true)
        loss = loss_data
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 1 == 0:
            print(
                f"Epoch {epoch:4d} | data_loss={loss_data.item():.3e} |  total={loss.item():.3e}"
            )
            loss_hist.append(loss.item())

    torch.save(model.state_dict(), modelSavePath)
    np.savetxt(f"model/loss{patch_size//1000}k.npy", np.array(loss_hist))
    print("✅ Training complete, model saved.")


out_gif = "cylinder_timeseries_all40k.gif"


# ===== 7. ANIMATE: SERIE TEMPORALE PATCH =====
def animate_patch_time_series(
    results,
    xynorm,
    uvnorm,
    rad_scaled,
    out_gif=out_gif,
):
    import matplotlib.tri as mtri
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation

    # --- Modello (una volta sola) ---
    t0, centers0, U0, neighbors0, edge_index0 = results[0]
    device = xynorm.mean.device  # prendi device dai normalizer

    # Decodifica centri cella su device corretto!
    centers0 = torch.tensor(centers0, dtype=torch.float32, device=device)
    centers0_dec = centers0.cpu().numpy()
    print("centers0_dec shape:", centers0_dec)
    xdec, ydec = centers0_dec[:, 0], centers0_dec[:, 1]

    # Maschera solo fluidi (per il plot, su CPU)
    cx, cy = 10000, 10000
    r_cyl_dec = 3000
    print(f"Radius cylinder (dec): {r_cyl_dec:.2f} (scaled from {rad_scaled})")
    mask_fluid = ((xdec - cx) ** 2 + (ydec - cy) ** 2) >= r_cyl_dec**2
    xdec_fluid = xdec[mask_fluid]
    ydec_fluid = ydec[mask_fluid]

    # Triangolazione
    triang = mtri.Triangulation(xdec_fluid, ydec_fluid)
    tri_points = np.stack(
        [xdec_fluid[triang.triangles], ydec_fluid[triang.triangles]], axis=-1
    )
    tri_centers_x = tri_points[:, :, 0].mean(axis=1)
    tri_centers_y = tri_points[:, :, 1].mean(axis=1)
    dist2 = (tri_centers_x - cx) ** 2 + (tri_centers_y - cy) ** 2
    mask_tri = dist2 < r_cyl_dec**2
    triang.set_mask(mask_tri)

    # --- Carica modello (device sicuro) ---
    input_dim = (
        results[0][1].shape[1] + results[0][2].shape[1] + 2
    )  # [x,y] + [u,v] + dist, inside (se usi)
    edge_dim = 3  # rel_x, rel_y, dist
    model = GNOModel(input_dim, hidden_channels, output_channels, edge_dim).to(device)
    model.load_state_dict(torch.load(modelSavePath, map_location=device))
    model.eval()

    # --- LOOP su tutti i tempi, decodifica predizioni ---
    mag_vals = []
    for t, centers, U, neighbors, edge_index in results:
        centers = torch.tensor(centers, dtype=torch.float32, device=device)
        U = torch.tensor(U, dtype=torch.float32, device=device)

        # Encode PRIMA di passare alla rete!
        # centers_norm = centers
        centers_norm = xynorm.encode(centers)
        U_norm = uvnorm.encode(U)
        U_norm = U

        g = create_graph_data_mesh(
            centers_norm,  # tensor, già su device
            U_norm,
            neighbors,
            add_cyl_features=True,
            cyl_center=(0, 0),
            cyl_radius=rad_scaled,
        )
        g = g.to(device)
        with torch.no_grad():
            uvp = model(g)
            # Qui decode su predizione per output fisico!
            # u_pred = uvp[:, 0].cpu().numpy()
            # v_pred = uvp[:, 1].cpu().numpy()
            u_pred = uvnorm.decode(uvp[:, 0], idx=0).cpu().numpy()
            v_pred = uvnorm.decode(uvp[:, 1], idx=1).cpu().numpy()
            mag = np.linalg.norm(np.stack([u_pred, v_pred], axis=-1), axis=1)
            mag_vals.append(mag[mask_fluid])
    mag_vals = np.array(mag_vals)
    vmin, vmax = mag_vals.min(), mag_vals.max()

    # --- PLOT & ANIM ---
    fig, ax = plt.subplots(figsize=(7, 6))
    cntr = ax.tricontourf(
        triang, mag_vals[0], levels=600, cmap="jet", vmin=vmin, vmax=vmax
    )
    cb = fig.colorbar(cntr, ax=ax, label="|u,v|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title("t = 0.00")
    ax.axis("equal")
    circle = Circle(
        (cx, cy),
        r_cyl_dec,
        color="k",
        fill=False,
        linewidth=0.0,
        linestyle="",
        zorder=10,
    )
    ax.add_patch(circle)

    def update(frame):
        for c in ax.collections:
            c.remove()
        cntr = ax.tricontourf(
            triang, mag_vals[frame], levels=600, cmap="jet", vmin=vmin, vmax=vmax
        )
        t_now = results[frame][0]
        title.set_text(f"t = {t_now:.3f}")
        ax.add_patch(circle)
        return []

    from matplotlib.animation import FuncAnimation

    ani = FuncAnimation(
        fig, update, frames=len(results), interval=60, blit=False, repeat=True
    )
    ani.save(out_gif, writer="imagemagick", fps=20)
    print("✅ Animation saved as", out_gif)
    return ani


def plot_loss():
    loss_hist = np.loadtxt(f"model/loss{patch_size//1000}k.npy")
    plt.figure(figsize=(8, 5))
    plt.plot(loss_hist, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.show()


# ===== 8. MAIN =====
if __name__ == "__main__":
    # TRAIN
    # train(graph_data, epochs=epochs)
    # ANIMATE
    print("xynorm mean:", xynorm.mean)
    print("uvnorm mean:", uvnorm.mean)
    print("max and min centers:", centers.max(), centers.min())
    animate_patch_time_series(results, xynorm, uvnorm, rad_scaled)
    plot_loss()
