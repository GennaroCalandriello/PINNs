import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

out_gif = "difference_animation.gif"

"""In questo file carico la patch di dati, i due modelli GNO e calcolo la differenza tra le loro predizioni """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
hidden_channels = 150  # è qui che gestisco la nonlinearità (?)
output_channels = 3  # u, v, p
epochs = 2000
scheduler_step = 500
os.makedirs("model", exist_ok=True)

num_layers = 8
# data loader
load_data = True  # Se True, carica i dati da file pickle, salvare i file da openPyMeshParallel.py, qui il multiprocessing non funziona bene
patch_size = 40000
path_pkl = f"patch_{patch_size//1000}k_uniform.pkl"
# modelSavePath = f"model/gno_modelGELU{patch_size//1000}k.pth"
model5k_path = "model/gno_modelGELU5k.pth"
model40k_path = "model/gno_modelGELU40k.pth"

rad_scaled = 0.51


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


# here I load the data for the two models
path_pkl1 = "patch_40k_uniform.pkl"

with open(path_pkl1, "rb") as f:
    data40k = pickle.load(f)

results40k = data40k["results"]
idx_cells40k = data40k["idx_cells"]

t0, centers, U, neighbors, edge_index = results40k[0]

xynorm = GaussianNormalizer(centers)
uvnorm = GaussianNormalizer(U)
xynorm.cuda()
uvnorm.cuda()
print("min and max of centers:", centers.min(), centers.max())
print("min and max of U:", U.min(), U.max())


def differences(ani_bool=False):

    import matplotlib.tri as mtri
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation

    # --- Modello (una volta sola) ---
    t0, centers0, U0, neighbors0, edge_index0 = results40k[0]
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
        results40k[0][1].shape[1] + results40k[0][2].shape[1] + 2
    )  # [x,y] + [u,v] + dist, inside (se usi)
    edge_dim = 3  # rel_x, rel_y, dist

    # loading the two models
    model40k = GNOModel(input_dim, hidden_channels, output_channels, edge_dim).to(
        device
    )
    model5k = GNOModel(input_dim, hidden_channels, output_channels, edge_dim).to(device)
    model40k.load_state_dict(torch.load(model40k_path, map_location=device))
    model5k.load_state_dict(torch.load(model5k_path, map_location=device))

    model40k.eval()
    model5k.eval()

    # --- LOOP su tutti i tempi, decodifica predizioni ---
    mag_vals40k = []
    mag_vals5k = []
    abs_errs = []
    l2s = []

    for t, centers, U, neighbors, edge_index in results40k:
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
            uvp40k = model40k(g)
            uvp5k = model5k(g)
            # Qui decode su predizione per output fisico!
            # u_pred = uvp[:, 0].cpu().numpy()
            # v_pred = uvp[:, 1].cpu().numpy()
            u_pred40k = uvnorm.decode(uvp40k[:, 0], idx=0).cpu().numpy()
            v_pred40k = uvnorm.decode(uvp40k[:, 1], idx=1).cpu().numpy()
            mag40k = np.linalg.norm(np.stack([u_pred40k, v_pred40k], axis=-1), axis=1)
            mag_vals40k.append(mag40k[mask_fluid])

            u_pred5k = uvnorm.decode(uvp5k[:, 0], idx=0).cpu().numpy()
            v_pred5k = uvnorm.decode(uvp5k[:, 1], idx=1).cpu().numpy()
            mag5k = np.linalg.norm(np.stack([u_pred5k, v_pred5k], axis=-1), axis=1)
            mag_vals5k.append(mag5k[mask_fluid])

    mag_vals40k = np.array(mag_vals40k)
    mag_vals5k = np.array(mag_vals5k)
    print("mag max 40k:", mag_vals40k.max(), "min:", mag_vals40k.min())
    print("mag max 5k:", mag_vals5k.max(), "min:", mag_vals5k.min())
    abs_err = np.abs(mag_vals40k - mag_vals5k)
    l2 = np.sqrt(np.mean(abs_err**2, axis=1))
    relative_err = abs_err / (abs(mag_vals40k) + 1e-8)  # evita divisione per zero
    print("Relative error max:", relative_err.max(), "min:", relative_err.min())
    print("mean relative error:", np.mean(relative_err))

    vminl2, vmaxl2 = l2.min(), l2.max()
    vmin = abs_err.min()
    vmax = abs_err.max()
    print(f"l2vmin: {vminl2}, l2vmax: {vmaxl2}")
    print(f"abs_err min: {vmin}, max: {vmax}")

    # --- PLOT & ANIM ---
    if ani_bool:
        fig, ax = plt.subplots(figsize=(7, 6))
        cntr = ax.tricontourf(
            triang, abs_err[0], levels=600, cmap="viridis", vmin=0, vmax=3
        )
        cb = fig.colorbar(cntr, ax=ax, label="difference (40k - 5k) model")
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
            ax.tricontourf(
                triang, abs_err[frame], levels=600, cmap="jet", vmin=0, vmax=3
            )
            t_now = results40k[frame][0]
            title.set_text(f"t = {t_now:.3f}")
            ax.add_patch(circle)
            return []

        from matplotlib.animation import FuncAnimation

        ani = FuncAnimation(
            fig, update, frames=len(results40k), interval=60, blit=False, repeat=True
        )
        ani.save(out_gif, writer="pillow", fps=20)
        print("✅ Animation saved as", out_gif)
        return ani


if __name__ == "__main__":
    differences()
# valori delle velocità predette e degli errori:
# mag max 40k: 9.104164 min: 0.00040980996
# mag max 5k: 7.892516 min: 0.0003323566
# l2vmin: 0.015834426507353783, l2vmax: 0.2051508128643036
# abs_err min: 1.4901161193847656e-08, max: 2.1203689575195312
