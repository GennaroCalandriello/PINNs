import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.data import Data
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

# ===== 2. Parameters =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
hidden_channels = 200
output_channels = 3  # u, v, p
k_neighbors = 3
epochs = 3000
sample_dim = 40000
scheduler_step = 500
os.makedirs("model", exist_ok=True)
modelSavePath = "model/gno_modelGELU3.pth"
num_layers = 5


# ===== 1. GaussianNormalizer =====
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


# ===== 3. GNO Model (now with edge_attr support) =====
class GNOLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim, edge_dim):
        super(GNOLayer, self).__init__(aggr="mean")
        # Message now gets edge features
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
        # x_i: [num_edges, in_channels], x_j: [num_edges, in_channels], edge_attr: [num_edges, edge_dim]
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


# ===== 4. Data Preparation =====
from openPy import DataSamplerOpenFoam

data_sampler = DataSamplerOpenFoam("cylinderFoam/velocity_*.dat")
print(f"Data shape: {data_sampler.data.shape}")
xynorm = GaussianNormalizer(data_sampler.data[:, 1:])
data_sampler.data[:, 1:] = xynorm.encode(data_sampler.data[:, 1:])
uvnorm = GaussianNormalizer(data_sampler.data[:, -2:])  # u, v
data_sampler.data[:, -2:] = uvnorm.encode(data_sampler.data[:, -2:])

sigma_x = xynorm.std[0].item()
rad = 3000
rad_scaled = rad / sigma_x
print(f"Scaled radius: {rad_scaled:.2f}")

X_batch, U_batch = data_sampler.sample(sample_dim)
xynorm.cuda()
uvnorm.cuda()


def compute_cyl_features(xy, center=(0.0, 0.0), radius=1.0):
    cx, cy = center
    dist = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    inside = (dist < radius).astype(np.float32)
    dist_norm = dist / radius
    return dist_norm[:, None], inside[:, None]


def create_graph_data(xyt, uv, cyl_center=(0.0, 0.0), cyl_radius=0.51):
    xy = xyt[:, 1:3]
    dist_norm, inside = compute_cyl_features(xy, center=cyl_center, radius=cyl_radius)
    features = np.concatenate([xyt, dist_norm, inside], axis=1)  # [N, 5]
    x = torch.tensor(features, dtype=torch.float, device=device)
    y = torch.tensor(uv, dtype=torch.float, device=device)
    edge_index = knn_graph(x[:, 1:3], k=k_neighbors, batch=None, loop=False).to(device)
    # -------- EDGE FEATURES --------
    src, dst = edge_index
    rel_pos = x[dst, 1:3] - x[src, 1:3]  # [num_edges, 2]
    dist = torch.norm(rel_pos, dim=1, keepdim=True)  # [num_edges, 1]
    edge_attr = torch.cat([rel_pos, dist], dim=1)  # [num_edges, 3]
    # -------------------------------
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


graph_data = create_graph_data(X_batch, U_batch, cyl_center=(0.0, 0.0))


# ===== 6. TRAINING (supervisiona solo u,v) =====
def train(data, epochs=epochs):
    cyl_center = torch.tensor([0.0, 0.0], device=device)
    cyl_radius = rad_scaled
    input_dim = data.x.shape[1]
    edge_dim = data.edge_attr.shape[1]
    model = GNOModel(input_dim, hidden_channels, output_channels, edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=0.9
    )
    data = data.to(device)
    loss_hist = []

    unique_t = torch.unique(data.x[:, 0])
    print(f"Unique t in batch: {unique_t.cpu().numpy()}")
    if unique_t.numel() < 2:
        print(
            "\n[ERRORE] Hai bisogno di più snapshot temporali per usare la loss fisica su t!\n"
        )
        return

    coords = data.x[:, 1:3]
    dist2 = ((coords - cyl_center) ** 2).sum(dim=1)
    mask_obs = dist2 <= cyl_radius**2
    mask_fluid = dist2 > cyl_radius**2

    for epoch in range(epochs):
        optimizer.zero_grad()
        data.x.requires_grad_(True)

        pred = model(data)
        p_pred = pred[:, 2]

        u_pred = uvnorm.decode(pred[:, 0], idx=0)
        v_pred = uvnorm.decode(pred[:, 1], idx=1)
        u_true = uvnorm.decode(data.y[:, 0], idx=0)
        v_true = uvnorm.decode(data.y[:, 1], idx=1)
        loss_data = F.mse_loss(u_pred, u_true) + F.mse_loss(v_pred, v_true)
        loss = loss_data
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | data_loss={loss_data.item():.3e} |  total={loss.item():.3e}"
            )
            loss_hist.append(loss.item())

    torch.save(model.state_dict(), modelSavePath)
    np.savetxt("model/loss.npy", np.array(loss_hist))
    print("✅ Training complete, model saved.")


def prediction_and_animate_tricontourf(
    graph_data, num_frames=100, xynorm=None, cyl_center=(0.0, 0.0), cyl_radius=0.51
):
    import matplotlib.tri as mtri
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation

    # Infer input_dim and edge_dim directly from training data!
    input_dim = graph_data.x.shape[1]
    edge_dim = graph_data.edge_attr.shape[1]
    model = GNOModel(input_dim, hidden_channels, output_channels, edge_dim).to(device)
    model.load_state_dict(torch.load(modelSavePath, map_location=device))
    model.eval()

    xy = graph_data.x[:, 1:3].detach().cpu().numpy()
    coords_tensor = torch.tensor(xy, dtype=torch.float32, device=device)

    # Denormalizza x, y
    if xynorm is not None:
        xy_torch = torch.from_numpy(xy).to(device)
        xdec = xynorm.decode(xy_torch[:, 0], idx=0).cpu().numpy()
        ydec = xynorm.decode(xy_torch[:, 1], idx=1).cpu().numpy()
    else:
        xdec = xy[:, 0]
        ydec = xy[:, 1]

    # Denormalizza centro e raggio
    if xynorm is not None:
        cx_dec = (
            xynorm.decode(torch.tensor(cyl_center[0], device=device), idx=0)
            .cpu()
            .numpy()[0]
        )
        cy_dec = (
            xynorm.decode(torch.tensor(cyl_center[1], device=device), idx=1)
            .cpu()
            .numpy()[1]
        )
        r_cyl_dec = cyl_radius * xdec.std()
    else:
        cx_dec, cy_dec = cyl_center
        r_cyl_dec = cyl_radius

    # Maschera solo fluidi (fuori dal cilindro)
    mask_fluid = ((xdec - cx_dec) ** 2 + (ydec - cy_dec) ** 2) >= r_cyl_dec**2
    xdec_fluid = xdec[mask_fluid]
    ydec_fluid = ydec[mask_fluid]
    coords_tensor_fluid = coords_tensor[mask_fluid]

    # Ricalcolo delle feature aggiuntive
    def compute_cyl_features(xy, center=(0.0, 0.0), radius=1.0):
        cx, cy = center
        dist = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
        inside = (dist < radius).astype(np.float32)
        dist_norm = dist / radius
        return dist_norm[:, None], inside[:, None]

    xy_fluid = np.stack([xdec_fluid, ydec_fluid], axis=1)
    dist_norm_fluid, inside_fluid = compute_cyl_features(
        xy_fluid, center=(cx_dec, cy_dec), radius=r_cyl_dec
    )

    t_samples = np.linspace(0, 1, num_frames)
    mag_vals = []
    with torch.no_grad():
        for t in t_samples:
            t_col = torch.full((coords_tensor_fluid.shape[0], 1), t, device=device)
            dist_col = torch.tensor(dist_norm_fluid, dtype=torch.float32, device=device)
            inside_col = torch.tensor(inside_fluid, dtype=torch.float32, device=device)
            x_pred = torch.cat(
                [t_col, coords_tensor_fluid, dist_col, inside_col], dim=1
            )  # shape (N, input_dim)

            # ===== Build edge_index and edge_attr exactly as in training =====
            edge_index_pred = knn_graph(
                x_pred[:, 1:3], k=k_neighbors, batch=None, loop=False
            ).to(device)
            src, dst = edge_index_pred
            rel_pos = x_pred[dst, 1:3] - x_pred[src, 1:3]  # [num_edges, 2]
            dist = torch.norm(rel_pos, dim=1, keepdim=True)  # [num_edges, 1]
            edge_attr_pred = torch.cat([rel_pos, dist], dim=1)  # [num_edges, 3]
            # ===============================================================

            data_pred = Data(
                x=x_pred, edge_index=edge_index_pred, edge_attr=edge_attr_pred
            ).to(device)
            uvp = model(data_pred)
            u_pred = uvnorm.decode(uvp[:, 0], idx=0)
            v_pred = uvnorm.decode(uvp[:, 1], idx=1)
            mag = np.linalg.norm(
                np.stack([u_pred.cpu().numpy(), v_pred.cpu().numpy()], axis=-1), axis=1
            )
            mag_vals.append(mag)
    mag_vals = np.array(mag_vals)
    vmin, vmax = mag_vals.min(), mag_vals.max()

    triang = mtri.Triangulation(xdec_fluid, ydec_fluid)
    tri_points = np.stack(
        [xdec_fluid[triang.triangles], ydec_fluid[triang.triangles]], axis=-1
    )
    tri_centers_x = tri_points[:, :, 0].mean(axis=1)
    tri_centers_y = tri_points[:, :, 1].mean(axis=1)
    dist2 = (tri_centers_x - cx_dec) ** 2 + (tri_centers_y - cy_dec) ** 2
    mask_tri = dist2 < r_cyl_dec**2
    triang.set_mask(mask_tri)

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
        (cx_dec, cy_dec),
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
        title.set_text(f"t = {frame/(num_frames-1):.2f}")
        ax.add_patch(circle)
        return []

    ani = FuncAnimation(
        fig, update, frames=num_frames, interval=60, blit=False, repeat=True
    )
    ani.save("cylinder_flow2.gif", writer="imagemagick", fps=20)
    # plt.show()
    return ani


if __name__ == "__main__":
    # Train the model
    train(graph_data, epochs=epochs)

    # Predict and animate
    ani = prediction_and_animate_tricontourf(
        graph_data,
        num_frames=100,
        xynorm=xynorm,
        cyl_center=(0.0, 0.0),
        cyl_radius=rad_scaled,
    )
# Usage example:
# train(graph_data, epochs=400)
