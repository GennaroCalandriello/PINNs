import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# PARAMS
hidden_channels = 150
output_channels = 3
k_neighbors = 10
epochs = 1000
sample_dim = 20000
scheduler_step = 1000
os.makedirs("model", exist_ok=True)
modelSavePath = "model/gno_model.pth"
input_dim = 4  # t, x, y, omega, dist no omega
num_layers = 4
cyl_center_np = (0.0, 0.0)  # Normalized
cyl_center = torch.tensor([0.0, 0.0], device=device)
num_frames = 200


# ==== NORMALIZER ====
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


# ==== MODEL ====
import torch.nn as nn


class GNOLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim):
        super(GNOLayer, self).__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_dim),
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

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        if self.residual:
            return out + x
        else:
            return out

    def message(self, x_i, x_j):
        return self.msg_mlp(torch.cat([x_i, x_j], dim=-1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class GNOModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=num_layers):
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
            [GNOLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.decoder(x)


# ==== GRAPH GRADIENTS ====
def weighted_graph_grad(field, x, edge_index, eps=1e-12):
    row, col = edge_index
    dx = x[col, 0] - x[row, 0]
    dy = x[col, 1] - x[row, 1]
    dist = torch.sqrt(dx**2 + dy**2) + eps
    dfield = field[col] - field[row]
    contribution = dfield / dist
    w0 = 1 / dist
    w = w0 * contribution
    num = torch.zeros_like(field).scatter_add(0, row, w)
    den = torch.zeros_like(field).scatter_add(0, row, w0)
    grad = num / (den + eps)
    grad[torch.isnan(grad)] = 0
    grad[torch.isinf(grad)] = 0
    return grad


def vorticity_feature(xyt, uv, edge_index):
    xy = xyt[:, 1:3]
    u = uv[:, 0]
    v = uv[:, 1]
    dvdx = weighted_graph_grad(v, xy, edge_index)
    dudy = weighted_graph_grad(u, xy, edge_index)
    return dvdx - dudy


def dist_feature(xyt, center, rad):
    xy = xyt[:, 1:3]
    dist = torch.linalg.norm(xy - center, dim=1)
    return dist - rad


# ==== DATA PREP ====
from openPy import DataSamplerOpenFoam

data_sampler = DataSamplerOpenFoam("cylinderFoam/velocity_*.dat")
print(f"Data shape: {data_sampler.data.shape}")

# Attenzione: normalizza SOLO x, y!
xynorm = GaussianNormalizer(data_sampler.data[:, 1:3])
data_sampler.data[:, 1:3] = xynorm.encode(data_sampler.data[:, 1:3])
uvnorm = GaussianNormalizer(data_sampler.data[:, -2:])
data_sampler.data[:, -2:] = uvnorm.encode(data_sampler.data[:, -2:])

sigma_x = xynorm.std[0].item()
rad = 3000
rad_scaled = rad / sigma_x
print(f"Scaled radius: {rad_scaled:.2f}")

X_batch, U_batch = data_sampler.sample(sample_dim)
X_batch = torch.tensor(X_batch, dtype=torch.float32, device=device)
U_batch = torch.tensor(U_batch, dtype=torch.float32, device=device)

xynorm.cuda()
uvnorm.cuda()

edge_index = knn_graph(X_batch[:, 1:], k=k_neighbors, batch=None, loop=False).to(device)
omega = vorticity_feature(X_batch, U_batch, edge_index)
dist = dist_feature(X_batch, cyl_center, rad_scaled)
print("omega min/max:", omega.min().item(), omega.max().item())

# Clamp values only if you find huge spikes in debugging!
omega_norm = (
    (GaussianNormalizer(omega.cpu().numpy()).encode(omega.cpu().numpy()))
    .float()
    .to(device)
)
dist_norm = (
    (GaussianNormalizer(dist.cpu().numpy()).encode(dist.cpu().numpy()))
    .float()
    .to(device)
)
print("diiiist", dist_norm)
print("diiist minmax", dist_norm.min().item(), dist_norm.max().item())
X_batch_feat = torch.cat([X_batch, dist_norm.unsqueeze(1)], dim=1)

print("X_batch_feat shape:", X_batch_feat.shape)


def create_graph_data(xyt, uv, edge_index):
    x = (
        xyt
        if torch.is_tensor(xyt)
        else torch.tensor(xyt, dtype=torch.float, device=device)
    )
    y = (
        uv
        if torch.is_tensor(uv)
        else torch.tensor(uv, dtype=torch.float, device=device)
    )
    return Data(x=x, edge_index=edge_index, y=y)


graph_data = create_graph_data(X_batch_feat, U_batch, edge_index)


# ==== TRAINING ====
def train():
    model = GNOModel(input_dim, hidden_channels, output_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=0.9
    )
    model.train()
    print("Training model...")
    loss_hist = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(graph_data)
        u_pred = uvnorm.decode(pred[:, 0], idx=0)
        v_pred = uvnorm.decode(pred[:, 1], idx=1)
        u_true = uvnorm.decode(graph_data.y[:, 0], idx=0)
        v_true = uvnorm.decode(graph_data.y[:, 1], idx=1)
        loss_data = F.mse_loss(u_pred, u_true) + F.mse_loss(v_pred, v_true)
        loss = loss_data
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | data_loss={loss_data.item():.3e} | total={loss.item():.3e}"
            )
            loss_hist.append(loss.item())
    torch.save(model.state_dict(), modelSavePath)
    np.savetxt("model/loss.npy", np.array(loss_hist))
    print("✅ Training complete, model saved.")


# ==== ANIMATION ====
def prediction_and_animate_tricontourf():
    cyl_radius = rad_scaled
    model = GNOModel(input_dim, hidden_channels, output_channels).to(device)
    model.load_state_dict(torch.load(modelSavePath, map_location=device))
    model.eval()
    xy = graph_data.x[:, 1:3].detach().cpu().numpy()
    if xynorm is not None:
        xy_torch = torch.from_numpy(xy).to(device)
        xdec = xynorm.decode(xy_torch[:, 0], idx=0).cpu().numpy()
        ydec = xynorm.decode(xy_torch[:, 1], idx=1).cpu().numpy()
    else:
        xdec, ydec = xy[:, 0], xy[:, 1]
    # Centro/raggio denormalizzati
    if xynorm is not None:
        cx_dec = 10000
        cy_dec = 10000
        r_cyl_dec = cyl_radius * xdec.std()
    else:
        cx_dec, cy_dec = cyl_center_np
        r_cyl_dec = cyl_radius
    mask_fluid = ((xdec - cx_dec) ** 2 + (ydec - cy_dec) ** 2) >= r_cyl_dec**2
    xdec_fluid = xdec[mask_fluid]
    ydec_fluid = ydec[mask_fluid]
    dist_fluid = dist_norm[mask_fluid]

    coords_tensor_fluid = torch.tensor(
        np.stack([xdec_fluid, ydec_fluid], axis=1), dtype=torch.float32, device=device
    )
    edge_index_fluid = knn_graph(
        coords_tensor_fluid, k=k_neighbors, batch=None, loop=False
    )
    t_samples = np.linspace(0, 1, num_frames)
    mag_vals = []
    with torch.no_grad():
        for t in t_samples:
            t_col = torch.full((coords_tensor_fluid.shape[0], 1), t, device=device)
            omega_col = torch.zeros((coords_tensor_fluid.shape[0], 1), device=device)
            dist_col = torch.zeros((coords_tensor_fluid.shape[0], 1), device=device)
            x_pred = torch.cat([t_col, coords_tensor_fluid, dist_norm], dim=1)
            data_pred = Data(x=x_pred, edge_index=edge_index_fluid).to(device)
            uvp = model(data_pred)
            u_pred = uvnorm.decode(uvp[:, 0], idx=0).cpu().numpy()
            v_pred = uvnorm.decode(uvp[:, 1], idx=1).cpu().numpy()
            mag = np.linalg.norm(np.stack([u_pred, v_pred], axis=-1), axis=1)
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
        linewidth=1.0,
        linestyle="--",
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


def plot_loss():
    loss = np.loadtxt("model/loss.npy")
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # train()
    prediction_and_animate_tricontourf()
    plot_loss()
