import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# ===== 2. Parameters =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
hidden_channels = 180
hidden_dim_gno_layer = 40
output_channels = 3  # u, v, p
k_neighbors = 12
epochs = 4000
sample_dim = 10000
scheduler_step = 1000
nu = 1.5e-5
sigma_kernel = 0.2
os.makedirs("model", exist_ok=True)
modelSavePath = "model/gno_modelGELU3.pth"
lambda_phys = 10.0
N_data_batch = 2048
N_obs_batch = 1000
lambda_obs = 10.0
num_layers = 4
input_dim = 5  #
# sigma = 0.01


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


# ===== 3. GNO Model (con LayerNorm, GELU, Dropout) =====
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


def estimate_sigma(x, edge_index):
    # x: (N, 2), edge_index: (2, E)
    row, col = edge_index
    dist = torch.sqrt(((x[row] - x[col]) ** 2).sum(dim=1))
    return dist.mean().item()


# ==== GRAPH GRADIENTS ====
def weighted_graph_grad(field, x, edge_index, eps=0):
    row, col = edge_index[0], edge_index[1]
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
    if grad is None:
        print("Warning: grad is None, returning zeros")
    grad[torch.isnan(grad)] = 0
    grad[torch.isinf(grad)] = 0
    return grad


def simple_graph_grad(field, x, edge_index, eps=0):
    """
    Calcola il gradiente "non pesato" di un campo scalare sul grafo.
    field: (N,)  campo scalare su ogni nodo
    x: (N, 2)    coordinate spaziali [x, y] (non tutte le feature!)
    edge_index: (2, E)
    """
    row, col = edge_index[0], edge_index[1]  # edge da row → col

    dx = x[col, 0] - x[row, 0]
    dy = x[col, 1] - x[row, 1]
    dist = torch.sqrt(dx**2 + dy**2) + eps

    dfield = field[col] - field[row]

    # grad_x: somma differenze normalizzate per ogni vicino (no pesi)
    grad_x = torch.zeros_like(field).scatter_add(0, row, dfield * dx / (dist**2 + eps))
    grad_y = torch.zeros_like(field).scatter_add(0, row, dfield * dy / (dist**2 + eps))

    # Contiamo il numero di contributi per ogni nodo per fare la media
    count = torch.zeros_like(field).scatter_add(0, row, torch.ones_like(dist))

    grad_x = grad_x / (count)
    grad_y = grad_y / (count)

    grad_x[torch.isnan(grad_x)] = 0
    grad_y[torch.isnan(grad_y)] = 0
    grad_x[torch.isinf(grad_x)] = 0
    grad_y[torch.isinf(grad_y)] = 0

    return grad_x, grad_y


def rbf_weights(x, edge_index, sigma):
    """
    Calcola i pesi RBF tra i nodi x e il centro x0.
    x: (N, 2)  coordinate spaziali dei nodi
    x0: (2,)   centro della funzione RBF
    sigma: float, deviazione standard della RBF
    """
    xi = x[edge_index[0]]  # coordinate del nodo di partenza
    xj = x[edge_index[1]]  # coordinate del nodo di destinazione
    r2 = torch.sum((xi - xj) ** 2, dim=1)
    w = torch.exp(-r2 / (2 * sigma**2))
    return w


def vorticity_feature(xyt, uv, edge_index):
    xy = xyt[:, 1:3]
    u = uv[:, 0]
    v = uv[:, 1]
    v = torch.tensor(v, device=device)
    u = torch.tensor(u, device=device)
    # dvdx = weighted_graph_grad(v, xy, edge_index)
    # dudy = weighted_graph_grad(u, xy, edge_index)
    dvdx, _ = simple_graph_grad(v, xy, edge_index)
    _, dudy = simple_graph_grad(u, xy, edge_index)
    return dvdx - dudy


def graph_div(u, v, x, edge_index, sigma, eps=1e-8):
    w = rbf_weights(x, edge_index, sigma=sigma)
    i, j = edge_index[0], edge_index[1]
    du = u[j] - u[i]
    dv = v[j] - v[i]
    dx = x[j, 0] - x[i, 0]
    dy = x[j, 1] - x[i, 1]
    dist2 = dx**2 + dy**2 + eps
    contrib = (du * dx + dv * dy) / dist2
    num = torch.zeros_like(u).scatter_add(0, i, w * contrib)
    den = torch.zeros_like(u).scatter_add(0, i, w)
    div = num / (den)
    div[torch.isnan(div)] = 0
    div[torch.isinf(div)] = 0
    return div


def graph_grad(p, x, edge_index, sigma, eps=1e-8):
    """
    Calcola il gradiente del campo scalare p su un grafo.
    """
    w = rbf_weights(x, edge_index, sigma=sigma)
    i, j = edge_index[0], edge_index[1]
    dp = p[j] - p[i]
    dx = x[j, 0] - x[i, 0]
    dy = x[j, 1] - x[i, 1]
    dist2 = dx**2 + dy**2 + eps

    grad_x = torch.zeros_like(p)
    grad_y = torch.zeros_like(p)
    grad_x = grad_x.scatter_add(0, i, w * dp * dx / dist2)
    grad_y = grad_y.scatter_add(0, i, w * dp * dy / dist2)
    den = torch.zeros_like(p).scatter_add(0, i, w)
    grad_x = grad_x / (den)
    grad_y = grad_y / (den)
    grad_x[torch.isnan(grad_x)] = 0
    grad_y[torch.isnan(grad_y)] = 0
    grad_x[torch.isinf(grad_x)] = 0
    grad_y[torch.isinf(grad_y)] = 0
    return grad_x, grad_y


def graph_laplacian(u, x, edge_index, sigma):
    """
    Calcola il laplaciano del campo scalare u su un grafo.
    u: (N,)  campo scalare su ogni nodo
    x: (N, 2)  coordinate spaziali [x, y] (non tutte le feature!)
    edge_index: (2, E)
    sigma: float, deviazione standard della RBF
    """
    w = rbf_weights(x, edge_index, sigma=sigma)
    i, j = edge_index[0], edge_index[1]
    N = u.size(0)
    num = torch.zeros(N, device=u.device)
    den = torch.zeros(N, device=u.device)
    num = num.scatter_add(0, i, w * (u[j] - u[i]))
    den = den.scatter_add(0, i, w)
    laplacian = num / (
        den
    )  # Aggiungi un piccolo epsilon per evitare divisione per zero
    return laplacian


def ddt(u, t):
    """Calcola la derivata temporale senza AUTOGRAD, con differenze finite"""
    u = torch.as_tensor(u, dtype=torch.float32, device=device)
    dt = t[1] - t[0]
    du_dt = (u[1:] - u[:-1]) / dt
    du_dt = torch.cat([du_dt, du_dt[-1:]], dim=0)  # Mantieni la stessa lunghezza
    return du_dt


# ===== 4. Data Preparation =====
# data_sampler = 0
# xynorm = 0
# uvnorm = 0
# sigma_x = 0
# rad = 3000
# rad_scaled = 0
# X_batch, U_batch = 0, 0


def physics_loss(u_pred, v_pred, p_pred, xyt, edge_index, sigma, nu=nu):
    t = xyt[:, 0]
    du_dt = ddt(u_pred, t)
    dv_dt = ddt(v_pred, t)

    lap_u = graph_laplacian(u_pred, xyt[:, 1:3], edge_index, sigma=sigma)
    lap_v = graph_laplacian(v_pred, xyt[:, 1:3], edge_index, sigma=sigma)

    px, py = graph_grad(p_pred, xyt[:, 1:3], edge_index, sigma=sigma)
    div_uv = graph_div(u_pred, v_pred, xyt[:, 1:3], edge_index, sigma=sigma)

    ux = graph_grad(u_pred, xyt[:, 1:3], edge_index, sigma=sigma)[0]
    uy = graph_grad(u_pred, xyt[:, 1:3], edge_index, sigma=sigma)[1]
    vx = graph_grad(v_pred, xyt[:, 1:3], edge_index, sigma=sigma)[0]
    vy = graph_grad(v_pred, xyt[:, 1:3], edge_index, sigma=sigma)[1]
    # print("ux", ux)
    res_u = du_dt + u_pred * ux + v_pred * uy - nu * lap_u + px
    res_v = dv_dt + u_pred * vx + v_pred * vy - nu * lap_v + py
    res_div = div_uv
    loss = (
        F.mse_loss(res_u, torch.zeros_like(res_u))
        + F.mse_loss(res_v, torch.zeros_like(res_v))
        + F.mse_loss(res_div, torch.zeros_like(res_div))
    )
    return loss


def compute_cyl_features(xy, center=(0.0, 0.0), radius=1.0):
    cx, cy = center
    dist = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    inside = (dist < radius).astype(np.float32)
    dist_norm = dist / radius
    return dist_norm[:, None], inside[:, None]


def create_graph_data(xyt, uv, cyl_center=(0.0, 0.0), cyl_radius=0.51):
    xy = xyt[:, 1:3]

    # Calcola feature geometriche
    dist_norm, inside = compute_cyl_features(xy, center=cyl_center, radius=cyl_radius)

    # Costruisci il tensore iniziale delle feature
    features_np = np.concatenate([xyt, dist_norm, inside], axis=1)  # shape (N, 5)
    features = torch.tensor(features_np, dtype=torch.float, device=device)
    y = torch.tensor(uv, dtype=torch.float, device=device)

    # edge_index SOLO su (x, y)
    edge_index = knn_graph(features[:, 1:3], k=k_neighbors, batch=None, loop=False).to(
        device
    )

    # Calcola la vorticità (occhio: passa xyt e uv come tensori)
    xyt_torch = torch.tensor(xyt, dtype=torch.float, device=device)
    # omega = vorticity_feature(xyt_torch, y, edge_index)  # shape (N,)

    # # Aggiungi la vorticità come feature
    # features = torch.cat([features, omega.unsqueeze(1)], dim=1)  # shape (N, 6)

    # # (facoltativo) ricostruisci edge_index su (x, y) anche dopo aver aggiunto la nuova feature
    # edge_index = knn_graph(features[:, 1:3], k=k_neighbors, batch=None, loop=False).to(
    #     device
    # )

    return Data(x=features, edge_index=edge_index, y=y)


def subgraph_by_mask(u, x, edge_index, mask):
    idx = torch.where(mask)[0]
    idx_map = -torch.ones(mask.shape[0], dtype=torch.long, device=mask.device)
    idx_map[idx] = torch.arange(idx.size(0), device=mask.device)
    edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
    edge_index_sub = edge_index[:, edge_mask]
    edge_index_sub = idx_map[edge_index_sub]
    return u[mask], x[mask], edge_index_sub


from openPy import DataSamplerOpenFoam

print("Loading data...")
data_sampler = DataSamplerOpenFoam("cylinderFoam/velocity_*.dat")
print(f"Data shape: {data_sampler.data.shape}")
xynorm = GaussianNormalizer(data_sampler.data[:, 1:])
data_sampler.data[:, 1:] = xynorm.encode(data_sampler.data[:, 1:])
uvnorm = GaussianNormalizer(data_sampler.data[:, -2:])  # Solo u, v
data_sampler.data[:, -2:] = uvnorm.encode(data_sampler.data[:, -2:])

sigma_x = xynorm.std[0].item()
rad = 3000
rad_scaled = rad / sigma_x
print(f"Scaled radius: {rad_scaled:.2f}")

X_batch, U_batch = data_sampler.sample(sample_dim)
xynorm.cuda()
uvnorm.cuda()
graph_data = create_graph_data(X_batch, U_batch, cyl_center=(0.0, 0.0))

sigma = estimate_sigma(graph_data.x[:, 1:3], graph_data.edge_index)
print("Estimated sigma:", sigma)


# (resta identico tutto il codice delle funzioni loss, train, pred, ecc...)

# ===== 5. Physics Loss: u,v dati, p solo fisica =====
# (Funzioni rbf_weights, graph_laplacian, graph_gradient, graph_divergence, time_derivative, loss_physics_ns... come sopra)


# ===== 6. TRAINING (supervisiona solo u,v) =====
def train(data, epochs=epochs):
    loss_data = 0
    cyl_center = torch.tensor([0.0, 0.0], device=device)
    cyl_radius = rad_scaled
    model = GNOModel(input_dim, hidden_channels, output_channels).to(
        device
    )  # input_dim=4
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

    idx_obs = torch.nonzero(mask_obs, as_tuple=False).squeeze()
    idx_data = torch.nonzero(mask_fluid, as_tuple=False).squeeze()

    for epoch in range(epochs):
        optimizer.zero_grad()

        pred = model(data)
        p_pred = pred[:, 2]

        # u_pred = uvnorm.decode(pred[:, 0], idx=0)
        # v_pred = uvnorm.decode(pred[:, 1], idx=1)
        # u_true = uvnorm.decode(data.y[:, 0], idx=0)
        # v_true = uvnorm.decode(data.y[:, 1], idx=1)
        u_pred = pred[:, 0]
        v_pred = pred[:, 1]
        u_true = data.y[:, 0]
        v_true = data.y[:, 1]
        # add physics loss
        loss_phys = physics_loss(
            u_pred, v_pred, p_pred, data.x, data.edge_index, sigma=sigma, nu=nu
        )
        loss_data = F.mse_loss(u_pred, u_true) + F.mse_loss(v_pred, v_true)
        # print("Loss physics:", loss_phys.item())
        if epoch < 0:
            loss = loss_data
            loss_phys = torch.tensor(
                0.0, device=device
            )  # No loss fisica nei primi 500 epoch
        else:
            loss = (1 - 0.2) * loss_data + loss_phys * 0.2

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | data_loss={loss_data.item():.3e} | phys_loss = {loss_phys.item():.3e} | total={loss.item():.3e}"
            )
            loss_hist.append(loss.item())

    torch.save(model.state_dict(), modelSavePath)
    np.savetxt("model/loss.npy", np.array(loss_hist))
    print("✅ Training complete, model saved.")


# ===== 7. Prediction & Plotting (u,v,p) =====
def prediction_and_animate_tricontourf(
    graph_data, num_frames=100, xynorm=None, cyl_center=(0.0, 0.0), cyl_radius=0.51
):
    import matplotlib.tri as mtri
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation

    model = GNOModel(input_dim, hidden_channels, output_channels).to(device)
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

    # Prepara le feature "distanza normalizzata" e "inside" anche in predizione
    xy_fluid = np.stack([xdec_fluid, ydec_fluid], axis=1)
    dist_norm_fluid, inside_fluid = compute_cyl_features(
        xy_fluid, center=(10000, 10000), radius=3000
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
            )  # shape (N,5)
            # knn SOLO su (x, y)
            edge_index_pred = knn_graph(
                x_pred[:, 1:3], k=k_neighbors, batch=None, loop=False
            ).to(device)
            data_pred = Data(x=x_pred, edge_index=edge_index_pred).to(device)
            # omega_pred = vorticity_feature(
            #     x_pred[:, :3], data_pred.x[:, 3:5], edge_index_pred
            # )
            # data_pred.x = torch.cat([data_pred.x, omega_pred.unsqueeze(1)], dim=1)
            # edge_index_pred = knn_graph(
            #     data_pred.x[:, 1:3], k=k_neighbors, batch=None, loop=False
            # ).to(device)
            # data_pred.edge_index = edge_index_pred

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
    ani.save("cylinder_flow2denorm.gif", writer="imagemagick", fps=20)
    # plt.show()
    return ani


# ===== 8. Plot Loss =====
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
    train(graph_data)
    prediction_and_animate_tricontourf(
        graph_data,
        num_frames=200,
        xynorm=xynorm,
        cyl_center=(0.0, 0.0),
        cyl_radius=rad_scaled,
    )
    plot_loss()
