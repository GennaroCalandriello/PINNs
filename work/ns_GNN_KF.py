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
num_layers = 4
hidden_dim = 50
latent_dim = 10
epochs = 3000
scheduler_step = 500
assign_dim = 5  # num_clusters for DiffPool
path_data = "patches/patch_max_uniformCyl.pkl"  # Update with actual path
# path_data = "patches/patch_5k_uniform.pkl"
radius = 3000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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

    t0, centers, U, neighbours, edge_index = results[0]
    print(f"Data shape: {U.shape}")
    xynorm = GaussianNormalizer(centers)
    uvnorm = GaussianNormalizer(U)
    xynorm.cuda()
    uvnorm.cuda()
    print("Data normalized successfully.")

    return t0, centers, U, neighbours, edge_index, xynorm, uvnorm


def geometryObject(xy, center, radius):
    """Create a geometric object (circle) for the given parameters."""
    cx, cy = center
    dist = torch.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    circle = dist < radius
    dist_norm = dist / radius
    return dist_norm.unsqueeze(1), circle.unsqueeze(1)


def createGraphData():
    """Create a PyTorch Geometric Data object from the normalized data."""
    results, idx_cells = dataLoader(path_data)
    t0, centers, U, neighbours, edge_index, xynorm, uvnorm = dataNormalizer(results)
    U_norm = uvnorm.encode(
        torch.tensor(U, dtype=torch.float32, device=uvnorm.mean.device)
    )
    centers_norm = xynorm.encode(
        torch.tensor(centers, dtype=torch.float32, device=xynorm.mean.device)
    )

    centers_norm = torch.as_tensor(centers_norm, dtype=torch.float32, device=device)
    U_norm = torch.as_tensor(U_norm, dtype=torch.float32, device=device)

    sigma_x = xynorm.std[0].item()
    sigma_y = xynorm.std[1].item()
    print(f"Sigma X: {sigma_x}, Sigma Y: {sigma_y}")
    rad_scaled = radius / (sigma_x)
    print(f"Scaled radius: {rad_scaled}")

    feature_list = [centers_norm, U_norm]
    dist_norm, circle = geometryObject(centers_norm, (0, 0), rad_scaled)
    feature_list.append(dist_norm)
    feature_list.append(circle)

    x = torch.cat(feature_list, dim=1)
    y = U_norm

    edge_src, edge_dist = [], []
    for i, nbs in enumerate(neighbours):
        for j in nbs:
            edge_src.append(i)
            edge_dist.append(j)

    edge_src = torch.tensor(edge_src, dtype=torch.long, device=device)
    edge_dist = torch.tensor(edge_dist, dtype=torch.long, device=device)
    edge_index = torch.stack([edge_src, edge_dist], dim=0)
    relative_positions = (
        centers_norm[edge_dist] - centers_norm[edge_src]
    )  # [num edges, 2]
    distances = torch.norm(relative_positions, dim=1, keepdim=True)  # [num edges, 1]
    edge_attr = torch.cat([relative_positions, distances], dim=1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


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
            z_filtered = torch.tensor(z_filtered, dtype=torch.float32, device=z.device)
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
            nn.Dropout(0.1),
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


def train(data, latent_dim=latent_dim, epochs=epochs):
    """Train the GNN autoencoder."""
    print("Starting training...")
    results, _ = dataLoader(path_data)
    t0, centers, U, neighbours, edge_index, xynorm, uvnorm = dataNormalizer(results)

    input_dim = data.x.shape[1]
    edge_dim = data.edge_attr.shape[1]
    output_dim = 3  # u, v, p

    model = GNNAutoencoder(input_dim, hidden_dim, latent_dim, output_dim, edge_dim).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=0.9
    )

    data = data.to(device)
    loss_history = []

    for epoch in range(epochs):
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

        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            loss_history.append(loss.item())
    print(" ✅ Training complete.")

    torch.save(model.state_dict(), "model/gnn_autoencoder.pth")
    np.save("model/loss_history.npy", np.array(loss_history))
    print(" ✅ Model and loss history saved.")


if __name__ == "__main__":
    data = createGraphData()
    train(data, latent_dim=latent_dim, epochs=epochs)
    print("Training finished and model saved.")
