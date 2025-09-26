import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from ns_GNN_KF import (
    GaussianNormalizer,
    createGraphData,
    dataLoader,
    dataNormalizer,
    geometryObject,
)
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from aggr import LSTMAggregatorTorch
from tqdm import tqdm

"""I'm working here!!!!"""

# ==== Hyperparameters ====
num_layers = 5
hidden_dim = 30
latent_dim = 10
epochs = 3000
scheduler_step = 50
assign_dim = 500  # num_clusters for DiffPool
radius = 3000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def dense_to_sparse(x_dense, mask, batch):

    if batch is None:
        return x_dense.squeeze(0)
    out = []
    for b in range(x_dense.size(0)):
        nb = int(mask[b].sum().item())
        out.append(x_dense[b, :nb, :])

    return torch.cat(out, dim=0)


# ==== GNOLayer (PyG-style, pure PyTorch, with aggregators) ====
class GNOLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        edge_dim,
        hidden_dim,
        aggregator_type="mean",
        activation=F.relu,
        dropout=0.1,
        bn=True,
        bias=True,
    ):
        super(GNOLayer, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_features + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + out_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        self.use_bn = bn
        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        self.aggregator_type = aggregator_type
        if aggregator_type == "lstm":
            self.aggregator = LSTMAggregatorTorch(out_features, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        """Dimensioni:
        x: [num_nodes, in_features]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_dim]
        """
        x = self.dropout(x)
        src, dst = edge_index
        x_src = x[
            src
        ]  # [num_edges, in_features] src = source nodes, dst = destination nodes
        x_dst = x[dst]  # [num_edges, in_features]
        m_input = torch.cat([x_src, x_dst, edge_attr], dim=1)
        # m is the message tensor
        m = self.edge_mlp(m_input)  # [num_edges, out_features]

        # === Aggregation ===
        if (
            self.aggregator_type == "maxpool"
        ):  # better for detecting "extreme" behaviors, but can be noisy. It captures strong local effects
            agg = torch_scatter.scatter_max(m, dst, dim=0, dim_size=x.shape[0])[0]
        elif self.aggregator_type == "mean":
            agg = torch_scatter.scatter_mean(m, dst, dim=0, dim_size=x.shape[0])
        elif self.aggregator_type == "lstm":
            agg = self.aggregator(m, dst, x.shape[0])

        node_input = torch.cat([x, agg], dim=1)

        """ h_new applies a multi-layer perceptron to transform the concatenated
        features into the new node representation. This MLP learns how to effectively
        combine the node's original features with the aggregated neighborhood information,
        essentially determining what aspects of both sources are most important for the final representation."""

        h_new = self.node_mlp(node_input)
        if self.use_bn:
            h_new = self.bn(h_new)
        if self.activation:
            h_new = self.activation(h_new)
        return h_new
        # Output: [num_nodes, out_features]


class DiffPoolEncoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, latent_dim, edge_dim, num_layers, num_clusters
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gno_layers = nn.ModuleList(
            [
                GNOLayer(hidden_dim, hidden_dim, edge_dim, hidden_dim)
                for _ in range(num_layers)
            ]
        )

        # use dense GNN on dense batches
        self.gnn_embed = DenseSAGEConv(hidden_dim, hidden_dim)
        self.gnn_assign = DenseSAGEConv(hidden_dim, num_clusters)
        self.num_clusters = num_clusters
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        """
        data: PyG Data object with fields:
            - x: node features [num_nodes, input_dim]
            - edge_index: edge indices [2, num_edges] matrice di adiacenza
            - edge_attr: edge features [num_edges, edge_dim] edge features
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.embedding(x)
        for layer in self.gno_layers:
            x = layer(x, edge_index, edge_attr)

        batch = getattr(data, "batch", None)
        x_dense, mask = to_dense_batch(x, batch)
        adj_dense = to_dense_adj(edge_index, batch=batch)

        Z = self.gnn_embed(x_dense, adj_dense)
        S = F.softmax(self.gnn_assign(x_dense, adj_dense), dim=-1)

        x_pooled, adj_pooled, _, _ = dense_diff_pool(
            Z, adj_dense, S, mask
        )  # output: [batch, num_clusters, hidden_dim], for single graph, batch=1, so squeeze
        # x_pooled = x_pooled.squeeze(0)
        z_latent = self.to_latent(x_pooled)

        return adj_pooled, z_latent, S, mask


# ==== Decoder ====
# class GNNDecoder(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.layer1 = nn.Linear(latent_dim, hidden_dim)
#         self.layer2 = nn.Linear(hidden_dim, output_dim)


#     def forward(self, z):
#         h = F.gelu(self.layer1(z))
#         return self.layer2(h)
class DiffPoolDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, edge_dim, num_layers):
        super().__init__()
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.dec_layers = nn.ModuleList(
            [
                GNOLayer(hidden_dim, hidden_dim, edge_dim, hidden_dim)
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, z_latent, S, mask, data):
        Zc = F.gelu(self.from_latent(z_latent))

        # unpooling
        x0_dense = torch.bmm(S, Zc)  # [1, num_nodes, hidden_dim]
        x0_sparse = dense_to_sparse(x0_dense, mask, getattr(data, "batch", None))

        for layer in self.dec_layers:
            x0_sparse = layer(x0_sparse, data.edge_index, data.edge_attr)

        y_pred = self.head(x0_sparse)
        return y_pred


class GNNAutoencoder2(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        output_dim,
        edge_dim,
        num_layers,
        num_clusters,
    ):
        super().__init__()
        self.encoder = DiffPoolEncoder(
            input_dim, hidden_dim, latent_dim, edge_dim, num_layers, num_clusters
        )
        self.decoder = DiffPoolDecoder(
            latent_dim, hidden_dim, output_dim, edge_dim, num_layers
        )

    def forward(self, data):
        """
        data: PyG Data object with fields:
            - x: node features [num_nodes, input_dim]
            - edge_index: edge indices [2, num_edges] matrice di adiacenza
            - edge_attr: edge features [num_edges, edge_dim] edge features
        """
        adj_pooled, z_latent, S, mask = self.encoder(data)
        reconstructed = self.decoder(z_latent, S, mask, data)

        return reconstructed, (adj_pooled, z_latent, S, mask)


# ==== Training Loop ====
def train(data, latent_dim=latent_dim, epochs=epochs):
    print("Starting training...")
    input_dim = data.x.shape[1]
    edge_dim = data.edge_attr.shape[1]
    # output_dim = data.y.shape[1]
    output_dim = 3  # Assuming u, v, p

    model = GNNAutoencoder2(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        edge_dim=edge_dim,
        num_layers=num_layers,
        num_clusters=assign_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=0.9
    )
    data = data.to(device)
    loss_history = []
    loop = tqdm(range(epochs), desc="Training Progress")

    for epoch in loop:
        optimizer.zero_grad()
        pred, _ = model(data)  # shape: [num_clusters, output_dim]
        # You need to map data.y to clusters if you want strict nodewise loss,
        # here just compare first k nodes for demonstration.
        # (You probably want to aggregate or interpolate targets in practice.)
        target = data.y[: pred.shape[0]]
        # loss = F.mse_loss(pred, target)
        loss = F.mse_loss(pred[:, :2], target)

        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            loss_history.append(loss.item())
    print(" ✅ Training complete.")

    torch.save(model.state_dict(), "model/gnn_autoencoder.pth")
    np.save("model/loss_history.npy", np.array(loss_history))
    print(" ✅ Model and loss history saved.")


if __name__ == "__main__":
    # Your createGraphData() must return a PyG Data object with .x, .edge_index, .edge_attr, .y
    data = createGraphData()
    train(data, latent_dim=latent_dim, epochs=epochs)
    print("Training finished and model saved.")
