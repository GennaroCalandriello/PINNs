from ns_GNN_KF import *
import torch
import matplotlib.pyplot as plt
import numpy as np


def get_latent_trajectory(model, results, data_static, uvnorm, device="cpu"):
    """
    This plot shows how the spatially-averaged latent variables (the encoded "modes") evolve over time.
    - Oscillations or patterns: may correspond to physical dynamics (e.g., vortex shedding, periodicity).
    - Flat or near-zero curves: indicate unused or redundant latent dimensions.
    - Similar curves: suggest redundancy in the latent space.
    - Smooth, well-separated curves: mean the encoder captures distinct, meaningful features.
    - Abrupt spikes: may signal anomalies, transitions, or bugs.
    Use this plot to:
    - Check if latent_dim is well chosen (reduce if many are flat).
    - Verify that the latent space evolves smoothly and tracks system dynamics.
    - Diagnose model or data issues before applying data assimilation.
    """

    # Clone original static features
    orig_x = data_static.x.clone()
    latents = []
    model.eval()
    with torch.no_grad():
        for t_idx in range(len(results)):
            # Get normalized velocity for this time step
            U_t = torch.tensor(
                results[t_idx][2], dtype=torch.float32, device=orig_x.device
            )
            U_norm = uvnorm.encode(U_t)
            # Update only u, v in node features (assume columns 2, 3)
            x_t = orig_x.clone()
            x_t[:, 2:4] = U_norm
            data_static.x = x_t
            # Pass to encoder
            z = model.encoder(data_static)  # [num_nodes, latent_dim]
            latents.append(z.cpu().numpy())
    return np.array(latents)  # [num_times, num_nodes, latent_dim]


def modes():
    # --- Load data and model ---
    results, idx_cells = dataLoader()
    t0, centers, U, neighbors, edge_index, xynorm, uvnorm = dataNormalizer(results)
    # Build the static graph once
    data_static = createGraphData()
    input_dim = data_static.x.shape[1]
    output_dim = 3  # [u, v, p]
    edge_dim = data_static.edge_attr.shape[1]
    model = GNNAutoencoder(
        input_dim, hidden_dim, latent_dim, output_dim, edge_dim, num_layers
    )
    model.load_state_dict(torch.load("model/gnn_autoencoder.pth"))
    model = model.to(data_static.x.device)

    # --- Extract latent trajectory efficiently ---
    Z_traj = get_latent_trajectory(
        model, results, data_static, uvnorm, device=data_static.x.device
    )
    # Global mode: mean over nodes
    z_global_traj = Z_traj.mean(axis=1)  # [num_times, latent_dim]
    return z_global_traj


# --- Visualization ---
Z = modes()
print(f"Latent trajectory shape: {Z.shape[1]}")  # [num_times, latent_dim]
plt.figure(figsize=(10, 5))
for i in range(min(Z.shape[1], 6)):
    plt.plot(Z[:, i], label=f"Latent dim {i}")
plt.xlabel("Time step")
plt.ylabel("Latent value")
plt.title("Evolution of Latent Dimensions")
plt.legend()
plt.show()
