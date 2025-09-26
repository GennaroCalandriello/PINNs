import matplotlib.pyplot as plt

# from ns_GNN_KF import geometryObject
from temp import *


def geometryObject(xy, center, radius):
    """Create a geometric object (circle) for the given parameters."""
    cx, cy = center
    dist = torch.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    circle = dist < radius
    dist_norm = dist / radius
    return dist_norm.unsqueeze(1), circle.unsqueeze(1)


def plotFields():
    data = createGraphData()  # Adjust time_idx as needed
    input_dim = data.x.shape[1]
    output_dim = 3  # Assuming u, v, p
    # output_dim = data.y.shape[1]
    edge_dim = data.edge_attr.shape[1]
    # Load model
    # model = GNNAutoencoder(input_dim, hidden_dim, latent_dim, output_dim, edge_dim).to(
    #     device
    # )
    model = GNNAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        edge_dim=edge_dim,
        num_layers=num_layers,
        num_clusters=assign_dim,
    ).to(device)
    model.load_state_dict(torch.load("model/gnn_autoencoder.pth"))
    model.eval()

    # For a given time index t_idx
    t_idx = 10  # for example
    # Build a Data object for this time slice (you may need to adapt createGraphData to take a time index)
    data_t = data  # your Data object for time t_idx
    data_t = data_t.to(device)

    with torch.no_grad():
        pred = model(data_t)
        u_pred, v_pred = pred[:, 0].cpu().numpy(), pred[:, 1].cpu().numpy()
        centers = data_t.x[:, :2].cpu().numpy()  # first two columns assumed to be x, y

    plt.figure(figsize=(8, 6))
    plt.quiver(centers[:, 0], centers[:, 1], u_pred, v_pred)
    plt.title(f"Predicted Velocity Field at t={t_idx}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def animate_patch_time_series_gnn(
    out_gif="velocity_rom.gif",
    model_path="model/gnn_autoencoder.pth",
):
    import matplotlib.tri as mtri
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation
    import torch

    results, _ = dataLoader()
    t0, centers, U, neighbors, edge_index, xynorm, uvnorm = dataNormalizer(results)
    data = createGraphData()

    sigma_x = xynorm.std[0].item()
    rad_scaled = radius / sigma_x
    edge_dim = data.edge_attr.shape[1]

    # --- Set up device
    device = xynorm.mean.device if hasattr(xynorm, "mean") else torch.device("cpu")

    # --- Initial geometry (for triangulation & masking) ---
    t0, centers0, U0, neighbors0, edge_index0 = results[0]
    centers0 = torch.tensor(centers0, dtype=torch.float32, device=device)
    centers0_dec = centers0.cpu().numpy()
    xdec, ydec = centers0_dec[:, 0], centers0_dec[:, 1]

    # Mask (fluid region)
    cx, cy = 10000, 10000
    r_cyl_dec = 3000
    mask_fluid = ((xdec - cx) ** 2 + (ydec - cy) ** 2) >= r_cyl_dec**2
    xdec_fluid = xdec[mask_fluid]
    ydec_fluid = ydec[mask_fluid]

    triang = mtri.Triangulation(xdec_fluid, ydec_fluid)
    tri_points = np.stack(
        [xdec_fluid[triang.triangles], ydec_fluid[triang.triangles]], axis=-1
    )
    tri_centers_x = tri_points[:, :, 0].mean(axis=1)
    tri_centers_y = tri_points[:, :, 1].mean(axis=1)
    dist2 = (tri_centers_x - cx) ** 2 + (tri_centers_y - cy) ** 2
    mask_tri = dist2 < r_cyl_dec**2
    triang.set_mask(mask_tri)

    # --- Load model ---
    input_dim = data.x.shape[1]
    output_dim = 3
    model = GNNAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        edge_dim=edge_dim,
        num_layers=num_layers,
        num_clusters=assign_dim,
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # --- Collect velocity magnitude for all frames ---
    mag_vals = []
    for t_idx, (t, centers, U, neighbors, edge_index) in enumerate(results):
        # Build data_t using your normalizer pipeline (this matches your createGraphData_t logic)
        centers = torch.tensor(centers, dtype=torch.float32, device=device)
        U = torch.tensor(U, dtype=torch.float32, device=device)
        centers_norm = xynorm.encode(centers)
        U_norm = uvnorm.encode(U)
        dist_norm, circle = geometryObject(centers_norm, (0, 0), rad_scaled)
        x = torch.cat([centers_norm, U_norm, dist_norm, circle], dim=1)
        y = U_norm
        # Edges
        edge_src, edge_dst = [], []
        for i, nbs in enumerate(neighbors):
            for j in nbs:
                edge_src.append(i)
                edge_dst.append(j)
        edge_src = torch.tensor(edge_src, dtype=torch.long, device=device)
        edge_dst = torch.tensor(edge_dst, dtype=torch.long, device=device)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        relative_positions = centers_norm[edge_dst] - centers_norm[edge_src]
        distances = torch.norm(relative_positions, dim=1, keepdim=True)
        edge_attr = torch.cat([relative_positions, distances], dim=1)
        # Data object
        from torch_geometric.data import Data

        data_t = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y).to(device)
        # Predict
        with torch.no_grad():
            # **RICOSTRUZIONE SULLO SPAZIO COMPLETO DEI NODI!**
            uvp_full = model.reconstruct_full(data_t)
            u_pred = uvnorm.decode(uvp_full[:, 0], idx=0).cpu().numpy()
            v_pred = uvnorm.decode(uvp_full[:, 1], idx=1).cpu().numpy()
            mag = np.linalg.norm(np.stack([u_pred, v_pred], axis=-1), axis=1)
            mag_vals.append(mag[mask_fluid])

    mag_vals = np.array(mag_vals)
    vmin, vmax = mag_vals.min(), mag_vals.max()

    # --- Animation setup ---
    import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    plotFields()
    animate_patch_time_series_gnn()
