import matplotlib.pyplot as plt
from workinghere import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def animate_patch_time_series_gnn(
    out_gif="mehmeh.gif",
    model_path="model/scalable_gnn_ae.pt",
):
    import matplotlib.tri as mtri
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation
    import torch
    import numpy as np
    from torch_geometric.data import Data

    radius = 3000

    # ===== Load & normalize data =====
    results, _ = dataLoader()
    t0, centers, U, neighbors, edge_index, xynorm, uvnorm = dataNormalizer(results)
    data = createGraphData()

    # --- scale raggio con le statistiche di xynorm (CPU va bene qui) ---
    sigma_x = xynorm.std[0].item()
    rad_scaled = radius / sigma_x

    in_ch = data.x.size(-1)
    edge_dim = data.edge_attr.size(-1)
    out_ch = data.y.size(-1) if hasattr(data, "y") else 3

    # ===== Model on chosen device =====
    model = SparseUNetAutoEncoder(in_ch, edge_dim, out_ch=out_ch).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ===== Geometry (CPU / numpy) =====
    t0, centers0, U0, neighbors0, edge_index0 = results[0]
    centers0 = torch.tensor(centers0, dtype=torch.float32)  # CPU
    centers0_np = centers0.numpy()
    xdec, ydec = centers0_np[:, 0], centers0_np[:, 1]

    # Mask (fluid region) in numpy
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

    # ===== Collect velocity magnitudes for all frames =====
    mag_vals = []
    with torch.inference_mode():
        for t_idx, (t, centers, U, neighbors, edge_index) in enumerate(results):
            # Tensors on the SAME device as the model
            centers_t = torch.tensor(centers, dtype=torch.float32, device=device)
            U_t = torch.tensor(U, dtype=torch.float32, device=device)

            # Encode on device (if your normalizers expect torch tensors they’ll follow the device of inputs)
            centers_norm = xynorm.encode(centers_t)
            U_norm = uvnorm.encode(U_t)

            # Geometry features (stay on device)
            dist_norm, circle = geometryObject(centers_norm, (0, 0), rad_scaled)

            # Node features x and target y
            x = torch.cat([centers_norm, U_norm, dist_norm, circle], dim=1)
            y = U_norm

            # Build edges on device
            edge_src, edge_dst = [], []
            for i, nbs in enumerate(neighbors):
                for j in nbs:
                    edge_src.append(i)
                    edge_dst.append(j)
            edge_src = torch.tensor(edge_src, dtype=torch.long, device=device)
            edge_dst = torch.tensor(edge_dst, dtype=torch.long, device=device)
            edge_index_t = torch.stack([edge_src, edge_dst], dim=0)

            relative_positions = centers_norm[edge_dst] - centers_norm[edge_src]
            distances = torch.norm(relative_positions, dim=1, keepdim=True)
            edge_attr = torch.cat([relative_positions, distances], dim=1)

            data_t = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr, y=y).to(
                device
            )

            # Predict on device
            uvp = model(data_t)

            # Decode on CPU to avoid any hidden cpu<->cuda mismatch inside normalizers
            u_pred = uvnorm.decode(uvp[:, 0].detach().cpu(), idx=0).numpy()
            v_pred = uvnorm.decode(uvp[:, 1].detach().cpu(), idx=1).numpy()
            mag = np.linalg.norm(np.stack([u_pred, v_pred], axis=-1), axis=1)

            # apply mask (same as triang nodes)
            mag_vals.append(mag[mask_fluid])

    mag_vals = np.array(mag_vals)
    vmin, vmax = mag_vals.min(), mag_vals.max()

    # ===== Animation setup =====
    fig, ax = plt.subplots(figsize=(7, 6))
    cntr = ax.tricontourf(
        triang, mag_vals[0], levels=600, cmap="jet", vmin=vmin, vmax=vmax
    )
    cb = fig.colorbar(cntr, ax=ax, label="|u,v|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title("t = 0.00")
    ax.axis("equal")
    circle_patch = Circle(
        (cx, cy), r_cyl_dec, color="k", fill=False, linewidth=0.8, zorder=10
    )
    ax.add_patch(circle_patch)

    def update(frame):
        # clear previous contourf collections (keep colorbar)
        for coll in [c for c in ax.collections if isinstance(c, plt.PolyCollection)]:
            coll.remove()
        cntr = ax.tricontourf(
            triang, mag_vals[frame], levels=600, cmap="jet", vmin=vmin, vmax=vmax
        )
        t_now = results[frame][0]
        title.set_text(f"t = {t_now:.3f}")
        return []

    ani = FuncAnimation(
        fig, update, frames=len(results), interval=60, blit=False, repeat=True
    )
    ani.save(out_gif, writer="imagemagick", fps=20)
    print("✅ Animation saved as", out_gif)
    return ani


if __name__ == "__main__":
    animate_patch_time_series_gnn()
