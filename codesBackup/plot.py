import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# from ns_GNN_KF import *
# from ns_gnn_diffpool import *
# from correct import *
# from correct import *
from EdgeNodeAttentionDiffPool import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

geometry = "rectangle"  # "circle" or "rectangle"


def animate_patch_time_series_gnn(
    out_gif="mehmeh.gif",
    model_path="model/gnn_ae.pth",
):
    radius = 3000
    import matplotlib.tri as mtri
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation
    import torch

    results, _ = dataLoader()
    t0, centers, U, neighbors, edge_index, xynorm, uvnorm = dataNormalizer(results)
    data = createGraphData()

    sigma_x = xynorm.std[0].item()
    rad_scaled = radius / sigma_x
    in_ch = data.x.size(-1)
    edge_dim = data.edge_attr.size(-1)
    out_ch = data.y.size(-1)

    model = GraphAutoEncoder(
        in_ch=in_ch,
        edge_dim=edge_dim,
        out_ch=out_ch,
        clusters_per_level=CLUSTERS_PER_LEVEL,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # --- Normalize data ---

    # --- Initial geometry (for triangulation & masking) ---
    t0, centers0, U0, neighbors0, edge_index0 = results[0]
    centers0 = torch.tensor(centers0, dtype=torch.float32, device=device)
    centers0_dec = centers0.cpu().numpy()
    xdec, ydec = centers0_dec[:, 0], centers0_dec[:, 1]

    # Mask (fluid region)
    cx, cy = 10000, 10000
    r_cyl_dec = 3000
    mask_fluid = ((xdec - cx) ** 2 + (ydec - cy) ** 2) >= 0
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
    # Figure out input_dim/output_dim from a single sample
    input_dim = data.x.shape[1]
    output_dim = 3

    # --- Collect velocity magnitude for all frames ---
    mag_vals = []
    for t_idx, (t, centers, U, neighbors, edge_index) in enumerate(results):
        # Build data_t using your normalizer pipeline (this matches your createGraphData_t logic)
        centers = torch.tensor(centers, dtype=torch.float32, device=device)
        U = torch.tensor(U, dtype=torch.float32, device=device)
        centers_norm = xynorm.encode(centers)
        U_norm = uvnorm.encode(U)
        # dist_norm, circle = geometryObject(centers_norm, (0, 0), rad_scaled)
        dist_norm, circle = geometryObject(
            centers_norm, (-0.4911, -1.2672), 0.120, 0.555
        )
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
            uvp = model(data_t)
            u_pred = uvnorm.decode(uvp[:, 0], idx=0).cpu().numpy()
            v_pred = uvnorm.decode(uvp[:, 1], idx=1).cpu().numpy()
            mag = np.linalg.norm(np.stack([u_pred, v_pred], axis=-1), axis=1)
            mag_vals.append(mag[mask_fluid])
    mag_vals = np.array(mag_vals)
    vmin, vmax = mag_vals.min(), mag_vals.max()

    # --- Animation setup ---

    fig, ax = plt.subplots(figsize=(7, 6))
    cntr = ax.tricontourf(
        triang, mag_vals[0], levels=600, cmap="jet", vmin=vmin, vmax=vmax
    )
    cb = fig.colorbar(cntr, ax=ax, label="|u,v|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.set_xlim(min(xdec), max(xdec))
    # ax.set_ylim(min(ydec), max(ydec))
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 3000)
    title = ax.set_title("t = 0.00")
    ax.axis("equal")
    if geometry == "circle":
        circle = Circle(
            (cx, cy),
            r_cyl_dec,
            color="k",
            fill=False,
            linewidth=0.0,
            linestyle="",
            zorder=10,
        )
        # ax.add_patch(circle)
    elif geometry == "rectangle":
        rect_x = 1000  # x-coordinate of the lower left corner
        rect_y = min(ydec)  # y-coordinate of the lower left corner
        rect_width = 100  # width of the rectangle
        rect_height = 200  # height of the rectangle

        rectangle = Rectangle(
            (rect_x, rect_y),
            rect_width,
            rect_height,
            color="k",  # black border
            fill=False,  # not filled
            linewidth=0,  # border thickness
            linestyle="",  # dashed border
            zorder=10,  # draw above the circle if both are present
        )

        ax.add_patch(rectangle)

    def update(frame):
        for c in ax.collections:
            c.remove()
        cntr = ax.tricontourf(
            triang, mag_vals[frame], levels=600, cmap="jet", vmin=vmin, vmax=vmax
        )
        t_now = results[frame][0]
        title.set_text(f"t = {t_now:.3f}")

        # ax.add_patch(circle)
        ax.add_patch(rectangle)
        return []

    ani = FuncAnimation(
        fig, update, frames=len(results), interval=60, blit=False, repeat=True
    )
    ani.save(out_gif, writer="imagemagick", fps=20)
    print("✅ Animation saved as", out_gif)
    return ani


if __name__ == "__main__":
    animate_patch_time_series_gnn()
