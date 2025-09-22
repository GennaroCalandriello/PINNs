from testROM import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from torch_geometric.data import Data

# ==== importa il TUO modello e le tue utility ====
# from test import GraphAutoencoderDiffPool, geometryObject, dataLoader, dataNormalizer
# Se già in namespace, puoi omettere.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
geometry = "rectangle"  # "circle" or "rectangle"
out_gif = "anim_testROM.gif"

# Rettangolo axis-aligned (angolo in basso-sx)
RECT_X0 = 1000.0
RECT_Y0 = None  # se None: usa min(ydec)
RECT_W = 100.0
RECT_H = 200.0


def build_fluid_masks(x, y, obstacles):

    inside_any = np.zeros_like(x, dtype=bool)

    def inside_rect_axis_aligned(x, y, x0, y0, w, h):
        return (x >= x0) & (x <= x0 + w) & (y >= y0) & (y <= y0 + h)

    for obs in obstacles:
        inside_any |= inside_rect_axis_aligned(
            x, y, obs["x0"], obs["y0"], obs["w"], obs["h"]
        )
    mask_points_fluid = ~inside_any

    def tri_mask_fn(tri_xc, tri_yc):
        tri_inside = np.zeros_like(tri_xc, dtype=bool)
        for obs in obstacles:

            tri_inside |= inside_rect_axis_aligned(
                tri_xc, tri_yc, obs["x0"], obs["y0"], obs["w"], obs["h"]
            )
        return tri_inside

    return mask_points_fluid, tri_mask_fn


def build_static_graph_and_norms(
    rect_center_phys=(1050, 100), rect_w_phys=100, rect_h_phys=200
):
    """Precompute TUTTO ciò che è statico: pos, edge_index, edge_attr, rect features, triangulation, normalizers."""
    # Carica dati grezzi e normalizzatori
    results, _ = dataLoader()  # -> lista di (t, centers, U, neighbors, edge_index)
    t0, centers_all, U_all, neighbors_all, edge_index_all, xynorm, uvnorm = (
        dataNormalizer(results)
    )

    # Usiamo il primo frame per pos/edges (la mesh è fissa)
    t0, centers0, U0, neighbors0, edge_index0 = results[0]
    centers0 = torch.tensor(centers0, dtype=torch.float32, device=device)

    # Normalizza pos (come in createGraphData)
    centers_norm = torch.as_tensor(
        xynorm.encode(centers0), dtype=torch.float32, device=device
    )

    # Precompute edge_index dalla neighbors list (coerente con createGraphData)
    edge_src, edge_dst = [], []
    for i, nbs in enumerate(neighbors0):
        for j in nbs:
            if i != j:
                edge_src.append(i)
                edge_dst.append(j)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)

    # Edge attr: [rel(2), dist(1), dir(2), invr(1)] = 6 dim (STATICHE)
    rel = centers_norm[edge_index[1]] - centers_norm[edge_index[0]]
    dist = torch.norm(rel, dim=1, keepdim=True)
    dirn = rel / (dist + 1e-12)
    invr = 1.0 / (dist + 1e-12)
    edge_attr = torch.cat([rel, dist, dirn, invr], dim=1).to(torch.float32)

    # Feature rettangolo (STATICHE, rettangolo fisso)
    sigma_x = xynorm.std[0].item()
    sigma_y = xynorm.std[1].item()
    cx_phys, cy_phys = rect_center_phys
    rect_center_norm = torch.tensor(
        [
            (cx_phys - xynorm.mean[0].item()) / sigma_x,
            (cy_phys - xynorm.mean[1].item()) / sigma_y,
        ],
        dtype=torch.float32,
        device=device,
    )
    rect_w_norm = rect_w_phys / sigma_x
    rect_h_norm = rect_h_phys / sigma_y
    rect_dist_norm, rect_mask = geometryObject(
        centers_norm,
        (rect_center_norm[0].item(), rect_center_norm[1].item()),
        rect_w_norm,
        rect_h_norm,
    )  # shape [N,1] + [N,1]

    # Triangolazione per plotting IN COORDINATE FISICHE (decodifica su assi reali)
    centers0_np = centers0.detach().cpu().numpy()
    xdec, ydec = centers0_np[:, 0], centers0_np[:, 1]
    # triangolazione sui soli punti fluido
    obstacles_dec = build_obstacles(xdec, ydec)
    mask_fluid, tri_mask_fn = build_fluid_masks(xdec, ydec, obstacles_dec)
    xdec_fluid = xdec[mask_fluid]
    ydec_fluid = ydec[mask_fluid]
    triang = mtri.Triangulation(xdec_fluid, ydec_fluid)
    tri_pts = np.stack(
        [xdec_fluid[triang.triangles], ydec_fluid[triang.triangles]], axis=-1
    )
    tri_centers_x = tri_pts[:, :, 0].mean(axis=1)
    tri_centers_y = tri_pts[:, :, 1].mean(axis=1)
    triang.set_mask(tri_mask_fn(tri_centers_x, tri_centers_y))
    triang = mtri.Triangulation(
        xdec, ydec
    )  # nessuna maschera, o aggiungila qui se vuoi

    static = {
        "centers_norm": centers_norm,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "rect_dist_norm": rect_dist_norm,
        "rect_mask": rect_mask,
        "triang": triang,
        "xdec": xdec,
        "ydec": ydec,
        "xynorm": xynorm,
        "uvnorm": uvnorm,
        "results": results,
    }
    return static


def build_obstacles(decoded_x, decoded_y):
    obstacles = []
    y0 = float(np.min(decoded_y)) if RECT_Y0 is None else float(RECT_Y0)
    obstacles.append(
        {
            "type": "rect",
            "x0": float(RECT_X0),
            "y0": y0,
            "w": float(RECT_W),
            "h": float(RECT_H),
        }
    )

    return obstacles


def build_frame_data(static, U_frame):
    """Costruisce Data per un frame: aggiorna SOLO U_norm (input x e target y)."""
    centers_norm = static["centers_norm"]
    edge_index = static["edge_index"]
    edge_attr = static["edge_attr"]
    rect_dist_norm = static["rect_dist_norm"]
    rect_mask = static["rect_mask"]
    uvnorm = static["uvnorm"]

    # Normalizza U frame-by-frame
    U_t = torch.tensor(U_frame, dtype=torch.float32, device=device)
    U_norm = torch.as_tensor(uvnorm.encode(U_t), dtype=torch.float32, device=device)

    # Node features come in training: [U_norm, rect_dist_norm, rect_mask]
    x = torch.cat([U_norm, rect_dist_norm, rect_mask], dim=1)

    # Target = U_norm (ricostruzione)
    y = U_norm

    data_t = Data(
        x=x, y=y, pos=centers_norm, edge_index=edge_index, edge_attr=edge_attr
    ).to(device)
    # (opzionale) sanificazione tipi
    data_t.x = data_t.x.float().contiguous()
    data_t.y = data_t.y.float().contiguous()
    data_t.edge_attr = data_t.edge_attr.float().contiguous()
    data_t.edge_index = data_t.edge_index.long().contiguous()
    return data_t


def animate_patch_time_series_gnn(
    out_gif=out_gif,
):
    # ==== precompute static ====
    static = build_static_graph_and_norms(
        rect_center_phys=(1050, 100), rect_w_phys=100, rect_h_phys=200
    )
    results = static["results"]
    triang = static["triang"]
    xdec, ydec = static["xdec"], static["ydec"]
    uvnorm = static["uvnorm"]
    obstacles_dec = build_obstacles(xdec, ydec)
    mask_fluid, tri_mask_fn = build_fluid_masks(xdec, ydec, obstacles_dec)
    # ==== prepara un Data "di base" per dedurre IN/EDGE/OUT dims ====
    _, _, U0, _, _ = results[0]
    data0 = build_frame_data(static, U0)
    IN_DIM = data0.x.size(1)  # 4 (U_norm(2)+rect 2)
    EDGE_DIM = data0.edge_attr.size(1)  # 6
    OUT_DIM = data0.y.size(1)  # 2 (u,v)

    # ==== modello ====
    model = GraphAutoEncoder(
        in_ch=IN_DIM,
        edge_dim=EDGE_DIM,
        out_ch=OUT_DIM,
        hidden=HIDDEN,
        latent=LATENT,
        clusters_per_level=CLUSTERS_PER_LEVEL,
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # ==== precomputo magnitudo predetta per ogni frame ====
    mag_vals = []
    with torch.no_grad():
        for t, centers, U, neighbors, edge_index_raw in results:
            data_t = build_frame_data(static, U)
            y_hat = model(data_t)  # (N,2)
            # decode in unità fisiche per plotting
            u_pred = uvnorm.decode(y_hat[:, 0], idx=0).cpu().numpy()
            v_pred = uvnorm.decode(y_hat[:, 1], idx=1).cpu().numpy()
            mag = np.linalg.norm(np.stack([u_pred, v_pred], axis=-1), axis=1)
            mag_vals.append(mag)
    mag_vals = np.array(mag_vals)
    vmin, vmax = np.nanmin(mag_vals), np.nanmax(mag_vals)

    # ==== setup figura ====
    fig, ax = plt.subplots(figsize=(7, 6))
    for obs in obstacles_dec:
        rectangle = Rectangle(
            (obs["x0"], obs["y0"]),
            obs["w"],
            obs["h"],
            color="k",
            fill=True,
            linewidth=0.0,
            linestyle="",
            zorder=10,
        )
        ax.add_patch(rectangle)

    cntr = ax.tricontourf(
        triang, mag_vals[0], levels=300, cmap="jet", vmin=vmin, vmax=vmax
    )
    cb = fig.colorbar(cntr, ax=ax, label="|u| (decoded)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.set_aspect("equal", adjustable="box")

    title = ax.set_title("t = 0.00")

    def update(frame):
        # pulisci vecchie collections del contour
        for c in ax.collections:
            c.remove()
        cntr = ax.tricontourf(
            triang, mag_vals[frame], levels=300, cmap="jet", vmin=vmin, vmax=vmax
        )
        t_now = results[frame][0]
        title.set_text(f"t = {t_now:.3f}")
        ax.add_patch(rectangle)
        return []

    ani = FuncAnimation(
        fig, update, frames=len(results), interval=60, blit=False, repeat=True
    )
    ani.save(out_gif, writer="pillow", fps=20)
    print("✅ Animation saved as", out_gif)
    return ani


def plotLoss():
    loss = np.loadtxt(LOSS_PATH)
    xep = np.linspace(1, EPOCHS, len(loss))
    plt.figure(figsize=(6, 4))

    plt.plot(loss)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.show()


if __name__ == "__main__":
    plotLoss()
    animate_patch_time_series_gnn()
