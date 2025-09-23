# animate_gnn_errors.py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from plotROM import (
    build_static_graph_and_norms,
    build_fluid_masks,
    build_obstacles,
    build_frame_data,
)

# ====== tuoi import ======
from ns_GNN_cav2 import dataLoader, dataNormalizer, createGraphData
from testROM import *

# ====== CONFIG ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Geometria ostacoli (coordinate DECODED/fisiche)
GEOMETRY = "rectangle"  # "circle" | "rectangle" | "both"
# cerchio
# rettangolo axis-aligned (angolo in basso-sx)
RECT_X0 = 1000.0
RECT_Y0 = None  # se None: usa min(ydec) del primo frame
RECT_W = 100.0
RECT_H = 200.0

# modello e output
OUT_GIF = "anim_errors_testROM.gif"
FPS = 20
CMAP = "magma"
LEVELS = 500

# cosa plottare: "mag" | "u" | "v" | "relmag"
ERROR_MODE = "mag"

# se in training avevi aggiunto cp nell'edge_attr, metti True (altrimenti deve restare False!)
USE_CP_FEATURE = False


# =========================================================
# helpers: ostacoli & maschere
# =========================================================


def inside_rect_axis_aligned(x, y, x0, y0, w, h):
    return (x >= x0) & (x <= x0 + w) & (y >= y0) & (y <= y0 + h)


# =========================================================
# helpers: feature geometriche in SPAZIO NORMALIZZATO
# =========================================================
def decoded_to_normalized_point(px, py, norm_obj):
    mx, sx = norm_obj.mean[0].item(), norm_obj.std[0].item()
    my, sy = norm_obj.mean[1].item(), norm_obj.std[1].item()
    return ((px - mx) / sx, (py - my) / sy)


def rect_to_normalized(o, norm_obj):
    x0, y0 = o["x0"], o["y0"]
    x1, y1 = x0 + o["w"], y0 + o["h"]
    (x0n, y0n) = decoded_to_normalized_point(x0, y0, norm_obj)
    (x1n, y1n) = decoded_to_normalized_point(x1, y1, norm_obj)
    xn0, yn0 = min(x0n, x1n), min(y0n, y1n)
    wn, hn = abs(x1n - x0n), abs(y1n - y0n)
    return {"type": "rect", "x0": xn0, "y0": yn0, "w": wn, "h": hn}


def obstacles_to_normalized(obstacles_decoded, norm_obj):
    out = []
    for o in obstacles_decoded:
        out.append(rect_to_normalized(o, norm_obj))
    return out


# =========================================================
# main: animazione degli errori
# =========================================================
def animate_error_time_series_gnn(
    out_gif=OUT_GIF, model_path=MODEL_PATH, error_mode=ERROR_MODE
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

    # --- accumula errori per tutti i frame (solo nodi fluido) ---
    err_vals = []

    eps = 1e-9
    with torch.no_grad():
        for t, centers, U, neighbors, edge_index_raw in results:

            data_t = build_frame_data(static, U)
            y_hat = model(data_t)  # (N,2) predetto in norm
            u_pred = uvnorm.decode(y_hat[:, 0], idx=0)
            v_pred = uvnorm.decode(y_hat[:, 1], idx=1)

            U_dec = torch.tensor(
                U, dtype=torch.float32, device=device
            )  # ground truth (decoded)

            u_gt = U[:, 0]
            v_gt = U[:, 1]
            u_gt = torch.tensor(u_gt, dtype=torch.float32, device=device)
            v_gt = torch.tensor(v_gt, dtype=torch.float32, device=device)

            u_pred = torch.tensor(u_pred, dtype=torch.float32, device=device)
            v_pred = torch.tensor(v_pred, dtype=torch.float32, device=device)

            # error modes
            if error_mode == "u":
                e = (u_pred - u_gt).abs().detach().cpu().numpy()
            elif error_mode == "v":
                e = (v_pred - v_gt).abs().detach().cpu().numpy()
            elif error_mode == "relmag":
                mag_gt = torch.sqrt(u_gt**2 + v_gt**2)
                mag_pred = torch.sqrt(u_pred**2 + v_pred**2)
                e = ((mag_pred - mag_gt).abs() / (mag_gt + eps)).detach().cpu().numpy()
            else:  # "mag"
                e = (
                    torch.sqrt((u_pred - u_gt) ** 2 + (v_pred - v_gt) ** 2)
                    .detach()
                    .cpu()
                    .numpy()
                )

                # conserva solo nodi fluido
                e_fluid = e
                err_vals.append(e_fluid)

    err_vals = np.array(err_vals)  # [T, N_fluid]
    vmin = np.nanmin(err_vals)
    # vmax = float(np.percentile(err_vals, 99.5))  # robust max per colormap
    vmax = np.nanmax(err_vals)

    # --- figura & patches ostacoli ---
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

    # primo frame
    quad = ax.tricontourf(
        triang, err_vals[0], levels=LEVELS, cmap=CMAP, vmin=vmin, vmax=vmax
    )
    cb = fig.colorbar(quad, ax=ax, label=f"Error ({error_mode})")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(np.min(xdec), np.max(xdec))
    ax.set_ylim(np.min(ydec), np.max(ydec))
    title = ax.set_title("t = 0.00")
    ax.set_aspect("equal", adjustable="box")

    # update: rimuove SOLO le collezioni del vecchio contourf
    def update(frame):
        nonlocal quad
        # rimuovi le collezioni del precedente contourf
        for coll in ax.collections:
            coll.remove()
        # ridisegna
        quad = ax.tricontourf(
            triang, err_vals[frame], levels=LEVELS, cmap=CMAP, vmin=vmin, vmax=vmax
        )
        t_now = results[frame][0]
        title.set_text(f"t = {t_now:.3f}")
        return []

    ani = FuncAnimation(
        fig, update, frames=len(results), interval=60, blit=False, repeat=True
    )

    # salvataggio: prova imagemagick, fallback a pillow
    ani.save(out_gif, writer=PillowWriter(fps=FPS))

    print("✅ Error animation saved as", out_gif)
    print(" Max and min errors over all frames:", err_vals.max(), err_vals.min())
    return ani


# =========================================================
if __name__ == "__main__":
    animate_error_time_series_gnn(
        out_gif=OUT_GIF, model_path=MODEL_PATH, error_mode=ERROR_MODE
    )
