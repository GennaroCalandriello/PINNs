# animate_gnn_errors.py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle as MPCircle, Rectangle as MPRect

# ====== tuoi import ======
from ns_GNN_KF import dataLoader, dataNormalizer, createGraphData
from EdgeNodeAttentionDiffPool import (
    GraphAutoEncoder,
    CLUSTERS_PER_LEVEL,
)  # usa la tua classe

# ====== CONFIG ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Geometria ostacoli (coordinate DECODED/fisiche)
GEOMETRY = "rectangle"  # "circle" | "rectangle" | "both"
# cerchio
CIRCLE_CX = 10000.0
CIRCLE_CY = 10000.0
CIRCLE_R = 3000.0
# rettangolo axis-aligned (angolo in basso-sx)
RECT_X0 = 1000.0
RECT_Y0 = None  # se None: usa min(ydec) del primo frame
RECT_W = 100.0
RECT_H = 200.0

# modello e output
MODEL_PATH = "model/gnn_ae.pth"
OUT_GIF = "errors.gif"
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
def inside_circle(x, y, cx, cy, r):
    return (x - cx) ** 2 + (y - cy) ** 2 <= r**2


def inside_rect_axis_aligned(x, y, x0, y0, w, h):
    return (x >= x0) & (x <= x0 + w) & (y >= y0) & (y <= y0 + h)


def build_obstacles(decoded_x, decoded_y):
    obs = []
    if GEOMETRY == "circle":
        obs.append({"type": "circle", "cx": CIRCLE_CX, "cy": CIRCLE_CY, "r": CIRCLE_R})
    elif GEOMETRY == "rectangle":
        y0 = float(np.min(decoded_y)) if RECT_Y0 is None else float(RECT_Y0)
        obs.append(
            {
                "type": "rect",
                "x0": float(RECT_X0),
                "y0": y0,
                "w": float(RECT_W),
                "h": float(RECT_H),
            }
        )
    elif GEOMETRY == "both":
        obs.append({"type": "circle", "cx": CIRCLE_CX, "cy": CIRCLE_CY, "r": CIRCLE_R})
        y0 = float(np.min(decoded_y)) if RECT_Y0 is None else float(RECT_Y0)
        obs.append(
            {
                "type": "rect",
                "x0": float(RECT_X0),
                "y0": y0,
                "w": float(RECT_W),
                "h": float(RECT_H),
            }
        )
    else:
        raise ValueError("GEOMETRY deve essere 'circle' | 'rectangle' | 'both'")
    return obs


def build_fluid_masks(x, y, obstacles):
    inside_any = np.zeros_like(x, dtype=bool)
    for o in obstacles:
        if o["type"] == "circle":
            inside_any |= inside_circle(x, y, o["cx"], o["cy"], o["r"])
        else:
            inside_any |= inside_rect_axis_aligned(
                x, y, o["x0"], o["y0"], o["w"], o["h"]
            )
    mask_points_fluid = ~inside_any

    def tri_mask_fn(tri_xc, tri_yc):
        tri_inside = np.zeros_like(tri_xc, dtype=bool)
        for o in obstacles:
            if o["type"] == "circle":
                tri_inside |= inside_circle(tri_xc, tri_yc, o["cx"], o["cy"], o["r"])
            else:
                tri_inside |= inside_rect_axis_aligned(
                    tri_xc, tri_yc, o["x0"], o["y0"], o["w"], o["h"]
                )
        return tri_inside

    return mask_points_fluid, tri_mask_fn


# =========================================================
# helpers: feature geometriche in SPAZIO NORMALIZZATO
# =========================================================
def decoded_to_normalized_point(px, py, norm_obj):
    mx, sx = norm_obj.mean[0].item(), norm_obj.std[0].item()
    my, sy = norm_obj.mean[1].item(), norm_obj.std[1].item()
    return ((px - mx) / sx, (py - my) / sy)


def circle_to_normalized(o, norm_obj):
    cxn, cyn = decoded_to_normalized_point(o["cx"], o["cy"], norm_obj)
    sx = norm_obj.std[0].item()
    rn = o["r"] / sx
    return {"type": "circle", "cx": cxn, "cy": cyn, "r": rn}


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
        out.append(
            circle_to_normalized(o, norm_obj)
            if o["type"] == "circle"
            else rect_to_normalized(o, norm_obj)
        )
    return out


def sdf_features_normalized(centers_norm, obstacles_norm):
    x = centers_norm[:, 0]
    y = centers_norm[:, 1]
    N = centers_norm.size(0)
    sdf = torch.full((N,), 1e6, device=centers_norm.device)
    inside_any = torch.zeros((N,), dtype=torch.bool, device=centers_norm.device)

    for o in obstacles_norm:
        if o["type"] == "circle":
            dx = x - o["cx"]
            dy = y - o["cy"]
            d = torch.sqrt(dx * dx + dy * dy) - o["r"]
            sdf = torch.minimum(sdf, d)
            inside_any |= d <= 0
        else:
            px = x
            py = y
            x0, y0, w, h = o["x0"], o["y0"], o["w"], o["h"]
            dx0 = torch.maximum(x0 - px, torch.zeros_like(px))
            dx1 = torch.maximum(px - (x0 + w), torch.zeros_like(px))
            dy0 = torch.maximum(y0 - py, torch.zeros_like(py))
            dy1 = torch.maximum(py - (y0 + h), torch.zeros_like(py))
            outside = torch.sqrt((dx0 + dx1) ** 2 + (dy0 + dy1) ** 2)
            inside = (px >= x0) & (px <= x0 + w) & (py >= y0) & (py <= y0 + h)
            d = torch.where(
                inside,
                -torch.minimum(
                    torch.minimum(px - x0, (x0 + w) - px),
                    torch.minimum(py - y0, (y0 + h) - py),
                ),
                outside,
            )
            sdf = torch.minimum(sdf, d)
            inside_any |= inside

    return sdf.unsqueeze(1), inside_any.float().unsqueeze(1)  # [N,1], [N,1]


# =========================================================
# main: animazione degli errori
# =========================================================
def animate_error_time_series_gnn(
    out_gif=OUT_GIF, model_path=MODEL_PATH, error_mode=ERROR_MODE
):
    # --- carica dati & normalizzatori ---
    results, _ = dataLoader()
    t0, centers, U, neighbors, edge_index, xynorm, uvnorm = dataNormalizer(results)

    # decoded coords del primo frame (per mesh & maschera)
    centers0_dec = (
        torch.tensor(results[0][1], dtype=torch.float32, device=device).cpu().numpy()
    )
    xdec, ydec = centers0_dec[:, 0], centers0_dec[:, 1]

    # ostacoli & maschere
    obstacles_dec = build_obstacles(xdec, ydec)
    mask_fluid, tri_mask_fn = build_fluid_masks(xdec, ydec, obstacles_dec)

    # triangolazione solo sui punti del fluido
    xdec_fluid = xdec[mask_fluid]
    ydec_fluid = ydec[mask_fluid]
    triang = mtri.Triangulation(xdec_fluid, ydec_fluid)
    tri_pts = np.stack(
        [xdec_fluid[triang.triangles], ydec_fluid[triang.triangles]], axis=-1
    )
    tri_centers_x = tri_pts[:, :, 0].mean(axis=1)
    tri_centers_y = tri_pts[:, :, 1].mean(axis=1)
    triang.set_mask(tri_mask_fn(tri_centers_x, tri_centers_y))

    # ostacoli anche in spazio NORMALIZZATO per feature
    obstacles_norm = obstacles_to_normalized(obstacles_dec, xynorm)

    # --- modello ---
    data_dim_probe = createGraphData().to(device)
    in_ch = data_dim_probe.x.size(-1)
    edge_dim = data_dim_probe.edge_attr.size(-1)
    out_ch = data_dim_probe.y.size(-1)

    try:
        model = GraphAutoEncoder(
            in_ch=in_ch,
            edge_dim=edge_dim,
            out_ch=out_ch,
            clusters_per_level=CLUSTERS_PER_LEVEL,
        ).to(device)
    except TypeError:
        model = GraphAutoEncoder(in_ch=in_ch, edge_dim=edge_dim, out_ch=out_ch).to(
            device
        )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- accumula errori per tutti i frame (solo nodi fluido) ---
    err_vals = []
    eps = 1e-9
    for t_idx, (t, centers_dec, U_dec, neighbors_i, edge_index_i) in enumerate(results):
        centers_dec = torch.tensor(centers_dec, dtype=torch.float32, device=device)
        U_dec = torch.tensor(
            U_dec, dtype=torch.float32, device=device
        )  # ground truth (decoded)

        centers_norm = xynorm.encode(centers_dec)  # [N,2]
        U_norm = uvnorm.encode(U_dec)  # [N,2] (se U_norm è parte delle feature input)

        # feature geometriche in norm
        dist_norm, indicator = sdf_features_normalized(centers_norm, obstacles_norm)

        # input features x (adatta a quello che hai usato in training)
        x = torch.cat([centers_norm, U_norm, dist_norm, indicator], dim=1)

        # edge_index & edge_attr
        edge_src, edge_dst = [], []
        for i, nbs in enumerate(neighbors_i):
            for j in nbs:
                edge_src.append(i)
                edge_dst.append(j)
        edge_src = torch.tensor(edge_src, dtype=torch.long, device=device)
        edge_dst = torch.tensor(edge_dst, dtype=torch.long, device=device)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        rel = centers_norm[edge_dst] - centers_norm[edge_src]  # [E,2]
        dist = torch.norm(rel, dim=1, keepdim=True)  # [E,1]

        if USE_CP_FEATURE:
            dvel = U_norm[edge_dst] - U_norm[edge_src]  # [E,2]
            cp = rel[:, 0:1] * dvel[:, 1:2] - rel[:, 1:2] * dvel[:, 0:1]  # [E,1]
            edge_attr = torch.cat([rel, dist, cp], dim=1)
        else:
            edge_attr = torch.cat([rel, dist], dim=1)

        from torch_geometric.data import Data

        data_t = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=U_norm).to(
            device
        )

        with torch.no_grad():
            pred_norm = model(data_t)  # [N, out_ch], supponiamo (u,v,...) normalizzati
            # decodifica ai valori fisici
            u_pred = uvnorm.decode(pred_norm[:, 0], idx=0)  # [N]
            v_pred = uvnorm.decode(pred_norm[:, 1], idx=1)  # [N]
            u_gt = U_dec[:, 0]
            v_gt = U_dec[:, 1]

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

            err_vals.append(e[mask_fluid])

    err_vals = np.array(err_vals)  # [T, N_fluid]
    vmin = 0.0
    vmax = float(np.percentile(err_vals, 99.5))  # robust max per colormap

    # --- figura & patches ostacoli ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for o in obstacles_dec:
        if o["type"] == "circle":
            patch = MPCircle(
                (o["cx"], o["cy"]),
                o["r"],
                color="k",
                fill=False,
                linewidth=1.0,
                zorder=10,
            )
        else:
            patch = MPRect(
                (o["x0"], o["y0"]),
                o["w"],
                o["h"],
                color="k",
                fill=False,
                linewidth=1.0,
                zorder=10,
            )
        ax.add_patch(patch)

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

    # update
    def update(frame):
        nonlocal quad
        for c in ax.collections:  # rimuovi SOLO il vecchio contourf
            c.remove()
        quad = ax.tricontourf(
            triang, err_vals[frame], levels=LEVELS, cmap=CMAP, vmin=vmin, vmax=vmax
        )
        t_now = results[frame][0]
        title.set_text(f"t = {t_now:.3f}")
        return []

    ani = FuncAnimation(
        fig, update, frames=len(results), interval=60, blit=False, repeat=True
    )
    ani.save(out_gif, writer="imagemagick", fps=FPS)
    print("✅ Error animation saved as", out_gif)
    print(" Max and min errors over all frames:", err_vals.max(), err_vals.min())
    return ani


# =========================================================
if __name__ == "__main__":
    animate_error_time_series_gnn = animate_error_time_series_gnn = (
        animate_error_time_series_gnn
        if "animate_error_time_series_gnn" in globals()
        else None
    )
    # esegui
    animate_error_time_series_gnn(
        out_gif=OUT_GIF, model_path=MODEL_PATH, error_mode=ERROR_MODE
    )
