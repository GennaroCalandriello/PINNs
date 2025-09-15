# animate_gnn_fields.py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MPCircle, Rectangle as MPRect
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation

# ====== TUOI IMPORT ======
from ns_GNN_KF import (
    dataLoader,
    dataNormalizer,
    createGraphData,
)
from EdgeNodeAttentionDiffPool import (
    GraphAutoEncoder,
    CLUSTERS_PER_LEVEL,
)  # usa la tua classe

# ====== CONFIG ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Geometria di ostacoli (in coordinate DECODED / fisiche)
geometry = "rectangle"  # "circle" | "rectangle" | "both"

# Cerchio
CIRCLE_CX = 10000.0
CIRCLE_CY = 10000.0
CIRCLE_R = 3000.0

# Rettangolo axis-aligned (angolo in basso-sx)
RECT_X0 = 1000.0
RECT_Y0 = None  # se None: usa min(ydec)
RECT_W = 100.0
RECT_H = 200.0

# Modello
MODEL_PATH = "model/gnn_ae.pth"
OUT_GIF = "mehmeh.gif"
FPS = 20
CMAP = "jet"
LEVELS = 600


# =========================================================
# Helpers: ostacoli & maschere
# =========================================================
def inside_circle(x, y, cx, cy, r):
    return (x - cx) ** 2 + (y - cy) ** 2 <= r**2


def inside_rect_axis_aligned(x, y, x0, y0, w, h):
    return (x >= x0) & (x <= x0 + w) & (y >= y0) & (y <= y0 + h)


def build_obstacles(decoded_x, decoded_y):
    """Ritorna la lista di ostacoli in coordinate DECODED, e normalizza RECT_Y0 se None."""
    obstacles = []
    if geometry == "circle":
        obstacles.append(
            {"type": "circle", "cx": CIRCLE_CX, "cy": CIRCLE_CY, "r": CIRCLE_R}
        )
    elif geometry == "rectangle":
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
    elif geometry == "both":
        obstacles.append(
            {"type": "circle", "cx": CIRCLE_CX, "cy": CIRCLE_CY, "r": CIRCLE_R}
        )
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
    else:
        raise ValueError("geometry deve essere 'circle' | 'rectangle' | 'both'")
    return obstacles


def build_fluid_masks(x, y, obstacles):
    """
    x,y: decoded coords (np)
    Ritorna:
      mask_points_fluid (True = fluido),
      tri_mask_fn(tri_xc, tri_yc) -> True = triangolo da mascherare (interno agli ostacoli).
    """
    inside_any = np.zeros_like(x, dtype=bool)
    for obs in obstacles:
        if obs["type"] == "circle":
            inside_any |= inside_circle(x, y, obs["cx"], obs["cy"], obs["r"])
        else:
            inside_any |= inside_rect_axis_aligned(
                x, y, obs["x0"], obs["y0"], obs["w"], obs["h"]
            )
    mask_points_fluid = ~inside_any

    def tri_mask_fn(tri_xc, tri_yc):
        tri_inside = np.zeros_like(tri_xc, dtype=bool)
        for obs in obstacles:
            if obs["type"] == "circle":
                tri_inside |= inside_circle(
                    tri_xc, tri_yc, obs["cx"], obs["cy"], obs["r"]
                )
            else:
                tri_inside |= inside_rect_axis_aligned(
                    tri_xc, tri_yc, obs["x0"], obs["y0"], obs["w"], obs["h"]
                )
        return tri_inside

    return mask_points_fluid, tri_mask_fn


# =========================================================
# Helpers: feature geometriche in SPAZIO NORMALIZZATO
# =========================================================
def decoded_to_normalized_point(px, py, norm_obj):
    # x_norm = (x_dec - mean) / std
    mx, sx = norm_obj.mean[0].item(), norm_obj.std[0].item()
    my, sy = norm_obj.mean[1].item(), norm_obj.std[1].item()
    return ((px - mx) / sx, (py - my) / sy)


def circle_to_normalized(circle_obs, norm_obj):
    cxn, cyn = decoded_to_normalized_point(circle_obs["cx"], circle_obs["cy"], norm_obj)
    # r normalizzato: scala su x (assumiamo isotropia o scala_x)
    sx = norm_obj.std[0].item()
    rn = circle_obs["r"] / sx
    return {"type": "circle", "cx": cxn, "cy": cyn, "r": rn}


def rect_to_normalized(rect_obs, norm_obj):
    # converti i 4 angoli e ricava x0,y0,w,h in normalizzato
    x0, y0 = rect_obs["x0"], rect_obs["y0"]
    x1, y1 = x0 + rect_obs["w"], y0 + rect_obs["h"]
    (x0n, y0n) = decoded_to_normalized_point(x0, y0, norm_obj)
    (x1n, y1n) = decoded_to_normalized_point(x1, y1, norm_obj)
    # ricostruisci rettangolo axis-aligned normalizzato
    xn0, yn0 = min(x0n, x1n), min(y0n, y1n)
    wn, hn = abs(x1n - x0n), abs(y1n - y0n)
    return {"type": "rect", "x0": xn0, "y0": yn0, "w": wn, "h": hn}


def obstacles_to_normalized(obstacles_decoded, norm_obj):
    out = []
    for obs in obstacles_decoded:
        if obs["type"] == "circle":
            out.append(circle_to_normalized(obs, norm_obj))
        else:
            out.append(rect_to_normalized(obs, norm_obj))
    return out


def sdf_features_normalized(centers_norm, obstacles_norm):
    """
    Ritorna:
      dist_norm: [N,1] signed distance (min sui vari ostacoli),
      indicator: [N,1] 1 se inside ANY obstacle, altrimenti 0
    """
    x = centers_norm[:, 0]
    y = centers_norm[:, 1]
    N = centers_norm.size(0)
    # inizializza con grandi positivi per min
    sdf = torch.full((N,), 1e6, device=centers_norm.device)
    inside_any = torch.zeros((N,), dtype=torch.bool, device=centers_norm.device)

    for obs in obstacles_norm:
        if obs["type"] == "circle":
            dx = x - obs["cx"]
            dy = y - obs["cy"]
            d = torch.sqrt(dx * dx + dy * dy) - obs["r"]
            sdf = torch.minimum(sdf, d)
            inside_any |= d <= 0
        else:
            # SDF rettangolo axis-aligned (centered a x0,y0 con width w,height h)
            # usiamo formula standard per box axis-aligned con corner inferiore (x0,y0)
            # riferimento: q = |p - c| - b, ma qui usiamo x0,y0,w,h in forma corner
            px = x
            py = y
            x0, y0, w, h = obs["x0"], obs["y0"], obs["w"], obs["h"]
            # distanza firmata a rettangolo axis-aligned:
            dx0 = torch.maximum(x0 - px, torch.zeros_like(px))
            dx1 = torch.maximum(px - (x0 + w), torch.zeros_like(px))
            dy0 = torch.maximum(y0 - py, torch.zeros_like(py))
            dy1 = torch.maximum(py - (y0 + h), torch.zeros_like(py))
            outside = torch.sqrt((dx0 + dx1) ** 2 + (dy0 + dy1) ** 2)
            # inside: max(max(x0-px, px-(x0+w)), max(y0-py, py-(y0+h))) <=0
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

    dist_norm = sdf.unsqueeze(1)  # [N,1]
    indicator = inside_any.float().unsqueeze(1)  # [N,1]
    return dist_norm, indicator


# =========================================================
# MAIN: animazione
# =========================================================
def animate_patch_time_series_gnn(
    out_gif=OUT_GIF,
    model_path=MODEL_PATH,
):
    # --- Carica dati e normalizzatori ---
    results, _ = dataLoader()
    xnorm, centers0_dec, _, _, _ = results[0]  # non usato: t0, neighbors ecc.
    t0, centers, U, neighbors, edge_index, xynorm, uvnorm = dataNormalizer(results)
    data = createGraphData().to(device)

    # decoded coords del primo frame (per costruire triangolazione/maschera)
    centers0_dec = (
        torch.tensor(centers0_dec, dtype=torch.float32, device=device).cpu().numpy()
    )
    xdec, ydec = centers0_dec[:, 0], centers0_dec[:, 1]

    # ostacoli in decoded space + maschere
    obstacles_dec = build_obstacles(xdec, ydec)
    mask_fluid, tri_mask_fn = build_fluid_masks(xdec, ydec, obstacles_dec)

    # triangolazione sui soli punti fluido
    xdec_fluid = xdec[mask_fluid]
    ydec_fluid = ydec[mask_fluid]
    triang = mtri.Triangulation(xdec_fluid, ydec_fluid)
    tri_pts = np.stack(
        [xdec_fluid[triang.triangles], ydec_fluid[triang.triangles]], axis=-1
    )
    tri_centers_x = tri_pts[:, :, 0].mean(axis=1)
    tri_centers_y = tri_pts[:, :, 1].mean(axis=1)
    mask_tri = tri_mask_fn(tri_centers_x, tri_centers_y)
    triang.set_mask(mask_tri)

    # ostacoli anche in spazio NORMALIZZATO per le feature
    obstacles_norm = obstacles_to_normalized(obstacles_dec, xynorm)

    # --- Modello ---
    in_ch = data.x.size(-1)
    edge_dim = data.edge_attr.size(-1)
    out_ch = data.y.size(-1)
    try:
        model = GraphAutoEncoder(
            in_ch=in_ch,
            edge_dim=edge_dim,
            out_ch=out_ch,
            clusters_per_level=CLUSTERS_PER_LEVEL,
        ).to(device)
    except TypeError:
        # fallback se il costruttore non richiede clusters_per_level
        model = GraphAutoEncoder(in_ch=in_ch, edge_dim=edge_dim, out_ch=out_ch).to(
            device
        )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- Colleziona il modulo della velocità per tutti i frame (solo nodi fluido) ---
    mag_vals = []
    for t_idx, (t, centers_dec, U_dec, neighbors_i, edge_index_i) in enumerate(results):
        centers_dec = torch.tensor(centers_dec, dtype=torch.float32, device=device)
        U_dec = torch.tensor(U_dec, dtype=torch.float32, device=device)

        # normalizza
        centers_norm = xynorm.encode(centers_dec)  # [N,2]
        U_norm = uvnorm.encode(U_dec)  # [N,2]

        # feature geometriche in norm
        dist_norm, indicator = sdf_features_normalized(centers_norm, obstacles_norm)

        # costruiamo x (stesse dimensioni attese in training: [centers_norm, U_norm, dist, ind])
        x = torch.cat([centers_norm, U_norm, dist_norm, indicator], dim=1)

        # edges
        edge_src, edge_dst = [], []
        for i, nbs in enumerate(neighbors_i):
            for j in nbs:
                edge_src.append(i)
                edge_dst.append(j)
        edge_src = torch.tensor(edge_src, dtype=torch.long, device=device)
        edge_dst = torch.tensor(edge_dst, dtype=torch.long, device=device)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        rel = centers_norm[edge_dst] - centers_norm[edge_src]
        dist = torch.norm(rel, dim=1, keepdim=True)
        edge_attr = torch.cat([rel, dist], dim=1)

        # Data per frame (solo per forward)
        from torch_geometric.data import Data

        data_t = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=U_norm).to(
            device
        )

        with torch.no_grad():
            uvp_norm = model(data_t)  # [N, out_ch], out_ch = 3? usiamo i primi 2
            u_pred = uvnorm.decode(uvp_norm[:, 0], idx=0).detach().cpu().numpy()
            v_pred = uvnorm.decode(uvp_norm[:, 1], idx=1).detach().cpu().numpy()
            mag = np.linalg.norm(np.stack([u_pred, v_pred], axis=-1), axis=1)
            mag_vals.append(mag[mask_fluid])

    mag_vals = np.array(mag_vals)  # [T, N_fluid]
    vmin, vmax = float(mag_vals.min()), float(mag_vals.max())

    # --- Plot iniziale ---
    fig, ax = plt.subplots(figsize=(7, 6))
    # disegna ostacoli (una sola volta)
    for obs in obstacles_dec:
        if obs["type"] == "circle":
            patch = MPCircle(
                (obs["cx"], obs["cy"]),
                obs["r"],
                color="k",
                fill=False,
                linewidth=1.0,
                zorder=10,
            )
        else:
            patch = MPRect(
                (obs["x0"], obs["y0"]),
                obs["w"],
                obs["h"],
                color="k",
                fill=False,
                linewidth=1.0,
                zorder=10,
            )
        ax.add_patch(patch)

    # primo contourf
    quad = ax.tricontourf(
        triang, mag_vals[0], levels=LEVELS, cmap=CMAP, vmin=vmin, vmax=vmax
    )
    cb = fig.colorbar(quad, ax=ax, label="|u,v|")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(np.min(xdec), np.max(xdec))
    ax.set_ylim(np.min(ydec), np.max(ydec))
    title = ax.set_title("t = 0.00")
    ax.set_aspect("equal", adjustable="box")

    # --- Update func ---
    # def update(frame):
    #     nonlocal quad
    #     # rimuovi SOLO le collezioni del contour precedente
    #     for c in quad.collections:
    #         c.remove()
    #     quad = ax.tricontourf(
    #         triang, mag_vals[frame], levels=LEVELS, cmap=CMAP, vmin=vmin, vmax=vmax
    #     )
    #     t_now = results[frame][0]
    #     title.set_text(f"t = {t_now:.3f}")
    #     return quad.collections
    def update(frame):
        for c in ax.collections:
            c.remove()
        cntr = ax.tricontourf(
            triang, mag_vals[frame], levels=600, cmap="jet", vmin=vmin, vmax=vmax
        )
        t_now = results[frame][0]
        title.set_text(f"t = {t_now:.3f}")

        # ax.add_patch(circle)
        return []

    ani = FuncAnimation(
        fig, update, frames=len(results), interval=60, blit=False, repeat=True
    )
    ani.save(out_gif, writer="imagemagick", fps=FPS)
    print("✅ Animation saved as", out_gif)
    return ani


# =========================================================
if __name__ == "__main__":
    animate_patch_time_series_gnn()
