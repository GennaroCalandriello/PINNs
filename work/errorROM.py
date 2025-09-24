# animate_gnn_errors.py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from trainAndPlotROM import (
    build_static_graph_and_norms,
    build_fluid_masks,
    build_obstacles,
    build_frame_data,
    plotLoss,
)

# ====== tuoi import ======
from ns_GNN_cav2 import dataLoader, dataNormalizer, createGraphData
from testROMmod import *  # deve esportare GraphAutoEncoder, HIDDEN, LATENT, CLUSTERS_PER_LEVEL, MODEL_PATH

# ====== CONFIG ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Geometria ostacoli (coordinate DECODED/fisiche)
GEOMETRY = "rectangle"  # "circle" | "rectangle" | "both"
RECT_X0 = 1000.0
RECT_Y0 = None  # se None: usa min(ydec) del primo frame
RECT_W = 100.0
RECT_H = 200.0

# modello e output
OUT_GIF = "plots/anim_errors_testROM.gif"
FPS = 20
CMAP = "magma"
LEVELS = 500

# cosa plottare: "mag" | "u" | "v" | "relmag"
ERROR_MODE = "mag"

# se in training avevi aggiunto cp nell'edge_attr, metti True (altrimenti deve restare False!)
USE_CP_FEATURE = False
SAVE_GROUND_TRUTH = False  # salva anche i valori GT, utile per debug
PLOT_SOME_SNAPSHOTS = False  # plotta snapshot statici (start/mid/end)


# =========================================================
# helpers: ostacoli & maschere
# =========================================================
def inside_rect_axis_aligned(x, y, x0, y0, w, h):
    return (x >= x0) & (x <= x0 + w) & (y >= y0) & (y <= y0 + h)


# =========================================================
# helpers: feature geometriche in SPAZIO NORMALIZZATO (se servono)
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
    results = static["results"]  # lista di frame
    triang: mtri.Triangulation = static["triang"]
    xdec, ydec = static["xdec"], static["ydec"]
    uvnorm = static["uvnorm"]
    obstacles_dec = build_obstacles(xdec, ydec)

    # Maschere punto/triangoli
    mask_fluid, tri_mask_fn = build_fluid_masks(xdec, ydec, obstacles_dec)

    # ==== prepara un Data "di base" per dedurre IN/EDGE/OUT dims ====
    _, _, U0, _, _ = results[0]
    data0 = build_frame_data(static, U0)
    IN_DIM = data0.x.size(1)
    EDGE_DIM = data0.edge_attr.size(1)
    OUT_DIM = data0.y.size(1)

    # ==== modello ====
    model = GraphAutoEncoder(
        in_ch=IN_DIM,
        edge_dim=EDGE_DIM,
        out_ch=OUT_DIM,
        hidden=HIDDEN,
        latent=LATENT,
        clusters_per_level=CLUSTERS_PER_LEVEL,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- maschera triangoli (una sola volta, triangolazione statica) ---
    tris = triang.triangles  # (nT, 3) indici di nodi
    tri_xc = (xdec[tris[:, 0]] + xdec[tris[:, 1]] + xdec[tris[:, 2]]) / 3.0
    tri_yc = (ydec[tris[:, 0]] + ydec[tris[:, 1]] + ydec[tris[:, 2]]) / 3.0
    tri_mask_centroid = tri_mask_fn(tri_xc, tri_yc).astype(bool)

    mf_np = mask_fluid.astype(bool)
    tri_touches_solid = ~mf_np[tris].all(axis=1)  # True se almeno un vertice è solido
    tri_mask = np.logical_or(tri_mask_centroid, tri_touches_solid)
    triang.set_mask(tri_mask)

    # --- accumula errori per tutti i frame (N nodi, NaN sui solidi) ---
    err_vals = []
    ground_truth_vals = []

    eps = 1e-9
    with torch.no_grad():
        for t, centers, U, neighbors, edge_index_raw in results:
            data_t = build_frame_data(static, U)
            y_hat = model(data_t)  # (N,2) pred NORMALIZZATI

            # decode pred con STESSE norme usate in training
            u_pred = uvnorm.decode(y_hat[:, 0], idx=0)
            v_pred = uvnorm.decode(y_hat[:, 1], idx=1)

            # ground truth in scala fisica (U già decoded)
            u_gt = torch.as_tensor(U[:, 0], dtype=torch.float32, device=device)
            v_gt = torch.as_tensor(U[:, 1], dtype=torch.float32, device=device)

            u_pred = torch.as_tensor(u_pred, dtype=torch.float32, device=device)
            v_pred = torch.as_tensor(v_pred, dtype=torch.float32, device=device)

            # DEBUG: range e bound
            max_gt_mag = torch.sqrt(u_gt**2 + v_gt**2).max().item()
            max_pred_mag = torch.sqrt(u_pred**2 + v_pred**2).max().item()
            bound_theoretical = max_pred_mag + max_gt_mag

            # errore per-nodo
            if error_mode == "u":
                e = (u_pred - u_gt).abs()
            elif error_mode == "v":
                e = (v_pred - v_gt).abs()
            elif error_mode == "relmag":
                mag_gt = torch.sqrt(u_gt**2 + v_gt**2)
                mag_pred = torch.sqrt(u_pred**2 + v_pred**2)
                e = (mag_pred - mag_gt).abs() / (mag_gt + eps)
            else:  # "mag"
                e = torch.sqrt((u_pred - u_gt) ** 2 + (v_pred - v_gt) ** 2)

            # metti NaN sui solidi (z deve restare lungo N)
            mf = torch.as_tensor(mask_fluid, device=device, dtype=torch.bool)
            e_full = e.clone()
            e_full[~mf] = float("nan")

            err_vals.append(e_full.detach().cpu().numpy())

            if SAVE_GROUND_TRUTH:
                if error_mode == "relmag":
                    ground_truth_vals.append(mag_gt.detach().cpu().numpy())
                elif error_mode in ("u", "v", "mag"):
                    # opzionale: salva magnitude GT per confronto visuale
                    mag_gt_tmp = torch.sqrt(u_gt**2 + v_gt**2)
                    ground_truth_vals.append(mag_gt_tmp.detach().cpu().numpy())

            # qualche print di debug
            idx_dbg = len(err_vals)
            if idx_dbg in (1, len(results) // 2, len(results)):
                print(
                    f"[t={t:.3f}] max|gt|≈{max_gt_mag:.3f}, max|pred|≈{max_pred_mag:.3f}, "
                    f"max error≈{np.nanmax(e_full.detach().cpu().numpy()):.3f}, bound≤{bound_theoretical:.3f}"
                )

    err_vals = np.array(err_vals)  # [T, N]
    if SAVE_GROUND_TRUTH:
        ground_truth_vals = np.array(ground_truth_vals)  # [T, N]

    # limiti colore robusti: usa solo valori finiti
    valid_vals = np.concatenate(
        [ev[np.isfinite(ev)] for ev in err_vals if np.any(np.isfinite(ev))]
    )
    vmin = float(np.min(valid_vals))
    vmax = float(np.percentile(valid_vals, 99.5))
    print("DEBUG true max error =", float(np.max(valid_vals)), " | vmax usato =", vmax)

    # --- figura & ostacoli ---
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

    # primo frame (usa masked array)
    z0 = np.ma.masked_invalid(err_vals[0])
    quad = ax.tricontourf(triang, z0, levels=LEVELS, cmap=CMAP, vmin=vmin, vmax=vmax)
    if SAVE_GROUND_TRUTH:
        z0_gt = np.ma.masked_invalid(ground_truth_vals[0])
        quad_gt = ax.tricontour(
            triang, z0_gt, levels=LEVELS, cmap="jet", linewidths=0.5
        )

    cb = fig.colorbar(quad, ax=ax, label=f"Error ({error_mode})")

    # --------------------------------- SNAPSHOTS STATICI (opzionale) -----------------------
    if PLOT_SOME_SNAPSHOTS:
        plt.close(fig)  # chiudi figura animazione per non sovrapporre

        snap_indices = [0, len(err_vals) // 2, len(err_vals) - 1]
        snap_labels = ["start", "mid", "end"]

        # 1) Salva ciascuno snapshot
        for idx, lbl in zip(snap_indices, snap_labels):
            fig_s, ax_s = plt.subplots(figsize=(5, 4))
            for obs in obstacles_dec:
                rect = Rectangle(
                    (obs["x0"], obs["y0"]),
                    obs["w"],
                    obs["h"],
                    color="k",
                    fill=True,
                    linewidth=0.0,
                    zorder=10,
                )
                ax_s.add_patch(rect)

            zf = np.ma.masked_invalid(err_vals[idx])
            cf = ax_s.tricontourf(
                triang, zf, levels=LEVELS, cmap=CMAP, vmin=vmin, vmax=vmax
            )
            ax_s.set_title(f"{error_mode} error (frame {idx})")
            ax_s.set_xlabel("x")
            ax_s.set_ylabel("y")
            ax_s.set_aspect("equal", adjustable="box")
            ax_s.set_xlim(np.min(xdec), np.max(xdec))
            ax_s.set_ylim(np.min(ydec), np.max(ydec))
            fig_s.colorbar(cf, ax=ax_s, label=f"Error ({error_mode})")
            out_path = f"plots/error_snapshot_{error_mode}_{lbl}.png"
            fig_s.savefig(out_path, dpi=300)
            plt.close(fig_s)
            print(f"✅ Salvato {out_path}")

        # 2) Triptych
        fig_c, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for ax_c, idx, lbl in zip(axes, snap_indices, snap_labels):
            for obs in obstacles_dec:
                rect = Rectangle(
                    (obs["x0"], obs["y0"]),
                    obs["w"],
                    obs["h"],
                    color="k",
                    fill=True,
                    linewidth=0.0,
                    zorder=10,
                )
                ax_c.add_patch(rect)
            zf = np.ma.masked_invalid(err_vals[idx])
            cf = ax_c.tricontourf(
                triang, zf, levels=LEVELS, cmap=CMAP, vmin=vmin, vmax=vmax
            )
            ax_c.set_title(lbl)
            ax_c.set_xlabel("x")
            ax_c.set_ylabel("y")
            ax_c.set_aspect("equal", adjustable="box")
            ax_c.set_xlim(np.min(xdec), np.max(xdec))
            ax_c.set_ylim(np.min(ydec), np.max(ydec))

        fig_c.colorbar(
            cf, ax=axes.ravel().tolist(), shrink=0.85, label=f"Error ({error_mode})"
        )
        combo_path = f"plots/error_snapshots_triptych_{error_mode}.png"
        fig_c.savefig(combo_path, dpi=300)
        plt.close(fig_c)
        print(f"✅ Salvato {combo_path}")

        plotLoss()
        return  # stop qui se fai solo snapshots
    # ---------------------------------------------------------------------------------------

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(np.min(xdec), np.max(xdec))
    ax.set_ylim(np.min(ydec), np.max(ydec))
    title = ax.set_title("t = 0.00")
    ax.set_aspect("equal", adjustable="box")

    # update: rimuove SOLO le collezioni del vecchio contourf e ridisegna
    def update(frame):
        print("Animating error frame", frame)
        nonlocal quad
        for coll in ax.collections:
            coll.remove()
        zf = np.ma.masked_invalid(err_vals[frame])
        quad = ax.tricontourf(
            triang, zf, levels=LEVELS, cmap=CMAP, vmin=vmin, vmax=vmax
        )
        t_now = results[frame][0]
        title.set_text(f"t = {t_now:.3f}")
        return []

    ani = FuncAnimation(
        fig, update, frames=len(results), interval=60, blit=False, repeat=True
    )

    # salvataggio (GIF con Pillow)
    ani.save(out_gif, writer=PillowWriter(fps=FPS))

    print("✅ Error animation saved as", out_gif)
    print(
        " Max and min errors over all frames:", np.nanmax(err_vals), np.nanmin(err_vals)
    )
    return ani


# =========================================================
if __name__ == "__main__":
    from trainAndPlotROM import *
    from testROMmod import *

    # main()
    # =========================================================
    animate_patch_time_series_gnn()
    # Se vuoi solo snapshots statici, lascia PLOT_SOME_SNAPSHOTS=True
    animate_error_time_series_gnn(
        out_gif=OUT_GIF, model_path=MODEL_PATH, error_mode=ERROR_MODE
    )
    plotLoss()
