# from ns_GNN_cav2 import createGraphData, dataLoader, dataNormalizer  # usa i tuoi moduli
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.tri as mtri
# from matplotlib.animation import FuncAnimation, PillowWriter
# from testROMscatter import *
# from tqdm import tqdm

# # MODEL_PATH = "model/gnn_autoencoder.pth"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OUT_GIF = "plots/anim_gnn.gif"


# # ---------- Helper: build static from createGraphData ----------
# def build_static_from_graph():
#     """
#     Usa createGraphData() come unica fonte statica.
#     Recupera results + normalizzatori solo per l'andamento temporale e la decodifica plotting.
#     """
#     # 1) Grafo statico
#     data_static = createGraphData().to(device)
#     N = data_static.x.size(0)

#     # identifico canali extra (oltre U_norm[2])
#     # Assumo layout: x = [U_norm(2), ... extra ...]
#     assert data_static.x.size(1) >= 2, "x deve contenere almeno U_norm (2 canali)."
#     extra_feats = data_static.x[:, 2:]  # [N, F_extra] (F_extra può essere 0)

#     # 2) Serie temporale (centri fisici + U per ogni t) e normalizzatori
#     results, _ = dataLoader()
#     # results[k] = (t, centers, U, neighbors, edge_index_raw) o con V se hai aggiornato lo schema
#     t0, centers0, U0, *_rest = results[0]
#     _, _, _, _, _, xynorm, uvnorm, *maybe_volnorm = dataNormalizer(results)

#     # Triangolazione in coordinate fisiche (per contourf)
#     centers0 = np.asarray(centers0)
#     xdec, ydec = centers0[:, 0], centers0[:, 1]
#     triang = mtri.Triangulation(xdec, ydec)

#     static = {
#         "data_static": data_static,  # pos, edge, edge_attr, layout features
#         "extra_feats": extra_feats,  # feature statiche (rettangolo/mask/V ecc.)
#         "results": results,  # serie temporale con U per ogni t
#         "triang": triang,  # triangolazione plotting
#         "xynorm": xynorm,
#         "uvnorm": uvnorm,  # normalizzatori
#     }
#     return static


# # ---------- Helper: costruisci Data per un frame (riusa edge, pos, edge_attr) ----------
# def build_frame_data(static, U_frame):
#     ds = static["data_static"]
#     uvnorm = static["uvnorm"]

#     # encode U al volo
#     U_t = torch.as_tensor(U_frame, dtype=torch.float32, device=device)
#     U_norm = torch.as_tensor(uvnorm.encode(U_t), dtype=torch.float32, device=device)

#     # ricompongo x come in training: [U_norm(2), extra_feats]
#     if static["extra_feats"].numel() > 0:
#         x = torch.cat([U_norm, static["extra_feats"]], dim=1)
#     else:
#         x = U_norm

#     y = U_norm  # autoencoder
#     data_t = ds.clone()
#     data_t.x = x
#     data_t.y = y
#     return data_t


# # ---------- Animazione ----------
# def animate_autoencoder(out_gif=OUT_GIF):

#     # progress bar

#     static = build_static_from_graph()
#     ds = static["data_static"]
#     results = static["results"]
#     triang = static["triang"]
#     uvnorm = static["uvnorm"]

#     IN_DIM = ds.x.size(1)
#     EDGE_DIM = ds.edge_attr.size(1)
#     OUT_DIM = ds.y.size(1)

#     # Modello
#     model = GraphAutoEncoder(
#         in_ch=IN_DIM,
#         edge_dim=EDGE_DIM,
#         out_ch=OUT_DIM,
#         hidden=HIDDEN,
#         latent=LATENT,
#         clusters_per_level=CLUSTERS_PER_LEVEL,
#     ).to(device)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.eval()

#     # Precompute magnitudo predetta per scalare il colormap
#     mag_vals = []
#     with torch.no_grad():
#         for tup in results:
#             # supporta entrambi gli schemi (con o senza V nei results)
#             t, centers, U = tup[0], tup[1], tup[2]
#             data_t = build_frame_data(static, U)
#             y_hat = model(data_t)  # (N,2)
#             u_pred = uvnorm.decode(y_hat[:, 0], idx=0).cpu().numpy()
#             v_pred = uvnorm.decode(y_hat[:, 1], idx=1).cpu().numpy()
#             mag = np.linalg.norm(np.stack([u_pred, v_pred], axis=-1), axis=1)
#             mag_vals.append(mag)
#     mag_vals = np.asarray(mag_vals)
#     vmin, vmax = np.nanmin(mag_vals), np.nanmax(mag_vals)

#     # Plot
#     fig, ax = plt.subplots(figsize=(7, 6))
#     cntr = ax.tricontourf(
#         triang, mag_vals[0], levels=300, cmap="jet", vmin=vmin, vmax=vmax
#     )
#     cb = fig.colorbar(cntr, ax=ax, label="|u| (decoded)")
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_aspect("equal", adjustable="box")
#     title = ax.set_title(f"t = {results[0][0]:.3f}")

#     def update(frame):
#         # rimuovi vecchie collezioni contour
#         # print("Animating frame", frame)
#         for c in ax.collections:
#             c.remove()
#         cntr = ax.tricontourf(
#             triang, mag_vals[frame], levels=300, cmap="jet", vmin=vmin, vmax=vmax
#         )
#         title.set_text(f"t = {results[frame][0]:.3f}")
#         return []

#     ani = FuncAnimation(
#         fig, update, frames=len(results), interval=60, blit=False, repeat=True
#     )
#     # progress bar
#     total_frames = len(results)
#     pbar = tqdm(total=total_frames, desc="Rendering frames")

#     def _progress(i, n):
#         # i parte da 0; n è il numero totale di frame
#         pbar.update(1)
#         if i + 1 == n:
#             pbar.close()

#     try:
#         # Tentativo 1: usa il callback (se supportato mostrerà la barra)
#         ani.save(out_gif, writer="pillow", fps=20, progress_callback=_progress)
#     except Exception:
#         # Tentativo 2: fallback manuale sempre affidabile
#         if not pbar.closed:
#             pbar.close()
#         writer = PillowWriter(fps=20)
#         with writer.saving(fig, out_gif, dpi=100):
#             for i in tqdm(range(total_frames), desc="Rendering frames (fallback)"):
#                 update(i)
#                 writer.grab_frame()

#     print("✅ Animation saved as", out_gif)
#     return ani


# if __name__ == "__main__":
#     main()
#     animate_autoencoder()

# # ------------------------------O L D    C O D E ------------------------------------------------------------
# from testROMscatter import *
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.tri as mtri
# from matplotlib.animation import FuncAnimation
# from matplotlib.patches import Rectangle
# from torch_geometric.data import Data

# # ==== importo il  modello e le utility ====
# # from test import GraphAutoencoderDiffPool, geometryObject, dataLoader, dataNormalizer
# # Se già in namespace, puoi omettere.

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# geometry = "rectangle"  # "circle" or "rectangle"
# out_gif = "plots/anim_testROM.gif"

# # Rettangolo axis-aligned (angolo in basso-sx)
# RECT_X0 = 1000.0
# RECT_Y0 = None  # se None: usa min(ydec)
# RECT_W = 100.0
# RECT_H = 200.0


# def build_fluid_masks(x, y, obstacles):

#     inside_any = np.zeros_like(x, dtype=bool)

#     def inside_rect_axis_aligned(x, y, x0, y0, w, h):
#         return (x >= x0) & (x <= x0 + w) & (y >= y0) & (y <= y0 + h)

#     for obs in obstacles:
#         inside_any |= inside_rect_axis_aligned(
#             x, y, obs["x0"], obs["y0"], obs["w"], obs["h"]
#         )
#     mask_points_fluid = ~inside_any

#     def tri_mask_fn(tri_xc, tri_yc):
#         tri_inside = np.zeros_like(tri_xc, dtype=bool)
#         for obs in obstacles:

#             tri_inside |= inside_rect_axis_aligned(
#                 tri_xc, tri_yc, obs["x0"], obs["y0"], obs["w"], obs["h"]
#             )
#         return tri_inside

#     return mask_points_fluid, tri_mask_fn


# def build_static_graph_and_norms(
#     rect_center_phys=(1050, 100), rect_w_phys=100, rect_h_phys=200
# ):
#     """Precompute TUTTO ciò che è statico: pos, edge_index, edge_attr, rect features, triangulation, normalizers."""
#     # Carica dati grezzi e normalizzatori
#     results, _ = dataLoader()  # -> lista di (t, centers, U, neighbors, edge_index)
#     (
#         t0,
#         centers_all,
#         U_all,
#         neighbors_all,
#         vol,
#         edge_index_all,
#         xynorm,
#         uvnorm,
#         vnorm,
#     ) = dataNormalizer(results)

#     # Usiamo il primo frame per pos/edges (la mesh è fissa)
#     t0, centers0, U0, neighbors0, edge_index0, V = results[0]
#     centers0 = torch.tensor(centers0, dtype=torch.float32, device=device)

#     # Normalizza pos (come in createGraphData)
#     centers_norm = torch.as_tensor(
#         xynorm.encode(centers0), dtype=torch.float32, device=device
#     )

#     # Precompute edge_index dalla neighbors list (coerente con createGraphData)
#     edge_src, edge_dst = [], []
#     for i, nbs in enumerate(neighbors0):
#         for j in nbs:
#             if i != j:
#                 edge_src.append(i)
#                 edge_dst.append(j)
#     edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)

#     # Edge attr: [rel(2), dist(1), dir(2), invr(1)] = 6 dim (STATICHE)
#     rel = centers_norm[edge_index[1]] - centers_norm[edge_index[0]]
#     dist = torch.norm(rel, dim=1, keepdim=True)
#     dirn = rel / (dist + 1e-12)
#     invr = 1.0 / (dist + 1e-12)
#     edge_attr = torch.cat([rel, dist, dirn, invr], dim=1).to(torch.float32)
#     ea_norm = GaussianNormalizer(edge_attr)
#     ea_norm.cuda()
#     edge_attr = ea_norm.encode(edge_attr)

#     # Feature rettangolo (STATICHE, rettangolo fisso)
#     sigma_x = xynorm.std[0].item()
#     sigma_y = xynorm.std[1].item()
#     cx_phys, cy_phys = rect_center_phys
#     rect_center_norm = torch.tensor(
#         [
#             (cx_phys - xynorm.mean[0].item()) / sigma_x,
#             (cy_phys - xynorm.mean[1].item()) / sigma_y,
#         ],
#         dtype=torch.float32,
#         device=device,
#     )
#     rect_w_norm = rect_w_phys / sigma_x
#     rect_h_norm = rect_h_phys / sigma_y
#     rect_dist_norm, rect_mask = geometryObject(
#         centers_norm,
#         (rect_center_norm[0].item(), rect_center_norm[1].item()),
#         rect_w_norm,
#         rect_h_norm,
#     )  # shape [N,1] + [N,1]

#     # Triangolazione per plotting IN COORDINATE FISICHE (decodifica su assi reali)
#     centers0_np = centers0.detach().cpu().numpy()
#     xdec, ydec = centers0_np[:, 0], centers0_np[:, 1]
#     # triangolazione sui soli punti fluido
#     obstacles_dec = build_obstacles(xdec, ydec)
#     mask_fluid, tri_mask_fn = build_fluid_masks(xdec, ydec, obstacles_dec)
#     xdec_fluid = xdec[mask_fluid]
#     ydec_fluid = ydec[mask_fluid]
#     triang = mtri.Triangulation(xdec_fluid, ydec_fluid)
#     tri_pts = np.stack(
#         [xdec_fluid[triang.triangles], ydec_fluid[triang.triangles]], axis=-1
#     )
#     tri_centers_x = tri_pts[:, :, 0].mean(axis=1)
#     tri_centers_y = tri_pts[:, :, 1].mean(axis=1)
#     triang.set_mask(tri_mask_fn(tri_centers_x, tri_centers_y))
#     triang = mtri.Triangulation(
#         xdec, ydec
#     )  # nessuna maschera, o aggiungila qui se vuoi

#     static = {
#         "centers_norm": centers_norm,
#         "edge_index": edge_index,
#         "edge_attr": edge_attr,
#         "rect_dist_norm": rect_dist_norm,
#         "rect_mask": rect_mask,
#         "triang": triang,
#         "xdec": xdec,
#         "ydec": ydec,
#         "xynorm": xynorm,
#         "uvnorm": uvnorm,
#         "vnorm": vnorm,
#         "results": results,
#     }
#     return static


# def build_obstacles(decoded_x, decoded_y):
#     obstacles = []
#     y0 = float(np.min(decoded_y)) if RECT_Y0 is None else float(RECT_Y0)
#     obstacles.append(
#         {
#             "type": "rect",
#             "x0": float(RECT_X0),
#             "y0": y0,
#             "w": float(RECT_W),
#             "h": float(RECT_H),
#         }
#     )

#     return obstacles


# def build_frame_data(static, U_frame, V_t):
#     """Costruisce Data per un frame: aggiorna SOLO U_norm (input x e target y)."""
#     centers_norm = static["centers_norm"]
#     edge_index = static["edge_index"]
#     edge_attr = static["edge_attr"]
#     rect_dist_norm = static["rect_dist_norm"]
#     rect_mask = static["rect_mask"]
#     uvnorm = static["uvnorm"]
#     vnorm = static["vnorm"]

#     # Normalizza U frame-by-frame
#     U_t = torch.tensor(U_frame, dtype=torch.float32, device=device)
#     V_t = torch.tensor(V_t, dtype=torch.float32, device=device)
#     U_norm = torch.as_tensor(uvnorm.encode(U_t), dtype=torch.float32, device=device)
#     Volnorm = torch.as_tensor(
#         vnorm.encode(V_t), dtype=torch.float32, device=device
#     ).view(-1, 1)

#     # Node features come in training: [U_norm, rect_dist_norm, rect_mask]
#     x = torch.cat([U_norm, rect_dist_norm, rect_mask, Volnorm], dim=1)

#     # Target = U_norm (ricostruzione)
#     y = U_norm
#     data_t = Data(
#         x=x, y=y, pos=centers_norm, edge_index=edge_index, edge_attr=edge_attr
#     ).to(device)
#     # (opzionale) sanificazione tipi
#     data_t.x = data_t.x.float().contiguous()
#     data_t.y = data_t.y.float().contiguous()
#     data_t.edge_attr = data_t.edge_attr.float().contiguous()
#     data_t.edge_index = data_t.edge_index.long().contiguous()
#     return data_t


# def animate_patch_time_series_gnn(
#     out_gif=out_gif,
# ):
#     # ==== precompute static ====
#     static = build_static_graph_and_norms(
#         rect_center_phys=(1050, 100), rect_w_phys=100, rect_h_phys=200
#     )
#     results = static["results"]
#     triang = static["triang"]
#     xdec, ydec = static["xdec"], static["ydec"]
#     uvnorm = static["uvnorm"]
#     obstacles_dec = build_obstacles(xdec, ydec)
#     mask_fluid, tri_mask_fn = build_fluid_masks(xdec, ydec, obstacles_dec)
#     # ==== prepara un Data "di base" per dedurre IN/EDGE/OUT dims ====
#     _, _, U0, _, _, V = results[0]
#     data0 = build_frame_data(static, U0, V)
#     IN_DIM = data0.x.size(1)  # 4 (U_norm(2)+rect 2)
#     EDGE_DIM = data0.edge_attr.size(1)  # 6
#     OUT_DIM = data0.y.size(1)  # 2 (u,v)

#     # ==== modello ====
#     model = GraphAutoEncoder(
#         in_ch=IN_DIM,
#         edge_dim=EDGE_DIM,
#         out_ch=OUT_DIM,
#         hidden=HIDDEN,
#         latent=LATENT,
#         clusters_per_level=CLUSTERS_PER_LEVEL,
#     ).to(device)

#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.eval()

#     # ==== precomputo magnitudo predetta per ogni frame ====
#     mag_vals = []
#     with torch.no_grad():
#         for t, centers, U, neighbors, edge_index_raw, _ in results:
#             data_t = build_frame_data(static, U)
#             y_hat = model(data_t)  # (N,2)
#             # decode in unità fisiche per plotting
#             u_pred = uvnorm.decode(y_hat[:, 0], idx=0).cpu().numpy()
#             v_pred = uvnorm.decode(y_hat[:, 1], idx=1).cpu().numpy()
#             mag = np.linalg.norm(np.stack([u_pred, v_pred], axis=-1), axis=1)
#             mag_vals.append(mag)
#     mag_vals = np.array(mag_vals)
#     vmin, vmax = np.nanmin(mag_vals), np.nanmax(mag_vals)

#     # ==== setup figura ====
#     fig, ax = plt.subplots(figsize=(7, 6))
#     for obs in obstacles_dec:
#         rectangle = Rectangle(
#             (obs["x0"], obs["y0"]),
#             obs["w"],
#             obs["h"],
#             color="k",
#             fill=True,
#             linewidth=0.0,
#             linestyle="",
#             zorder=10,
#         )
#         ax.add_patch(rectangle)

#     cntr = ax.tricontourf(
#         triang, mag_vals[0], levels=300, cmap="jet", vmin=vmin, vmax=vmax
#     )
#     cb = fig.colorbar(cntr, ax=ax, label="|u| (decoded)")
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.axis("equal")
#     ax.set_aspect("equal", adjustable="box")

#     title = ax.set_title("t = 0.00")

#     def update(frame):
#         # pulisci vecchie collections del contour
#         print("Animating frame", frame)
#         for c in ax.collections:
#             c.remove()
#         cntr = ax.tricontourf(
#             triang, mag_vals[frame], levels=300, cmap="jet", vmin=vmin, vmax=vmax
#         )
#         t_now = results[frame][0]
#         title.set_text(f"t = {t_now:.3f}")
#         ax.add_patch(rectangle)
#         return []

#     ani = FuncAnimation(
#         fig, update, frames=len(results), interval=60, blit=False, repeat=True
#     )
#     ani.save(out_gif, writer="pillow", fps=20)
#     print("✅ Animation saved as", out_gif)
#     return ani


# def plotLoss():
#     loss = np.loadtxt(LOSS_PATH)
#     xep = np.linspace(1, EPOCHS, len(loss))
#     plt.figure(figsize=(6, 4))

#     plt.plot(loss)
#     plt.yscale("log")
#     plt.xlabel("Epoch")
#     plt.ylabel("MSE Loss")
#     plt.show()


# if __name__ == "__main__":
#     # main()

#     animate_patch_time_series_gnn()
#     plotLoss()
from testROMscatter import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
geometry = "rectangle"
out_gif = "plots/anim_testROM.gif"

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
    """
    Precompute STATICI: pos_norm, edge_index, edge_attr_norm, feature rettangolo, triangolazione, normalizzatori.
    """
    results, _ = dataLoader()  # lista: (t, centers, U, neighbors, edge_index, V)

    # Normalizzatori e dati (DEVONO essere coerenti col training)
    (
        t0,
        centers_all,
        U_all,
        neighbors_all,
        vol_all,
        edge_index_all,
        xynorm,
        uvnorm,
        vnorm,
    ) = dataNormalizer(results)

    # Primo frame per geometria
    t0, centers0, U0, neighbors0, edge_index0, V0 = results[0]
    centers0 = torch.tensor(centers0, dtype=torch.float32, device=device)

    # pos normalizzate
    centers_norm = torch.as_tensor(
        xynorm.encode(centers0), dtype=torch.float32, device=device
    )

    # edge_index da neighbors (coerente con training)
    edge_src, edge_dst = [], []
    for i, nbs in enumerate(neighbors0):
        for j in nbs:
            if i != j:
                edge_src.append(i)
                edge_dst.append(j)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)

    # edge_attr: [rel(2), dist(1), dir(2), invr(1)] = 6 → NORMALIZZO (come in training)
    rel = centers_norm[edge_index[1]] - centers_norm[edge_index[0]]
    dist = torch.norm(rel, dim=1, keepdim=True)
    dirn = rel / (dist + 1e-12)
    invr = 1.0 / (dist + 1e-12)
    edge_attr = torch.cat([rel, dist, dirn], dim=1).to(torch.float32)
    ea_norm = GaussianNormalizer(edge_attr)
    ea_norm.cuda()
    edge_attr = ea_norm.encode(edge_attr)

    # feature rettangolo statiche (in spazio normalizzato)
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
    )  # [N,1], [N,1]

    # triangolazione in coordinate FISICHE
    centers0_np = centers0.detach().cpu().numpy()
    xdec, ydec = centers0_np[:, 0], centers0_np[:, 1]
    obstacles_dec = build_obstacles(xdec, ydec)
    mask_fluid, tri_mask_fn = build_fluid_masks(xdec, ydec, obstacles_dec)
    # se vuoi mascherare i triangoli: crea triang con soli punti fluid e setta mask; qui mantengo tutto:
    triang = mtri.Triangulation(xdec, ydec)

    static = {
        "centers_norm": centers_norm,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "rect_dist_norm": rect_dist_norm,
        "rect_mask": rect_mask.float(),
        "triang": triang,
        "xdec": xdec,
        "ydec": ydec,
        "xynorm": xynorm,
        "uvnorm": uvnorm,
        "vnorm": vnorm,
        "results": results,
        "obstacles": obstacles_dec,
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


def build_frame_data(static, U_frame, V_frame):
    """Costruisce Data per un frame: aggiorna SOLO U_norm e V_norm (input x e target y)."""
    centers_norm = static["centers_norm"]
    edge_index = static["edge_index"]
    edge_attr = static["edge_attr"]
    rect_dist_norm = static["rect_dist_norm"]
    rect_mask = static["rect_mask"]
    uvnorm = static["uvnorm"]
    vnorm = static["vnorm"]

    # Normalizza U e V frame-by-frame
    U_t = torch.tensor(U_frame, dtype=torch.float32, device=device)  # [N,2]
    V_t = torch.tensor(V_frame, dtype=torch.float32, device=device).view(-1, 1)  # [N,1]
    U_norm = torch.as_tensor(uvnorm.encode(U_t), dtype=torch.float32, device=device)
    V_norm = torch.as_tensor(vnorm.encode(V_t), dtype=torch.float32, device=device)

    # Node features come in training: [U_norm, rect_dist_norm, rect_mask, V_norm]
    x = torch.cat([U_norm, rect_dist_norm, rect_mask, V_norm], dim=1)

    # Target = U_norm (ricostruzione)
    y = U_norm
    data_t = Data(
        x=x, y=y, pos=centers_norm, edge_index=edge_index, edge_attr=edge_attr
    ).to(device)

    # sanificazione tipi
    data_t.x = data_t.x.float().contiguous()
    data_t.y = data_t.y.float().contiguous()
    data_t.edge_attr = data_t.edge_attr.float().contiguous()
    data_t.edge_index = data_t.edge_index.long().contiguous()
    return data_t


def animate_patch_time_series_gnn(out_gif=out_gif):
    # ==== precompute static ====
    static = build_static_graph_and_norms(
        rect_center_phys=(1050, 100), rect_w_phys=100, rect_h_phys=200
    )
    results = static["results"]
    triang = static["triang"]
    uvnorm = static["uvnorm"]

    # ==== Data "base" per dedurre IN/EDGE/OUT dims ====
    _, _, U0, _, _, V0 = results[0]
    data0 = build_frame_data(static, U0, V0)
    IN_DIM = data0.x.size(1)  # es. 5: U(2)+rect(2)+V(1)
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

    # ==== precompute magnitudo predetta per vmin/vmax ====
    mag_vals = []
    with torch.no_grad():
        for t, centers, U, neighbors, edge_index_raw, V in results:
            data_t = build_frame_data(static, U, V)
            y_hat = model(data_t)  # (N,2)
            u_pred = uvnorm.decode(y_hat[:, 0], idx=0).cpu().numpy()
            v_pred = uvnorm.decode(y_hat[:, 1], idx=1).cpu().numpy()
            mag = np.linalg.norm(np.stack([u_pred, v_pred], axis=-1), axis=1)
            mag_vals.append(mag)
    mag_vals = np.array(mag_vals)
    vmin, vmax = np.nanmin(mag_vals), np.nanmax(mag_vals)

    # ==== setup figura ====
    fig, ax = plt.subplots(figsize=(7, 6))

    # disegna ostacoli UNA volta (statici)
    for obs in static["obstacles"]:
        ax.add_patch(
            Rectangle(
                (obs["x0"], obs["y0"]),
                obs["w"],
                obs["h"],
                color="k",
                fill=True,
                linewidth=0.0,
                zorder=10,
            )
        )

    cntr = ax.tricontourf(
        triang, mag_vals[0], levels=300, cmap="jet", vmin=vmin, vmax=vmax
    )
    cb = fig.colorbar(cntr, ax=ax, label="|u| (decoded)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.set_aspect("equal", adjustable="box")
    title = ax.set_title(f"t = {results[0][0]:.3f}")

    # handle mutabile per rimuovere SOLO le collections del contour
    cntr_ref = [cntr]

    # def update(frame):
    #     # rimuovi SOLO le collections del contour precedente
    #     for coll in cntr_ref[0].collections:
    #         coll.remove()
    #     cntr_new = ax.tricontourf(
    #         triang, mag_vals[frame], levels=300, cmap="jet", vmin=vmin, vmax=vmax
    #     )
    #     cntr_ref[0] = cntr_new
    #     cb.update_normal(cntr_new)
    #     title.set_text(f"t = {results[frame][0]:.3f}")
    #     return []

    def update(frame):
        #         # pulisci vecchie collections del contour
        print("Animating frame", frame)
        for c in ax.collections:
            c.remove()
        cntr = ax.tricontourf(
            triang, mag_vals[frame], levels=300, cmap="jet", vmin=vmin, vmax=vmax
        )
        t_now = results[frame][0]
        title.set_text(f"t = {t_now:.3f}")
        # ax.add_patch(rectangle)
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
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    animate_patch_time_series_gnn()
    plotLoss()
