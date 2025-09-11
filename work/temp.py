import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

# Patch-1 (DATA): aggiunge SDF (distanza firmata dal bordo), indicatore (inside), normale

#  alle feature dei nodi; opzionale cp (edge feature orientata per la swirl).

# Patch-2 (POOL): rende il pooling boundary-aware (non butta i nodi vicini al muro) per TopK e ENAD.


# Patch-3 (LOSS): aggiunge pesi sulla loss vicino al muro + termini di BC (no-penetration/no-slip) in training.
# ---- SDF & indicator in spazio normalizzato ----
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

    return sdf.unsqueeze(1), inside_any.float().unsqueeze(1)  # [N,1],[N,1]


# ---- gradiente SDF → normale stimata ----
def estimate_normals_from_sdf(pos, edge_index, sdf):
    src, dst = edge_index
    rij = pos[dst] - pos[src]  # [E,2]
    lij2 = (rij**2).sum(dim=1, keepdim=True).clamp_min(1e-12)  # [E,1]
    dphi = sdf[dst] - sdf[src]  # [E,1]
    gij = (dphi / lij2) * rij  # [E,2]
    g = scatter_add(gij, dst, dim=0, dim_size=pos.size(0))  # [N,2]
    n = g / (g.norm(dim=1, keepdim=True).clamp_min(1e-9))
    n[n != n] = 0.0
    return n  # [N,2]


# ---- edge features: rel, dist, (opzionale) cp orientato per vortici ----
def build_edge_attr(centers_norm, U_norm, edge_index, use_cp=False):
    src, dst = edge_index
    rel = centers_norm[dst] - centers_norm[src]  # [E,2]
    dist = torch.norm(rel, dim=1, keepdim=True)  # [E,1]
    if not use_cp:
        return torch.cat([rel, dist], dim=1)  # edge_dim = 3
    dvel = U_norm[dst] - U_norm[src]  # [E,2]
    cp = rel[:, 0:1] * dvel[:, 1:2] - rel[:, 1:2] * dvel[:, 0:1]  # [E,1]
    return torch.cat([rel, dist, cp], dim=1)  # edge_dim = 4


# x = [centers_norm(2), U_norm(2), sdf(1), indicator(1), n_hat(2), u_n(1), u_t(1)]  => in_ch = 9
sdf, indicator = sdf_features_normalized(centers_norm, obstacles_norm)  # [N,1],[N,1]
n_hat = estimate_normals_from_sdf(centers_norm, edge_index, sdf)  # [N,2]
u_n = (U_norm * n_hat).sum(dim=1, keepdim=True)  # [N,1]
t_hat = torch.stack([-n_hat[:, 1], n_hat[:, 0]], dim=1)  # [N,2]
u_t = (U_norm * t_hat).sum(dim=1, keepdim=True)  # [N,1]
x = torch.cat([centers_norm, U_norm, sdf, indicator, n_hat, u_n, u_t], dim=1)

# edge_attr coerente con training:
edge_attr = build_edge_attr(centers_norm, U_norm, edge_index, use_cp=False)


logits = self.assign(x)  # [N,C]
SDF_COL = 4
dist_norm = x[:, SDF_COL : SDF_COL + 1]
bd_mask = dist_norm.squeeze(1) < 0.03
K = max(1, logits.size(1) // 8)  # ~12.5% cluster dedicati al boundary
logits[bd_mask, :K] += 8.0
S = F.softmax(logits, dim=-1)


center = batch.batch_size  # center nodes (NeighborLoader)
pred_u, pred_v = pred[:center, 0], pred[:center, 1]
gt_u, gt_v = batch.y[:center, 0], batch.y[:center, 1]

# estrai sdf e normale dal batch (adegua gli indici/nomi)
SDF_COL = 4
NORM_X_COL = 6
NORM_Y_COL = 7
sdf_c = batch.x[:center, SDF_COL : SDF_COL + 1]
n_hat_c = batch.x[:center, NORM_X_COL : NORM_X_COL + 2]  # [center,2]
t_hat_c = torch.stack([-n_hat_c[:, 1], n_hat_c[:, 0]], dim=1)  # [center,2]

# pesi boundary (γ e δ da regolare)
gamma, delta = 4.0, 0.03
w_boundary = 1.0 + (gamma - 1.0) * torch.exp(-(sdf_c.clamp_min(0.0) / delta))

# data loss (pesata sul boundary)
err = torch.stack([(pred_u - gt_u) ** 2, (pred_v - gt_v) ** 2], dim=1)  # [center,2]
loss_data = (w_boundary * err).mean()

# BC fisiche (no-penetration / no-slip in fascia)
u_pred = torch.stack([pred_u, pred_v], dim=1)
u_n = (u_pred * n_hat_c).sum(dim=1, keepdim=True)  # componente normale predetta
u_t = (u_pred * t_hat_c).sum(dim=1, keepdim=True)  # componente tangenziale predetta
mask_bc = (sdf_c < 0.03).float()  # fascia vicino al muro

lambda_np = 1.0  # no-penetration
lambda_ns = 0.2  # no-slip (più morbido, se flusso laminare)
loss_np = (mask_bc * (u_n**2)).mean()
loss_ns = (mask_bc * (u_t**2)).mean()

loss = loss_data + lambda_np * loss_np + lambda_ns * loss_ns
loss.backward()
