import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1) Import your trained PINN definition:
from NS_2D_circle2 import LSTMPINN  

# 2) Load model + weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMPINN().to(device)
model.load_state_dict(torch.load("models/LSTM_NS2d.pth", map_location=device))
model.eval()

# 3) Problem parameters (must match training!)
T_final = 4.0
cx, cy, r = 0, 0, 0.4  # circle center & radius

# 4) Grid & times
nx, ny = 300, 300
x = np.linspace(-1,1,nx)
y = np.linspace(-1,1,ny)
X, Y = np.meshgrid(x,y)
xy = np.stack([X.ravel(), Y.ravel()], axis=1)
nt = 50
times = np.linspace(0, T_final, nt)

# 5) Evaluator (exactly as before, just masking the interior)
def eval_uv_speed(t):
    # build input tensors *with* requires_grad so we can differentiate
    xx = torch.tensor(xy[:,0:1], dtype=torch.float32, device=device, requires_grad=True)
    yy = torch.tensor(xy[:,1:2], dtype=torch.float32, device=device, requires_grad=True)
    tt = torch.full_like(xx, float(t),                 device=device, requires_grad=True)

    inp = torch.cat([xx, yy, tt], dim=1).unsqueeze(1)    # (N,1,3)

    # 1) forward pass *WITHOUT* torch.no_grad()
    out = model(inp).squeeze(1)                         # (N,2): [ψ, p]
    psi = out[:, 0:1]

    # 2) compute u, v via autograd
    u = torch.autograd.grad(psi, yy,
                            grad_outputs=torch.ones_like(psi),
                            create_graph=False, retain_graph=True)[0]
    v = -torch.autograd.grad(psi, xx,
                             grad_outputs=torch.ones_like(psi),
                             create_graph=False, retain_graph=True)[0]

    # 3) detach & to‐numpy
    u = u.detach().cpu().numpy().flatten()
    v = v.detach().cpu().numpy().flatten()
    speed = np.sqrt(u**2 + v**2)

    # 4) mask inside the circle
    dist2 = (xy[:,0]-cx)**2 + (xy[:,1]-cy)**2
    mask = dist2 < r**2
    u[   mask] = np.nan
    v[   mask] = np.nan
    speed[mask] = np.nan

    return u.reshape(ny,nx), v.reshape(ny,nx), speed.reshape(ny,nx)


# 6) Build figure + add the circle patch once
fig, ax = plt.subplots(figsize=(6,5))
speed0 = eval_uv_speed(times[0])[2]
im = ax.imshow(speed0, origin='lower', extent=(0,1,0,1), cmap='viridis')
cb = fig.colorbar(im, ax=ax, label='speed |v|')
ax.set_xlabel('x'); ax.set_ylabel('y')
# draw the obstacle
# circle = Circle((cx,cy), r, facecolor='white', edgecolor='black', lw=1.5)
# ax.add_patch(circle)
title = ax.set_title(f"speed |v|(t={times[0]:.2f})")

def animate(i):
    ax.clear()
    u, v, sp = eval_uv_speed(times[i])
    im = ax.imshow(sp, origin='lower', extent=(0,1,0,1), cmap='viridis')
    # im = ax.pcolormesh(
    #     x, y,
    #     sp.T,
    #     shading='auto', cmap='viridis',
    #     vmin=0, vmax=1
    # )
    # redraw the circle
    circle = Circle((cx,cy), r, facecolor='white', edgecolor='black', lw=1.5)
    ax.add_patch(circle)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f"speed |v|(t={times[i]:.2f})")
    return [im]

ani = animation.FuncAnimation(fig, animate,
                              frames=nt, interval=200, blit=True)

plt.tight_layout()
plt.show()

# def animate_p_slice(save_path="pressure_slice.gif"):
#     fig, ax = plt.subplots(figsize=(6,4))
#     pcm = ax.pcolormesh(
#         x, y,
#         P_masked[0, z_idx].T,
#         shading='auto', cmap='coolwarm',
#         vmin=vmin_p, vmax=vmax_p
#     )
#     cbar = fig.colorbar(pcm, ax=ax, label='p')
#     ax.set_xlabel('x'); ax.set_ylabel('y')
