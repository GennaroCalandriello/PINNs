import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1) Riproponi la definizione del modello
class LSTMPINN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, num_layers=2, out_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h)

# 2) Carica il modello allenato
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMPINN().to(device)
model.load_state_dict(torch.load("models/LSTM_PINN_NS2D.pth", map_location=device))
model.eval()

# 3) Griglia spaziale e tempi
nx, ny = 50, 50
x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)
X, Y = np.meshgrid(x,y)
xy_flat = np.stack([X.ravel(), Y.ravel()], axis=1)

nt = 100
T_final = 1.0
times = np.linspace(0, T_final, nt)

# 4) Funzione per valutare (U,V,ω) in un dato t
def eval_uvomega(t_scalar):
    t_col = np.full((nx*ny,1), t_scalar, dtype=np.float32)
    inp = np.concatenate([xy_flat.astype(np.float32), t_col], axis=1)
    with torch.no_grad():
        T = torch.from_numpy(inp).to(device).unsqueeze(1)  # (N,1,3)
        uvp = model(T).squeeze(1).cpu().numpy()           # (N,3)
    U = uvp[:,0].reshape((ny,nx))
    V = uvp[:,1].reshape((ny,nx))
    dx = x[1]-x[0]; dy = y[1]-y[0]
    dvdx = np.gradient(V, axis=1) / dx
    dudy = np.gradient(U, axis=0) / dy
    omega = dvdx - dudy
    return U, V, omega

# 5) Setup figura e animazione
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# frame iniziale
U0, V0, ω0 = eval_uvomega(times[0])
speed0 = np.sqrt(U0**2 + V0**2)

im1 = ax1.imshow(ω0, origin='lower', extent=(0,1,0,1), cmap='RdBu')
cb1 = fig.colorbar(im1, ax=ax1, label='Vorticità')
title1 = ax1.set_title(f"$\\omega$, t={times[0]:.2f}")

im2 = ax2.imshow(speed0, origin='lower', extent=(0,1,0,1))
cb2 = fig.colorbar(im2, ax=ax2, label='|u,v|')
title2 = ax2.set_title(f"|v|, t={times[0]:.2f}")

for ax in (ax1, ax2):
    ax.set_xlabel('x'); ax.set_ylabel('y')

def update(frame):
    t = times[frame]
    U, V, ω = eval_uvomega(t)
    speed = np.sqrt(U**2 + V**2)
    im1.set_data(ω)
    im2.set_data(speed)
    title1.set_text(f"$\\omega$, t={t:.2f}")
    title2.set_text(f"|v|, t={t:.2f}")
    return im1, im2, title1, title2

ani = FuncAnimation(fig, update, frames=nt, interval=50, blit=False)

# per salvare:
# ani.save("ns2d_vorticita_velocita.mp4", writer="ffmpeg", dpi=150)
# ani.save("ns2d_vorticita_velocita.gif", writer="pillow", fps=20)

plt.tight_layout()
plt.show()
