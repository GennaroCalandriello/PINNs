# ns3d_analysis.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from LSTM_NS3D import LSTM_PINN_NS3D, sphere_noslip
import hyperpar as hp

# 1) Hyperparameters & device
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, L       = hp.HIDDEN_SIZE, hp.NUM_LAYERS
input_size = 4  # input size (x,y,z,t)
model_path = "models/LSTM_NS.pth"

# Domain bounds & grid resolution
x_lb, x_ub = -3.0, 3.0
y_lb, y_ub = x_lb, x_ub
z_lb, z_ub = x_lb, x_ub
t_lb, t_ub =  0.0, 1.0
Nx, Ny, Nz, Nt = 80, 80, 80, 10   # tune as you like

# 2) Load the trained 3D NS PINN
model = LSTM_PINN_NS3D(input_size=input_size, hidden_size=H, num_layers=L).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 3) Build the 4D grid
x = np.linspace(x_lb, x_ub, Nx)
y = np.linspace(y_lb, y_ub, Ny)
z = np.linspace(z_lb, z_ub, Nz)
t = np.linspace(t_lb, t_ub, Nt)
Xg, Yg, Zg, Tg = np.meshgrid(x, y, z, t, indexing='ij')  # [Nx,Ny,Nz,Nt]

# 4) Fluid mask: True outside the spherical obstacle
mask_xyz = ~sphere_noslip(
    torch.tensor(Xg[...,0], dtype=torch.float32),
    torch.tensor(Yg[...,0], dtype=torch.float32),
    torch.tensor(Zg[...,0], dtype=torch.float32),
).numpy()  # shape (Nx,Ny,Nz)
# mask_xyz =mask_xyz.transpose(2, 1, 0)
# 5) Flatten to feed the model
pts = Xg.ravel(), Yg.ravel(), Zg.ravel(), Tg.ravel()
x_flat = torch.tensor(pts[0], dtype=torch.float32).view(-1,1).to(device)
y_flat = torch.tensor(pts[1], dtype=torch.float32).view(-1,1).to(device)
z_flat = torch.tensor(pts[2], dtype=torch.float32).view(-1,1).to(device)
t_flat = torch.tensor(pts[3], dtype=torch.float32).view(-1,1).to(device)

# 6) Batch‐wise evaluation to avoid OOM
batch_size = hp.BATCH_SIZE
u_list = []
v_list = []
w_list = []
p_list = []

with torch.no_grad():
    for i in range(0, x_flat.shape[0], batch_size):
        xb = x_flat[i:i+batch_size]
        yb = y_flat[i:i+batch_size]
        zb = z_flat[i:i+batch_size]
        tb = t_flat[i:i+batch_size]
        u_b, v_b, w_b, p_b = model(xb,yb,zb,tb)
        u_list.append(u_b.cpu().numpy())
        v_list.append(v_b.cpu().numpy())
        w_list.append(w_b.cpu().numpy())
        p_list.append(p_b.cpu().numpy())

u_flat = np.vstack(u_list)
v_flat = np.vstack(v_list)
w_flat = np.vstack(w_list)
p_flat = np.vstack(p_list)

# 7) Reshape into (Nx,Ny,Nz,Nt) then transpose to (Nt,Nz,Ny,Nx)
U_xyz = u_flat.reshape(Nx, Ny, Nz, Nt)
U     = np.transpose(U_xyz, (3,2,1,0))  # U[t_idx, z_idx, y_idx, x_idx]
# apply mask: NaN inside obstacle sphere
U_masked = np.where(mask_xyz[None,:,:,:], U, np.nan)

# choose a fixed z‐slice to visualize (e.g. middle)
z_idx = Nz//2
vmin, vmax = np.nanmin(U_masked), np.nanmax(U_masked)

# --- Visualization routines ---

def plot_slice(t_idx=0):
    plt.figure(figsize=(6,4))
    pcm = plt.pcolormesh(
        x, y, U_masked[t_idx, z_idx].T,
        shading='auto', cmap='viridis',
        vmin=vmin, vmax=vmax
    )
    plt.colorbar(pcm, label='u(x,y,z={:.2f},t={:.2f})'.format(z[z_idx], t[t_idx]))
    plt.xlabel('x'); plt.ylabel('y')
    plt.title('Slice at z={:.2f}, t={:.2f}'.format(z[z_idx], t[t_idx]))
    plt.tight_layout(); plt.show()

def animate_slice(save_path="ns3d_slice.gif"):
    fig, ax = plt.subplots(figsize=(6,4))
    pcm = ax.pcolormesh(
        x, y, U_masked[0, z_idx].T,
        shading='auto', cmap='viridis',
        vmin=vmin, vmax=vmax
    )
    cbar = fig.colorbar(pcm, ax=ax, label='u')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    def update(frame):
        pcm.set_array(U_masked[frame, z_idx].T.ravel())
        ax.set_title(f'z={z[z_idx]:.2f}, t={t[frame]:.2f}')
        return pcm,

    ani = animation.FuncAnimation(
        fig, update, frames=Nt, interval=100, blit=True
    )
    ani.save(save_path, writer='pillow', fps=10)
    plt.close(fig)

#==========================P R E S S U R E===========================
#provo ad animare la pressione
P_xyz    = p_flat.reshape(Nx, Ny, Nz, Nt)                    # [Nx,Ny,Nz,Nt]
P        = np.transpose(P_xyz, (3,2,1,0))                    # [Nt,Nz,Ny,Nx]
P_masked = np.where(mask_xyz[None], P, np.nan)  
vmin_p, vmax_p = np.nanmin(P_masked), np.nanmax(P_masked)

def plot_p_slice(t_idx):
    plt.figure(figsize=(6,4))
    pcm = plt.pcolormesh(
        x, y,
        P_masked[t_idx, z_idx].T,
        shading='auto', cmap='coolwarm',
        vmin=vmin_p, vmax=vmax_p
    )
    plt.colorbar(pcm, label='p(x,y,z={:.2f},t={:.2f})'.format(z[z_idx], t[t_idx]))
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'Pressure slice at z={z[z_idx]:.2f}, t={t[t_idx]:.2f}')
    plt.tight_layout(); plt.show()


 
def animate_p_slice(save_path="pressure_slice.gif"):
    fig, ax = plt.subplots(figsize=(6,4))
    pcm = ax.pcolormesh(
        x, y,
        P_masked[0, z_idx].T,
        shading='auto', cmap='coolwarm',
        vmin=vmin_p, vmax=vmax_p
    )
    cbar = fig.colorbar(pcm, ax=ax, label='p')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    def update(frame):
        pcm.set_array(P_masked[frame, z_idx].T.ravel())
        ax.set_title(f'z={z[z_idx]:.2f}, t={t[frame]:.2f}')
        return pcm,

    ani = animation.FuncAnimation(
        fig, update, frames=Nt, interval=200, blit=True
    )
    ani.save(save_path, writer='pillow', fps=5)
    plt.close(fig)

# --- Main ---
if __name__ == "__main__":
    animate_p_slice("pressure_slice.gif")
    # plot_slice(0)
    # animate_slice("ns3d_slice.gif")
    # print("Done.")
    
