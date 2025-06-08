import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import hyperpar as hp
from realBurgers2D import obstacle_, LSTM_PINN

# 1) Hyperparameters & device
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = hp.HIDDEN_SIZE
num_layers  = hp.NUM_LAYERS
x_lb, x_ub  = hp.X_LB, hp.X_UB
y_lb, y_ub  = hp.Y_LB, hp.Y_UB
t_lb, t_ub  = hp.T_LB, hp.T_UB
Nx, Ny, Nt  = 400, 400, 20   # grid resolution

# 2) Load trained model
model = LSTM_PINN(input_size=3, hidden_size=hidden_size, num_layers=num_layers).to(device)
model.load_state_dict(torch.load("model/lstm_pinn2D.pth", map_location=device))
model.eval()

# 3) Build (x,y,t) grid
x = np.linspace(x_lb, x_ub, Nx)
y = np.linspace(y_lb, y_ub, Ny)
t = np.linspace(t_lb, t_ub, Nt)
Xg, Yg, Tg = np.meshgrid(x, y, t, indexing='ij')  # [Nx,Ny,Nt]

# 4) Obstacle mask on XY plane: True = fluid region
o_mask = obstacle_(Xg[:,:,0], Yg[:,:,0], hp.cx, hp.cy, 0)
mask_xy = ~o_mask  # invert so fluid is outside the circle

# 5) Flatten to feed through the network
x_flat = torch.tensor(Xg.ravel(), dtype=torch.float32).view(-1,1).to(device)
y_flat = torch.tensor(Yg.ravel(), dtype=torch.float32).view(-1,1).to(device)
t_flat = torch.tensor(Tg.ravel(), dtype=torch.float32).view(-1,1).to(device)

# 6) Evaluate in batches
batch_size = 1000
uv_list = []
with torch.no_grad():
    for i in range(0, x_flat.shape[0], batch_size):
        xb = x_flat[i: i+batch_size]
        yb = y_flat[i: i+batch_size]
        tb = t_flat[i: i+batch_size]
        uv_batch = model(xb, yb, tb).cpu().numpy()  # [B,2]
        uv_list.append(uv_batch)

uv_flat = np.vstack(uv_list)  # shape (Nx*Ny*Nt, 2)

# 7) Reshape & mask
UV = uv_flat.reshape(Nx, Ny, Nt, 2)       # [Nx,Ny,Nt,2]
U = np.transpose(UV[:,:,:,0], (2,0,1))    # [Nt,Nx,Ny]
V = np.transpose(UV[:,:,:,1], (2,0,1))    # [Nt,Nx,Ny]

U_mask = np.where(mask_xy[None,:,:], U, np.nan)
V_mask = np.where(mask_xy[None,:,:], V, np.nan)

# 8) Compute magnitude of velocity
g_speed = np.sqrt(U_mask**2 + V_mask**2)
s_vmin, s_vmax = np.nanmin(g_speed), np.nanmax(g_speed)

# Precompute color scale limits for u, v
u_vmin, u_vmax = np.nanmin(U_mask), np.nanmax(U_mask)
v_vmin, v_vmax = np.nanmin(V_mask), np.nanmax(V_mask)

# --- Plot routines ---
def plot_field(field, x, y, t, t_idx=0, name='u', vmin=None, vmax=None):
    plt.figure(figsize=(6,4))
    pcm = plt.pcolormesh(x, y, field[t_idx].T,
                         shading='auto', cmap='viridis',
                         vmin=vmin, vmax=vmax)
    plt.colorbar(pcm, label=f'{name}(x,y)')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'{name}(x,y) at t = {t[t_idx]:.3f}')
    plt.tight_layout()
    plt.show()

# Animation helper
def animate(field, name, vmin, vmax, save_path):
    fig, ax = plt.subplots(figsize=(6,4))
    pcm = ax.pcolormesh(x, y, field[0].T,
                        shading='auto', cmap='viridis',
                        vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label=name)
    ax.set_xlabel('x'); ax.set_ylabel('y')

    def update(frame):
        pcm.set_array(field[frame].T.ravel())
        ax.set_title(f'{name} at t = {t[frame]:.3f}')
        return pcm,

    ani = animation.FuncAnimation(fig, update,
                                  frames=Nt, interval=100, blit=True)
    ani.save(save_path, writer='pillow', fps=10)
    plt.close(fig)

# --- Main ---
if __name__ == '__main__':
    # Initial condition plots
    plot_field(U_mask, x, y, t, t_idx=0, name='u', vmin=u_vmin, vmax=u_vmax)
    plot_field(V_mask, x, y, t, t_idx=0, name='v', vmin=v_vmin, vmax=v_vmax)
    plot_field(g_speed, x, y, t, t_idx=0, name='speed', vmin=s_vmin, vmax=s_vmax)

    # Animations
    # animate(U_mask, 'u', u_vmin, u_vmax, "burgers2d_u.gif")
    # animate(V_mask, 'v', v_vmin, v_vmax, "burgers2d_v.gif")
    animate(g_speed, 'speed', s_vmin, s_vmax, "burgers2d_speed.gif")

    # Plot losses
    try:
        loss_dict = np.load("loss/lossLSTM_2D.npy", allow_pickle=True).item()
        plt.figure(figsize=(6,4))
        plt.plot(loss_dict['total'], label='Total Loss')
        plt.plot(loss_dict['data'],  label='Data Loss')
        plt.plot(loss_dict['pde'],   label='PDE Loss')
        if 'obs' in loss_dict:
            plt.plot(loss_dict['obs'], label='Obs Loss')
        plt.xlabel('Epochs'); plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend(); plt.tight_layout(); plt.show()
    except FileNotFoundError:
        print("No loss file found.")

    print("♠♠ Finished ♠♠")
