# burgers2d_analysis.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import hyperpar as hp
from LSTM_Burgers2D_BC import obstacle_, LSTM_PINN
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

# 4) Obstacle mask on XY plane: True = _fluid_ region = outside the circle
mask_xy = ~obstacle_(Xg[:,:,0], Yg[:,:,0], hp.cx, hp.cy, hp.r)  # invert so fluid is outside

# 5) Flatten to feed through the network
pts = Xg.ravel(), Yg.ravel(), Tg.ravel()
x_flat = torch.tensor(pts[0], dtype=torch.float32).view(-1,1).to(device)
y_flat = torch.tensor(pts[1], dtype=torch.float32).view(-1,1).to(device)
t_flat = torch.tensor(pts[2], dtype=torch.float32).view(-1,1).to(device)

# 6) Evaluate
model.eval()
batch_size = 1000    # tune this so it fits in your GPU memory
u_list = []

#Considerando che la gpu satura, ho inserito un batch_size in cui valutare il modello
# in modo da non saturare la memoria. Il sistema viene poi ricostruito
# concatenando i risultati di ogni batch
with torch.no_grad():
    for i in range(0, x_flat.shape[0], batch_size):
        xb = x_flat[i : i+batch_size]
        yb = y_flat[i : i+batch_size]
        tb = t_flat[i : i+batch_size]
        u_batch = model(xb, yb, tb).cpu().numpy()
        u_list.append(u_batch[0])

u_flat = np.vstack(u_list)   # shape (Nx*Ny*Nt, 1)

# 7) Reshape & apply mask: NaN _inside_ obstacle
U_xyz    = u_flat.reshape(Nx, Ny, Nt)             # [Nx,Ny,Nt]
U        = np.transpose(U_xyz, (2,0,1))           # [Nt,Nx,Ny]
U_masked = np.where(mask_xy[None,:,:], U, np.nan) # fluid outside circle

# --- Plotting routines ---

def plot_colormap(t_idx=0):
    plt.figure(figsize=(6,4))
    pcm = plt.pcolormesh(x, y, U_masked[t_idx].T,
                         shading='auto', cmap='viridis',
                         vmin=np.nanmin(U_masked), vmax=np.nanmax(U_masked))
    plt.colorbar(pcm, label='u(x,y)')
    # obstacle boundary
    theta = np.linspace(0, 2*np.pi, 200)
    # plt.plot(0.5*np.cos(theta),
    #          0.5*np.sin(theta),
    #          'k--', lw=1)
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'u(x,y) at t = {t[t_idx]:.3f}')
    plt.tight_layout()
    plt.show()

def plot_surface(t_idx=0):
    fig = plt.figure(figsize=(6,5))
    ax  = fig.add_subplot(111, projection='3d')
    X2d, Y2d = np.meshgrid(x, y, indexing='xy')
    Z = U_masked[t_idx].T
    surf = ax.plot_surface(X2d, Y2d, Z, cmap='viridis',
                           vmin=np.nanmin(U_masked), vmax=np.nanmax(U_masked))
    fig.colorbar(surf, ax=ax, label='u')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
    ax.set_title(f'u(x,y) at t = {t[t_idx]:.3f}')
    plt.tight_layout()
    plt.show()

# Precompute vmin/vmax so color scales don’t jump
vmin, vmax = np.nanmin(U_masked), np.nanmax(U_masked)

def animate_colormap(save_path="burgers2d_colormap.gif"):
    fig, ax = plt.subplots(figsize=(6,4))
    pcm = ax.pcolormesh(x, y, U_masked[0].T,
                        shading='auto', cmap='viridis',
                        vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(pcm, ax=ax, label='u')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    def update(frame):
        pcm.set_array(U_masked[frame].T.ravel())
        ax.set_title(f'u(x,y) at t = {t[frame]:.3f}')
        return pcm,

    ani = animation.FuncAnimation(fig, update,
                                  frames=Nt, interval=100, blit=True)
    # Pillow writer will always be available for GIF
    ani.save(save_path, writer='pillow', fps=10)
    plt.close(fig)


def animate_surface(save_path="burgers2d_surface.gif"):
    fig = plt.figure(figsize=(6,5))
    ax  = fig.add_subplot(111, projection='3d')
    X2d, Y2d = np.meshgrid(x, y, indexing='xy')

    def update(frame):
        ax.clear()  # clear the entire axes
        Z = U_masked[frame].T
        surf = ax.plot_surface(
            X2d, Y2d, Z,
            cmap='viridis', vmin=vmin, vmax=vmax,
            linewidth=0, antialiased=False
        )
        ax.set_zlim(vmin, vmax)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
        ax.set_title(f'u(x,y) at t = {t[frame]:.3f}')
        return surf,

    ani = animation.FuncAnimation(fig, update,
                                  frames=Nt, interval=100, blit=False)
    # Save as GIF with pillow (MP4 without ffmpeg will fail)
    #showing
    plt.show()
    ani.save(save_path, writer='pillow', fps=10)
    plt.close(fig)

def plot_losses():
    loss_dict = np.load("loss/lossLSTM_2D.npy", allow_pickle=True).item()
    plt.figure(figsize=(6,4))
    plt.plot(loss_dict['total'], label='Total Loss')
    plt.plot(loss_dict['data'], label='Data Loss')
    plt.plot(loss_dict['pde'],  label='PDE Loss')
    if 'obs' in loss_dict:
        plt.plot(loss_dict['obs'], label='Obs Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend(); plt.tight_layout(); plt.show()

# --- Main ---
if __name__ == "__main__":
    plot_colormap()
    animate_colormap("burgers2d_colormap.gif")
    animate_surface("burgers2d_surface.gif")
    # plot_losses()
    print("♠♠ Finished ♠♠")
    x = np.linspace(hp.X_LB, hp.X_UB, hp.NX)
    y = np.linspace(hp.Y_LB, hp.Y_UB, hp.NX)
    X, Y = np.meshgrid(x, y, indexing='xy')
    U = np.sin(np.pi * X) * np.sin(np.pi * Y)
    #plot the initial condition
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.pcolormesh(X, Y, U, shading='auto', cmap='viridis')
    plt.colorbar(label='u(x,y)')
    plt.title('Initial Condition at t=0')
    plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout()
    plt.show()
    