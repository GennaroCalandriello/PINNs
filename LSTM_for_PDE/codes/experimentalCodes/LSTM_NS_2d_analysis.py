import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from LSTM_NS_2D import LSTM_PINN_NS2D, circle_mask
from temp2 import LSTM_PINN_NS2D, circle_mask, circle_bool
import hyperpar as hp

# 1) Hyperparameters & device
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = hp.HIDDEN_SIZE
num_layers  = hp.NUM_LAYERS
x_lb, x_ub  = hp.X_LB, hp.X_UB
y_lb, y_ub  = hp.Y_LB, hp.Y_UB
t_lb, t_ub  = hp.T_LB, hp.T_UB

Nx, Ny, Nt  = 200, 200, 40      # grid resolution
batch_size  = 10000              # evaluation batch size
xc, yc, r   = hp.cx, hp.cy, hp.r  # circle center and radius

if circle_bool == False: r = 0.0

# 2) Load trained model
def plotModel():
    model = LSTM_PINN_NS2D(input_size=3, hidden_size=hidden_size,
                           num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(
        "models/LSTM_NS2d_tg_cyl.pth", map_location=device))
    model.eval()

    # 3) Build (x,y,t) grid
    x = np.linspace(x_lb, x_ub, Nx)
    y = np.linspace(y_lb, y_ub, Ny)
    t = np.linspace(t_lb, t_ub, Nt)
    Xg, Yg, Tg = np.meshgrid(x, y, t, indexing='ij')  # [Nx,Ny,Nt]

    # 4) Fluid mask
    mask_xy = ~circle_mask(
        torch.tensor(Xg[:,:,0], dtype=torch.float32),
        torch.tensor(Yg[:,:,0], dtype=torch.float32), xc, yc, r
    ).numpy()
    mask_tiled = np.tile(mask_xy[:,:,None], (1,1,Nt))  # [Nx,Ny,Nt]
    flat_mask = mask_tiled.ravel()

    # 5) Flatten fluid coords
    x_flat = Xg.ravel()[flat_mask].reshape(-1,1)
    y_flat = Yg.ravel()[flat_mask].reshape(-1,1)
    t_flat = Tg.ravel()[flat_mask].reshape(-1,1)
    x_ft = torch.tensor(x_flat, dtype=torch.float32).to(device)
    y_ft = torch.tensor(y_flat, dtype=torch.float32).to(device)
    t_ft = torch.tensor(t_flat, dtype=torch.float32).to(device)

    # 6) Predict
    u_pred = np.full((Nx*Ny*Nt,1), np.nan, dtype=np.float32)
    v_pred = np.full((Nx*Ny*Nt,1), np.nan, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, x_ft.shape[0], batch_size):
            xb = x_ft[i:i+batch_size]; yb = y_ft[i:i+batch_size]; tb = t_ft[i:i+batch_size]
            u_b, v_b, _ = model(xb, yb, tb)
            idx = np.where(flat_mask)[0][i:i+batch_size]
            u_pred[idx,0] = u_b.cpu().numpy().ravel()
            v_pred[idx,0] = v_b.cpu().numpy().ravel()

    # reshape
    U = u_pred.reshape(Nx,Ny,Nt)
    V = v_pred.reshape(Nx,Ny,Nt)
    speed = np.sqrt(U**2 + V**2)
    speed[~mask_tiled] = np.nan

    # 7) Compute vorticity ω = ∂v/∂x - ∂u/∂y via finite differences
    dx = (x_ub - x_lb)/(Nx-1)
    dy = (y_ub - y_lb)/(Ny-1)
    # Preallocate
    omega = np.full((Nx,Ny,Nt), np.nan, dtype=np.float32)
    for k in range(Nt):
        # central differences, pad boundaries with nan
        dv_dx = np.full((Nx,Ny), np.nan)
        du_dy = np.full((Nx,Ny), np.nan)
        dv_dx[1:-1,:] = (V[2:,:,k] - V[:-2,:,k])/(2*dx)
        du_dy[:,1:-1] = (U[:,2:,k] - U[:,:-2,k])/(2*dy)
        omega[:,:,k] = dv_dx - du_dy
        omega[:,:,k][~mask_xy] = np.nan

    # 8) Animate speed and vorticity side by side
    vmin_s, vmax_s = np.nanmin(speed), np.nanmax(speed)*0.4
    vmin_w, vmax_w = np.nanmin(omega), np.nanmax(omega)
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    pcm1 = axes[0].pcolormesh(x, y, speed[:,:,0].T, shading='auto', cmap='viridis', vmin=vmin_s, vmax=vmax_s)
    axes[0].set_title('|velocity|'); axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    fig.colorbar(pcm1, ax=axes[0])
    pcm2 = axes[1].pcolormesh(x, y, omega[:,:,0].T, shading='auto', cmap='seismic', vmin=vmin_w, vmax=vmax_w)
    axes[1].set_title('vorticity ω'); axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    fig.colorbar(pcm2, ax=axes[1])

    def update(frame):
        pcm1.set_array(speed[:,:,frame].T.ravel())
        pcm2.set_array(omega[:,:,frame].T.ravel())
        fig.suptitle(f't = {t[frame]:.3f}')
        return pcm1, pcm2

    ani = animation.FuncAnimation(fig, update, frames=Nt, interval=100, blit=True)
    plt.tight_layout(); plt.show()

# 9) Plot training losses
def plotLoss():
    data = np.load("models/LSTM_NS2d_tg_cyl_loss.npy", allow_pickle=True).item()
    plt.figure(figsize=(6,5))
    for key in ['total_loss','pde_loss','bc_loss','ic_loss','obs_loss']:
        plt.plot(data[key], label=key)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.show()

if __name__ == "__main__":
    plotModel()
    plotLoss()
    
