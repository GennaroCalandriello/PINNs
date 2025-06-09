import os
import torch
import torch.nn as nn
import numpy as np
import hyperpar as hp


# 0) Disable CuDNN for double-backward through LSTM
torch.backends.cudnn.enabled = False

# 1) Hyperparameters & device
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nu          = hp.NU
hidden_size = hp.HIDDEN_SIZE
num_layers  = hp.NUM_LAYERS
lr          = hp.LR
epochs      = 5000
N_data      = hp.N_IC
N_colloc    = hp.N_COLLOCATION
N_bc = hp.N_BC
N_obs = hp.N_OBSTACLE
lambda_data      = hp.LAMBDA_DATA
lambda_pde       = hp.LAMBDA_PDE
lambda_obs      = hp.LAMBDA_OBS  # add this in hyperpar.py

# Domain bounds
x_ub, x_lb =  hp.X_UB, hp.X_LB
y_ub, y_lb =  hp.Y_UB, hp.Y_LB
t_lb, t_ub =  hp.T_LB, hp.T_UB

#obstacle function
def obstacle_(x, y, xc=0.0, yc=0.0, r=1):
    return ((x - xc)**2 + (y - yc)**2) < r**2



#creo una maschera per le condizioni iniziali
# 

x_bc_l = torch.full((N_bc,1), x_lb, device=device)
print("some elements of x_bc_l:", x_bc_l[:5])
x_bc_u = torch.full((N_bc,1), x_ub, device=device)
y_bc_l = torch.full((N_bc,1), y_lb, device=device)
y_bc_u = torch.full((N_bc,1), y_ub, device=device)

t_bc = torch.rand(N_bc,1, device=device)*(t_ub-t_lb) + t_lb

x_bc_l, y_bc_l, t_bc = [t.to(device) for t in (x_bc_l, y_bc_l, t_bc)]


# 3) Sample collocation points in the full domain




def sampling_obs_(N):
    """Sto provando ad inserire delle condizioni no-slip per i walls dell'ostacolo"""
    x = (x_ub-x_lb) *torch.rand(N, 1)+x_lb
    y = (y_ub-y_lb)*torch.rand(N, 1)+y_lb
    mask_noslip=obstacle_(x, y)
    x_obs = x[mask_noslip].reshape(-1,1)
    y_obs = y[mask_noslip].reshape(-1,1)
    t_obs = (t_ub-t_lb)*torch.rand(x_obs.shape[0], 1)+t_lb
    return [f.to(device)for f in (x_obs, y_obs, t_obs)]

def sampling_obs_around(N):
    theta = np.linspace(0, 2 * np.pi, N)
    x_obs = hp.cx + hp.r * np.cos(theta)
    y_obs = hp.cy + hp.r * np.sin(theta)
    t_obs = (t_ub-t_lb)*torch.rand(x_obs.shape[0], 1)+t_lb
    x_obs, y_obs, t_obs = [torch.tensor(t, dtype=torch.float32).view(-1, 1).to(device) for t in (x_obs, y_obs, t_obs)]
    
    return [f.to(device) for f in (x_obs, y_obs, t_obs)]
# 4) LSTM-PINN model for (x,y,t)→u
class LSTM_PINN(nn.Module):
    def __init__(self, input_size=3, hidden_size=hidden_size, num_layers=num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # FC ora proietta su 2 valori (u, v)
        self.fc   = nn.Linear(hidden_size, 2)

    def forward(self, x, y, t):
        # cat → [B,3] → add seq-dim → [B,1,3]
        seq = torch.cat([x, y, t], dim=-1).unsqueeze(1)
        out, _ = self.lstm(seq)               # → [B,1,hidden]
        uv = self.fc(out[:, -1, :])           # → [B,2]
        u, v = uv[:, :1], uv[:, 1:]           # separo i due componenti se serve
        return uv  # oppure return u, v se preferisci due tensori distinti


model = LSTM_PINN(3, hidden_size, num_layers).to(device)
def sampling_data_():
    
    x_data = (x_ub - x_lb) * torch.rand(N_data,1) + x_lb
    y_data = (y_ub - y_lb) * torch.rand(N_data,1) + y_lb
    t_data = torch.zeros_like(x_data)
    mask_data = ~obstacle_(x_data, y_data)
    u_data = torch.sin(np.pi * x_data) * torch.sin(np.pi * y_data)
    v_data = torch.sin(np.pi * x_data) * torch.sin(np.pi * y_data)
    x_data = x_data[mask_data].reshape(-1,1)
    y_data = y_data[mask_data].reshape(-1,1)
    t_data = t_data[mask_data].reshape(-1,1)
    u_data = u_data[mask_data].reshape(-1, 1)
    v_data = v_data[mask_data].reshape(-1, 1)
    x_data, y_data, t_data, u_data, v_data = [t.to(device) for t in (x_data,y_data,t_data,u_data,v_data)]
    
    return x_data, y_data, t_data, u_data, v_data
# 5) PDE residual for 2D Burgers: u_t + u*(u_x+u_y) - ν*(u_xx+u_yy)
def pde_res(model):
    # 1) Genero i punti di collocazione
    x_coll = (x_ub - x_lb) * torch.rand(N_colloc,1) + x_lb
    y_coll = (y_ub - y_lb) * torch.rand(N_colloc,1) + y_lb
    t_coll = (t_ub - t_lb) * torch.rand(N_colloc,1) + t_lb
    x_coll, y_coll, t_coll = [t.to(device) for t in (x_coll,y_coll,t_coll)]
    
    # 2) (Opzionale) maschera sugli ostacoli
    mask = ~obstacle_(x_coll, y_coll)
    x_coll, y_coll, t_coll = [c[mask].reshape(-1,1) for c in (x_coll, y_coll, t_coll)]
    
    # 3) Richiedo gradienti per PDE
    x = x_coll.clone().detach().requires_grad_(True)
    y = y_coll.clone().detach().requires_grad_(True)
    t = t_coll.clone().detach().requires_grad_(True)
    
    # 4) Modello fornisce entrambi i componenti u, v
    uv = model(x, y, t)          # shape [N, 2]
    u = uv[:, 0:1]
    v = uv[:, 1:2]
    
    # 5) Derivate prime
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    # 6) Derivate seconde
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    
    # 7) Residui PDE per u e per v
    res_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)
    res_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)
    
    # 8) Ritorno entrambi i residui (shape [N,2])
    return res_u, res_v


# 6) Loss function
mse = nn.MSELoss()
def compute_losses():

    x_data, y_data, t_data, u_data, v_data = sampling_data_()
    
    uv_pred_data = model(x_data, y_data, t_data)   # [N_data,2]
    u_pred, v_pred = uv_pred_data[:,0:1], uv_pred_data[:,1:2]
    Ld = mse(u_pred, u_data) + mse(v_pred, v_data)
    # physics loss over collocation
    ru, rv  = pde_res(model)
    Lp = mse(ru, torch.zeros_like(ru))+ mse(rv, torch.zeros_like(rv))
    # no-slip loss over the obstacle
    x_obs, y_obs, t_obs = sampling_obs_(N_obs)
    x_obs_around, y_obs_around, t_obs_around = sampling_obs_around(N_obs)
    uv_obs_pred_around = model(x_obs_around, y_obs_around, t_obs_around)
    L_obs = mse(uv_obs_pred_around[:,0:1], torch.zeros_like(uv_obs_pred_around[:,0:1])) + \
                   mse(uv_obs_pred_around[:,1:2], torch.zeros_like(uv_obs_pred_around[:,1:2]))
    uv_obs_pred = model(x_obs, y_obs, t_obs)
    L_obs += mse(uv_obs_pred[:,0:1], torch.zeros_like(uv_obs_pred[:,0:1])) + \
            mse(uv_obs_pred[:,1:2], torch.zeros_like(uv_obs_pred[:,1:2]))
    # weighted losses over each component
    L_yb_l = mse(model(x_bc_l, y_bc_l, t_bc), torch.zeros_like(model(x_bc_l, y_bc_l, t_bc)))
    L_yb_u = mse(model(x_bc_u, y_bc_u, t_bc), torch.zeros_like(model(x_bc_u, y_bc_u, t_bc)))
    L_xb_l = mse(model(x_bc_l, y_bc_l, t_bc), torch.zeros_like(model(x_bc_l, y_bc_l, t_bc)))
    L_xb_u = mse(model(x_bc_u, y_bc_u, t_bc), torch.zeros_like(model(x_bc_u, y_bc_u, t_bc)))
    L_bc = (L_yb_l + L_yb_u + L_xb_l + L_xb_u) / 4.0  # average over 4 boundaries
    
    return Ld+Lp+2*L_obs, Ld, Lp, L_obs

# 7) Training loop
def train():
    history = {"total":[], "data":[], "pde":[], "obs":[]}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(1, epochs+1):
        optimizer.zero_grad()
        loss, Ld, Lp, Lo = compute_losses()
        loss.backward()
        optimizer.step()

        history["total"].append(loss.item())
        history["data"].append(Ld.item())
        history["pde"].append(Lp.item())
        history["obs"].append(Lo.item())

        if e % 20 == 0:
            print(f"Epoch {e:4d}/{epochs}  "
                  f"Loss={loss.item():.3e}  "
                  f"Data={Ld.item():.3e}  "
                  f"PDE ={Lp.item():.3e} "
                  f"Obs ={Lo.item():.3e} ")

    # ensure dirs
    os.makedirs("model", exist_ok=True)
    os.makedirs("loss",  exist_ok=True)
    # save
    torch.save(model.state_dict(), "model/lstm_pinn2D.pth")
    np.save("loss/lossLSTM_2D.npy", history)
    print("✔️  Training complete, model + history saved.")

if __name__ == "__main__":
    train()
    #plot this to compare: u_data = torch.sin(np.pi * x_data) * torch.sin(np.pi * y_data)
   #1 2 7 3 10 4 12 11 8
   # 5 6 9 13