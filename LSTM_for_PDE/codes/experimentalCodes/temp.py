import os
import torch
import torch.nn as nn
import numpy as np
import hyperpar as hp

# 0) Disable CuDNN for double‐backward
torch.backends.cudnn.enabled = False

"""This is the main code DO NOT TOUCH!! GODDAMN"""
want_obstacle = True
# 1) Hyperparameters & device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nu = hp.NU  # kinematic viscosity
rho = getattr(hp, 'RHO', 1.0)
hidden_size = hp.HIDDEN_SIZE
num_layers = hp.NUM_LAYERS
lr = hp.LR
epochs = 300
N_ic = hp.N_IC
N_colloc = hp.N_COLLOCATION
N_bc = hp.N_BC
N_obs = hp.N_OBSTACLE  # # obstacle‐condition points
input_size = 3  # (x,y,t)

# Domain bounds
x_lb, x_ub, y_lb, y_ub = hp.X_LB, hp.X_UB, hp.Y_LB, hp.Y_UB
t_lb, t_ub = hp.T_LB, hp.T_UB

# Cylinder parameters
cx, cy, r = hp.cx, hp.cy, hp.r

# 2) Taylor–Green analytic solution
def taylor_green(x, y, t):
    exp2 = torch.exp(-2 * nu * t)
    exp4 = torch.exp(-4 * nu * t)
    u = torch.cos(x) * torch.sin(y)*exp2
    v = -torch.sin(x) * torch.cos(y)*exp2
    p = -rho / 4 * (torch.cos(2*x) + torch.cos(2*y))*exp4
    return u, v, p

def circle_mask(x, y, xc, yc, r):
    """Circle mask for obstacle"""
    
    return (x-xc)**2 + (y-yc)**2 < r**2

# 3) Sampling
# 3.1) Initial condition: interior, t=0
x_ic = torch.rand(N_ic, 1)*(x_ub - x_lb) + x_lb
y_ic = torch.rand(N_ic, 1)*(y_ub - y_lb) + y_lb
t_ic = torch.zeros_like(x_ic)
u_ic, v_ic, _ = taylor_green(x_ic, y_ic, t_ic)


# 3.2) Collocation points (interior)
x_coll = torch.rand(N_colloc,1)*(x_ub-x_lb) + x_lb
y_coll = torch.rand(N_colloc,1)*(y_ub-y_lb) + y_lb
t_coll = torch.rand(N_colloc,1)*(t_ub-t_lb) + t_lb
# Remove points inside the circle
if want_obstacle:
    mask_coll = ~circle_mask(x_coll, y_coll, cx, cy, r)
    #zero around the circle, IC does respect the BC at time t=0
    x_coll = x_coll[mask_coll].reshape(-1,1)
    y_coll = y_coll[mask_coll].reshape(-1,1)
    t_coll = t_coll[mask_coll].reshape(-1,1)
    
    mask_ic = ~circle_mask(x_ic, y_ic, cx, cy, r)
    x_ic = x_ic[mask_ic].reshape(-1,1)
    y_ic = y_ic[mask_ic].reshape(-1,1)
    t_ic = t_ic[mask_ic].reshape(-1,1)
    u_ic = u_ic[mask_ic].reshape(-1,1)
    v_ic = v_ic[mask_ic].reshape(-1,1)

x_ic, y_ic, t_ic, u_ic, v_ic = [t.to(device) for t in (x_ic, y_ic, t_ic, u_ic, v_ic)]
x_coll, y_coll, t_coll = [t.to(device) for t in (x_coll, y_coll, t_coll)]

# 3.3) Boundary: four walls
def sample_boundary(N):
    # N per side
    xb = torch.linspace(x_lb, x_ub, N).unsqueeze(-1)
    yb = y_lb * torch.ones_like(xb)
    tb = torch.rand(N,1)*(t_ub-t_lb) + t_lb
    xt, yt, tt = xb, y_ub*torch.ones_like(xb), torch.rand(N,1)*(t_ub-t_lb) + t_lb
    yl = torch.linspace(y_lb, y_ub, N).unsqueeze(-1)
    xl, tl = x_lb * torch.ones_like(yl), torch.rand(N,1)*(t_ub-t_lb) + t_lb
    yr, xr, tr = yl, x_ub * torch.ones_like(yl), torch.rand(N,1)*(t_ub-t_lb) + t_lb
    x_b = torch.cat([xb, xt, xl, xr], dim=0)
    y_b = torch.cat([yb, yt, yl, yr], dim=0)
    t_b = torch.cat([tb, tt, tl, tr], dim=0)
    return x_b.to(device), y_b.to(device), t_b.to(device)

x_bc, y_bc, t_bc = sample_boundary(N_bc)
u_bc, v_bc, _ = taylor_green(x_bc, y_bc, t_bc)

# 3.4) Cylinder obstacle noslip
def sample_obstacle(N):
    theta = 2*torch.pi*torch.rand(N,1)
    x_o = cx + r * torch.cos(theta)
    y_o = cy + r * torch.sin(theta)
    t_o = torch.rand(N,1)*(t_ub-t_lb) + t_lb
    return x_o.to(device), y_o.to(device), t_o.to(device)

x_obs, y_obs, t_obs = sample_obstacle(N_obs)
# noslip target on obstacle surface (zero velocity)
u_obs = torch.zeros_like(x_obs)
v_obs = torch.zeros_like(y_obs)


# 4) Define LSTM‐PINN model
class LSTM_PINN_NS2D(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3),
            nn.Tanh()
        )
    def forward(self, x, y, t):
        seq = torch.cat([x,y,t], dim=-1).unsqueeze(1)
        h, _ = self.rnn(seq)
        uvp = self.fc(h[:, -1])
        return uvp[:,0:1], uvp[:,1:2], uvp[:,2:3]

model = LSTM_PINN_NS2D(input_size, hidden_size, num_layers).to(device)

# 5) PDE residuals
def NS_res(x,y,t):
    for g in (x,y,t): g.requires_grad_(True)
    u,v,p = model(x,y,t)
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=ones, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=ones, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=ones, create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=ones, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=ones, create_graph=True)[0]
    continuity = u_x + v_y
    pu = u_t + (u*u_x + v*u_y) + p_x - nu*(u_xx+u_yy)
    pv = v_t + (u*v_x + v*v_y) + p_y - nu*(v_xx+v_yy)
    return pu, pv, continuity

# 6) Loss
mse = nn.MSELoss()
def loss_functions():
    # IC loss vs analytic
    u0, v0, _ = model(x_ic, y_ic, t_ic)
    L_ic = mse(u0, u_ic) + mse(v0, v_ic)
    # PDE residual loss
    pu,pv,cont = NS_res(x_coll, y_coll, t_coll)
    L_pde = mse(pu, torch.zeros_like(pu)) + mse(pv, torch.zeros_like(pv)) + mse(cont, torch.zeros_like(cont))
    # Wall BC loss vs analytic
    u_b_pred, v_b_pred, _ = model(x_bc, y_bc, t_bc)
    L_bc = mse(u_b_pred, u_bc) + mse(v_b_pred, v_bc)
    # Obstacle noslip loss
    u_o_pred, v_o_pred, _ = model(x_obs, y_obs, t_obs)
    L_obs = mse(u_o_pred, u_obs) + mse(v_o_pred, v_obs)
    # Total loss
    L = hp.LAMBDA_DATA*L_ic + hp.LAMBDA_PDE*L_pde + hp.LAMBDA_BC*L_bc + hp.LAMBDA_OBS*L_obs
    return L, L_ic, L_pde, L_bc, L_obs

# 7) Training loop
def train_noBatches():
    history={
        "total_loss": [],
        "pde_loss": [],
        "bc_loss": [],
        "ic_loss": [],
        "obs_loss": []}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    for ep in range(1, epochs+1):
        opt.zero_grad()
        L, L_ic, L_pde, L_bc, L_obs = loss_functions()
        L.backward()
        opt.step()
        if ep % 10 == 0:
            print(f"Ep {ep}/{epochs}: L={L.item():.2e}, IC={L_ic.item():.2e}, PDE={L_pde.item():.2e}, BC={L_bc.item():.2e}, OBS={L_obs.item():.2e}")
        # registro le losses
        history["total_loss"].append(L.item())
        history["pde_loss"].append(L_pde.item())
        history["bc_loss"].append(L_bc.item())
        history["ic_loss"].append(L_ic.item())
        history["obs_loss"].append(L_obs.item())
    # save
    # save model and los
    if hp.SAVE_MODEL:
        try:
            os.remove("models/LSTM_NS2d_tg_cyl.pth")
        except OSError:
            pass
        torch.save(model.state_dict(), "models/LSTM_NS2d_tg_cyl.pth")
        print("Model saved to models/LSTM_NS2d_tg_cyl.pth")
    if hp.SAVE_LOSS:
        try:
            os.remove("models/LSTM_NS2d_tg_cyl_loss.npy")
        except OSError:
            pass
        np.save("models/LSTM_NS2d_tg_cyl_loss.npy", history)
        print("Loss saved to models/LSTM_NS2d_tg_cyl_loss.npy")

def train():
    history={
        "total_loss": [],
        "pde_loss": [],
        "bc_loss": [],
        "ic_loss": [],
        "obs_loss": []}
    batch_size = 128
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        idx_ic  = np.random.choice(N_ic, batch_size, replace=False)
        idx_coll = np.random.choice(N_colloc, batch_size, replace=False)
        idx_bc   = np.random.choice(N_bc*4, batch_size, replace=False)
        idx_obs  = np.random.choice(N_obs, batch_size, replace=False)
        
        # batch tensors xb, yb, tb are batches of x, y, t
        xb_ic, yb_ic, tb_ic = x_ic[idx_ic], y_ic[idx_ic], t_ic[idx_ic]
        ub_ic, vb_ic = u_ic[idx_ic], v_ic[idx_ic]
        xb_coll, yb_coll,tb_coll = x_coll[idx_coll], y_coll[idx_coll], t_coll[idx_coll]
        xb_bc, yb_bc, tb_bc, ub_bc, vb_bc = x_bc[idx_bc], y_bc[idx_bc], t_bc[idx_bc], u_bc[idx_bc], v_bc[idx_bc]
        xb_obs, yb_obs, tb_obs, ub_obs, vb_obs = x_obs[idx_obs], y_obs[idx_obs], t_obs[idx_obs], u_obs[idx_obs], v_obs[idx_obs]
        # losses
        opt.zero_grad()
        # IC
        u0, v0, _ = model(xb_ic, yb_ic, tb_ic)
        L_ic = mse(u0, ub_ic) +mse(v0, vb_ic)
        # PDE
        pu, pv, continuity = NS_res(xb_coll, yb_coll, tb_coll)
        L_pde = mse(pu, 0*pu) + mse(pv, 0*pv) +mse(continuity, 0*continuity)
        # BC walls
        ub_pred, vb_pred, _ = model(xb_bc, yb_bc, tb_bc)
        L_bc = mse(ub_pred, ub_bc) + mse(vb_pred, vb_bc)
        # BC obs
        uo_pred, vo_pred, _ = model(xb_obs, yb_obs, tb_obs)
        L_obs = mse(uo_pred, ub_obs) + mse(vo_pred, vb_obs)
        # total
        L = hp.LAMBDA_DATA*L_ic + hp.LAMBDA_PDE*L_pde + hp.LAMBDA_OBS*L_obs
        L.backward()
        opt.step()
        # registro le losses
        history["total_loss"].append(L.item())
        history["pde_loss"].append(L_pde.item())
        history["bc_loss"].append(L_bc.item())
        history["ic_loss"].append(L_ic.item())
        history["obs_loss"].append(L_obs.item())
        # print losses
        if ep % 10 == 0:
            print(f"Ep {ep}/{epochs}: L={L.item():.2e}, IC={L_ic.item():.2e}, PDE={L_pde.item():.2e}, BC={L_bc.item():.2e}, OBS={L_obs.item():.2e}")
    # save
    if hp.SAVE_MODEL:
        try:
            os.remove("models/LSTM_NS2d_tg_cyl.pth")
        except OSError:
            pass
        torch.save(model.state_dict(), "models/LSTM_NS2d_tg_cyl.pth")
        print("Model saved to models/LSTM_NS2d_tg_cyl.pth")
    if hp.SAVE_LOSS:
        try:
            os.remove("models/LSTM_NS2d_tg_cyl_loss.npy")
        except OSError:
            pass
        np.save("models/LSTM_NS2d_tg_cyl_loss.npy", history)
        print("Loss saved to models/LSTM_NS2d_tg_cyl_loss.npy")
if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs("loss", exist_ok=True)
    train_noBatches()
    # train()
