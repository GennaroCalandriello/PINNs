import os
import torch
import torch.nn as nn
import numpy as np
import hyperpar as hp

# 0) Disable CuDNN for double‐backward
torch.backends.cudnn.enabled = False

"""This is the main code DO NOT TOUCH!! GODDAMN"""

# 1) Hyperparameters & device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nu = hp.NU  # kinematic viscosity
rho = getattr(hp, 'RHO', 1.0)
hidden_size = hp.HIDDEN_SIZE
num_layers = hp.NUM_LAYERS
lr = hp.LR
epochs = 200
N_ic = hp.N_IC
N_colloc = hp.N_COLLOCATION
N_bc = hp.N_BC
N_obs = hp.N_OBSTACLE  # # obstacle‐condition points
input_size = 3  # (x,y,t)

# Domain bounds
x_lb, x_ub, y_lb, y_ub = hp.X_LB, hp.X_UB, hp.Y_LB, hp.Y_UB
t_lb, t_ub = hp.T_LB, hp.T_UB
circle_bool = True
# Cylinder parameters
cx, cy, r = hp.cx, hp.cy, hp.r
    
def circle_mask(x, y, xc, yc, r):
    """Circle mask for obstacle"""
    
    return (x-xc)**2 + (y-yc)**2 < r**2

def sample_collocation(N):
    x = torch.rand(N, 1)*(x_ub-x_lb) + x_lb
    y = torch.rand(N, 1)*(y_ub-y_lb) + y_lb
    t = torch.rand(N, 1)*(t_ub-t_lb) + t_lb
    if circle_bool:
        mask = ~circle_mask(x, y, cx, cy, r)
        x = x[mask].reshape(-1, 1)
        y = y[mask].reshape(-1, 1)
        t = t[mask].reshape(-1, 1)
    return x.to(device), y.to(device), t.to(device)

def sample_initial_conditions(N):
    x = torch.rand(N, 1)*(x_ub-x_lb) + x_lb
    y = torch.rand(N, 1)*(y_ub-y_lb) + y_lb
    t = torch.zeros(N, 1, device=device)
    #gaussian U centered in the domain
    u = hp.A*torch.exp(-((x-hp.x_gauss)**2 + (y-hp.y_gauss)**2)/hp.sigma**2)
    #gaussian V centered in the domain
    v = hp.A*torch.exp(-((x-hp.x_gauss)**2 + (y-hp.y_gauss)**2)/hp.sigma**2)
    # v= torch.zeros_like(u)
    
    # if circle_bool:
    #     mask = ~circle_mask(x, y, cx, cy, r)
    #     x = x[mask].reshape(-1, 1)
    #     y = y[mask].reshape(-1, 1)
    #     t = t[mask].reshape(-1, 1)
    #     u = u[mask].reshape(-1, 1)
    #     v = v[mask].reshape(-1, 1)
        
    return x.to(device), y.to(device), t.to(device), u.to(device), v.to(device)

def sample_initial_conditions2(x, y):
    #gaussian U centered in the domain
    u = hp.A*torch.exp(-((x-hp.x_gauss)**2 + (y-hp.y_gauss)**2)/hp.sigma**2)
    #gaussian V centered in the domain
    v = hp.A*torch.exp(-((x-hp.x_gauss)**2 + (y-hp.y_gauss)**2)/hp.sigma**2)
    return u, v

def sample_circle(N):
    theta = torch.rand(N, 1)*(2*np.pi)
    r = torch.sqrt(torch.rand(N, 1)) * (x_ub-x_lb)/2
    x = cx + r * torch.cos(theta)
    y = cy + r * torch.sin(theta)
    t = torch.rand(N, 1)*(t_ub-t_lb) + t_lb
    return x.to(device), y.to(device), t.to(device)

def sample_dirichlet(N):
    #incremento i bound di un epsilon
    eps = 0.0
    x_lb1, x_ub1, y_lb1, y_ub1 = x_lb+eps, x_ub-eps, y_lb+eps, y_ub-eps
    # basso: y = y_lb, x in [x_lb,x_ub]
    xb = torch.rand(N,1,device=device)*(x_ub1-x_lb1) + x_lb1
    yb = torch.full((N,1), y_lb1, device=device)
    tb = torch.rand(N,1,device=device)*(t_ub-t_lb) + t_lb

    # alto: y = y_ub
    xt = torch.rand(N,1,device=device)*(x_ub1-x_lb1) + x_lb1
    yt = torch.full((N,1), y_ub1, device=device)
    tt = torch.rand(N,1,device=device)*(t_ub-t_lb) + t_lb

    # sinistra: x = x_lb
    xl = torch.full((N,1), x_lb1, device=device)
    yl = torch.rand(N,1,device=device)*(y_ub1-y_lb1) + y_lb1
    tl = torch.rand(N,1,device=device)*(t_ub-t_lb) + t_lb

    # destra: x = x_ub
    xr = torch.full((N,1), x_ub1, device=device)
    yr = torch.rand(N,1,device=device)*(y_ub1-y_lb1) + y_lb1
    tr = torch.rand(N,1,device=device)*(t_ub-t_lb) + t_lb

    # concateno tutto
    x_bc = torch.cat([xb, xt, xl, xr], dim=0).requires_grad_(True)
    y_bc = torch.cat([yb, yt, yl, yr], dim=0).requires_grad_(True)
    t_bc = torch.cat([tb, tt, tl, tr], dim=0).requires_grad_(True)
    return x_bc, y_bc, t_bc

# 4) Define LSTM‐PINN model
class LSTM_PINN_NS2D(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output_layer =nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            # nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            # nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            # nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            # nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            # nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            # nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 3))
    def forward(self, x, y, t):
        seq = torch.cat([x,y,t], dim=-1).unsqueeze(1)
        h, _ = self.rnn(seq)
        uvp = self.output_layer(h[:, -1])
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
    umean = torch.mean(u)
    vmean = torch.mean(v)
    entropy = (u-umean)*pu + (v-vmean)*pv
    return pu, pv, continuity, entropy

# 6) Loss
mse = nn.MSELoss()
working = False
xi = torch.rand(N_ic, 1)*(x_ub-x_lb) + x_lb
yi = torch.rand(N_ic, 1)*(y_ub-y_lb) + y_lb
ti = torch.zeros(N_ic, 1, device=device)
xi, yi, ti = xi.to(device), yi.to(device), ti.to(device)
xb, yb, tb = sample_dirichlet(5000)
x, y, t = sample_collocation(20000)

def loss_functions():
    
    ui, vi = sample_initial_conditions2(xi, yi)
    u_pred, v_pred, _ = model(xi, yi, ti)
    L_ic = mse(u_pred, ui) + mse(v_pred, vi)
    u_bc, v_bc, _ = model(xb, yb, tb)
    L_bc = mse(u_bc, torch.zeros_like(u_bc)) + mse(v_bc, torch.zeros_like(v_bc))
    pu, pv, continuity, entropy = NS_res(x,y,t)
    L_pde = mse(pu, torch.zeros_like(pu)) + mse(pv, torch.zeros_like(pv)) + mse(continuity, torch.zeros_like(continuity))
    L_e = mse(entropy, torch.zeros_like(entropy))
    
    if circle_bool:
            xc, yc, tc = sample_circle(5000)
            u_c, v_c, _ = model(xc, yc, tc)
            L_bc += (mse(u_c, torch.zeros_like(u_c)) + mse(v_c, torch.zeros_like(v_c)))*0.1
    
    loss = L_ic+L_bc
    
    return loss, L_ic, L_ic, L_ic

# 7) Training loop
epoche = 400
def train_noBatches():
    history={
        "total_loss": [],
        "pde_loss": [],
        "bc_loss": [],
        "ic_loss": [],
        "obs_loss": []}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    #pretrain on initial conditions
    for ep in range(1, epoche+1):
        opt.zero_grad()
        loss, L_ic, L_bc, L_pde = loss_functions()
        loss.backward()
        opt.step()
        if ep % 10 == 0:
            print(f"Ep {ep}/{epoche}: loss={loss.item():.2e}, loss pde={L_pde.item():.2e}, loss bc={L_bc.item():.2e}, loss ic={L_ic.item():.2e}")
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
    import LSTM_NS_2d_analysis as an
    an.plotModel()
    # train()
