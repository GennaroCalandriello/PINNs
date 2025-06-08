import os
import torch
import torch.nn as nn
import numpy as np
import hyperpar as hp

# 0) Riproducibilità
torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Parametri PDE e rete
nu          = hp.NU
hidden_size = hp.HIDDEN_SIZE
num_layers  = hp.NUM_LAYERS
lr          = hp.LR
epochs      = hp.EPOCHS

# 2) Numeri di punti
N_ic     = hp.N_IC
N_coll   = hp.N_COLLOCATION
N_bc_obs = hp.N_OBSTACLE   # punti SUL contorno del cerchio
# (non più “interni” ma proprio sul perimetro)

# 3) Dominio
x_lb, x_ub = hp.X_LB, hp.X_UB
y_lb, y_ub = hp.Y_LB, hp.Y_UB
t_lb, t_ub = hp.T_LB, hp.T_UB
lambda_ic = hp.LAMBDA_IC
lambda_pde = hp.LAMBDA_PDE
lambda_bc = hp.LAMBDA_BC

# 4) Sampling dei punti

# 4.1) Initial condition: qui mettiamo il profilo di inlet uniforme
#      come condizione in t=0 su tutto il dominio fluido.
x0 = x_lb + (x_ub - x_lb) * torch.rand(N_ic,1)
y0 = y_lb + (y_ub - y_lb) * torch.rand(N_ic,1)
t0 = torch.zeros_like(x0)

# fuori ostacolo
mask0 = ((x0 - 0.5)**2 + (y0 - 0.0)**2) >= 0.4**2
x0, y0, t0 = [v[mask0].reshape(-1, 1) for v in (x0, y0, t0)]

# target IC: flusso uniforme U_INLET
u0_target = torch.full_like(x0, hp.U_INLET)
v0_target = torch.zeros_like(x0)

x0, y0, t0, u0_target, v0_target = [v.to(device) for v in (x0,y0,t0,u0_target,v0_target)]


# 4.2) Collocation (interno fluido, random)
xc = x_lb + (x_ub - x_lb) * torch.rand(N_coll,1)
yc = y_lb + (y_ub - y_lb) * torch.rand(N_coll,1)
tc = t_lb + (t_ub - t_lb) * torch.rand(N_coll,1)
maskc = ((xc - 0.5)**2 + (yc - 0.0)**2) >= 0.4**2
xc, yc, tc = [v[maskc].reshape(-1, 1) for v in (xc, yc, tc)]
#device
xc, yc, tc = [v.to(device).requires_grad_() for v in (xc, yc, tc)]


# 4.3) Boundary sul cerchio (no-slip)
theta = 2*np.pi * torch.rand(N_bc_obs,1)
xb = 0.5 + 0.4*torch.cos(theta)
yb = 0.0 + 0.4*torch.sin(theta)
tb = t_lb + (t_ub - t_lb)*torch.rand_like(xb)

xb, yb, tb = [v.to(device).requires_grad_() for v in (xb,yb,tb)]


# 5) Definizione del modello
class LSTM_PINN_NS2D(nn.Module):
    def __init__(self, in_dim=3, hidden=64, layers=2):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden, layers, batch_first=True)
        self.fc  = nn.Linear(hidden, 3)
    def forward(self, x, y, t):
        # costruisco sequenza (batch,seq=1,3)
        seq = torch.cat([x,y,t], dim=-1).unsqueeze(1)
        h, _ = self.rnn(seq)                # → (batch,1,hidden)
        uvp = self.fc(h[:, -1, :])          # → (batch,3)
        u = uvp[:,0:1];  v = uvp[:,1:2];  p = uvp[:,2:3]
        return u, v, p

model = LSTM_PINN_NS2D(3, hidden_size, num_layers).to(device)


# 6) Residui Navier–Stokes
def NS_residuals(x,y,t):
    # assicuro i gradienti
    for v in (x,y,t):
        v.requires_grad_(True)
    u,v,p = model(x,y,t)
    one = torch.ones_like(u)

    # derivate prime
    u_t = torch.autograd.grad(u, t, grad_outputs=one, create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=one, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=one, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=one, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=one, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=one, create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=one, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=one, create_graph=True)[0]

    # laplaciani
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=one, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=one, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=one, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=one, create_graph=True)[0]

    # equazioni
    continuity = u_x + v_y
    momentum_x = u_t + (u*u_x + v*u_y) + p_x - nu*(u_xx + u_yy)
    momentum_y = v_t + (u*v_x + v*v_y) + p_y - nu*(v_xx + v_yy)

    return momentum_x, momentum_y, continuity


# 7) Funzione di loss
mse = nn.MSELoss()
def compute_losses():
    # 7.1 IC loss
    u0_pred, v0_pred, _ = model(x0,y0,t0)
    L_ic = mse(u0_pred, u0_target) + mse(v0_pred, v0_target)

    # 7.2 PDE loss
    mx, my, cont = NS_residuals(xc,yc,tc)
    L_pde = mse(mx, torch.zeros_like(mx)) \
          + mse(my, torch.zeros_like(my)) \
          + mse(cont, torch.zeros_like(cont))

    # 7.3 BC Loss (no-slip cerchio)
    ub_pred, vb_pred, _ = model(xb,yb,tb)
    L_bc = mse(ub_pred, torch.zeros_like(ub_pred)) \
         + mse(vb_pred, torch.zeros_like(vb_pred))

    # combinazione
    L = lambda_ic*L_ic + lambda_pde*L_pde + lambda_bc*L_bc
    return L, L_ic, L_pde, L_bc


# 8) Training loop
def train():
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        opt.zero_grad()
        L, L_ic, L_pde, L_bc = compute_losses()
        L.backward()
        opt.step()

        if ep % 20 == 0:
            print(f"[{ep:4d}/{epochs}] "
                  f"Total={L.item():.2e} "
                  f"IC={L_ic.item():.2e} "
                  f"PDE={L_pde.item():.2e} "
                  f"BC={L_bc.item():.2e}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/LSTM_NS2d.pth")
    print("✅ Saved model")

if __name__=="__main__":
    train()
