import os
import torch
import torch.nn as nn
import numpy as np
import hyperpar as hp

# 0) Disable CuDNN for double‐backward
torch.backends.cudnn.enabled = False

# 1) Hyperparameters & device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nu = hp.NU  # kinematic viscosity
hidden_size = hp.HIDDEN_SIZE
num_layers = hp.NUM_LAYERS
lr = hp.LR
epochs = hp.EPOCHS
N_ic = hp.N_IC  # # initial‐condition points
N_colloc = hp.N_COLLOCATION
N_bc = hp.N_BC  # # boundary‐condition points
N_obs = hp.N_OBSTACLE  # # obstacle points
lambda_ic = hp.LAMBDA_DATA
lambda_pde = hp.LAMBDA_PDE
lambda_bc = hp.LAMBDA_BC
lambda_obs = hp.LAMBDA_OBS
input_size = 4  # input size (x,y,z,t)

#Domain
x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = hp.X_LB, hp.X_UB, hp.Y_LB, hp.Y_UB, hp.Z_LB, hp.Z_UB
t_lb, t_ub = hp.T_LB, hp.T_UB

# 2) Mask on obstacle
def sphere_noslip(x, y, z, xc =0, yc =0, zc =0, r = 0.15):
    return ((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2) < 0.0

# 3) Sampling
# 3.1) IC
x_ic = torch.rand(N_ic, 1)*(x_ub -x_lb) + x_lb
y_ic = torch.rand(N_ic, 1)*(y_ub-y_lb) +y_lb
z_ic = torch.rand(N_ic, 1)*(z_ub-z_lb) + z_lb
t_ic = torch.zeros_like(x_ic)

mask_ic = ~sphere_noslip(x_ic, y_ic, z_ic)
x_ic, y_ic, z_ic, t_ic = [v[mask_ic].reshape(-1, 1) for v in (x_ic, y_ic, z_ic, t_ic)]
u_ic = torch.zeros_like(x_ic)
v_ic = torch.zeros_like(x_ic)
w_ic = torch.zeros_like(x_ic)

#device
x_ic, y_ic, z_ic, t_ic, u_ic, v_ic, w_ic = [n.to(device) for n in (x_ic, y_ic, z_ic, t_ic, u_ic, v_ic, w_ic)]

# 3.2) Collocation points
x_coll = torch.rand(N_colloc, 1)*(x_ub-x_lb) +x_lb
y_coll = torch.rand(N_colloc, 1)*(y_ub-y_lb) + y_lb
z_coll = torch.rand(N_colloc, 1)*(z_ub-z_lb) +z_lb
t_coll = torch.rand(N_colloc, 1) * (t_ub-t_lb) + t_lb

mask_coll = ~sphere_noslip(x_coll, y_coll, z_coll)
x_coll, y_coll, z_coll, t_coll = [v[mask_coll].reshape(-1, 1) for v in (x_coll, y_coll, z_coll, t_coll)]

#device
x_coll, y_coll, z_coll, t_coll = [n.to(device) for n in (x_coll, y_coll, z_coll, t_coll)]

# 3.3) BC
def boundary(N):
    """qui campiono N punti sui boundary, sulle facce del dominio e 
    sull'ostacolo (collocation points on the boundaries)"""
    pts = []
    #qui per ogni faccia del cubo
    for (lb, ub, axis) in [x_lb, x_ub, 0], [y_lb, y_ub, 1], [z_lb, z_ub, 2]:
        fixed = torch.full((N, 1), lb if axis ==0 else (lb if axis ==1 else lb))
    
        ran1 = torch.rand (N,1)*((x_ub-x_lb) if axis !=0 else (y_ub-y_lb)) + (x_lb if axis !=0 else y_lb)
        ran2 = torch.rand(N,1)*((z_ub-z_lb)) +z_lb
        #qui decido come ordinare le coordinate in base all'asse. Lascio una coordinata fixed
        if axis ==0:
            x_b, y_b, z_b = fixed, ran1, ran2
        elif axis ==1:
            x_b, y_b, z_b = ran1, fixed, ran2
        else:
            x_b, y_b, z_b = ran1, ran2, fixed
        t_b = torch.rand(N,1)*(t_ub-t_lb) +t_lb
        pts.append((x_b, y_b, z_b, t_b))
    
    #campiono i punti sull'ostacolo
    x_obst = torch.rand(N*2, 1)*(x_ub-x_lb) +x_lb
    y_obst = torch.rand(N*2,1)*(y_ub-y_lb) +y_lb
    z_obst = torch.rand(N*2, 1) * (z_ub-z_lb) +z_lb
    
    #setting the mask (obstacle points, the equation doesn't exist here)
    mask_obst = sphere_noslip(x_obst, y_obst, z_obst)
    
    x_obst, y_obst, z_obst = [v[mask_obst].reshape(-1,1) for v in (x_obst, y_obst, z_obst)]
    t_obst = torch.rand(x_obst.shape[0], 1)*(t_ub-t_lb) +t_lb
    pts.append((x_obst, y_obst, z_obst, t_obst))
    #concatenate all points
    xboundary = torch.cat([p[0] for p in pts], dim = 0)
    yboundary = torch.cat([p[1] for p in pts], dim = 0)
    zboundary = torch.cat([p[2] for p in pts], dim = 0)
    tboundary = torch.cat([p[3] for p in pts], dim=0)
    
    return [m.to(device) for m in (xboundary, yboundary, zboundary, tboundary)]

x_bc, y_bc, z_bc, t_bc = boundary(N_bc)

# 4) definisco il modello LSTM-PINN

class LSTM_PINN_NS3D(nn.Module):
    #__init__ is the constructor, called when a new instance of the class is created
    def __init__(self, input_size, hidden_size, num_layers):
        #calls the constructor of the parent class (nn.Module)
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)
    
    def forward(self, x, y, z, t):
        seq = torch.cat([x, y, z, t], dim = -1).unsqueeze(1)
        h, _ = self.rnn(seq)
        #qui definisco i 3 vettori velocità (RANS) e la pressione
        uvwp = self.fc(h[:, -1, :])
        return uvwp[:, 0:1], uvwp[:, 1:2], uvwp[:, 2:3], uvwp[:, 3:4]

#5) inizializzo il modello
model = LSTM_PINN_NS3D(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers).to(device)
# model = LSTM_PINN_NS3D().to(device)
# 6) residui, qui sta la fisica!

def inlet_velocity_field(N):
    """Prescrive un campo di velocità uniforme in x, zero in y e z"""
    x_in = torch.full((N,1), hp.x_lb_inlet)
    y_in = torch.rand(N, 1)*(y_ub-y_lb) +y_lb
    z_in = torch.rand(N, 1) * (z_ub-z_lb) +z_lb
    t_in = torch.rand(N, 1)*(t_ub-t_lb) +t_lb
    
    return [v.to(device) for v in (x_in, y_in, z_in, t_in)]
    
    
def NS_res(x,y,z,t):
    
    #require gradient
    for g in (x,y,z,t):
        g.requires_grad_(True)
        
    u,v,w,p = model(x,y,z,t)
    ones = torch.ones_like(u)
    
    #time derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs = ones, create_graph = True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=ones, create_graph=True)[0]
    w_t = torch.autograd.grad(w, t, grad_outputs = ones, create_graph=True)[0]
    
    #spatial derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs= ones, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs= ones, create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs= ones, create_graph=True)[0]
    
    v_x = torch.autograd.grad(v, x, grad_outputs= ones, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs= ones, create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs= ones, create_graph=True)[0]
    
    w_x = torch.autograd.grad(w, x, grad_outputs= ones, create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs= ones, create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs= ones, create_graph=True)[0]
    
    p_x = torch.autograd.grad(p, x, grad_outputs= ones, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs= ones, create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs= ones, create_graph=True)[0]
    
    #Laplacian
    #laplaciano: xx components
    u_xx = torch.autograd.grad(u_x, x, grad_outputs= ones, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs= ones, create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs= ones, create_graph=True)[0]
    
    #laplaciano: yy components
    u_yy = torch.autograd.grad(u_y, y, grad_outputs= ones, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs= ones, create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs= ones, create_graph=True)[0]
    
    #laplaciano: zz components
    u_zz = torch.autograd.grad(u_z, z, grad_outputs= ones, create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs= ones, create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs= ones, create_graph=True)[0]
    
    #equazione di continuità
    continuity = u_x + v_y + w_z
    
    #momentum equations
    p_u = u_t + (u*u_x + v*u_y + w*u_z) + p_x -nu*(u_xx + u_yy + u_zz)
    p_v = v_t + (u*v_x +v*v_y+w*v_z) + p_y -nu*(v_xx +v_yy +v_zz)
    p_w = w_t + (u*w_x + v*w_y +w*w_z) +p_z - nu* (w_xx+w_yy+w_zz)
    
    return p_u, p_v, p_w, continuity

# 7) Loss function
mse = nn.MSELoss()
def loss_functions():
    """Funzione di loss, che calcola la loss totale e le singole componenti"""
    """Tutte le parti del dominio devono essere campionate dal modello"""
    
    #Initia condition loss
    u0, v0, w0, _ = model(x_ic, y_ic, z_ic, t_ic)
    #MSE := 1/N SUM ((u0-u_ic)^2 + (v0-v_ic)^2 + (w0-w_ic)^2)
    L_ic = mse(u0, u_ic) + mse(v0, v_ic) + mse(w0, w_ic)
    
    #PDE collocation loss
    p_u, p_v, p_w, continuity = NS_res(x_coll, y_coll, z_coll, t_coll)
    #Forcing the residuals to be zero, mse(p_i, 0*p_i), ma non so se è la soluzione migliore
    L_pde = mse(p_u, 0*p_u) +mse(p_v, 0*p_v) +mse(p_w, 0*p_w) +mse(continuity, 0*continuity)
    
    #BC loss no-slip
    u_bc, v_bc, w_bc, _ = model(x_bc, y_bc, z_bc, t_bc)
    L_bc = mse(u_bc, 0*u_bc) + mse(v_bc, 0*v_bc) + mse(w_bc, 0*w_bc)
    
    #BC loss inlet
    x_in, y_in, z_in, t_in = inlet_velocity_field(hp.N_INLET)
    u_in_pred, v_in_pred, w_in_pred, _ = model(x_in, y_in, z_in, t_in)
    
    #prescrivo una velocità uniforme U_INLET in x, zero in y e z
    U_target = torch.full_like(u_in_pred, hp.U_INLET)
    V_target = torch.zeros_like(v_in_pred)
    W_target = torch.zeros_like(w_in_pred)
    L_inlet = (mse(u_in_pred, U_target) +mse(v_in_pred, V_target) + mse(w_in_pred, W_target))
    
    #Total loss
    L = lambda_ic*L_ic + lambda_pde*L_pde+lambda_obs*L_bc + hp.LAMBDA_INLET*L_inlet
    
    return L, L_ic, L_pde, L_bc, L_inlet

# 8) training function
def train_NS():
    # torch.cuda.empty_cache()
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    history = {"total":[], "ic":[], "pde":[], "bc":[], "inlet":[]}
    for ep in range(1, epochs+1):
        opt.zero_grad()
        L, L_ic, L_pde, L_bc, L_inlet = loss_functions()
        L.backward()
        opt.step()
        history["total"].append(L.item())
        history["ic"].append(L_ic.item())
        history["pde"].append(L_pde.item())
        history["bc"].append(L_bc.item())
        history["inlet"].append(L_inlet.item())
        if ep % 20 == 0:
            print(f"Epoch {ep}/{epochs}, Loss: {L.item():.4e}, IC Loss: {L_ic.item():.4e}, PDE Loss: {L_pde.item():.4e}, BC Loss: {L_bc.item():.4e}, Inlet Loss: {L_inlet.item():.4e}")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("loss", exist_ok=True)
    
    #salva il modello
    torch.save(model.state_dict(), "models/LSTM_NS.pth")
    print("♪♪♪ Model saved ♪♪♪")
    np.save("loss/loss_NS.npy", history)
    print("©©© Loss saved ©©©")
    
if __name__ == "__main__":
    train_NS()
        