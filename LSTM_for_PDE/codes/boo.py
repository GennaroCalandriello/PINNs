import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

# 2D incompressible Navier-Stokes PINN with cylinder obstacle and no-slip BC
# Domain: x,y in [0,1], t in [0,1]
# Cylinder centered at (0.5,0.5) with radius r

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.net = nn.ModuleList(
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)
        )

    def forward(self, xyt):
        out = xyt
        for layer in self.net[:-1]:
            out = self.activation(layer(out))
        out = self.net[-1](out)
        u = out[..., 0:1]
        v = out[..., 1:2]
        p = out[..., 2:3]
        return u, v, p

# Robust derivative utility

def derivative(u, x, order=1):
    # u: tensor [N,1], x: tensor [N,1]
    for _ in range(order):
        # create grad_outputs matching u
        grad_outputs = torch.ones_like(u)
        grads = grad(u, x, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
        # replace None grads with zeros
        if grads is None:
            grads = torch.zeros_like(u)
        u = grads
    return u

# PDE residuals using robust derivatives
def navier_stokes_res(model, x, y, t, nu):
    x = x.view(-1,1)
    y = y.view(-1,1)
    t = t.view(-1,1)
    xyt = torch.cat([x,y,t], dim=1).requires_grad_(True)
    u, v, p = model(xyt)

    # first derivatives
    u_t = derivative(u, xyt[:,2:3], 1)
    u_x = derivative(u, xyt[:,0:1], 1)
    u_y = derivative(u, xyt[:,1:2], 1)
    v_t = derivative(v, xyt[:,2:3], 1)
    v_x = derivative(v, xyt[:,0:1], 1)
    v_y = derivative(v, xyt[:,1:2], 1)
    p_x = derivative(p, xyt[:,0:1], 1)
    p_y = derivative(p, xyt[:,1:2], 1)
    # second derivatives
    u_xx = derivative(u, xyt[:,0:1], 2)
    u_yy = derivative(u, xyt[:,1:2], 2)
    v_xx = derivative(v, xyt[:,0:1], 2)
    v_yy = derivative(v, xyt[:,1:2], 2)

    continuity = u_x + v_y
    momentum_u = u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
    momentum_v = v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
    return continuity, momentum_u, momentum_v

# (Sampler and training code unchanged)

# Sample collocation inside domain excluding cylinder

def sampler_domain(N, r, center):
    x = torch.rand(N,1)
    y = torch.rand(N,1)
    mask = ((x-center[0])**2 + (y-center[1])**2) >= r**2
    while mask.sum() < N:
        x2 = torch.rand(N,1); y2 = torch.rand(N,1)
        m2 = ((x2-center[0])**2 + (y2-center[1])**2) >= r**2
        x = torch.cat([x[mask], x2[m2]],0)[:N]
        y = torch.cat([y[mask], y2[m2]],0)[:N]
        mask = ((x-center[0])**2 + (y-center[1])**2) >= r**2
    t = torch.rand(N,1)
    return x, y, t

# Sample cylinder surface for no-slip

def sampler_cylinder(N, r, center):
    theta = 2*torch.pi*torch.rand(N,1)
    x = center[0] + r*torch.cos(theta)
    y = center[1] + r*torch.sin(theta)
    t = torch.rand(N,1)
    return x, y, t

# Inlet BC at x=0

def sampler_inlet(N):
    x = torch.zeros(N,1)
    y = torch.rand(N,1)
    t = torch.rand(N,1)
    u0 = torch.ones(N,1)
    v0 = torch.zeros(N,1)
    return x,y,t,u0,v0

# Initial BC at t=0 in fluid region

def initial_bc(N, r, center):
    x = torch.rand(N,1)
    y = torch.rand(N,1)
    mask = ((x-center[0])**2 + (y-center[1])**2) >= r**2
    x = x[mask][:N]; y = y[mask][:N]
    t = torch.zeros(N,1)
    u0 = torch.zeros_like(x)
    v0 = torch.zeros_like(x)
    return x, y, t, u0, v0

# Setup and training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nu = 0.01
center=(0.5,0.5); radius=0.15
layers = [3]+[50]*4+[3]
model = PINN(layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs=5000; N_f=20000; N_b=2000; N_i=2000; N_in=2000
for epoch in range(1, epochs+1):
    optimizer.zero_grad()
    # Physics loss
    x_f,y_f,t_f = sampler_domain(N_f, radius, center)
    x_f,y_f,t_f = x_f.to(device), y_f.to(device), t_f.to(device)
    c_f, mu_f, mv_f = navier_stokes_res(model, x_f, y_f, t_f, nu)
    loss_f = (c_f**2 + mu_f**2 + mv_f**2).mean()
    # No-slip on cylinder
    x_b,y_b,t_b = sampler_cylinder(N_b, radius, center)
    x_b,y_b,t_b = x_b.to(device), y_b.to(device), t_b.to(device)
    u_b, v_b, _ = model(torch.cat([x_b,y_b,t_b],dim=1))
    loss_b = (u_b**2 + v_b**2).mean()
    # Inlet BC
    x_in,y_in,t_in,u_in,v_in = sampler_inlet(N_in)
    x_in,y_in,t_in,u_in,v_in = [t.to(device) for t in (x_in,y_in,t_in,u_in,v_in)]
    u_pred_in, v_pred_in, _ = model(torch.cat([x_in,y_in,t_in],dim=1))
    loss_in = ((u_pred_in - u_in)**2 + (v_pred_in - v_in)**2).mean()
    # Initial BC
    x_i,y_i,t_i,u_i,v_i = initial_bc(N_i, radius, center)
    x_i,y_i,t_i,u_i,v_i = [t.to(device) for t in (x_i,y_i,t_i,u_i,v_i)]
    u_pred_i, v_pred_i, _ = model(torch.cat([x_i,y_i,t_i],dim=1))
    loss_i = ((u_pred_i - u_i)**2 + (v_pred_i - v_i)**2).mean()
    # Total loss
    loss = loss_f + loss_b + loss_in + loss_i
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
print("Training complete.")