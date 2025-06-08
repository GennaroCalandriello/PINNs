import torch
import torch.nn as nn
import torch.optim as optim

torch.backends.cudnn.enabled = False

# 1) PINN model: LSTM + FC
class LSTMPINN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, num_layers=2, out_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, in_dim)
        h, _ = self.lstm(x)           # → (batch, seq_len, hidden_dim)
        y = self.fc(h)                # → (batch, seq_len, out_dim)
        return y

# 2) Physics residuals NS
def navier_stokes_residual(model, x, y, t, nu):
    # ensure grads can flow from x,y,t → u,v,p
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    
    inp = torch.cat([x, y, t], dim=1).unsqueeze(1)  # (N,1,3)
    uvp = model(inp).squeeze(1)                     # (N,3)
    u, v, p = uvp[:,0:1], uvp[:,1:2], uvp[:,2:3]
    
    # first derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]

    # second derivatives
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

    # PDE residuals
    r1 = u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
    r2 = v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
    r3 = u_x + v_y

    return r1, r2, r3


def TaylorGreenVortex(x, y):
    # example initial condition: Taylor–Green vortex
    u0 = -0.1*torch.cos(torch.pi*x)*torch.sin(torch.pi*y)
    v0 =  0.1*torch.sin(torch.pi*x)*torch.cos(torch.pi*y)
    return u0, v0

# 3) Training loop
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # PDE parameters
    nu = 0.0001
    T_final = 1.0
    
    # collocation points
    N_f = 10000
    x_f = torch.rand(N_f,1, device=device)
    y_f = torch.rand(N_f,1, device=device)
    t_f = torch.rand(N_f,1, device=device) * T_final
    
    # initial condition points (t=0)
    N0 = 2000
    x0 = torch.rand(N0,1, device=device)
    y0 = torch.rand(N0,1, device=device)
    t0 = torch.zeros_like(x0)
    # e.g. Taylor–Green vortex IC
    u0, v0 = TaylorGreenVortex(x0, y0)
    
   
    
    mse = nn.MSELoss()
    epochs = 1000
    
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        
        # PDE‐residual loss
        r1, r2, r3 = navier_stokes_residual(model, x_f, y_f, t_f, nu)
        loss_pde = mse(r1, 0*r1) + mse(r2, 0*r2) + mse(r3, 0*r3)
        
        # **IC loss** ← simple forward pass, no autograd.grad here!
        inp0 = torch.cat([x0, y0, t0], dim=1).unsqueeze(1)
        uvp0 = model(inp0).squeeze(1)
        u_pred0, v_pred0 = uvp0[:,0:1], uvp0[:,1:2]
        loss_ic = mse(u_pred0, u0) + mse(v_pred0, v0)
        
        #total loss
        loss = loss_pde + loss_ic
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"[{epoch}/{epochs}] PDE_loss={loss_pde.item():.2e}, IC_loss={loss_ic.item():.2e}")
    
    return model

if __name__ == "__main__":
    trained = train()
    torch.save(trained.state_dict(), "models/LSTM_PINN_NS2D.pth")
    print("✅ Done.")
