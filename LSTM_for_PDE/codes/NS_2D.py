import torch
import torch.nn as nn
import numpy as np

# Disable cuDNN (optional, for reproducibility)
torch.backends.cudnn.enabled = False

# PDE parameter
nu = 0.01
T_final = 1.0

# Number of points
N_f = 20000   # total collocation points
N_i = 2000    # total initial‐condition points

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) LSTM-based PINN: outputs stream-function ψ and pressure p
class LSTMPINN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, num_layers=2, out_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        # x: (batch, seq_len, in_dim)
        h, _ = self.lstm(x)            # → (batch, seq_len, hidden_dim)
        return self.fc(h)              # → (batch, seq_len, out_dim)

# 2) Trainer for Navier–Stokes PINN with collocation + IC + minibatches
class NavierStokesLSTM:
    def __init__(self, cx = 0.5, cy = 0.5, r = 0.4, N_b = 2000):
        # Collocation points (x,y,t) ∈ [0,1]×[0,1]×[0,T], excluding the circle
        # centered at (cx,cy) with radius r
        self.x_f = torch.rand(N_f,1, device=device, requires_grad=True)
        self.y_f = torch.rand(N_f,1, device=device, requires_grad=True)
        self.t_f = torch.rand(N_f,1, device=device, requires_grad=True) * T_final

        
        # Initial‐condition points at t=0
        self.x0 = torch.rand(N_i,1, device=device, requires_grad=True)
        self.y0 = torch.rand(N_i,1, device=device, requires_grad=True)
        self.t0 = torch.zeros(N_i,1, device=device, requires_grad=True)

        # Taylor–Green vortex initial velocity
        self.u0 = -0.2*torch.cos(np.pi*self.x0)*torch.sin(np.pi*self.y0)
        self.v0 =  0.4*torch.sin(np.pi*self.x0)*torch.cos(np.pi*self.y0)

        # Null targets for PDE residuals
        self.null_f = torch.zeros(N_f,1, device=device)
        self.null_g = torch.zeros(N_f,1, device=device)

        # Build network and optimizer
        self.net = LSTMPINN().to(device)
        self.mse = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def residuals(self, x, y, t):
        """
        Given tensors x,y,t of shape (batch,1), compute:
          u, v, p, f = momentum_x_residual, g = momentum_y_residual
        """
        inp = torch.cat([x, y, t], dim=1).unsqueeze(1)   # (batch,1,3)
        out = self.net(inp).squeeze(1)                   # (batch,2)
        psi, p = out[:,0:1], out[:,1:2]

        # velocities from stream-function
        u =  torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        # first derivatives
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        # second derivatives
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        # PDE residuals (momentum equations)
        f = u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
        g = v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)

        return u, v, p, f, g

    def train(self, epochs, batch_f=2000):
        for epoch in range(1, epochs+1):
            # 1) sample minibatch of collocation points
            idx_f = torch.randperm(N_f, device=device)[:batch_f]
            x_f_batch = self.x_f[idx_f];   y_f_batch = self.y_f[idx_f];   t_f_batch = self.t_f[idx_f]

            # 2) compute PDE residuals on batch
            _, _, _, f_pred, g_pred = self.residuals(x_f_batch, y_f_batch, t_f_batch)
            loss_f = self.mse(f_pred, self.null_f[:batch_f])
            loss_g = self.mse(g_pred, self.null_g[:batch_f])

            # 3) compute IC loss on all initial points
            u0_pred, v0_pred, _, _, _ = self.residuals(self.x0, self.y0, self.t0)
            loss_ic = self.mse(u0_pred, self.u0) + self.mse(v0_pred, self.v0)

            # total loss
            loss = loss_f + loss_g + loss_ic

            # gradient step
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"[{epoch:5d}/{epochs}] "
                      f"PDE_loss=({loss_f.item():.2e},{loss_g.item():.2e})  "
                      f"IC_loss={loss_ic.item():.2e}")

        # save final model
        torch.save(self.net.state_dict(), "model/ns2d_adam.pth")
        print("✅ Training complete, model saved to lstm_pinn_ns2d_adam.pth")

if __name__ == "__main__":
    pinn = NavierStokesLSTM()
    pinn.train(epochs=2000, batch_f=2000)
