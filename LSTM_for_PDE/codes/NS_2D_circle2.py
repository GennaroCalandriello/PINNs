import torch
import torch.nn as nn
import numpy as np
import hyperpar as hp
# Disable cuDNN (for strict reproducibility)
torch.backends.cudnn.enabled = False

# PDE parameter
nu      = 0.001
T_final = 1.0

# Numbers of points
N_f = 40000   # collocation
N_i = 10000    # initial‐condition
N_b = 2000    # boundary
epochs = 500
num_layers = 2

x_lb, x_ub = hp.X_LB, hp.X_UB
y_lb, y_ub = hp.Y_LB, hp.Y_UB
cx, cy, r = hp.cx, hp.cy, hp.r

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) LSTM‐based PINN: outputs stream‐function ψ and pressure p
class LSTMPINN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, num_layers=num_layers, out_dim=2):
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


# 2) PINN trainer with circular obstacle
class NavierStokesLSTM:
    def __init__(self, cx=cx, cy=cy, r=r, x_lb=x_lb, x_ub=x_ub, y_lb=y_lb, y_ub=y_ub):
        #save parameters
        self.cx, self.cy, self.r = cx, cy, r
        self.x_lb, self.x_ub = x_lb, x_ub
        self.y_lb, self.y_ub = y_lb, y_ub
        
        # ---- collocation points outside the circle ----
        
        Xf =x_lb+(x_ub-x_lb)* torch.rand(int(1.2*N_f),1, device=device)
        Yf =y_lb+(y_ub-y_lb)* torch.rand(int(1.2*N_f),1, device=device)
        Tf = torch.rand(int(1.2*N_f),1, device=device) * T_final
        mask_f = ((Xf - cx)**2 + (Yf - cy)**2) >= r**2

        self.x_f = Xf[mask_f][:N_f].unsqueeze(1).requires_grad_()
        self.y_f = Yf[mask_f][:N_f].unsqueeze(1).requires_grad_()
        self.t_f = Tf[mask_f][:N_f].unsqueeze(1).requires_grad_()

        # ---- initial‐condition points outside circle, at t=0 ----
        X0 =x_lb+(x_ub-x_lb)* torch.rand(int(1.2*N_i),1, device=device)
        Y0 =y_lb+(y_ub-y_lb) *torch.rand(int(1.2*N_i),1, device=device)
        T0 = torch.rand(int(1.2*N_i),1, device=device) * T_final
        mask_0 = ((X0 - cx)**2 + (Y0 - cy)**2) >= r**2

        self.x0 = X0[mask_0][:N_i].unsqueeze(1).requires_grad_()
        self.y0 = Y0[mask_0][:N_i].unsqueeze(1).requires_grad_()
        self.t0 = T0[mask_0][:N_i].unsqueeze(1).requires_grad_()
        
        decay = torch.exp(-2*(np.pi**2)*nu*self.t0)

        # Taylor–Green vortex initial velocity
        self.u0 =  -0.1*torch.cos(np.pi*self.x0) * torch.sin(np.pi*self.y0)*decay
        self.v0 =  0.1*torch.sin(np.pi*self.x0) * torch.cos(np.pi*self.y0)*decay

        #Random initial condition
        # self.u0 = torch.rand(self.x0.size(0),1, device=device)
        # self.v0 = torch.rand(self.x0.size(0),1, device=device)
        
        # ---- boundary (circle) points for no‐slip ----
        theta = 2*np.pi * torch.rand(N_b,1, device=device)
        self.x_b = (cx + r*torch.cos(theta)).requires_grad_()
        self.y_b = (cy + r*torch.sin(theta)).requires_grad_()
        self.t_b = (torch.rand(N_b,1, device=device) * T_final).requires_grad_()
        self.null_b = torch.zeros(N_b,1, device=device)

        # ---- null targets for PDE residuals ----
        self.null_f = torch.zeros(N_f,1, device=device)
        self.null_g = torch.zeros(N_f,1, device=device)

        # build network & optimizer
        self.net = LSTMPINN().to(device)
        self.mse = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)


    def residuals(self, x, y, t):
        # x,y,t: (batch,1)
        inp = torch.cat([x, y, t], dim=1).unsqueeze(1)   # → (batch,1,3)
        out = self.net(inp).squeeze(1)                   # → (batch,2)
        psi, p = out[:,0:1], out[:,1:2]

        # velocities via stream‐function
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

        # momentum‐equation residuals
        f = u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
        g = v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
        
        #continuity equation
        continuity = u_x + v_y

        return u, v, p, f, g, continuity


    def train(self, epochs, batch_f=2000, batch_b=2000, lambda_bc=1.0):
        
        # Phase 1: pretrain on initial condition only
        pretrain_ic = 200
         # Phase 1: pretrain on initial condition only
        for e in range(1, pretrain_ic+1):
            u0_pred, v0_pred = self.residuals(self.x0, self.y0, self.t0)[2:4]
            loss_ic = self.mse(u0_pred, self.u0) + self.mse(v0_pred, self.v0)
            self.optimizer.zero_grad()
   
            # retain_graph to allow multiple backward passes in pretraining
            loss_ic.backward(retain_graph=True)
            self.optimizer.step()
            if e % 20 == 0:
                print(f"Pretrain IC [{e}/{pretrain_ic}] IC loss: {loss_ic:.3e}")

                
        for epoch in range(1, epochs+1):
            # 1) fluid‐domain collocation minibatch
            idx_f = torch.randperm(self.x_f.size(0), device=device)[:batch_f]
            x_f, y_f, t_f = self.x_f[idx_f], self.y_f[idx_f], self.t_f[idx_f]

            # PDE residual losses
            _, _, _, f_pred, g_pred, continuity = self.residuals(x_f, y_f, t_f)
            #provo ad aggiungere un noise per dare una "spinta" al modello
            noise_std = 1e-3
            noise_f = torch.randn_like(f_pred)*noise_std
            noise_g = torch.randn_like(g_pred)*noise_std
            loss_f = self.mse(f_pred, noise_f)
            loss_g = self.mse(g_pred, noise_g)
            loss_continuity = self.mse(continuity, 0*continuity)

            # 2) IC loss (all IC points)
            u0_p, v0_p, _, _, _, _ = self.residuals(self.x0, self.y0, self.t0)
            loss_ic = self.mse(u0_p, self.u0) + self.mse(v0_p, self.v0)

            # 3) no‐slip BC loss
            idx_b = torch.randperm(self.x_b.size(0), device=device)[:batch_b]
            u_b, v_b, _, _, _, _ = self.residuals(
                self.x_b[idx_b], self.y_b[idx_b], self.t_b[idx_b]
            )
            loss_bc = self.mse(u_b, self.null_b[:batch_b]) + self.mse(v_b, self.null_b[:batch_b])
            
            # 3.1) stream lossp boundary conditions
            xb, yb, tb = self.x_b[idx_b], self.y_b[idx_b], self.t_b[idx_b]
            input_bc = torch.cat([xb, yb, tb], dim=1).unsqueeze(1)  # (N,1,3)
            out_bc = self.net(input_bc).squeeze(1)                 # (N,2)
            psi_pred = out_bc[:,0:1]
            loss_psi_bc = self.mse(psi_pred, torch.zeros_like(psi_pred))
            # 4) total loss
            loss = loss_f + loss_g + loss_continuity +loss_ic + loss_bc

            # 5) backward & step
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # 6) logging
            if epoch % 1 == 0:
                print(f"[{epoch:5d}/{epochs}] "
                      f"PDE=({loss_f:.2e},{loss_g:.2e})  "
                      f"IC={loss_ic:.2e}  BC={loss_bc:.2e} "
                      f"total={loss:.2e} ")

        # save model
        torch.save(self.net.state_dict(), "model/ns2d_lstm_circle.pth")
        print("✅ Training complete, model saved to ns2d_lstm_circle.pth")


if __name__ == "__main__":
    
    pinn = NavierStokesLSTM(cx=0.5, cy=0.5, r=0.4)
    pinn.train(epochs=epochs, batch_f=2000, batch_b=2000, lambda_bc=1.0)
    
   
