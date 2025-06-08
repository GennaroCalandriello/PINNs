import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 0) Disable CuDNN for double‐backward
torch.backends.cudnn.enabled = False
# Domain bounds
x_lb, x_ub = 0.0, 1.0
y_lb, y_ub = 0.0, 1.0
t_lb, t_ub = 0.0, 1.0
nu = 0.01  # Viscosity

class LSTM_PINN_NS2D(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2):
        super().__init__()
        self.input_layer = nn.Linear(3, hidden_size)  # (x, y, t)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 3)  # (u, v, p)

    def forward(self, xyt):  # xyt: (N, 3)
        N = xyt.shape[0]
        xyt_seq = xyt.unsqueeze(1)  # (N, 1, 3)
        enc = self.input_layer(xyt_seq)  # (N, 1, H)
        lstm_out, _ = self.lstm(enc)     # (N, 1, H)
        uvp = self.output_layer(lstm_out.squeeze(1))  # (N, 3)
        return uvp

def sample_collocation(N):
    x = torch.rand(N, 1) * (x_ub - x_lb) + x_lb
    y = torch.rand(N, 1) * (y_ub - y_lb) + y_lb
    t = torch.rand(N, 1) * (t_ub - t_lb) + t_lb
    xyt = torch.cat([x, y, t], dim=1).to(device).requires_grad_(True)  # (N, 3)
    return xyt

def compute_physics_loss(model, xyt):
    x = xyt[:, 0:1]
    y = xyt[:, 1:2]
    t = xyt[:, 2:3]

    uvp = model(xyt)  # (N, 3)
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True, retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True, retain_graph=True)[0]

    res_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    res_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    div = u_x + v_y

    loss = (res_u ** 2).mean() + (res_v ** 2).mean() + (div ** 2).mean()
    return loss



def train():
    
    model = LSTM_PINN_NS2D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(10000):
        model.train()
        xyt = sample_collocation(5000)

        loss = compute_physics_loss(model, xyt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.6f}")

if __name__ == "__main__":
    train()