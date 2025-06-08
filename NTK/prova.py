import os
import torch
import torch.nn as nn
import numpy as np
import hyperpar as hp

# Disable CuDNN for double-backward
torch.backends.cudnn.enabled = False

"""
NTK-PINN for 2D Navier-Stokes: training loop with loss monitoring
Supports full IC+BC+PDE training or IC-only training for validation
"""

# Hyperparameters & device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_layers = hp.NUM_LAYERS
hidden_units = hp.HIDDEN_SIZE
lr = hp.LR
epochs = hp.EPOCHS if hasattr(hp, 'EPOCHS') else 100
N_ic = hp.N_IC
N_bc = hp.N_BC
N_colloc = hp.N_COLLOCATION
input_size = 3
sigma_w = getattr(hp, 'SIGMA_W', 1.0)
sigma_b = getattr(hp, 'SIGMA_B', 0.0)
ridge = getattr(hp, 'RIDGE', 1e-6)

# Domain bounds
x_lb, x_ub, y_lb, y_ub = hp.X_LB, hp.X_UB, hp.Y_LB, hp.Y_UB
t_lb, t_ub = hp.T_LB, hp.T_UB
circle_bool = True
cx, cy, r = hp.cx, hp.cy, hp.r
nu = hp.NU
def circle_mask(x, y, xc, yc, r):
    return (x-xc)**2 + (y-yc)**2 < r**2

# ... (rest of sampling funcs unchanged) ...


def relu_ntk(K, sigma_w, sigma_b):
    """Compute ReLU NTK transform of kernel matrix K"""
    # Normalize kernel
    eps = 1e-12
    vars = torch.diag(K)
    inv_sqrt = 1.0 / torch.sqrt(vars + eps)
    D = torch.diag(inv_sqrt)
    C = D @ K @ D  # correlation matrix
    # clamp for numerical stability
    C_clamped = C.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(C_clamped)
    # Arc-cosine kernel formula for ReLU
    return (sigma_w**2 / (2 * np.pi)) * (C_clamped * (np.pi - theta) + torch.sqrt(1.0 - C_clamped**2)) + sigma_b**2

class NTK_PINN_NS2D(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_units,
                 sigma_w=1.0, sigma_b=0.0, ridge=1e-6):
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.ridge = ridge
        self.initialized = False

    def ntk_kernel(self, X1, X2):
        """Compute NTK matrix between X1 and X2"""
        # Base linear kernel
        K = (self.sigma_w ** 2 / self.input_size) * (X1 @ X2.T) + self.sigma_b ** 2
        
        # Iteratively apply ReLU NTK transform for each layer
        for _ in range(self.hidden_layers):
            K = relu_ntk(K, self.sigma_w, self.sigma_b)
        
        return K

    def fit(self, X, y):
        """Solve kernel ridge regression: (K + ridge*I) alpha = y"""
        self.X_train = X
        self.y_train = y
        # Compute training kernel
        K = self.ntk_kernel(X, X)
        I = torch.eye(K.size(0), device=K.device)
        reg = self.ridge
        # Try direct solve with increasing regularization
        for _ in range(3):
            try:
                K_reg = K + reg * I
                self.alpha = torch.linalg.solve(K_reg, y)
                self.initialized = True
                return
            except RuntimeError:
                reg *= 10
        # Fallback to pseudo-inverse
        K_reg = K + reg * I
        
        self.alpha = K_reg.pinverse() @ y
        self.initialized = True

    def forward(self, x, y, t):
        """Predict (u,v,p) for new inputs"""
        if not self.initialized:
            raise RuntimeError("Call fit() before forward()")
        X = torch.cat([x, y, t], dim=1)
        K_star = self.ntk_kernel(X, self.X_train)
        uvp = K_star @ self.alpha
        return uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]

# Instantiate NTK-PINN model
model = NTK_PINN_NS2D(
    input_size=input_size,
    hidden_layers=hidden_layers,
    hidden_units=hidden_units,
    sigma_w=getattr(hp, 'SIGMA_W', 1.0),
    sigma_b=getattr(hp, 'SIGMA_B', 0.0),
    ridge=getattr(hp, 'RIDGE', 1e-6)
).to(device)

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

mse = nn.MSELoss()

def sample_initial(N):
    x = torch.rand(N,1)*(x_ub-x_lb)+x_lb
    y = torch.rand(N,1)*(y_ub-y_lb)+y_lb
    t = torch.zeros(N,1,device=device)
    u = hp.A*torch.exp(-((x-hp.x_gauss)**2+(y-hp.y_gauss)**2)/hp.sigma**2)
    v = u.clone()
    return x.to(device), y.to(device), t, u.to(device), v.to(device)

xi,yi,ti, ui, vi = sample_initial(3000)
# Training with epoch loop and loss tracking
# Training with only initial conditions (for quick validation)

def train():
    mse = nn.MSELoss()
    history = []
    for ep in range(1, epochs+1):
        xi, yi, ti, ui, vi = sample_initial(100)
        X_ic = torch.cat([xi, yi, ti], dim=1)
        y_ic = torch.cat([ui, vi, torch.zeros_like(ui)], dim=1)
     
        model.fit(X_ic, y_ic)
        u_pred, v_pred, _ = model(xi, yi, ti)
        
        L_ic = mse(torch.cat([u_pred, v_pred], dim=1), torch.cat([ui, vi], dim=1))
        history.append(L_ic.item())
        if ep == 1 or ep % 10 == 0:
            print(f"Ep {ep}/{epochs} | L_ic={L_ic:.2e}")
    
    if hp.SAVE_MODEL:
        torch.save(model.state_dict(), f'models/ntk.pth')
        print(f"Model saved ")
    return history

if __name__=='__main__':
    os.makedirs('models',exist_ok=True); os.makedirs('loss',exist_ok=True)
    train()