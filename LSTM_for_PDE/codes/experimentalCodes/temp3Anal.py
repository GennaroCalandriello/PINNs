import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Define the PINN LSTM architecture (must match the training definition)
class PINN_LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PINN_LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_seq, hidden=None):
        y_seq, hidden = self.lstm(x_seq, hidden)
        out = self.fc(y_seq)
        return out, hidden

# Load model
nu = 0.01
L, H, T = 1.0, 0.5, 1.0
model = PINN_LSTM(input_dim=3, hidden_dim=50, num_layers=2, output_dim=3)
model.load_state_dict(torch.load('pinn_lstm_ns2d.pth'))
model.eval()

# Create spatial grid
nx, ny = 100, 50
x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)
X, Y = np.meshgrid(x, y)
points = np.column_stack([X.ravel(), Y.ravel()])

# Time steps for evolution
time_steps = np.linspace(0, T, num=50)

# Compute u-velocity field over time
U_frames = []
with torch.no_grad():
    for t_val in time_steps:
        x_t = torch.tensor(points[:,0], dtype=torch.float32)
        y_t = torch.tensor(points[:,1], dtype=torch.float32)
        t_t = torch.full_like(x_t, t_val)
        inp = torch.stack([x_t, y_t, t_t], dim=1).unsqueeze(1)
        pred, _ = model(inp)
        U = pred[:,0,0].cpu().numpy().reshape(ny, nx)
        U_frames.append(U)

# Set up plot and animation
fig, ax = plt.subplots()
c = ax.pcolormesh(X, Y, U_frames[0])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Velocity $u$ at $t={time_steps[0]:.2f}$')

def update(frame):
    c.set_array(frame.ravel())
    # ax.set_title(f'Velocity $u$ at $t={time_steps[U_frames.index(frame)]:.2f}$')
    return c,

anim = FuncAnimation(fig, update, frames=U_frames, interval=100)
plt.show()
HTML(anim.to_jshtml())
