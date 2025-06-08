import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags, eye, kron, csc_matrix
from scipy.sparse.linalg import spsolve

# Parameters
nx, ny = 100, 100       # grid size (smaller for fast demo)
nt = 500              # time steps
Lx, Ly = 2.0, 2.0
dx, dy = Lx/(nx-1), Ly/(ny-1)
nu = 0.01
dt = 0.001  # Try increasing if stable (e.g., dt=0.005)

# Grids
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

def ic_u(x, y): return np.sin(np.pi * x) * np.cos(np.pi * y)
def ic_v(x, y): return  np.cos(np.pi * x) * np.sin(np.pi * y)
u = ic_u(X, Y)
v = ic_v(X, Y)

# For animation
u_frames, v_frames, mag_frames = [], [], []

# 1D Laplacian matrices (Dirichlet 0 BC)
main_diag = -2 * np.ones(nx)
side_diag = np.ones(nx-1)
Lx_mat = diags([main_diag, side_diag, side_diag], [0, -1, 1], shape=(nx, nx)).toarray() / dx**2
main_diag = -2 * np.ones(ny)
side_diag = np.ones(ny-1)
Ly_mat = diags([main_diag, side_diag, side_diag], [0, -1, 1], shape=(ny, ny)).toarray() / dy**2

# 2D Laplacian
Ix = eye(nx)
Iy = eye(ny)
Laplacian = kron(Iy, csc_matrix(Lx_mat)) + kron(csc_matrix(Ly_mat), Ix)  # Shape (nx*ny, nx*ny)

I_big = eye(nx*ny)
A = (I_big - 0.5 * nu * dt * Laplacian).tocsc()
B = (I_big + 0.5 * nu * dt * Laplacian).tocsc()

def flatten(f):
    return f.reshape(-1)

def reshape(f):
    return f.reshape((nx, ny))

for it in range(nt):
    u_frames.append(u.copy())
    v_frames.append(v.copy())
    mag_frames.append(np.sqrt(u**2 + v**2))

    # Nonlinear terms (explicit)
    u_x = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)
    u_y = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dy)
    v_x = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2*dx)
    v_y = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2*dy)
    nonlinear_u = -u * u_x - v * u_y
    nonlinear_v = -u * v_x - v * v_y

    # Right-hand side for CN step
    rhs_u = flatten(B @ flatten(u) + dt * flatten(nonlinear_u))
    rhs_v = flatten(B @ flatten(v) + dt * flatten(nonlinear_v))

    # Solve for next u, v
    u_new = spsolve(A, rhs_u)
    v_new = spsolve(A, rhs_v)

    u = reshape(u_new)
    v = reshape(v_new)

# Animate |u,v|
fig, ax = plt.subplots(figsize=(6,5))
cax = ax.pcolormesh(X, Y, mag_frames[0], shading='auto', cmap='viridis', vmin=0, vmax=np.max(mag_frames))
title = ax.set_title("t = 0")
plt.colorbar(cax, ax=ax, label=r'$|\vec{u}|$')
ax.set_xlabel('x'); ax.set_ylabel('y')

def update(frame):
    cax.set_array(mag_frames[frame].ravel())
    title.set_text(f't = {frame*dt:.3f}')
    return cax, title

ani = animation.FuncAnimation(fig, update, frames=nt, blit=False, interval=40)
plt.show()
