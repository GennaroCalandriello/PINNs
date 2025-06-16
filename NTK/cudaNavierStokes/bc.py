import numpy as np
import navier2d
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

import numpy as np

eps = 0.0  # controlla le BC, campiona più internamente
dim = navier2d.dim
x_min = navier2d.x_min
x_max = navier2d.x_max
y_min = navier2d.y_min
y_max = navier2d.y_max


def get_left_bc_from_cfd(x):
    """
    Interpola la condizione al bordo sinistro (x=const).
    x: batch di punti (N, 3), colonna x[:,2] = y
    Ritorna (N,2): [u_x, u_y] interpolati in y
    """
    bc = np.array(navier2d.get_left_bc())  # [t, x, y, ux, uy]
    y_bc = bc[:, 2]
    ux_bc = bc[:, 3]
    uy_bc = bc[:, 4]
    y_query = x[:, 2]
    # ux_out = np.interp(y_query, y_bc, ux_bc)
    # uy_out = np.interp(y_query, y_bc, uy_bc)
    ux_out = ux_bc
    uy_out = uy_bc
    return np.vstack([ux_out, uy_out]).T


def get_right_bc_from_cfd(x):
    """
    Interpola la condizione al bordo destro (x=const).
    """
    bc = np.array(navier2d.get_right_bc())
    y_bc = bc[:, 2]
    ux_bc = bc[:, 3]
    uy_bc = bc[:, 4]
    y_query = x[:, 2]
    ux_out = np.interp(y_query, y_bc, ux_bc)
    uy_out = np.interp(y_query, y_bc, uy_bc)
    # ux_out = ux_bc
    # uy_out = uy_bc
    return np.vstack([ux_out, uy_out]).T


def get_top_bc_from_cfd(x):
    """
    Interpola la condizione al bordo superiore (y=const).
    x: batch di punti (N, 3), colonna x[:,1] = x
    Ritorna (N,2): [u_x, u_y] interpolati in x
    """
    bc = np.array(navier2d.get_top_bc())  # [t, x, y, ux, uy]
    x_bc = bc[:, 1]
    ux_bc = bc[:, 3]
    uy_bc = bc[:, 4]
    x_query = x[:, 1]
    ux_out = np.interp(x_query, x_bc, ux_bc)
    uy_out = np.interp(x_query, x_bc, uy_bc)
    # ux_out = ux_bc
    # uy_out = uy_bc
    return np.vstack([ux_out, uy_out]).T


def get_bottom_bc_from_cfd(x):
    """
    Interpola la condizione al bordo inferiore (y=const).
    """
    bc = np.array(navier2d.get_bottom_bc())
    x_bc = bc[:, 1]
    ux_bc = bc[:, 3]
    uy_bc = bc[:, 4]
    x_query = x[:, 1]
    ux_out = np.interp(x_query, x_bc, ux_bc)
    uy_out = np.interp(x_query, x_bc, uy_bc)
    # ux_out = ux_bc
    # uy_out = uy_bc
    return np.vstack([ux_out, uy_out]).T


def get_left_bc_from_history(x, t_grid=None):
    """
    x: (batch, 3) punti [t, x, y]
    history: (N_time, N_points, 5) [t, x, y, ux, uy]
    t_grid: (N_time,) i tempi a cui hai salvato la storia delle BC

    Ritorna (batch, 2): u_x, u_y interpolati in (t, y)
    """
    history = np.array(navier2d.get_left_bc_history())
    if t_grid is None:
        t_grid = history[:, 0, 0]  # i tempi salvati
    # Estrai batch info
    t_query = x[:, 0]
    y_query = x[:, 2]
    # Interpola in tempo e spazio
    ux_vals = []
    uy_vals = []
    for t, y in zip(t_query, y_query):
        # trova frame temporale più vicino
        t_idx = np.argmin(np.abs(t_grid - t))
        bc_frame = history[t_idx]
        # Interpola in y
        y_bc = bc_frame[:, 2]
        ux_bc = bc_frame[:, 3]
        uy_bc = bc_frame[:, 4]
        ux_vals.append(np.interp(y, y_bc, ux_bc))
        uy_vals.append(np.interp(y, y_bc, uy_bc))
    return np.vstack([ux_vals, uy_vals]).T


def get_right_bc_from_history(x, t_grid=None):
    """
    x: (batch, 3) punti [t, x, y]
    history: (N_time, N_points, 5) [t, x, y, ux, uy]
    t_grid: (N_time,) tempi della storia BC

    Ritorna (batch, 2): u_x, u_y interpolati in (t, y)
    """
    history = np.array(navier2d.get_right_bc_history())
    if t_grid is None:
        t_grid = history[:, 0, 0]  # tempi salvati
    t_query = x[:, 0]
    y_query = x[:, 2]
    ux_vals = []
    uy_vals = []
    for t, y in zip(t_query, y_query):
        t_idx = np.argmin(np.abs(t_grid - t))
        bc_frame = history[t_idx]
        y_bc = bc_frame[:, 2]
        ux_bc = bc_frame[:, 3]
        uy_bc = bc_frame[:, 4]
        ux_vals.append(np.interp(y, y_bc, ux_bc))
        uy_vals.append(np.interp(y, y_bc, uy_bc))
    return np.vstack([ux_vals, uy_vals]).T


def get_top_bc_from_history(x, t_grid=None):
    """
    x: (batch, 3) punti [t, x, y]
    history: (N_time, N_points, 5) [t, x, y, ux, uy]
    t_grid: (N_time,) tempi storia BC

    Ritorna (batch, 2): u_x, u_y interpolati in (t, x)
    """
    history = np.array(navier2d.get_top_bc_history())
    if t_grid is None:
        t_grid = history[:, 0, 0]  # tempi salvati
    t_query = x[:, 0]
    x_query = x[:, 1]
    ux_vals = []
    uy_vals = []
    for t, xq in zip(t_query, x_query):
        t_idx = np.argmin(np.abs(t_grid - t))
        bc_frame = history[t_idx]
        x_bc = bc_frame[:, 1]
        ux_bc = bc_frame[:, 3]
        uy_bc = bc_frame[:, 4]
        ux_vals.append(np.interp(xq, x_bc, ux_bc))
        uy_vals.append(np.interp(xq, x_bc, uy_bc))
    return np.vstack([ux_vals, uy_vals]).T


def get_bottom_bc_from_history(x, t_grid=None):
    """
    x: (batch, 3) punti [t, x, y]
    history: (N_time, N_points, 5) [t, x, y, ux, uy]
    t_grid: (N_time,) tempi storia BC

    Ritorna (batch, 2): u_x, u_y interpolati in (t, x)
    """
    history = np.array(navier2d.get_bottom_bc_history())
    if t_grid is None:
        t_grid = history[:, 0, 0]  # tempi salvati
    t_query = x[:, 0]
    x_query = x[:, 1]
    ux_vals = []
    uy_vals = []
    for t, xq in zip(t_query, x_query):
        t_idx = np.argmin(np.abs(t_grid - t))
        bc_frame = history[t_idx]
        x_bc = bc_frame[:, 1]
        ux_bc = bc_frame[:, 3]
        uy_bc = bc_frame[:, 4]
        ux_vals.append(np.interp(xq, x_bc, ux_bc))
        uy_vals.append(np.interp(xq, x_bc, uy_bc))
    return np.vstack([ux_vals, uy_vals]).T


import numpy as np


def get_ic_from_snapshot(filename="snapshots.txt"):
    """
    Leggi la condizione iniziale (IC) da file snapshot.
    Restituisce (x, y, ux0, uy0) shape (dim*dim, 4)
    """
    data = np.loadtxt(filename, delimiter=",")
    N_space, M_times = data.shape
    assert N_space == 2 * dim * dim, "Dimensione dati errata"
    # IC = prima colonna (t=0)
    ux0 = data[: dim * dim, 0]
    uy0 = data[dim * dim : 2 * dim * dim, 0]
    # Coordinate griglia
    x_lin = np.linspace(-1, 1, dim)
    y_lin = np.linspace(-1, 1, dim)
    X, Y = np.meshgrid(x_lin, y_lin)
    x_flat = X.ravel()
    y_flat = Y.ravel()
    # Stack in shape (dim*dim, 4): x, y, ux, uy
    ic = np.column_stack([x_flat, y_flat, ux0, uy0])
    return ic


def interpolate_ic(points):
    """
    Interpola la IC estratta dal file snapshot nei punti (t, x, y) dati.
    Args:
        ic: array shape (dim*dim, 4): x, y, ux, uy (dati CFD)
        points: array shape (N, 3): t, x, y (punti della rete)
    Returns:
        interp_ic: array shape (N, 2): ux, uy interpolati nei punti richiesti

    """
    print("chiama")
    ic = get_ic_from_snapshot()  # shape (dim*dim, 4)

    x_grid = np.linspace(x_min, x_max, dim)
    y_grid = np.linspace(y_min, y_max, dim)
    ux_grid = ic[:, 2].reshape(dim, dim)
    uy_grid = ic[:, 3].reshape(dim, dim)

    # Nota: CFD solitamente [x, y], points[:,1]=x, points[:,2]=y
    ux_interp = RegularGridInterpolator(
        (x_grid, y_grid), ux_grid.T, bounds_error=False, fill_value=0.0
    )
    uy_interp = RegularGridInterpolator(
        (x_grid, y_grid), uy_grid.T, bounds_error=False, fill_value=0.0
    )

    xq = points[:, 1]
    yq = points[:, 2]
    query_pts = np.stack([xq, yq], axis=-1)
    ux_out = ux_interp(query_pts)
    uy_out = uy_interp(query_pts)

    return np.vstack([ux_out, uy_out]).T  # shape (N, 2)


if __name__ == "__main__":
    navier2d.setupNS2d()
    navier2d.mainNS2d()
    x = np.random.rand(10000, 3) * np.array(
        [1.0, x_max - x_min, y_max - y_min]
    ) + np.array(
        [0.0, x_min, y_min]
    )  # batch di punti (N, 3) [t, x, y]
    # ic = interpolate_ic(x)
    ic = navier2d.interpolate_ic_from_snapshots(x)

    # Plot ux and uy as scatter colormaps
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    sc0 = axs[0].scatter(x[:, 1], x[:, 2], c=ic[:, 0], cmap="viridis")
    axs[0].set_title("ux")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    plt.colorbar(sc0, ax=axs[0])

    sc1 = axs[1].scatter(x[:, 1], x[:, 2], c=ic[:, 1], cmap="viridis")
    axs[1].set_title("uy")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    plt.colorbar(sc1, ax=axs[1])

    sc2 = axs[2].scatter(
        x[:, 1], x[:, 2], c=np.sqrt(ic[:, 1] ** 2 + ic[:, 0] ** 2), cmap="viridis"
    )
    axs[2].set_title("uy")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    plt.colorbar(sc2, ax=axs[2])

    plt.tight_layout()
    plt.show()
