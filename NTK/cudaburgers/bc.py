import numpy as np
import burgers2d
from scipy.interpolate import griddata

import numpy as np
eps = 0.0 #controlla le BC, campiona più internamente

def get_left_bc_from_cfd(x):
    """
    Interpola la condizione al bordo sinistro (x=const).
    x: batch di punti (N, 3), colonna x[:,2] = y
    Ritorna (N,2): [u_x, u_y] interpolati in y
    """
    bc = np.array(burgers2d.get_left_bc())  # [t, x, y, ux, uy]
    y_bc = bc[:,2]
    ux_bc = bc[:,3]
    uy_bc = bc[:,4]
    y_query = x[:,2]
    # ux_out = np.interp(y_query, y_bc, ux_bc)
    # uy_out = np.interp(y_query, y_bc, uy_bc)
    ux_out = ux_bc
    uy_out = uy_bc
    return np.vstack([ux_out, uy_out]).T

def get_right_bc_from_cfd(x):
    """
    Interpola la condizione al bordo destro (x=const).
    """
    bc = np.array(burgers2d.get_right_bc())
    y_bc = bc[:,2]
    ux_bc = bc[:,3]
    uy_bc = bc[:,4]
    y_query = x[:,2]
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
    bc = np.array(burgers2d.get_top_bc())  # [t, x, y, ux, uy]
    x_bc = bc[:,1]
    ux_bc = bc[:,3]
    uy_bc = bc[:,4]
    x_query = x[:,1]
    ux_out = np.interp(x_query, x_bc, ux_bc)
    uy_out = np.interp(x_query, x_bc, uy_bc)
    # ux_out = ux_bc
    # uy_out = uy_bc
    return np.vstack([ux_out, uy_out]).T

def get_bottom_bc_from_cfd(x):
    """
    Interpola la condizione al bordo inferiore (y=const).
    """
    bc = np.array(burgers2d.get_bottom_bc())
    x_bc = bc[:,1]
    ux_bc = bc[:,3]
    uy_bc = bc[:,4]
    x_query = x[:,1]
    ux_out = np.interp(x_query, x_bc, ux_bc)
    uy_out = np.interp(x_query, x_bc, uy_bc)
    # ux_out = ux_bc
    # uy_out = uy_bc
    return np.vstack([ux_out, uy_out]).T

def get_left_bc_from_history(x,t_grid=None):
    """
    x: (batch, 3) punti [t, x, y]
    history: (N_time, N_points, 5) [t, x, y, ux, uy]
    t_grid: (N_time,) i tempi a cui hai salvato la storia delle BC

    Ritorna (batch, 2): u_x, u_y interpolati in (t, y)
    """
    history=np.array(burgers2d.get_left_bc_history()) 
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
    history = np.array(burgers2d.get_right_bc_history())
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
    history = np.array(burgers2d.get_top_bc_history())
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
    history = np.array(burgers2d.get_bottom_bc_history())
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

    
if __name__ == "__main__":    
    burgers2d.setupB2d()  # Assicurati che la simulazione sia configurata correttamente
    burgers2d.mainB2d()
    
    # x = 0+(2-0)*np.random.rand(100000, 3)
    # bch= get_top_bc_from_history(x)
    # print("Top BC from history:", bch)
    # #salva su file:
    # np.savetxt("bottom_bc_history.txt", bch)

