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
    y_lb = 0.0
    y_ub = 2.0
    x_lb = 0.0
    # Esempio di test:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import burgers2d
    
    burgers2d.setupB2d()  # Assicurati che la simulazione sia configurata correttamente
    # burgers2d.mainB2d()
    burgers2d.stepB2d()  # Esegui un passo della simulazione per generare le BC
    print(burgers2d.get_top_bc())  # Assicurati che le BC siano disponibili
    u = get_left_bc_from_cfd(np.array([[0, 0, 1]]))  # Esempio di input
    print("Left BC u_x:", u)

    # --- Setup ---
    # burgers2d.setupB2d()
    # burgers2d.mainB2d()
    

    # y_query = np.linspace(y_lb, y_ub, 400)   # o come hai definito il dominio
    # Nsteps_per_frame = 100                    # quante step CFD tra ogni frame animato
    # Nframes = 100                            # lunghezza animazione

    # fig, ax = plt.subplots(figsize=(6, 4))
    # line, = ax.plot([], [], lw=2, color='blue', label='u_x from CFD (bottom BC)')
    # ax.set_xlabel('x')
    # ax.set_ylabel('u_x')
    # ax.set_title('Evoluzione della bottom BC (u_x vs x)')
    # ax.set_xlim(y_lb, y_ub)
    # ax.set_ylim(-0.2, 0.2)  # Adatta in base ai tuoi range reali

    # def init():
    #     line.set_data([], [])
    #     return line,

    # def animate(frame):
    #     # Avanza la simulazione di Nsteps_per_frame
    #     for _ in range(Nsteps_per_frame):
    #         burgers2d.stepB2d()
    #     bc = np.array(burgers2d.get_bottom_bc(eps))  # shape: (dim, 5): t, x, y, u_x, u_y
    #     x_vals = bc[:, 1]
    #     ux_vals = bc[:, 3]
    #     line.set_data(x_vals, ux_vals)
    #     ax.set_title(f'Evoluzione della bottom BC (u_x vs x), frame {frame}')
    #     ax.set_ylim(np.min(ux_vals), np.max(ux_vals))  # Aggiorna l'asse y dinamicamente
    #     return line,

    # ani = animation.FuncAnimation(
    #     fig, animate, frames=Nframes, init_func=init, blit=True, interval=50
    # )

    # plt.legend()
    # plt.tight_layout()
    # plt.show()

