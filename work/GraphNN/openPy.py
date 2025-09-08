import numpy as np
import os
import glob


class DataSamplerOpenFoam:
    def __init__(self, filepath_pattern):
        """
        Carica tutti i file velocity_<time>.dat generati da OpenFOAM.
        Costruisce un array (N, 5): t, x, y, u, v
        """
        self.data = self.load_velocity_files(filepath_pattern)

    def load_velocity_files(self, pattern):
        data_all = []
        files = sorted(
            glob.glob(pattern),
            key=lambda f: float(os.path.basename(f).split("_")[1].replace(".dat", "")),
        )
        for file in files:
            t_str = os.path.basename(file).split("_")[1].replace(".dat", "")
            try:
                t = float(t_str)
            except ValueError:
                continue

            raw = np.loadtxt(file, skiprows=1)
            x, y = raw[:, 0], raw[:, 1]
            u, v = raw[:, 2], raw[:, 3]

            t_arr = np.full_like(x, t)
            stacked = np.column_stack([t_arr, x, y, u, v])
            data_all.append(stacked)
            # print("tarr", t_arr)
        return np.vstack(data_all)

    def sample(self, N):
        """
        Estrae N punti casuali per training.
        Ritorna X: (N, 3) [t, x, y]  e  Y: (N, 2) [u, v]
        """
        idx = np.random.choice(len(self.data), size=N, replace=False)
        batch = self.data[idx]
        return batch[:, :3], batch[:, 3:]


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata


def animate_magnitude(data_sampler, grid_res=300, interval=10, save_as="cyl.gif"):
    """
    Crea un'animazione della magnitude sqrt(u^2 + v^2) per tutti i time-step disponibili.
    data_sampler: istanza di DataSamplerOpenFoam già caricata.
    grid_res: risoluzione della griglia (default 100x100).
    interval: intervallo tra frame in ms.
    save_as: se vuoi salvare l'animazione, specifica il nome file .mp4
    """

    data = data_sampler.data
    times = np.unique(data[:, 0])

    # Estrai i limiti delle coordinate
    x_min, x_max = data[:, 1].min(), data[:, 1].max()
    y_min, y_max = data[:, 2].min(), data[:, 2].max()
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_res), np.linspace(y_min, y_max, grid_res)
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = None

    def get_frame(t):
        # Seleziona tutti i punti a tempo t
        frame = data[data[:, 0] == t]
        x, y, u, v = frame[:, 1], frame[:, 2], frame[:, 3], frame[:, 4]
        mag = np.sqrt(u**2 + v**2)
        # Interpola la magnitude sulla griglia per una visualizzazione smooth
        mag_grid = griddata(
            (x, y), mag, (grid_x, grid_y), method="linear", fill_value=np.nan
        )
        return mag_grid

    def update(i):
        ax.clear()
        t = times[i]
        mag_grid = get_frame(t)
        im = ax.imshow(
            mag_grid,
            extent=(x_min, x_max, y_min, y_max),
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        ax.set_title(f"t = {t:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return [im]

    ani = FuncAnimation(fig, update, frames=len(times), interval=interval, blit=False)

    plt.tight_layout()
    if save_as is not None:
        ani.save(save_as, writer="ffmpeg")
    else:
        plt.show()


if __name__ == "__main__":
    sampler = DataSamplerOpenFoam("cylinderFoam/velocity_*.dat")
    animate_magnitude(sampler, grid_res=300)
