import pyvista as pv
import numpy as np
import glob
import os
import pickle
import concurrent.futures

# patch_size = 40000
# path_pkl = f"patch_{patch_size//1000}k_uniform.pkl"
path_time = "experiments_"
dataSamplerPath = "VTKs/VTK7/experiments_*"
savingPath = "patches/experiments.pkl"
# path_time = "cylinderFlux_"

total_train_time = 1000


def _read_snapshot(snap_dir, total_train_time=total_train_time):
    vtu_path = os.path.join(snap_dir, "internal.vtu")
    print(f"Multiprocessing: Reading snapshot from {vtu_path}")
    if not os.path.exists(vtu_path):
        return None
    t_str = os.path.basename(snap_dir).replace(path_time, "")
    try:
        t = float(t_str)
    except ValueError:
        return None
    if t is None or not (0 < t <= total_train_time):
        return None
    t = t / 1000
    mesh = pv.read(vtu_path)
    pts = mesh.cell_centers().points
    cell_centers = pts[:, [0, 2]]  # (x, z)
    # cell_centers = mesh.cell_centers().points[:, :2]
    # print(cell_centers)
    # U = mesh["U"][:, :2]
    U = mesh["U"][:, [0, 2]]  # keep components matching x,z
    # print(U)
    neighbors = [mesh.cell_neighbors(i) for i in range(mesh.n_cells)]
    return {"t": t, "cell_centers": cell_centers, "U": U, "neighbors": neighbors}


class DataSamplerVTK:
    def __init__(self, vtk_dir_pattern, n_workers=10):
        self.snap_dirs = sorted(
            glob.glob(vtk_dir_pattern), key=lambda s: float(s.split("_")[-1])
        )
        print(
            f"Found {len(self.snap_dirs)} snapshots, using {n_workers} workers for load..."
        )

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_read_snapshot, self.snap_dirs))
        self.data_all = [r for r in results if r is not None]
        self.time_steps = [d["t"] for d in self.data_all]
        assert len(self.data_all) > 0, "No data loaded!"
        self.Nsnapshots = len(self.data_all)
        self.Ncells = self.data_all[0]["cell_centers"].shape[0]

    # ---- BFS mesh-connected patch, centrata se vuoi ----
    def sample_local_patch(self, N, center=None, random_time=True):
        idx_time = np.random.randint(self.Nsnapshots) if random_time else 0
        d = self.data_all[idx_time]
        if center is not None:
            dist = np.linalg.norm(d["cell_centers"] - np.array(center), axis=1)
            start = np.argmin(dist)
        else:
            start = np.random.randint(self.Ncells)
        visited = set([start])
        to_visit = list(d["neighbors"][start])
        # ♦prendo tutta la mesh
        while len(visited) < N and to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
            visited.add(current)
            to_visit.extend(
                [
                    nb
                    for nb in d["neighbors"][current]
                    if nb not in visited and nb not in to_visit
                ]
            )
        if len(visited) < N:
            extra = set(np.random.choice(self.Ncells, N - len(visited), replace=False))
            visited |= extra
        idx = np.array(list(visited))[:N]
        return self._extract_patch(idx, d, t=d["t"])

    # ---- PATCH UNIFORME (sparsa) ----
    def sample_uniform_patch(self, N, snapshot_idx=0):
        d = self.data_all[snapshot_idx]
        idx = np.random.choice(self.Ncells, N, replace=False)
        return self._extract_patch(idx, d, t=d["t"])

    # ---- PATCH A GRIGLIA ----
    def sample_grid_patch(self, N, grid_nx=10, grid_ny=10, snapshot_idx=0):
        d = self.data_all[snapshot_idx]
        cc = d["cell_centers"]
        x_bins = np.linspace(0, 20000, grid_nx + 1)
        y_bins = np.linspace(0, 20000, grid_ny + 1)
        idx = []
        for i in range(grid_nx):
            for j in range(grid_ny):
                mask = (
                    (cc[:, 0] >= x_bins[i])
                    & (cc[:, 0] < x_bins[i + 1])
                    & (cc[:, 1] >= y_bins[j])
                    & (cc[:, 1] < y_bins[j + 1])
                )
                idx_in_bin = np.where(mask)[0]
                if len(idx_in_bin) > 0:
                    pick = np.random.choice(idx_in_bin)
                    idx.append(pick)
        idx = np.array(idx)[:N]
        return self._extract_patch(idx, d, t=d["t"])

    # ---- ESTRAE DATI DI UNA PATCH GIA' SCELTA ----
    def _extract_patch(self, idx, d, t=None):
        cell_centers = d["cell_centers"][idx]
        U = d["U"][idx]
        if t is None:
            t = d["t"]
        idx_map = {orig_idx: i for i, orig_idx in enumerate(idx)}
        neighbors_batch = []
        edge_index = []
        for i, orig_idx in enumerate(idx):
            nbs = d["neighbors"][orig_idx]
            nbs_in_patch = [idx_map[nb] for nb in nbs if nb in idx_map]
            neighbors_batch.append(nbs_in_patch)
            for nb in nbs:
                if nb in idx_map:
                    edge_index.append([i, idx_map[nb]])
        edge_index = np.array(edge_index).T
        return t, cell_centers, U, neighbors_batch, edge_index, idx

    # ---- SERIE TEMPORALE della stessa PATCH ----
    def sample_time_series_patch(
        self,
        N,
        idx_cells=None,
        mode="local",
        center=None,
        grid_nx=10,
        grid_ny=10,
        save_path=None,
    ):
        """
        mode: "local", "uniform", "grid"
        """
        if save_path is not None and os.path.exists(save_path):
            print(f"[DataSampler] Cancello '{save_path}'")
            os.remove(save_path)
            with open(save_path, "rb") as f:
                data = pickle.load(f)
            return data["results"], data["idx_cells"]

        N = len(self.data_all[0]["cell_centers"]) if N is None else N

        # SCEGLI la patch una volta per tutte
        if idx_cells is None:
            if mode == "local":
                t0, _, _, _, _, idx_cells = self.sample_local_patch(
                    N, center=center, random_time=False
                )
            elif mode == "uniform":
                t0, _, _, _, _, idx_cells = self.sample_uniform_patch(N)
            elif mode == "grid":
                t0, _, _, _, _, idx_cells = self.sample_grid_patch(N, grid_nx, grid_ny)
            else:
                raise ValueError(f"Unknown patch mode: {mode}")

        results = []
        for d in self.data_all:
            cell_centers = d["cell_centers"][idx_cells]
            U = d["U"][idx_cells]
            t = d["t"]
            print("times", t)
            neighbors_full = d["neighbors"]
            idx_map = {orig_idx: i for i, orig_idx in enumerate(idx_cells)}
            neighbors_batch = []
            edge_index = []
            for i, orig_idx in enumerate(idx_cells):
                nbs = neighbors_full[orig_idx]
                nbs_in_patch = [idx_map[nb] for nb in nbs if nb in idx_map]
                neighbors_batch.append(nbs_in_patch)
                for nb in nbs:
                    if nb in idx_map:
                        edge_index.append([i, idx_map[nb]])
            edge_index = np.array(edge_index).T
            results.append((t, cell_centers, U, neighbors_batch, edge_index))
        print(
            f"[DataSampler] Patch di {len(idx_cells)} celle, {len(results)} snapshot caricati"
        )
        if save_path is not None:
            print(f"[DataSampler] Salvo patch su '{save_path}'")
            with open(save_path, "wb") as f:
                pickle.dump({"results": results, "idx_cells": idx_cells}, f)
        return results, idx_cells

    # ---- DEBUG PLOT per vedere il campionamento ----
    def plot_patch(self, idx, snapshot_idx=0):
        d = self.data_all[snapshot_idx]
        cc = d["cell_centers"]
        import matplotlib.pyplot as plt

        plt.scatter(cc[:, 0], cc[:, 1], s=1, label="mesh")
        plt.scatter(cc[idx, 0], cc[idx, 1], c="r", s=8, label="sampled")
        plt.legend()
        plt.axis("equal")
        plt.title("Distribuzione patch")
        plt.show()

    def plot_sampled_patch(
        self, idx_cells, snapshot_idx=0, plot_edges=True, show_indices=False
    ):
        """
        Plot only the sampled patch (submesh) with connectivity.
        idx_cells: indices of the sampled cells (the patch), as returned by sampler.sample_time_series_patch(...)
        snapshot_idx: which snapshot to plot from
        plot_edges: if True, draw lines for neighbor connectivity
        show_indices: if True, show the node indices as text
        """
        d = self.data_all[snapshot_idx]
        cc = d["cell_centers"][idx_cells]  # Only the patch cell centers
        neighbors = d["neighbors"]
        idx_map = {orig_idx: i for i, orig_idx in enumerate(idx_cells)}

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.scatter(
            cc[:, 0], cc[:, 1], s=12, color="r", zorder=2, label="Sampled cells"
        )

        # Plot edges (connectivity in the patch)
        if plot_edges:
            for i, orig_idx in enumerate(idx_cells):
                for nb in neighbors[orig_idx]:
                    if nb in idx_map:
                        j = idx_map[nb]
                        x = [cc[i, 0], cc[j, 0]]
                        y = [cc[i, 1], cc[j, 1]]
                        plt.plot(x, y, "gray", linewidth=0.8, alpha=0.6, zorder=1)

        # Optionally show indices
        if show_indices:
            for i in range(len(idx_cells)):
                plt.text(cc[i, 0], cc[i, 1], str(i), fontsize=7, color="k")

        plt.title("Sampled mesh patch with neighbor connectivity")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_patch_real_mesh(vtu_path, idx_cells):
    # Carica tutta la mesh
    mesh = pv.read(vtu_path)
    # Crea una submesh con SOLO le celle campionate
    # pv.extract_cells prende una maschera booleana o indici
    submesh = mesh.extract_cells(idx_cells)

    # Plotta la submesh (mesh reale, celle vere, bordi veri)
    pl = pv.Plotter()
    pl.add_mesh(
        submesh, show_edges=True, color="white", edge_color="black", opacity=1.0
    )
    pl.show_grid()
    pl.show()


if __name__ == "__main__":
    import pickle

    # path_pkl = savingPath  # file pickle con i dati della patch
    # with open(path_pkl, "rb") as f:
    #     data = pickle.load(f)

    # results = data["results"]
    # idx_cells = data["idx_cells"]
    # plot_patch_real_mesh(savingPath, idx_cells)
    sampler = DataSamplerVTK(dataSamplerPath)
    results, idx_cells = sampler.sample_time_series_patch(
        N=None, mode="uniform", save_path=savingPath
    )
    # # # sampler.plot_patch(idx_cells, snapshot_idx=0)
    # sampler.plot_sampled_patch(
    #     idx_cells, snapshot_idx=0, plot_edges=True, show_indices=True
    # )
    # with open(path_pkl, "rb") as f:
    #     data = pickle.load(f)

    # results = data["results"]
    # idx_cells = data["idx_cells"]

    # results, idx_cells = sampler.sample_time_series_patch(
    #     N=60000, mode="local", center=(10000, 10000), save_path="patch_60k_local.pkl"
    # )
