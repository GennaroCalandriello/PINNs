import pyvista as pv
import numpy as np
import glob
import os
import pickle

total_train_time = 1000


class DataSamplerVTK:
    def __init__(self, vtk_dir_pattern="VTK/cylinderFlux_*"):
        """
        Carica tutti gli snapshot internal.vtu ordinati per tempo.
        """
        self.snap_dirs = sorted(
            glob.glob(vtk_dir_pattern), key=lambda s: float(s.split("_")[-1])
        )
        self.data_all = []
        self.time_steps = []

        for snap_dir in self.snap_dirs:
            vtu_path = os.path.join(snap_dir, "internal.vtu")
            if not os.path.exists(vtu_path):
                continue
            t_str = os.path.basename(snap_dir).replace("cylinderFlux_", "")
            try:
                t = float(t_str)
            except ValueError:
                t = None
            if t > 0 and t <= total_train_time:
                print(f"Opening snapshot at t={t} (<= 100)")
                t = t / 1000
                mesh = pv.read(vtu_path)
                cell_centers = mesh.cell_centers().points[:, :2]
                U = mesh["U"][:, :2]
                neighbors = [mesh.cell_neighbors(i) for i in range(mesh.n_cells)]
                self.data_all.append(
                    {
                        "t": t,
                        "cell_centers": cell_centers,
                        "U": U,
                        "neighbors": neighbors,
                    }
                )
                self.time_steps.append(t)

        self.Nsnapshots = len(self.data_all)
        self.Ncells = self.data_all[0]["cell_centers"].shape[0]

    def sample_local_patch(self, N, center=None, random_time=True):
        """
        Estrae una patch locale con N celle, centrata su 'center' se specificato.
        """
        # 1. Scegli snapshot
        idx_time = np.random.randint(self.Nsnapshots) if random_time else 0
        d = self.data_all[idx_time]

        # 2. Trova la cella più vicina al centro desiderato, altrimenti random
        if center is not None:
            dist = np.linalg.norm(d["cell_centers"] - np.array(center), axis=1)
            start = np.argmin(dist)
        else:
            start = np.random.randint(self.Ncells)

        visited = set([start])
        to_visit = list(d["neighbors"][start])

        # 3. BFS per raccogliere una patch connessa di N celle
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

        # Se meno di N, riempi random (opzionale)
        if len(visited) < N:
            extra = set(np.random.choice(self.Ncells, N - len(visited), replace=False))
            visited |= extra

        idx = np.array(list(visited))[:N]
        cell_centers = d["cell_centers"][idx]
        U = d["U"][idx]
        t = d["t"]

        # Costruisci edge_index e neighbor list interni alla patch
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
        edge_index = np.array(edge_index).T  # shape [2, num_edges]

        return t, cell_centers, U, neighbors_batch, edge_index, idx

    def sample_time_series_patch(
        self, N, idx_cells=None, center=None, save_path="patch_data_60k.pkl"
    ):
        """
        Estrae la serie temporale di una patch locale (o di idx_cells prefissati) per tutti i time step.
        """
        if idx_cells is None:
            t0, _, _, _, _, idx_cells = self.sample_local_patch(
                N, center=center, random_time=False
            )
        results = []
        for d in self.data_all:
            cell_centers = d["cell_centers"][idx_cells]
            U = d["U"][idx_cells]
            t = d["t"]
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
        if save_path is not None:
            print(f"[DataSampler] Salvo patch su '{save_path}'")
            with open(save_path, "wb") as f:
                pickle.dump({"results": results, "idx_cells": idx_cells}, f)
        return results, idx_cells


if __name__ == "__main__":

    sampler = DataSamplerVTK("VTK/cylinderFlux_*")

    print(f"Loaded {sampler.Nsnapshots} snapshots with {sampler.Ncells} cells each.")
    results, idx_cells = sampler.sample_time_series_patch(N=10)
    print(results)
# # Estrai una patch locale di 1024 celle mesh-connected in un time step random
# t, centers, U, neighbors, edge_index, idx = sampler.sample_local_patch(N=100)
# print("Tempo:", t)
# print("Centro cella 0:", centers[0])
# print("Neighbors batch 0:", neighbors[0])  # sempre >0 se mesh normale!
# print("Edge_index shape:", edge_index.shape)

# # Serie temporale della stessa patch
# (results, idx_cells) = sampler.sample_time_series_patch(N=128)
# for t, centers, U, neighbors, edge_index in results:
#     print(f"t={t}, u/v prima cella:", U[0], "neighbors:", neighbors[0])
#     print(f"t={t}, u/v sec cella:", U[1], "neighbors:", neighbors)
