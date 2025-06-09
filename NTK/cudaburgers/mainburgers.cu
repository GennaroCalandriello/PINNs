#include "functions.h"
#include "obstacles.h"
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

const char *SNAPSHOT_FILENAME = "snapshots.txt";
const bool implicit = true; // true for implicit, false for explicit

// Assumed available:
// struct Vector2f { float x, y; ... };
// struct Vector3f { float x, y, z; ... };
// int IND(int i, int j, int dim);
// Vector2f bilinearInterpolation(Vector2f pos, const Vector2f *field, int dim);
// ... periodic, etc ...
__global__ void BurgersConvectionStep(Vector2f *ustar, const Vector2f *u,
                                      float dt, float dx, int dim) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= dim || j >= dim)
    return;

  int idx = i * dim + j;

  if (i == 0 || j == 0 || i == dim - 1 || j == dim - 1) {
    ustar[idx].x = 0.0f;
    ustar[idx].y = 0.0f;
    return;
  }

  int idx_ip = (i + 1) * dim + j;
  int idx_im = (i - 1) * dim + j;
  int idx_jp = i * dim + (j + 1);
  int idx_jm = i * dim + (j - 1);

  float uij = u[idx].x;
  float vij = u[idx].y;

  float dudx = (u[idx_ip].x - u[idx_im].x) / (2.0f * dx);
  float dudy = (u[idx_jp].x - u[idx_jm].x) / (2.0f * dx);
  float dvdx = (u[idx_ip].y - u[idx_im].y) / (2.0f * dx);
  float dvdy = (u[idx_jp].y - u[idx_jm].y) / (2.0f * dx);

  ustar[idx].x = uij - dt * (uij * dudx + vij * dudy);
  ustar[idx].y = vij - dt * (uij * dvdx + vij * dvdy);
}

__global__ void BurgersDiffusionJacobi(Vector2f *unew, const Vector2f *ustar,
                                       float nu, float dt, float dx, int dim) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= dim || j >= dim)
    return;

  int idx = i * dim + j;

  if (i == 0 || j == 0 || i == dim - 1 || j == dim - 1) {
    unew[idx].x = 0.0f;
    unew[idx].y = 0.0f;
    return;
  }

  int idx_ip = (i + 1) * dim + j;
  int idx_im = (i - 1) * dim + j;
  int idx_jp = i * dim + (j + 1);
  int idx_jm = i * dim + (j - 1);

  float alpha = dx * dx / (nu * dt);
  float beta = 4.0f + alpha;

  unew[idx].x = (ustar[idx_ip].x + ustar[idx_im].x + ustar[idx_jp].x +
                 ustar[idx_jm].x + alpha * ustar[idx].x) /
                beta;

  unew[idx].y = (ustar[idx_ip].y + ustar[idx_im].y + ustar[idx_jp].y +
                 ustar[idx_jm].y + alpha * ustar[idx].y) /
                beta;
}

__global__ void BurgersExplicitKernel(Vector2f *unew, const Vector2f *u,
                                      float nu, float dt, float dx, int dim) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int idx = i * dim + j;

  // Dirichlet boundary: u = 0 at the border
  if (i == 0 || j == 0 || i == dim - 1 || j == dim - 1) {
    unew[idx].x = 0.0f;
    unew[idx].y = 0.0f;
    return;
  }

  // Indici vicini
  int ip = (i + 1);
  int im = (i - 1);
  int jp = (j + 1);
  int jm = (j - 1);

  int idx_ip = ip * dim + j;
  int idx_im = im * dim + j;
  int idx_jp = i * dim + jp;
  int idx_jm = i * dim + jm;

  float uij = u[idx].x;
  float vij = u[idx].y;

  float dudx = (u[idx_ip].x - u[idx_im].x) / (2.0f * dx);
  float dudy = (u[idx_jp].x - u[idx_jm].x) / (2.0f * dx);
  float dvdx = (u[idx_ip].y - u[idx_im].y) / (2.0f * dx);
  float dvdy = (u[idx_jp].y - u[idx_jm].y) / (2.0f * dx);

  float lapu =
      (u[idx_ip].x + u[idx_im].x + u[idx_jp].x + u[idx_jm].x - 4.0f * uij) /
      (dx * dx);
  float lapv =
      (u[idx_ip].y + u[idx_im].y + u[idx_jp].y + u[idx_jm].y - 4.0f * vij) /
      (dx * dx);

  unew[idx].x = uij - dt * (uij * dudx + vij * dudy) + nu * dt * lapu;
  unew[idx].y = vij - dt * (uij * dvdx + vij * dvdy) + nu * dt * lapv;
}

int main(int argc, char **argv) {
  // QUESTO MAIN è PER UNO SCHEMA ESPLICITO
  int framecount = 0;

  Vector2f *u = (Vector2f *)malloc(dim * dim * sizeof(Vector2f));
  Vector2f *unew = (Vector2f *)malloc(dim * dim * sizeof(Vector2f));
  Vector2f *dev_ustar; // per il passo convettivo

  // Condizioni iniziali
  float M_PI = 3.1415926535f;

  float x_min = 0.0f, x_max = 2.0f;
  float y_min = 0.0f, y_max = 2.0f;
  float dx = (x_max - x_min) / (dim - 1);
  float dy = (y_max - y_min) / (dim - 1);

  for (unsigned i = 0; i < dim; i++) {
    float x = x_min + i * dx;
    for (unsigned j = 0; j < dim; j++) {
      float y = y_min + j * dy;
      unsigned idx = i * dim + j;
      u[idx].x = sinf(M_PI * x) * cosf(M_PI * y);
      u[idx].y = cosf(M_PI * x) * sinf(M_PI * y);
      unew[idx].x = 0.0f;
      unew[idx].y = 0.0f;
    }
  }

  // Allocazione device
  Vector2f *dev_u, *dev_unew;
  cudaMalloc(&dev_u, dim * dim * sizeof(Vector2f));
  cudaMalloc(&dev_unew, dim * dim * sizeof(Vector2f));
  cudaMalloc(&dev_ustar, dim * dim * sizeof(Vector2f));
  cudaMemcpy(dev_u, u, dim * dim * sizeof(Vector2f), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_unew, unew, dim * dim * sizeof(Vector2f),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_ustar, unew, dim * dim * sizeof(Vector2f),
             cudaMemcpyHostToDevice);

  // Per salvataggio snapshot
  vector<vector<float>>
      snapshots; // ogni snapshot: [ux(0), ..., ux(N), uy(0), ..., uy(N)]

  dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
  dim3 blocks((dim + BLOCKSIZEX - 1) / BLOCKSIZEX,
              (dim + BLOCKSIZEY - 1) / BLOCKSIZEY);

  // --- MAIN SIMULATION LOOP ---
  if (implicit) {
    printf("Running implicit scheme...\n");
    while (framecount < MAX_FRAMES) {
      BurgersConvectionStep<<<blocks, threads>>>(dev_ustar, dev_u, timestep, dx,
                                                 dim);
      cudaDeviceSynchronize();

      for (int iter = 0; iter < NUM_OF_DIFFUSION_STEPS; ++iter) {
        BurgersDiffusionJacobi<<<blocks, threads>>>(
            dev_unew, dev_ustar, viscosity, timestep, dx, dim);
        cudaDeviceSynchronize();
        swap(dev_unew, dev_ustar);
      }

      swap(dev_u, dev_ustar);

      if (framecount % SNAPSHOT_INTERVAL == 0) {
        cudaMemcpy(u, dev_u, dim * dim * sizeof(Vector2f),
                   cudaMemcpyDeviceToHost);
        vector<float> snapshot;
        snapshot.reserve(dim * dim * 2);
        for (int i = 0; i < dim * dim; ++i)
          snapshot.push_back(u[i].x);
        for (int i = 0; i < dim * dim; ++i)
          snapshot.push_back(u[i].y);
        snapshots.push_back(snapshot);
      }
      framecount++;
    }
  } else {
    printf("Running explicit scheme...\n");
    while (framecount < MAX_FRAMES) {
      // Step esplicito unico
      BurgersExplicitKernel<<<blocks, threads>>>(dev_unew, dev_u, viscosity,
                                                 timestep, dx, dim);
      cudaDeviceSynchronize();

      // swap buffer
      std::swap(dev_u, dev_unew);

      // Salva snapshot a intervalli regolari
      if (framecount % SNAPSHOT_INTERVAL == 0) {
        cudaMemcpy(u, dev_u, dim * dim * sizeof(Vector2f),
                   cudaMemcpyDeviceToHost);
        vector<float> snapshot;
        snapshot.reserve(dim * dim * 2);
        for (int i = 0; i < dim * dim; ++i)
          snapshot.push_back(u[i].x);
        for (int i = 0; i < dim * dim; ++i)
          snapshot.push_back(u[i].y);
        snapshots.push_back(snapshot);
      }
      framecount++;
    }
  }

  // --- SCRITTURA SU FILE ---
  int M_times = snapshots.size();
  int N_space = snapshots[0].size();

  std::ofstream out(SNAPSHOT_FILENAME);
  std::cout << "Snapshot matrix size (time slices x spatial): " << M_times
            << " x " << N_space << "\n";
  for (int i = 0; i < N_space; ++i) {
    for (int j = 0; j < M_times; ++j) {
      out << snapshots[j][i];
      if (j < M_times - 1)
        out << ",";
    }
    out << "\n";
  }
  out.close();
  printf("Snapshots saved in %s\n", SNAPSHOT_FILENAME);

  // Cleanup
  free(u);
  free(unew);
  cudaFree(dev_u);
  cudaFree(dev_unew);

  return 0;
}