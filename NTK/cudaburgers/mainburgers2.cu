#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

#define BLOCKSIZEX 16
#define BLOCKSIZEY 16
#define MAX_FRAMES 500
#define SNAPSHOT_INTERVAL 50
#define NUM_OF_DIFFUSION_STEPS 20

const float viscosity = 0.1f;
const float timestep = 0.001f;
#const int dim = 128;

struct Vector2f { float x, y; };

__global__ void BurgersConvectionStep(Vector2f *ustar, const Vector2f *u,
                                      float dt, float dx, int dim) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= dim || j >= dim)
    return;

  int idx = i * dim + j;

  if (i == 0 || j == 0 || i == dim - 1 || j == dim - 1) {
    ustar[idx] = {0.0f, 0.0f};
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
    unew[idx] = {0.0f, 0.0f};
    return;
  }

  int idx_ip = (i + 1) * dim + j;
  int idx_im = (i - 1) * dim + j;
  int idx_jp = i * dim + (j + 1);
  int idx_jm = i * dim + (j - 1);

  float alpha = dx * dx / (nu * dt);
  float beta = 4.0f + alpha;

  unew[idx].x = (ustar[idx_ip].x + ustar[idx_im].x + ustar[idx_jp].x +
                 ustar[idx_jm].x + alpha * ustar[idx].x) / beta;

  unew[idx].y = (ustar[idx_ip].y + ustar[idx_im].y + ustar[idx_jp].y +
                 ustar[idx_jm].y + alpha * ustar[idx].y) / beta;
}

int main() {
  Vector2f *u = new Vector2f[dim * dim];
  Vector2f *dev_u, *dev_unew, *dev_ustar;

  float x_min = 0.0f, x_max = 2.0f;
  float dx = (x_max - x_min) / (dim - 1);

  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      int idx = i * dim + j;
      float x = x_min + i * dx;
      float y = x_min + j * dx;
      u[idx].x = 0.1f * sinf(M_PI * x) * cosf(M_PI * y);
      u[idx].y = 0.1f * cosf(M_PI * x) * sinf(M_PI * y);
      if (i == 0 || j == 0 || i == dim - 1 || j == dim - 1)
        u[idx] = {0.0f, 0.0f};
    }
  }

  cudaMalloc(&dev_u, dim * dim * sizeof(Vector2f));
  cudaMalloc(&dev_unew, dim * dim * sizeof(Vector2f));
  cudaMalloc(&dev_ustar, dim * dim * sizeof(Vector2f));
  cudaMemcpy(dev_u, u, dim * dim * sizeof(Vector2f), cudaMemcpyHostToDevice);

  dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
  dim3 blocks((dim + BLOCKSIZEX - 1) / BLOCKSIZEX,
              (dim + BLOCKSIZEY - 1) / BLOCKSIZEY);

  std::vector<std::vector<float>> snapshots;

  for (int framecount = 0; framecount < MAX_FRAMES; ++framecount) {
    BurgersConvectionStep<<<blocks, threads>>>(dev_ustar, dev_u, timestep, dx, dim);
    cudaDeviceSynchronize();

    for (int iter = 0; iter < NUM_OF_DIFFUSION_STEPS; ++iter) {
      BurgersDiffusionJacobi<<<blocks, threads>>>(dev_unew, dev_ustar, viscosity,
                                                  timestep, dx, dim);
      cudaDeviceSynchronize();
      std::swap(dev_unew, dev_ustar);
    }

    std::swap(dev_u, dev_ustar);

    if (framecount % SNAPSHOT_INTERVAL == 0) {
      cudaMemcpy(u, dev_u, dim * dim * sizeof(Vector2f), cudaMemcpyDeviceToHost);
      std::vector<float> snapshot(dim * dim * 2);
      for (int idx = 0; idx < dim * dim; ++idx) {
        snapshot[idx] = u[idx].x;
        snapshot[dim * dim + idx] = u[idx].y;
      }
      snapshots.push_back(snapshot);
    }
  }

  delete[] u;
  cudaFree(dev_u);
  cudaFree(dev_unew);
  cudaFree(dev_ustar);

  return 0;
}
