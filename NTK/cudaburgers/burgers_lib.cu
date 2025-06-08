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
const bool implicit = true; // Set to true for implicit scheme

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
