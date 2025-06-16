#include "navierStokes_lib.cu"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm> // for std::min/max
#include <vector>
#include <array>

namespace py = pybind11;

// Variabili globali
int framecount = 0;
float t_current = 0;
float eps = 0.0f; // per le condizioni al contorno
int shift =10;    // shift delle BC verso l'interno o l'esterno

Vector2f *u, *dev_u;
float *p, *c, *dev_p, *dev_c;
int *obstacleField, *dev_obstacleField;
float M_PI = 3.1415926535f;
bool wantWrite = false; // se true, scrive snapshot su file

float x_min = -1.0f, x_max = 1.0f;
float y_min = -1.0f, y_max = 1.0f;
float dx = (x_max - x_min) / (dim - 1);
float dy = (y_max - y_min) / (dim - 1);
vector<vector<float>>
    snapshots; // ogni snapshot: [ux(0), ..., ux(N), uy(0), ..., uy(N)]
// In cima al file, array globale per la storia delle BC
std::vector<std::vector<std::array<float, 5>>> left_bc_history;
std::vector<std::vector<std::array<float, 5>>> right_bc_history;
std::vector<std::vector<std::array<float, 5>>> top_bc_history;
std::vector<std::vector<std::array<float, 5>>> bottom_bc_history;

void setupNS2d() {

  // Alloca host e device arrays
  obstacleField = (int *)malloc(dim * dim * sizeof(int));
  initializeObstacle(obstacleField, dim, obstacleCenterX, obstacleCenterY, obstacleRadius);

  u = (Vector2f *)malloc(dim * dim * sizeof(Vector2f));
  p = (float*)malloc(dim * dim * sizeof(float));
  c = (float*)malloc(dim * dim * sizeof(float));

  cudaMalloc(&dev_u, dim * dim * sizeof(Vector2f));
  cudaMalloc(&dev_p, dim * dim * sizeof(float));
  cudaMalloc(&dev_c, dim * dim * sizeof(float));
  cudaMalloc((void**)&dev_obstacleField, dim * dim * sizeof(int));

  // Inizializza condizioni iniziali
  for (unsigned i = 0; i < dim; i++) {
    float x = x_min + i * dx;
    for (unsigned j = 0; j < dim; j++) {
      float y = y_min + j * dy;
      unsigned idx = i * dim + j;
      u[idx].x =sinf(M_PI * x) * cosf(M_PI * y);
      u[idx].y = cosf(M_PI * x) * sinf(M_PI * y);
    }
  }
  for (unsigned i = 0; i < dim * dim; i++) {
    p[i] = 0.0f;
    c[i] = 0.0f;
    // obstacleField[i] = 0; // Inizializza il campo degli ostacoli a 0
  }

  cudaMemcpy(dev_u, u, dim * dim * sizeof(Vector2f), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_p, p, dim * dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c, dim * dim * sizeof(float),
             cudaMemcpyHostToDevice);
              // Copy obstacle field to device
  cudaMemcpy(dev_obstacleField, obstacleField, dim * dim * sizeof(int), cudaMemcpyHostToDevice);

  framecount = 0;
  // Se vuoi, svuota anche snapshots se ne tieni traccia
}
// --- BC su x = 0 ---
std::vector<std::array<float, 5>> get_left_bc() {
  std::vector<std::array<float, 5>> bc(dim);

  // Calcola l'indice più vicino alla posizione eps
  // int i = int(std::round(eps / dx)); // i=0 (bordo), i=1,2... (più dentro)
  int i = shift;
  float x_val = x_min + i * dx;

  for (int j = 0; j < dim; ++j) {
    float y_val = y_min + j * dy;
    int idx = i * dim + j;
    bc[j] = {t_current, x_val, y_val, u[idx].x, u[idx].y};
  }
  return bc;
}

std::vector<std::array<float, 5>> get_right_bc() {
  std::vector<std::array<float, 5>> bc(dim);

  int i = (dim - 1) - shift; // x=max per shift=0, più interno per shift>0
  float x_val = x_min + i * dx;

  for (int j = 0; j < dim; ++j) {
    float y = y_min + j * dy;
    int idx = i * dim + j; // riga i, colonna j
    bc[j] = {t_current, x_val, y, u[idx].x, u[idx].y};
  }
  return bc;
}

std::vector<std::array<float, 5>> get_top_bc() {
  std::vector<std::array<float, 5>> bc(dim);
  int j = (dim - 1) - shift; // y=max per shift=0, più interno per shift>0
  float y_val = y_min + j * dy;
  for (int i = 0; i < dim; ++i) {
    float x = x_min + i * dx;
    int idx = i * dim + j; // riga i, colonna j
    bc[i] = {t_current, x, y_val, u[idx].x, u[idx].y};
  }
  return bc;
}

std::vector<std::array<float, 5>> get_bottom_bc() {
  std::vector<std::array<float, 5>> bc(dim);
  int j = shift; // y=0 per shift=0, più interno per shift>0
  for (int i = 0; i < dim; ++i) {
    float x = x_min + i * dx;
    int idx = i * dim + j; // y=0
    bc[i] = {t_current, x, y_min, u[idx].x, u[idx].y};
  }
  return bc;
}
void mainNS2d() {
  printf("Starting 2D Navier-Stokes simulation...\n");
  //generate a random float number

  float betabouyancy = BETA_BOUYANCY; // Buoyancy coefficient
  float gravity = -9.81f;

  // Obstacle parameters
  // float obstacleCenterX = dim / 2.0f; // Center of the domain
  // float obstacleCenterY = dim / 2.0f;
  // float obstacleRadius = dim / 10.0f; // Adjust as needed
  // CUDA grid and block dimensions
  dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
  dim3 blocks((dim + BLOCKSIZEX - 1) / BLOCKSIZEX, (dim + BLOCKSIZEY - 1) / BLOCKSIZEY);
  // initializeCylinder(obstacleField, dim, obstacleCenterX, obstacleCenterY, obstacleRadius);
  initializeObstacle(obstacleField, dim, obstacleCenterX, obstacleCenterY, obstacleRadius);
  cudaMemcpy(dev_obstacleField, obstacleField, dim * dim * sizeof(int), cudaMemcpyHostToDevice);

  // ------------------------------------------------------------S I M U L A T I O N    L O O P--------------------------------------------------------------------------------
  //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  while (framecount < MAX_FRAMES) {

      // Time step
      // float randomFloat = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      // float c_ambient = randomFloat;    // Ambient value of c
      float c_ambient = C_AMBIENT;

      // if (PERIODIC_FORCE == 1) {
      //     F = Vector2f(magnitude * sin(time), 0.0f); // Initial force
      // }

      // C = Vector2f(dim / 2.0f + 50.0f * sinf(glfwGetTime()), dim / 2.0f);

      // Execute the Navier-Stokes kernel
      NSkernel<<<blocks, threads>>>(dev_u, dev_p, dev_c, dev_obstacleField, c_ambient, gravity, betabouyancy, dx, viscosity, C, F, timestep, r, dim);

      // // Check for CUDA errors
      // cudaError_t err = cudaGetLastError();
      // if (err != cudaSuccess) {
      //     printf("CUDA Error after NSkernel: %s\n", cudaGetErrorString(err));
      //     return 1;
      // }

      cudaDeviceSynchronize();

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
        // Salva le condizioni al contorno
        left_bc_history.push_back(get_left_bc());
        right_bc_history.push_back(get_right_bc());
        top_bc_history.push_back(get_top_bc());
        bottom_bc_history.push_back(get_bottom_bc());
      }

      framecount++;
      t_current += timestep;
  }
  // --- SCRITTURA SU FILE ---
  if (wantWrite) {
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
  }

  // Free memory
  free(u);
  free(p);
  free(c);
  free(obstacleField);
  cudaFree(dev_u);
  cudaFree(dev_p);
  cudaFree(dev_c);
  cudaFree(dev_obstacleField);
}

std::vector<std::vector<std::array<float, 5>>> get_left_bc_history() {
  return left_bc_history;
}
std::vector<std::vector<std::array<float, 5>>> get_right_bc_history() {
  return right_bc_history;
}
std::vector<std::vector<std::array<float, 5>>> get_top_bc_history() {
  return top_bc_history;
}
std::vector<std::vector<std::array<float, 5>>> get_bottom_bc_history() {
  return bottom_bc_history;
}

void writeSnapshots(){
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
}

// esponi u
std::vector<std::vector<float>> get_u() {
  std::vector<std::vector<float>> u_array(dim, std::vector<float>(dim * 2));
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      int idx = i * dim + j;
      u_array[i][j] = u[idx].x;       // componente x
      u_array[i][j + dim] = u[idx].y; // componente y
    }
  }
  return u_array;
}

std::vector<std::array<float, 4>> get_ic_from_snapshots() {
  std::vector<std::array<float, 4>> ic(dim * dim);
  //==================================================== NOT TESTED=========================================================
  if (snapshots.empty()) {
      printf("Snapshots vector is empty!\n");
      return ic;
  }

  // La prima snapshot contiene [ux(0), ..., ux(N), uy(0), ..., uy(N)]
  const std::vector<float>& first_snapshot = snapshots[0];

  // Safety check
  if (first_snapshot.size() != 2 * dim * dim) {
      printf("First snapshot size mismatch: expected %d, got %zu\n", 2*dim*dim, first_snapshot.size());
      return ic;
  }

  // Griglia
  for (int i = 0; i < dim; ++i) {
      float x = x_min + i * dx;
      for (int j = 0; j < dim; ++j) {
          float y = y_min + j * dy;
          int idx = i * dim + j;
          float ux = first_snapshot[idx];             // primi dim*dim sono ux
          float uy = first_snapshot[idx + dim*dim];   // dopo ci sono gli uy
          ic[idx] = {x, y, ux, uy};
      }
  }
  return ic;
}

py::array_t<float> interpolate_ic_from_snapshots(py::array_t<float, py::array::c_style | py::array::forcecast> query) {
  // query: shape (N, 2) (x, y)
  if (snapshots.empty())
      throw std::runtime_error("Snapshots vector is empty!");
  const std::vector<float>& first_snapshot = snapshots[0];
  if (first_snapshot.size() != 2 * dim * dim)
      throw std::runtime_error("First snapshot size mismatch.");
  printf("shape of first snapshot: %zu\n", first_snapshot.size());
  // float *ux_grid;
  // float* uy_grid ;
  std::vector<float> ux_grid(dim * dim);
  std::vector<float> uy_grid(dim * dim);
  for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
          int idx = i * dim + j;
          ux_grid[idx] = first_snapshot[idx];             // primi dim*dim sono ux
          uy_grid[idx] = first_snapshot[idx + dim*dim];   // dopo ci sono gli uy
      }
    }
    // float dx = (x_max - x_min) / (dim - 1);
    // float dy = (y_max - y_min) / (dim - 1);

    auto buf = query.request();
  if (buf.ndim != 2 || buf.shape[1] < 2)
      throw std::runtime_error("Input shape must be (N, 2+)");

  int N = buf.shape[0];
  auto ptr = static_cast<const float*>(buf.ptr);

  // Crea output (N, 2)
  py::array_t<float> result({N, 2});
  auto r = result.mutable_unchecked<2>();

  for (int n = 0; n < N; ++n) {
      float x = ptr[n * buf.shape[1]];
      float y = ptr[n * buf.shape[1] + 1];

      int ix = static_cast<int>((x - x_min) / dx);
      int iy = static_cast<int>((y - y_min) / dy);

      ix = std::min(std::max(ix, 0), static_cast<int>(dim) - 2);
      iy = std::min(std::max(iy, 0), static_cast<int>(dim) - 2);


      float x1 = x_min + ix * dx;
      float y1 = y_min + iy * dy;
      float fx = (x - x1) / dx;
      float fy = (y - y1) / dy;

      // int idx11 = ix * dim + iy;
      // int idx12 = ix * dim + (iy + 1);
      // int idx21 = (ix + 1) * dim + iy;
      // int idx22 = (ix + 1) * dim + (iy + 1);
      // x = x_min + ix * dx
// y = y_min + iy * dy
      int idx11 = ix * dim + iy;         // (ix, iy)
      int idx12 = ix * dim + (iy + 1);   // (ix, iy+1)
      int idx21 = (ix + 1) * dim + iy;   // (ix+1, iy)
      int idx22 = (ix + 1) * dim + (iy + 1); // (ix+1, iy+1)


      // int idx11 = iy * dim + ix; // scambia x e y!
      // int idx12 = (ix + 1) * dim + iy;
      // int idx21 = ix * dim + (iy + 1);
      // int idx22 = (iy + 1) * dim + (ix + 1);

      float ux_val = (1-fx)*(1-fy)*ux_grid[idx11]
                   + (1-fx)*fy*ux_grid[idx12]
                   + fx*(1-fy)*ux_grid[idx21]
                   + fx*fy*ux_grid[idx22];
      float uy_val = (1-fx)*(1-fy)*uy_grid[idx11]
                   + (1-fx)*fy*uy_grid[idx12]
                   + fx*(1-fy)*uy_grid[idx21]
                   + fx*fy*uy_grid[idx22];

      r(n, 0) = ux_val;
      r(n, 1) = uy_val;
  }
  return result;
}



// --- PYBIND11 EXPORT ---


PYBIND11_MODULE(navier2d, m) {
  m.def("setupNS2d", &setupNS2d, "Setup CFD 2D");
  m.def("mainNS2d", &mainNS2d, "Setup CFD");
  m.def("get_left_bc", &get_left_bc, "Restituisce il profilo BC su x=0");
  m.def("get_right_bc", &get_right_bc, "Restituisce il profilo BC su x=max");
  m.def("get_top_bc", &get_top_bc, "Restituisce il profilo BC su y=max");
  m.def("get_bottom_bc", &get_bottom_bc, "Restituisce il profilo BC su y=0");
  m.def("get_left_bc_history", &get_left_bc_history,
        "Restituisce la storia delle BC su x=0");
  m.def("get_right_bc_history", &get_right_bc_history,
        "Restituisce la storia delle BC su x=max");
  m.def("get_top_bc_history", &get_top_bc_history,
        "Restituisce la storia delle BC su y=max");
  m.def("get_bottom_bc_history", &get_bottom_bc_history,
        "Restituisce la storia delle BC su y=0");
  m.attr("t_current") = &t_current;
  m.attr("dim") = &dim;
  m.attr("dx") = &dx;
  m.attr("dy") = &dy;
  m.attr("x_min") = &x_min;
  m.attr("x_max") = &x_max;
  m.attr("y_min") = &y_min;
  m.attr("y_max") = &y_max;
  m.def("get_u", &get_u, "Restituisce il campo u come array (dim,dim,2)");
  m.def("get_ic_from_snapshots", &get_ic_from_snapshots, "Restituisce la condizione iniziale completa");
  m.def("interpolate_ic_from_snapshots", &interpolate_ic_from_snapshots, "Interpola le IC CFD sui punti (x,y) richiesti");
  m.def("writeSnapshots", &writeSnapshots, "Scrive gli snapshots su file");
}
