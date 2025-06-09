#include "burgers_lib.cu"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Variabili globali
int framecount = 0;
float t_current = 0;
float eps = 0.0f; // per le condizioni al contorno
int shift = 2;    // shift delle BC verso l'interno o l'esterno

Vector2f *u, *unew, *dev_u, *dev_unew, *dev_ustar;
float M_PI = 3.1415926535f;
bool wantWrite = true; // se true, scrive snapshot su file

float x_min = 0.0f, x_max = 2.0f;
float y_min = 0.0f, y_max = 2.0f;
float dx = (x_max - x_min) / (dim - 1);
float dy = (y_max - y_min) / (dim - 1);
vector<vector<float>>
    snapshots; // ogni snapshot: [ux(0), ..., ux(N), uy(0), ..., uy(N)]
// In cima al file, array globale per la storia delle BC
std::vector<std::vector<std::array<float, 5>>> left_bc_history;
std::vector<std::vector<std::array<float, 5>>> right_bc_history;
std::vector<std::vector<std::array<float, 5>>> top_bc_history;
std::vector<std::vector<std::array<float, 5>>> bottom_bc_history;

void setupB2d() {

  // Alloca host e device arrays
  u = (Vector2f *)malloc(dim * dim * sizeof(Vector2f));
  unew = (Vector2f *)malloc(dim * dim * sizeof(Vector2f));
  cudaMalloc(&dev_u, dim * dim * sizeof(Vector2f));
  cudaMalloc(&dev_unew, dim * dim * sizeof(Vector2f));
  cudaMalloc(&dev_ustar, dim * dim * sizeof(Vector2f));

  // Inizializza condizioni iniziali
  for (unsigned i = 0; i < dim; i++) {
    float x = x_min + i * dx;
    for (unsigned j = 0; j < dim; j++) {
      float y = y_min + j * dy;
      unsigned idx = i * dim + j;
      u[idx].x = 0.1f * sinf(M_PI * x) * cosf(M_PI * y);
      u[idx].y = 0.1f * cosf(M_PI * x) * sinf(M_PI * y);
      unew[idx].x = 0.0f;
      unew[idx].y = 0.0f;
    }
  }
  cudaMemcpy(dev_u, u, dim * dim * sizeof(Vector2f), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_unew, unew, dim * dim * sizeof(Vector2f),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_ustar, unew, dim * dim * sizeof(Vector2f),
             cudaMemcpyHostToDevice);

  framecount = 0;
  // Se vuoi, svuota anche snapshots se ne tieni traccia
}

// --- STEP ---
void stepB2d() {
  dim3 threads(16, 16);
  dim3 blocks((dim + 15) / 16, (dim + 15) / 16);

  if (implicit) {
    BurgersConvectionStep<<<blocks, threads>>>(dev_ustar, dev_u, timestep, dx,
                                               dim);
    cudaDeviceSynchronize();
    for (int iter = 0; iter < NUM_OF_DIFFUSION_STEPS; ++iter) {
      BurgersDiffusionJacobi<<<blocks, threads>>>(dev_unew, dev_ustar,
                                                  viscosity, timestep, dx, dim);
      cudaDeviceSynchronize();
      std::swap(dev_unew, dev_ustar);
    }
    std::swap(dev_u, dev_unew);
  } else {
    BurgersExplicitKernel<<<blocks, threads>>>(dev_unew, dev_u, viscosity,
                                               timestep, dx, dim);
    cudaDeviceSynchronize();
    std::swap(dev_u, dev_unew);
  }

  t_current += timestep;
  framecount += 1;
  cudaMemcpy(u, dev_u, dim * dim * sizeof(Vector2f), cudaMemcpyDeviceToHost);
  // print all values of u
  // for (int i = 0; i < dim * dim; ++i) {
  //   printf("u[%d] = (%f, %f)\n", i, u[i].x, u[i].y);
  // }
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
void mainB2d() {
  printf("Starting 2D Burgers simulation...\n");

  // Per salvataggio snapshot

  dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
  dim3 blocks((dim + BLOCKSIZEX - 1) / BLOCKSIZEX,
              (dim + BLOCKSIZEY - 1) / BLOCKSIZEY);

  // --- MAIN SIMULATION LOOP ---
  if (implicit) {
    printf("Running implicit scheme...\n");
    while (framecount < MAX_FRAMES) {
      // Step esplicito unico
      BurgersConvectionStep<<<blocks, threads>>>(dev_ustar, dev_u, timestep, dx,
                                                 dim);
      cudaDeviceSynchronize();
      // Jacobi iterato N volte, ping-pong fra dev_unew/dev_ustar
      for (int iter = 0; iter < NUM_OF_DIFFUSION_STEPS; ++iter) {
        BurgersDiffusionJacobi<<<blocks, threads>>>(
            dev_unew, dev_ustar, viscosity, timestep, dx, dim);
        cudaDeviceSynchronize();
        std::swap(dev_unew, dev_ustar);
      }
      std::swap(dev_u, dev_ustar);
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
        // Salva le condizioni al contorno
        left_bc_history.push_back(get_left_bc());
        right_bc_history.push_back(get_right_bc());
        top_bc_history.push_back(get_top_bc());
        bottom_bc_history.push_back(get_bottom_bc());
      }
      framecount++;
      t_current += timestep;
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
      cudaMemcpy(u, dev_unew, dim * dim * sizeof(Vector2f),
                 cudaMemcpyDeviceToHost);
      // printf("some values of u: %f, %f, %f, %f\n", u[12].x, u[12].y,
      //        u[dim * dim - 1].x, u[dim * dim - 1].y);
      // Salva snapshot a intervalli regolari
      if (framecount % SNAPSHOT_INTERVAL == 0) {

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

    // cudaMemcpy(dev_u, u, dim * dim * sizeof(Vector2f),
    // cudaMemcpyHostToDevice); cudaMemcpy(dev_unew, unew, dim * dim *
    // sizeof(Vector2f),
    //            cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_ustar, unew, dim * dim * sizeof(Vector2f),
    //            cudaMemcpyHostToDevice);
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

  // Cleanup
  free(u);
  free(unew);
  cudaFree(dev_u);
  cudaFree(dev_unew);
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
// --- PYBIND11 EXPORT ---
namespace py = pybind11;

PYBIND11_MODULE(burgers2d, m) {
  m.def("setupB2d", &setupB2d, "Setup CFD 2D");
  m.def("mainB2d", &mainB2d, "Setup CFD");
  m.def("stepB2d", &stepB2d, "Esegue un timestep");
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
}