// g++ -o cuda_particle_filter cuda_particle_filter.cu -lcudart -lcurand
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>

#define N 100000  // Numero di particelle
#define T 20      // Numero di passi temporali
#define BLOCK_SIZE 256

/* ================================
   CUDA Particle Filter:
   Stima della media a posteriori di uno stato nascosto (v) di un sistema
   lineare dinamico: v_{j+1} = a * v_j + xi_j,        con xi_j ~ N(0, sigma^2)
   Osservazioni rumorose:
       y_{j+1} = v_{j+1} + eta_{j+1},   con eta_{j+1} ~ N(0, obs_sigma^2)
   ================================ */

// Inizializza il generatore di numeri casuali su ogni thread (per
// randomizzazione GPU)
__global__ void init_rng(curandState *state, unsigned long seed) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < N) curand_init(seed, id, 0, &state[id]);
}

// STEP 1: Propagazione dinamica delle particelle (Prediction)
// v_{j+1}^{(i)} = a * v_j^{(i)} + xi_j^{(i)}
__global__ void predict(float *v, curandState *state, float a, float sigma) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < N) {
    float noise =
        sigma * curand_normal(&state[id]);  // xi_j^{(i)} ~ N(0, sigma^2)
    v[id] = a * v[id] + noise;
  }
}

// STEP 2: Calcolo dei pesi secondo la likelihood (Analysis)
// w_j^{(i)} ∝ exp( -0.5 * (y_j - v_j^{(i)})^2 / obs_sigma^2 )
// Formula: P(y | v) = N(y; v, obs_sigma^2)
__global__ void compute_weights(float *v, float *weights, float y,
                                float obs_sigma) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < N) {
    float diff = y - v[id];
    weights[id] = expf(-0.5f * diff * diff / (obs_sigma * obs_sigma));
    // Qui manca la costante di normalizzazione, ma non serve: i pesi vengono
    // poi normalizzati!
  }
}

// STEP 3: Normalizzazione dei pesi (in modo che sommino a 1)
__global__ void normalize_weights(float *weights, float sum) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < N) {
    weights[id] /= sum;
  }
}

// STEP 4: Resampling multinomiale delle particelle
// Scelta di nuove particelle in base ai pesi normalizzati
__global__ void resample(float *v, float *v_new, float *weights,
                         curandState *state) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < N) {
    float u = curand_uniform(&state[id]);  // Numero random tra 0 e 1
    float cum_sum = 0.0f;
    for (int i = 0; i < N; ++i) {
      cum_sum += weights[i];
      if (cum_sum > u) {
        v_new[id] = v[i];
        break;
      }
    }
  }
}

// Calcola la media delle particelle su host
float mean_host(float *d_v) {
  float *h_v = new float[N];
  cudaMemcpy(h_v, d_v, N * sizeof(float), cudaMemcpyDeviceToHost);
  double sum = 0;
  for (int i = 0; i < N; ++i) sum += h_v[i];
  delete[] h_v;
  return sum / N;
}

/*
    Funzione principale del filtro:
    Input: vettore di osservazioni obs[T]
    Output: media a posteriori finale delle particelle
*/
void run_filter(float *obs, float *out_mean, float a = 0.9, float sigma = 1.0,
                float obs_sigma = 1.0) {
  float *d_v, *d_vnew, *d_weights;
  curandState *d_state;
  cudaMalloc(&d_v, N * sizeof(float));
  cudaMalloc(&d_vnew, N * sizeof(float));
  cudaMalloc(&d_weights, N * sizeof(float));
  cudaMalloc(&d_state, N * sizeof(curandState));
  init_rng<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_state, 1234);

  // INIZIALIZZAZIONE: campiona particelle da prior N(0,1)
  predict<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_v, d_state, 0.0,
                                                             1.0);

  for (int t = 0; t < T; ++t) {
    // STEP 1: Predizione (propagazione dinamica)
    // v_{j+1}^{(i)} = a * v_j^{(i)} + xi_j^{(i)}
    predict<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_v, d_state, a,
                                                               sigma);

    // STEP 2: Calcola pesi di likelihood rispetto all'osservazione attuale
    // w_j^{(i)} ∝ exp( -0.5 * (y_j - v_j^{(i)})^2 / obs_sigma^2 )
    compute_weights<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_v, d_weights, obs[t], obs_sigma);

    // Somma i pesi sul CPU (questo può essere ottimizzato con parallel
    // reduction su GPU)
    float *h_weights = new float[N];
    cudaMemcpy(h_weights, d_weights, N * sizeof(float), cudaMemcpyDeviceToHost);
    double sumw = 0;
    for (int i = 0; i < N; ++i) sumw += h_weights[i];
    // STEP 3: Normalizza i pesi
    normalize_weights<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_weights, sumw);

    // STEP 4: Resampling delle particelle secondo i pesi
    resample<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_v, d_vnew, d_weights, d_state);
    std::swap(d_v, d_vnew);
    delete[] h_weights;
  }
  // Calcola la media finale delle particelle (stima a posteriori di E^{\mu} v)
  *out_mean = mean_host(d_v);

  cudaFree(d_v);
  cudaFree(d_vnew);
  cudaFree(d_weights);
  cudaFree(d_state);
}

/*
    MAIN: Esegue due filtri su due dataset quasi identici, e confronta le medie
   finali (Verifica numerica del corollario di stabilità)
*/
int main() {
  float y1[T], y2[T];
  float v_true = 0;
  float a = 0.9, sigma = 1.0, obs_sigma = 1.0;

  // Simula una traiettoria reale + due dataset osservativi (y2 = y1 + piccolo
  // rumore)
  for (int t = 0; t < T; ++t) {
    v_true = a * v_true +
             sigma * ((float)rand() / RAND_MAX - 0.5f);  // Stato nascosto
    y1[t] = v_true +
            obs_sigma * ((float)rand() / RAND_MAX - 0.5f);  // Osservazione 1
    y2[t] =
        y1[t] + 0.1f;  // Osservazione 2: y2 = y1 + 0.1 (piccola perturbazione)
  }
  float mean1, mean2;
  // Esegui il filtro particellare su ciascun dataset
  run_filter(y1, &mean1, a, sigma, obs_sigma);
  run_filter(y2, &mean2, a, sigma, obs_sigma);

  // Stampa la differenza tra le due stime a posteriori
  printf("Mean with y1: %f, Mean with y2: %f, Difference: %f\n", mean1, mean2,
         fabs(mean1 - mean2));
  printf(
      "Differenza teoricamente controllata dal corollario: |E^mu v - E^mu' v| "
      "<= c|y - y'|\n");
  return 0;
}
