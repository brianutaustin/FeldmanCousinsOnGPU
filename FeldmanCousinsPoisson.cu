#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>

#define N       100
#define nrange  20
#define bkgd    3
#define CL      0.9

__global__ void   kernel(double*, int*, double*);
__device__ double poissonP(double, double);
__device__ double factorial(double n);

__global__ void kernel(double* mu, int* n, double* R) {
  int thId = threadIdx.x;
  int blId = blockIdx.x;
  int atId = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ double cacheR[nrange];

  cacheR[thId] = poissonP(mu[blId], thId)/poissonP(max(0, thId - bkgd), thId);
  __syncthreads();

  n[atId] = thId;
  R[atId] = cacheR[thId];
}

__device__ double poissonP(double mu, double n) {
  return pow(mu + 3., n)*exp(-(mu + 3.))/factorial(n);
}

__device__ double factorial(double n) {
  double fn = 1.;
  if (n == 0) {
    return 1.;
  } else {
    for (int i = 1; i < n + 1; i++) {
      fn *= (double)i;
    }
  }
  return fn;
}

int main() {

  double* mu    = new double[N];
  double* R     = new double[N*nrange];
  int*    n     = new int[N*nrange];
  double* dev_mu;
  double* dev_R;
  int*    dev_n;
  cudaMalloc((void**)&dev_mu,   N*sizeof(double));
  cudaMalloc((void**)&dev_R,    N*nrange*sizeof(double));
  cudaMalloc((void**)&dev_n,    N*nrange*sizeof(int));

  double muMax = 10;
  double muMin = 0;
  double step  = (muMax - muMin)/N;
  for (int i = 0; i < N; i++) {
    mu[i] = muMin + (double)i * step;
  }
  cudaMemcpy(dev_mu, mu, N*sizeof(double), cudaMemcpyHostToDevice);

  kernel<<<N,nrange>>>(dev_mu, dev_n, dev_R);
  cudaMemcpy(R, dev_R, N*nrange*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(n, dev_n, N*nrange*sizeof(int), cudaMemcpyDeviceToHost);

  std::ofstream ofs;
  ofs.open ("ul.dat", std::ofstream::out | std::ofstream::app);
  for (int i = 0; i < N; i++) {
    ofs << mu[i];
    for (int j = 0; j < nrange; j++) {
      ofs << "," << n[j + i*nrange] << "," << R[j + i * nrange];
    }
    ofs << std::endl;
  }
  ofs.close();

  cudaFree(dev_mu);
  cudaFree(dev_n);
  cudaFree(dev_R);
  return 0;
}
