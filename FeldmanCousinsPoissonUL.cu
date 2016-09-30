#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>

#define N       10000
#define nrange  20
#define bkgd    3
#define CL      0.9

__global__ void   kernel(double*, int*, double*);
__device__ double poissonP(double, double);
__device__ double factorial(double n);

__global__ void kernel(double* mu, int* n, double* P) {
  int thId = threadIdx.x;
  int blId = blockIdx.x;
  int atId = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ double cacheR[nrange];
  __shared__ double cacheP[nrange];
  __shared__ int    cacheI[nrange];

  cacheR[thId] = poissonP(mu[blId], thId)/poissonP(max(0, thId - bkgd), thId);
  cacheP[thId] = poissonP(mu[blId], thId);
  cacheI[thId] = thId;
  __syncthreads();

  if (thId == 0) {
    for (int i = 0; i < nrange; i++) {
      double  rpRValTemp = cacheR[i];
      double  rpPValTemp = cacheP[i];
      int     rpIValTemp = cacheI[i];
      double  maxValTemp = cacheR[i];
      int     maxIdxTemp = i;
      for (int j = i + 1; j < nrange; j++) {
        if (cacheR[j] > maxValTemp) {
          maxValTemp = cacheR[j];
          maxIdxTemp = j;
        }
      }
      cacheR[i] = cacheR[maxIdxTemp];
      cacheP[i] = cacheP[maxIdxTemp];
      cacheI[i] = cacheI[maxIdxTemp];
      cacheR[maxIdxTemp] = rpRValTemp;
      cacheP[maxIdxTemp] = rpPValTemp;
      cacheI[maxIdxTemp] = rpIValTemp;
    }
  }

  n[atId] = cacheI[thId];
  P[atId] = cacheP[thId];
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
  double* P     = new double[N*nrange];
  int*    n     = new int[N*nrange];
  double* dev_mu;
  double* dev_P;
  int*    dev_n;
  cudaMalloc((void**)&dev_mu,   N*sizeof(double));
  cudaMalloc((void**)&dev_P,    N*nrange*sizeof(double));
  cudaMalloc((void**)&dev_n,    N*nrange*sizeof(int));

  double muMax = 10;
  double muMin = 0;
  double step  = (muMax - muMin)/N;
  for (int i = 0; i < N; i++) {
    mu[i] = muMin + (double)i * step;
  }
  cudaMemcpy(dev_mu, mu, N*sizeof(double), cudaMemcpyHostToDevice);

  kernel<<<N,nrange>>>(dev_mu, dev_n, dev_P);
  cudaMemcpy(P, dev_P, N*nrange*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(n, dev_n, N*nrange*sizeof(int), cudaMemcpyDeviceToHost);

  std::ofstream ofs;
  ofs.open ("ulUL.dat", std::ofstream::out | std::ofstream::app);
  for (int i = 0; i < N; i++) {
    ofs << mu[i];
    for (int j = 0; j < nrange; j++) {
      ofs << "," << n[j + i*nrange] << "," << P[j + i * nrange];
    }
    ofs << std::endl;
  }
  ofs.close();

  cudaFree(dev_mu);
  cudaFree(dev_n);
  cudaFree(dev_P);

  return 0;
}
