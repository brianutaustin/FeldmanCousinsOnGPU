#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>

#define N       100
#define nrange  20
#define bkgd    3
#define CL      0.9

__global__ void   kernel(double*, int*, double*);
__device__ double poissonP(double, int);
__device__ double factorial(double n);

__global__ void kernel(double* mu, int* n, double* R) {
  int thId = threadIdx.x;
  int blId = blockIdx.x;
  int atId = threadIdx.x + blockIdx.x * gridDim.x;

  __shared__ double cacheR[nrange];
  __shared__ double cacheP[nrange];
  __shared__ int    cacheI[nrange];

  cacheR[thId] = poissonP(mu[blId], thId)/poissonP(max(0, thId - bkgd), thId);
  cacheP[thId] = poissonP(mu[blId], thId);
  cacheI[thId] = thId;
  __syncthreads();

/*
  if (thId == 0) {
    for (int i = 0; i < nrange; i++) {
      double  rpRValTemp = cacheR[i];
      double  rpPValTemp = cacheP[i];
      int     rpIValTemp = cacheI[i];
      double  maxValTemp = 0;
      int     maxIdxTemp = 0;
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

    double  aggrP = 0;
    int     count = 0;
    while (aggrP < CL) {
      aggrP += cacheP[count];
      count ++;
    }

    for (int i = 0; i < count + 1; i++) {
      if (cacheI[i] > ul) ul = cacheI[i];
      if (cacheI[i] < ll) ll = cacheI[i];
    }
  }
*/
  llim[blId] = thId;
}

__device__ double poissonP(double mu, int n) {
  return pow(mu + 3., n)*exp(-(mu + 3.))/factorial((double)n);
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
  double* R;
  int*    dev_n;

  cudaMalloc((void**)&dev_mu,   N*sizeof(double));
  cudaMalloc((void**)&dev_R,    N*nrange*sizeof(double));
  cudaMalloc((void**)&dev_n,    N*nrange*sizeof(int));

  double muMax = 10;
  double muMin = 0;
  double step  = (muMax - muMin)/N;
  for (int i = 0; i < N+1; i++) {
    mu[i] = muMin + i * step;
  }

  cudaMemcpy(dev_mu, mu, N*sizeof(double), cudaMemcpyHostToDevice);

  kernel<<<N+1,nrange>>>(dev_mu, dev_n,  dev_R);

  cudaMemcpy(R, dev_R, N*nrange*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(n, dev_n, N*sizeof(int),        cudaMemcpyDeviceToHost);
/*
  std::ofstream ofs;
  ofs.open ("ul.dat", std::ofstream::out | std::ofstream::app);
  for (int i = 0; i < N+1; i++) {
    ofs << mu[i] << "\t\t" << llim[i] << "\t\t" << ulim[i] << std::endl;
  }
  ofs.close();
*/
  cudaFree(dev_mu);
  cudaFree(dev_ulim);
  cudaFree(dev_llim);

  return 0;
}
