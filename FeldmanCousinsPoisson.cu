#include <iostream>
#include <cmath>
#include <algorthm>
#include <fstream>

#define N       10000
#define nrange  20
#define bkgd    3

__global__ void       kernel(double*, int*);
__device__ double     poissonP(double, int);
__device__ long long  factorial(int n);

__global__ void kernel(double* mu, int* ulim, int* llim) {
  unsigned int thId = threadIdx.x;
  unsigned int blId = blockIdx.x;

  __shared__ double cacheR[nrange];
  __shared__ double cacheP[nrange];

  cacheR[thId] = poissonP(mu[blId], thId)/poissonP(max(0, thId - bkgd), thId);
  cacheP[thId] = poissonP(mu[blId], thId);
  __syncthreads();

  if (thId == 0) {
    for (int i = 0; i < nrange; i++) {
      double  rpRValTemp = cacheR[i];
      double  rpPValTemp = cacheP[i];
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
      cacheR[maxIdxTemp] = rpRValTemp;
      cacheP[maxIdxTemp] = rpPValTemp;
    }
  }

}

__device__ double poissonP(double mu, int n) {
  return pow(mu + bkgd, n)*exp(-(mu + bkgd))/factorial(n);
}

__device__ long long factorial(n) {
  long long fn = 1;
  if (n == 0) {
    return 1;
  } else {
    for (int i = 1; i < n + 1; i++) {
      fn *= i;
    }
  }
  
  return fn;
}

int main() {
  
  double* mu    = new double[N];
  int*    ulim  = new int[N];
  int*    llim  = new int[N];
  
  double* dev_mu;
  double* dev_ulim;
  double* dev_llim;

  cudaMalloc((void**)&dev_mu,   N*sizeof(double));
  cudaMalloc((void**)&dev_ulim, N*sizeof(int));
  cudaMalloc((void**)&dev_llim, N*sizeof(int));

  double muMax = 10;
  double muMin = 0;
  double step  = (muMax - muMin)/N; 
  for (int i = 0; i < N+1; i++) {
    mu[i] = muMin + i * step;
  }

  cudaMemcpy(dev_mu, mu, N*sizeof(double), cudaMemcpyHostToDevice);

  kernel<<<N,nrange>>>(dev_mu, dev_ulim, dev_llim);

  cudaMemcpy(ulim, dev_ulim, N*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(llim, dev_llim, N*sizeof(int), cudaMemcpyDeviceToHost);

  // Output chisq to file
  std::ofstream ofs;
  ofs.open ("ul.dat", std::ofstream::out | std::ofstream::app);
  for (int i = 0; i < Ndmsq; i++) {
    ofs << mu[i] << "\t\t" << llim[i] << "\t\t" << ulim[i] << std::endl;
  }
  ofs.close();

  cudaFree(dev_mu);
  cudaFree(dev_ulim);
  cudaFree(dev_llim);

  return 0;
}
