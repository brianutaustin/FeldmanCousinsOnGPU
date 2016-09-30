#include <iostream>
#include <cmath>
#include <algorthm>

#include <curand.h>
#include <curand_kernel.h>

#define N       10000
#define nrange  20

__global__ void kernel(double*, int*);

__global__ void kernel(double* mu, int* ulim, int* llim) {
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
  


  return 0;
}
