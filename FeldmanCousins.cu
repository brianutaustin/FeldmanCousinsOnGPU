#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

#include <curand.h>
#include <curand_kernel.h>

#define Ndmsq   1000
#define Nssq2th 1000
#define Ngrid   Ndmsq*Nssq2th
#define Nfexp   1000

#define Bkgd    100

__global__ void fakeexps(unsigned int, double*, double*, double*);
__device__ double oscprob(double, double, double);
__device__ double chisquare(double*, double*, double*, double*);

__global__ void fakeexps(unsigned int seed, double* dmsq, double* ssq2th, double* chisq) {
  // Intra-block indexing
  unsigned int blockID   = blockIdx.y * gridDim.x + blockIdx.x;

  // Random generator initiation
  curandState_t state;
  curand_init(seed, 0, 0, &state);

  // cacheChisq with length equals to number of threads (or number of fake experiments)
  __shared__ double cacheChisq[Nfexp];

  // cache Ebin
  __shared__ double ebin[5];
  if (threadIdx.x == 0) {
    for (int k = 0; k < 5; k++) {
      ebin[k] = 15 + k*10;
    }
  }

  double en[5];
  double bkgd[5];
  double mu[5];
  double n[5];
  double mubf[5];
  for (int i = 0; i < 5; i++) {
    en[i]   = ebin[i] + (curand_uniform(&state) - 0.5) * 10;
    bkgd[i] = (double) Bkgd;
    mu[i]   = oscprob(dmsq[blockIdx.y], ssq2th[blockIdx.x], en[i]) * 10000.;
    n[i]    = (double) curand_poisson(&state, mu[i] + bkgd[i]);
    mubf[i] = fmax(0., n[i] - bkgd[i]);
  }

  cacheChisq[threadIdx.x] = chisquare(bkgd, mu, n, mubf);
  __syncthreads();

  if (threadIdx.x == 1) {
    chisq[blockID] = cacheChisq[90];
  }
}

__device__ double oscprob(double dmsq, double ssq2th, double en) {
  return 2*ssq2th*(0.2+0.197*(en/dmsq)*(sin(1.5228*(dmsq/en))-sin(2.538*(dmsq/en))));
}

__device__ double chisquare(double* b, double* mu, double* n, double* mubf) {
  double temp = 0;
  for (int i = 0; i < 5; i++) {
    temp += 2*(mu[i] - mubf[i] + n[i]*log((mubf[i] + b[i]) / (mu[i] + b[i])));
  }
  return temp;
}

int main() {
  // chisq array to save the chi square values at every point on the grid
  double* chisq = new double[Ngrid];

  // Device pointers
  double* dev_dmsq;
  double* dev_ssq2th;
  double* dev_chisq;

  cudaMalloc((void**)&dev_dmsq, Ndmsq*sizeof(double));
  cudaMalloc((void**)&dev_ssq2th, Nssq2th*sizeof(double));
  cudaMalloc((void**)&dev_chisq, Nfexp*Ngrid*sizeof(double));

  // Initiate input values for kernel (grid)
  double dmsqmin = 1;
  double dmsqmax = 1000;
  double ssq2thmin = 0.0001;
  double ssq2thmax = 1;
  double* dmsq = new double[Ndmsq];
  double* ssq2th = new double[Nssq2th];
  for (unsigned int i = 0; i < Ndmsq; i++) {
    dmsq[i] = std::exp(std::log10(dmsqmin) + i*(std::log10(dmsqmax) - std::log10(dmsqmin))/Ndmsq);
  }
  for (unsigned int i = 0; i < Nssq2th; i++) {
    ssq2th[i] = std::exp(std::log10(ssq2thmin) + i*(std::log10(ssq2thmax) - std::log10(ssq2thmin))/Nssq2th);
  }

  cudaMemcpy(dev_dmsq, dmsq, Ndmsq*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_ssq2th, ssq2th, Nssq2th*sizeof(double), cudaMemcpyHostToDevice);

  // Call kernel (using grid of deltamsq and ssq2th)
  dim3 grid(Nssq2th, Ndmsq);
  fakeexps<<<grid,Nfexp>>>(time(NULL), dev_dmsq, dev_ssq2th, dev_chisq);
  cudaMemcpy(chisq, dev_chisq, Ngrid*sizeof(double), cudaMemcpyDeviceToHost);

  // Output chisq to file
  std::ofstream ofs;
  ofs.open ("chi.dat", std::ofstream::out | std::ofstream::app);
  for (int i = 0; i < Ndmsq; i++) {
    for (int j = 0; j < Nssq2th; j++) {
      ofs << dmsq[i] << "\t\t"<< ssq2th[j] << "\t\t" << chisq[j + i*Ndmsq] << std::endl;
    }
  }
  ofs.close();

  return 0;
}
