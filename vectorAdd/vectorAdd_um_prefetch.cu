#include <stdio.h>
#include <cassert>
#include <iostream>
#include <chrono>
#include <random>

using namespace std;

//===========================       prototypes des fonctions    ========================================
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x; 
	int j = blockIdx.y*blockDim.y + threadIdx.y; 
	int k = j*gridDim.x * blockDim.x + i;  

	if (k < N) c[k] = a[k] + b[k]; 
}

auto get_time() { return chrono::high_resolution_clock::now(); }





//===========================       fuction main      ===================================================
int main() {
  
  const int N = 1000 << 16;
  size_t bytes = N * sizeof(int);
  int BLOCK_SIZE = 1 << 10;
  int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  

  int *a, *b, *c;
  int id = cudaGetDevice(&id);

  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);
  
  
  

  cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemPrefetchAsync(c, bytes, id);

  for (int i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }
  
  cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
  cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
  cudaMemPrefetchAsync(a, bytes, id);
  cudaMemPrefetchAsync(b, bytes, id);

  auto start = get_time();
  vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);
  cudaDeviceSynchronize();

  cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

  auto finish = get_time();
  auto duration =
      chrono::duration_cast<std::chrono::milliseconds>(finish - start);

  cout << "temps écoulé en kernel = " << duration.count() << " ms\n";
  
  for (int i = 0; i < N; i++) {
    assert(c[i] == a[i] + b[i]);
  }

  
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  cout << "terminé avec succès"<<endl;


  return 0;
}
