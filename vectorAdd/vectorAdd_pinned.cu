#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>
#include <chrono>
#include <random>

using namespace std;
//===========================       kernel    ========================================
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x; 
	int j = blockIdx.y*blockDim.y + threadIdx.y; 
	int k = j*gridDim.x * blockDim.x + i;  

	if (k < N) c[k] = a[k] + b[k]; 
}

auto get_time() { return chrono::high_resolution_clock::now(); }



//===========================       fuction main      ===================================================
int main() {
  constexpr int N = 1000 << 16;
  size_t bytes = sizeof(int) * N;
  int NUM_THREADS = 1 << 10;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  // CPU
  int *h_a, *h_b, *h_c;
  cudaMallocHost(&h_a, bytes);
  cudaMallocHost(&h_b, bytes);
  cudaMallocHost(&h_c, bytes);

  for (int i = 0; i < N; i++) // initialisation les vacteurs a ,b
  {
    h_a[i]=rand() % 100;
    h_b[i]=rand() % 100;
  }
  
  
  //GPU
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // CPU ---> GPU
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);


  auto start = get_time();

  vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

  // GPU ---> CPU
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
  auto finish = get_time();
  auto duration =
      chrono::duration_cast<std::chrono::milliseconds>(finish - start);

  cout << "temps écoulé en kernel = " << duration.count() << " ms\n";

  for (int i = 0; i < N; i++) {
    assert(h_c[i] == h_a[i] + h_b[i]);
  }


  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
 
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  cout << "terminé avec succès"<<endl;

  return 0;
}
