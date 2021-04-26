#include <cassert>
#include <cstdlib>
#include <iostream>
#include <chrono>

using namespace std;
#define MASK_LENGTH 7

__constant__ int mask[MASK_LENGTH];
__global__ void convolution_1d(int *array, int *result, int n);
void verify_result(int *array, int *mask, int *result, int n);
auto get_time() { return chrono::high_resolution_clock::now(); }


int main() 
{
  int n = 1000 << 16;
  int bytes_n = n * sizeof(int);
  size_t bytes_m = MASK_LENGTH * sizeof(int);
  int r = MASK_LENGTH / 2;
  int n_p = n + r * 2;
  size_t bytes_p = n_p * sizeof(int);

  // CPU
  int *h_array = new int[n_p];
  int *h_mask = new int[MASK_LENGTH];
  int *h_result = new int[n];

  for (int i = 0; i < n_p; i++) 
    if ((i < r) || (i >= (n + r))) 
      h_array[i] = 0;
    else 
      h_array[i] = rand() % 100;
    

  for (int i = 0; i < MASK_LENGTH; i++) 
    h_mask[i] = rand() % 10;
  




  // GPU
  int *d_array, *d_result;
  cudaMalloc(&d_array, bytes_p);
  cudaMalloc(&d_result, bytes_n);

  // CPU --> GPU
  cudaMemcpy(d_array, h_array, bytes_p, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask, h_mask, bytes_m);

  int THREADS = 256;
  int GRID = (n + THREADS - 1) / THREADS;
  size_t SHMEM = (THREADS + r * 2) * sizeof(int);

  auto start = get_time();
  convolution_1d<<<GRID, THREADS, SHMEM>>>(d_array, d_result, n);

  // GPU --> CPU
  cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);
  auto finish = get_time();
  auto duration =
      chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  
  cout << "temps écoulé en kernel = " << duration.count() << " ms\n";
  

  verify_result(h_array, h_mask, h_result, n);
  cout << "terminé avec succès"<<endl;

  cudaFree(d_array);
  cudaFree(d_result);
  delete[] h_array;
  delete[] h_result;
  delete[] h_mask;
  

  return 0;
}











__global__ void convolution_1d(int *array, int *result, int n) 
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int s_array[];

  int r = MASK_LENGTH / 2;
  int d = 2 * r;
  int n_padded = blockDim.x + d;
  int offset = threadIdx.x + blockDim.x;
  int g_offset = blockDim.x * blockIdx.x + offset;

  s_array[threadIdx.x] = array[tid];

  if (offset < n_padded) 
    s_array[offset] = array[g_offset];
  
  __syncthreads();

  int temp = 0;

  for (int j = 0; j < MASK_LENGTH; j++) 
    temp += s_array[threadIdx.x + j] * mask[j];
  
  result[tid] = temp;
}


void verify_result(int *array, int *mask, int *result, int n) 
{
  int temp;
  for (int i = 0; i < n; i++)
  {
    temp = 0;
    for (int j = 0; j < MASK_LENGTH; j++) 
      temp += array[i + j] * mask[j];
    assert(temp == result[i]);
  }
}