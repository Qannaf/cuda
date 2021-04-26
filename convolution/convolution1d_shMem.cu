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


int main() {

  int n = 1000 << 16;                                    // n=2^16*1 = 1024
  int bytes_n = n * sizeof(int);                      // bytes_n = 1024*4
  int bytes_m = MASK_LENGTH * sizeof(int);            // bytes_m = 7*4
  int r = MASK_LENGTH / 2;                            // r = 7/2 = 3
  int n_p = n + r * 2;                                // n_p = 1024 + 3*2 = 1030
  int bytes_p = n_p * sizeof(int);                    // bytes_p = 1030*4
  
  // CPU
  int *h_array = new int[n_p];
  int *h_result = new int[n];
  int *h_mask = new int[MASK_LENGTH];


  for (int i = 0; i < n_p; i++) 
    if((i < r) || (i >= (n + r))) 
      h_array[i] = 0;
    else 
      h_array[i] = rand() % 100;
    
      
  
  for (int i = 0; i < MASK_LENGTH; i++)
    h_mask[i] = rand() % 10;
  

  
  

  // GPU
  int *d_array, *d_result;
  cudaMalloc(&d_array, bytes_p);
  cudaMalloc(&d_result, bytes_n);

  // CPU ---> GPU
  cudaMemcpy(d_array, h_array, bytes_p, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask, h_mask, bytes_m);

  int THREADS = 256;
  int GRID = (n + THREADS - 1) / THREADS;
  size_t SHMEM = THREADS * sizeof(int);

  auto start = get_time();
  convolution_1d<<<GRID, THREADS, SHMEM>>>(d_array, d_result, n);

  // GPU ---> CPU
  cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

 auto finish = get_time();
  auto duration =
      chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  
  cout << "temps écoulé en kernel = " << duration.count() << " ms\n";

  verify_result(h_array, h_mask, h_result, n);

  cout << "Terminé avec succès"<<endl;

  // Free allocated memory on the device and host
  delete[] h_array;
  delete[] h_result;
  delete[] h_mask;
  cudaFree(d_array);
  cudaFree(d_result);

  return 0;
}








__global__ void convolution_1d(int *array, int *result, int n) 
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int s_array[];
  s_array[threadIdx.x] = array[tid];
  __syncthreads();

  int temp = 0;

  for (int j = 0; j < MASK_LENGTH; j++) 
    if ((threadIdx.x + j) >= blockDim.x) 
      temp += array[tid + j] * mask[j];
    else 
      temp += s_array[threadIdx.x + j] * mask[j];

  result[tid] = temp;
}

void verify_result(int *array, int *mask, int *result, int n) {
  int temp;
  for (int i = 0; i < n; i++)
  {  temp = 0;
    for (int j = 0; j < MASK_LENGTH; j++) 
      temp += array[i + j] * mask[j];
    assert(temp == result[i]);
  }
}