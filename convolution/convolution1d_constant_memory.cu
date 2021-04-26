#include <cassert>
#include <cstdlib>
#include <iostream>
#include <chrono>


using namespace std;

#define MASK_LENGTH 7
__constant__ int mask[MASK_LENGTH];


__global__ void convolution_1d(int *array, int *result, int n);
void verify_result(int *array, int *mask, int *result, int n) ;
auto get_time() { return chrono::high_resolution_clock::now(); }


int main() {
  
  int n = 1000 << 16;
  int bytes_n = n * sizeof(int);
  size_t bytes_m = MASK_LENGTH * sizeof(int);


  // CPU
  int *h_array = new int[n];
  int *h_mask = new int[MASK_LENGTH];
  int *h_result = new int[n];


  for (int i = 0; i < n; i++) 
    h_array[i] = rand() % 100;
  
  
  for (int i = 0; i < MASK_LENGTH; i++) 
    h_mask[i] = rand() % 10;
  

 
  // GPU
  int *d_array, *d_result;
  cudaMalloc(&d_array, bytes_n);
  cudaMalloc(&d_result, bytes_n);

  // GPU ---> CPU
  cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask, h_mask, bytes_m);

  
  int THREADS = 256;
  int GRID = (n + THREADS - 1) / THREADS;
  auto start = get_time();
  convolution_1d<<<GRID, THREADS>>>(d_array, d_result, n);

  // CPU ---> GPU
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
  cudaFree(d_result);

  return 0;
}







__global__ void convolution_1d(int *array, int *result, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int r = MASK_LENGTH / 2;
  int start = tid - r;

  int temp = 0;
  for (int j = 0; j < MASK_LENGTH; j++) 
    if (((start + j) >= 0) && (start + j < n)) 
      temp += array[start + j] * mask[j];
      
  result[tid] = temp;
}


void verify_result(int *array, int *mask, int *result, int n) {
  int radius = MASK_LENGTH / 2;
  int temp;
  int start;
  for (int i = 0; i < n; i++) {
    start = i - radius;
    temp = 0;
    for (int j = 0; j < MASK_LENGTH; j++) {
      if ((start + j >= 0) && (start + j < n)) {
        temp += array[start + j] * mask[j];
      }
    }
    assert(temp == result[i]);
  }
}