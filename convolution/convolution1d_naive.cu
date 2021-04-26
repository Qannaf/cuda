#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

__global__ void convolution_1d(int *array, int *mask, int *result, int n,int m);
void verify_result(int *array, int *mask, int *result, int n, int m);
auto get_time() { return chrono::high_resolution_clock::now(); }



int main() {
  
  int n = 1000 << 16;
  int bytes_n = n * sizeof(int);
  int m = 7;
  int bytes_m = m * sizeof(int);

  // CPU
 vector<int> h_array(n);
 vector<int> h_mask(m);
 vector<int> h_result(n);
  generate(begin(h_array), end(h_array), [](){ return rand() % 100; });
  generate(begin(h_mask), end(h_mask), [](){ return rand() % 10; });

  
  // GPU
  int *d_array, *d_mask, *d_result;
  cudaMalloc(&d_array, bytes_n);
  cudaMalloc(&d_mask, bytes_m);
  cudaMalloc(&d_result, bytes_n);

  // CPU --> GPU
  cudaMemcpy(d_array, h_array.data(), bytes_n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask.data(), bytes_m, cudaMemcpyHostToDevice);

  //kernel
  int THREADS = 256;
  int GRID = (n + THREADS - 1) / THREADS;
  auto start = get_time();
  convolution_1d<<<GRID, THREADS>>>(d_array, d_mask, d_result, n, m);

  // GPU --> CPU
  cudaMemcpy(h_result.data(), d_result, bytes_n, cudaMemcpyDeviceToHost);
  auto finish = get_time();
  auto duration =
      chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  
  cout << "temps écoulé en kernel = " << duration.count() << " ms\n";
  
  verify_result(h_array.data(), h_mask.data(), h_result.data(), n, m);
  cout << "Terminé avec succès"<<endl;

  
  cudaFree(d_result);
  cudaFree(d_mask);
  cudaFree(d_array);

  return 0;
}












 
__global__ void convolution_1d(int *array, int *mask, int *result, int n,int m) 
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int r = m / 2;
  int start = tid - r;
  int temp = 0;

  for (int j = 0; j < m; j++) 
    if (((start + j) >= 0) && (start + j < n)) 
      temp += array[start + j] * mask[j];

  result[tid] = temp;
}


void verify_result(int *array, int *mask, int *result, int n, int m)
{
  int radius = m / 2;
  int temp;
  int start;

  for (int i = 0; i < n; i++) 
  {
    start = i - radius;
    temp = 0;
    for (int j = 0; j < m; j++) 
      if ((start + j >= 0) && (start + j < n)) 
        temp += array[start + j] * mask[j];
    assert(temp == result[i]);
  }
}