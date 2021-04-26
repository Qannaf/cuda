#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;


// =========            Prototypes des functions               =============

__global__ void matrixMul(const int *a, const int *b, int *c, int N);
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N);
auto get_time() { return chrono::high_resolution_clock::now(); }



// =========                  Foction main                     =============
int main() 
{
  int N = 1 << 10;                                       //1024
  size_t bytes = N * N * sizeof(int);                    //1024*1024*4

 
  // CPU
  vector<int> h_a(N * N);         
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; }); 
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  // GPU
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // CPU --->  GPU
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  int THREADS = 1<< 5;                                       //2^5 = 32 thrads/bloc
  int BLOCKS = N / THREADS;                                  //1024/32 = 32 bloc

  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);
  auto start = get_time();
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

  // GPU --->  CPU
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
  auto finish = get_time();
  auto duration =
      chrono::duration_cast<chrono::milliseconds>(finish - start);
    
  cout << "temps écoulé en kernel = " << duration.count() << " ms\n";
  cout << "terminé avec succès"<<endl;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}









// ==========            Corps des functions               ===================

__global__ void matrixMul(const int *a, const int *b, int *c, int N) 
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  c[row * N + col] = 0;

  for (int k = 0; k < N; k++) 
    c[row * N + col] += a[row * N + k] * b[k * N + col];
  
}


void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
  for (int i = 0; i < N; i++) 
    for (int j = 0; j < N; j++) 
    {
      int tmp = 0;

      for (int k = 0; k < N; k++) 
        tmp += a[i * N + k] * b[k * N + j];

      assert(tmp == c[i * N + j]);
    } 
}