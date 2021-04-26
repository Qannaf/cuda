#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

const int N = 1 << 3;
const int SHMEM_SIZE = 1 << 3;


__global__ void matrixMul(const int *a, const int *b, int *c);
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c);
void afficheMatrix(vector<int>& m,int line, int colone);
auto get_time() { return chrono::high_resolution_clock::now(); }

int main() 
{
  size_t bytes = N * N * sizeof(int);

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

  // CPU ---> GPU
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  
  int THREADS = 2;
  int BLOCKS = N / THREADS;
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  auto start = get_time();
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  auto finish = get_time();
  auto duration =
      chrono::duration_cast<chrono::milliseconds>(finish - start);
    
  cout << "temps écoulé en kernel = " << duration.count() << " ms\n";
  afficheMatrix(h_a,N,N);
  afficheMatrix(h_b,N,N);
  afficheMatrix(h_c,N,N);
 

  verify_result(h_a, h_b, h_c);
  cout << "terminé avec succès"<<endl;

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}






__global__ void matrixMul(const int *a, const int *b, int *c) 
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int s_a[SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE];

  int tmp = 0;
  for (int i = 0; i < N; i += blockDim.x) 
  {
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];
    __syncthreads();

    for (int j = 0; j < blockDim.x; j++) 
      tmp +=s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    __syncthreads();
  }

  c[row * N + col] = tmp;
}



void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) 
{
  for (int i = 0; i < N; i++) 
    for (int j = 0; j < N; j++) 
    {
      int tmp = 0;
      for (int k = 0; k < N; k++) 
        tmp += a[i * N + k] * b[k * N + j];
  
      assert(tmp == c[i * N + j]);
    }
}


void afficheMatrix(vector<int>& m,int line, int colone)
{ 
  for (int i = 0; i <line;  i++) 
  {
    for (int j = 0; j < colone; j++)
    { 
      cout<<m[i]<<" ";
    }
  cout<<endl;
  }
  cout<<"\n_______________________________________"<<endl;
}
