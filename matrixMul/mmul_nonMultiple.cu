#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
constexpr int M = 1 << 3 ;        //M =8
constexpr int N = 1 << 3;
constexpr int K = 1 << 3;
constexpr int THREADS = 1 << 2;   


constexpr int M_padded = M + THREADS - M % THREADS;   //8+4 - 8%4 = 12
constexpr int N_padded = N + THREADS - N % THREADS;
constexpr int K_padded = K + THREADS - K % THREADS;

constexpr int SHMEM_SIZE = THREADS * THREADS;         // size de MemSh par threads

__global__ void matrixMul(const int *a, const int *b, int *c);
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c);
void initMatrix(vector<int>& ,int, int);
void afficheMatrix(vector<int>& m,int line, int colone);
auto get_time() { return chrono::high_resolution_clock::now(); }

int main() 
{
  size_t bytes_a = M_padded * K_padded * sizeof(int);        // MxN = MxK * KxN
  size_t bytes_b = K_padded * N_padded * sizeof(int);
  size_t bytes_c = M * N * sizeof(int);

  // CPU
  vector<int> h_a(M_padded * K_padded);                   //12*12
  vector<int> h_b(K_padded * N_padded);                   // 12*12
  vector<int> h_c(M * N);                                 //8*8

  
  initMatrix(h_a,M_padded,K_padded);
  initMatrix(h_b,K_padded,N_padded);
  

  // GPU
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes_a);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_c, bytes_c);

  // CPU ---> GPU
  cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice);

  
  int BLOCKS_X = N_padded / THREADS;                           // 12/4 = 3
  int BLOCKS_Y = M_padded / THREADS;                            // 12/4 =3
  dim3 threads(THREADS, THREADS);                               // (4,4)
  dim3 blocks(BLOCKS_X, BLOCKS_Y);                              // (3,3)
  auto start = get_time();
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);                //<<< (3,3), (4,4) >>>
  cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);
  auto finish = get_time();
  auto duration =
      chrono::duration_cast<chrono::milliseconds>(finish - start);
    
  cout << "temps écoulé en kernel = " << duration.count() << " ms\n";
  afficheMatrix(h_c,M,N);
  verify_result(h_a, h_b, h_c);

  cout << "terminé avec succès"<<endl;

  
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

  for (int i = 0; i < K_padded; i += blockDim.x) 
  {
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];
    __syncthreads();

    for (int j = 0; j < blockDim.x; j++) 
      tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    __syncthreads();

  }

  if (row < M && col < N) c[row * N + col] = tmp;
}


void verify_result(vector<int> &a, vector<int> &b, vector<int> &c)
{
  for (int row = 0; row < M_padded; row++) 
  {
    if (row >= M) continue;
    for (int col = 0; col < N_padded; col++) 
    {
      if (col >= N) continue;
      int tmp = 0;
      for (int i = 0; i < K_padded; i++) 
        tmp += a[row * K + i] * b[i * N + col];
      assert(tmp == c[row * N + col]);
    }
  }
}



void initMatrix(vector<int>& m,int line, int colone)
{ 
  for (int i = 0; i <line;  i++) 
  {
    for (int j = 0; j < colone; j++)
    { 
      if (i < M && j < K) m[i * K + j] = rand() % 100;
      else m[i * K + j] = 0;
      cout<<m[i]<<" ";
    }
  cout<<endl;
  }
  cout<<"\n_______________________________________"<<endl;
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
