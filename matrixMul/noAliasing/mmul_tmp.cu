#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using namespace std;
__global__ void matrixMul(const int *a, const int *b, int *c, int N);
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N);
void afficheMatrix(vector<int>& m,int line, int colone);


int main() 
{
  int N = 1 << 3;
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

  // CPU --->  GPU
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  
  int THREADS = 1 << 2;
  int BLOCKS = N / THREADS;
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

  // CPU ---> GPU
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  
  verify_result(h_a, h_b, h_c, N);

  cout << "terminé avec succès"<<endl;

  afficheMatrix(h_a,N,N);
  afficheMatrix(h_b,N,N);
  afficheMatrix(h_c,N,N);


  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}



__global__ void matrixMul(const int *a, const int *b, int *c, int N) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Iterate over row, and down column
  int tmp = 0;
  for (int k = 0; k < N; k++) {
    // Accumulate results for a single element
    tmp += a[row * N + k] * b[k * N + col];
  }

  // Write back the result
  c[row * N + col] = tmp;
}

// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
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
