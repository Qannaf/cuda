#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace std;

//===========================       prototypes des fonctions     ===================================================
__global__ void vectorAddBaseLine(const int *__restrict a, const int *__restrict b,int *__restrict c, int N) ;
void verify_result( vector<int> &a,  vector<int> &b,  vector<int> &c);
auto get_time() { return chrono::high_resolution_clock::now(); }





//===========================       fuction main      ===================================================
int main() {
  constexpr int N = 1000 << 16;       // size 2^16*1 éléments
  constexpr size_t bytes = sizeof(int) * N;
  int NUM_THREADS = 1<< 10;        // 2^10*1 threads/bloc
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  //CPU
  vector<int> a;a.reserve(N);
  vector<int> b;b.reserve(N);
  vector<int> c;c.reserve(N);

 

  for (int i = 0; i < N; i++) // initialisation les vacteurs a ,b
  {
    a.push_back(rand() % 100);
    b.push_back(rand() % 100);
  }
  
  
  //GPU
  
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // CPU -----> GPU
  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

  auto start = get_time();
  vectorAddBaseLine<<<NUM_BLOCKS, NUM_THREADS >>>(d_a, d_b, d_c, N);  //kernel

  // GPU ---> CPU
  cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  auto finish = get_time();
  auto duration =
      chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  
  cout << "temps écoulé en kernel = " << duration.count() << " ms\n";

  // vérification 
  verify_result(a, b, c);

  // libérer la mémoire GPU
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  cout << "terminé avec succès"<<endl;

  
  
  return 0; // Free memory on device


}

// =======================          kernel          ==================================
__global__ void vectorAddBaseLine(const int *__restrict a, const int *__restrict b,int *__restrict c, int N) 
{
  int i = blockIdx.x*blockDim.x + threadIdx.x; 
	int j = blockIdx.y*blockDim.y + threadIdx.y; 
	int k = j*gridDim.x * blockDim.x + i;  

	if (k < N) c[k] = a[k] + b[k]; 
}

//=========================        vérification       ==========================================
void verify_result( vector<int> &a,  vector<int> &b,  vector<int> &c)
{
  for (int i = 0; i < a.size(); i++) 
    assert(c[i] == a[i] + b[i]);
}
