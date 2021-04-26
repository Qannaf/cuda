#include <iostream>
#include <cuda.h>
#include <algorithm>
#include <cstdlib>
using namespace std;

__global__ void fun1(float *d_out, float *d_in)
 {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x, x_;
  if(i > 0)
  { x =d_in[i];
    x_ = d_in[i-1];
    d_out[i] = x+ x_;
  }
}

int main()
{
    // initial les constanates
    const int N = 1024;
    const float n_byte = N*sizeof(int);

    //CPU variables
    float *h_in = new float(N);
    float *h_out = new float(N);
    for(size_t i =0; i<N;i++)
        h_in[i] = 1;

    for(size_t i =0; i<N;i++)
        std::cout<<h_in[i];
    // GPU variables
    float *d_in ;
    float *d_out ;
    cudaMalloc( (void**) &d_in, n_byte );
	cudaMalloc( (void**) &d_out, n_byte );

    // GPU ----> CPU
    cudaMemcpy(d_in, h_in, n_byte, cudaMemcpyHostToDevice);

    // lancer le kernel
    dim3 t_grid(2,2,1);
	dim3 t_bloc(256,1,1);
    fun1<<< t_grid, t_bloc >>>(d_out, d_in);

    // GPU ---->   CPU
	cudaMemcpy(h_out, d_out, n_byte, cudaMemcpyDeviceToHost);
	for(size_t i=0; i<N; ++i) 
		cout<<h_out[i]<<endl;


	// 6) libération de la mémoire sur le GPU
	cudaFree( d_in );cudaFree( d_out );
    // 7) libération de la mémoire centrale
    delete [] h_in;delete [] h_out;
    
    exit(EXIT_FAILURE);
}
    





 
 