#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdio.h>
#include "cuda.h"

__global__ void Sum( int* index)

{
	int row = blockIdx.z * blockDim.z + threadIdx.z;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int depth = blockIdx.x * blockDim.x + threadIdx.x;
	
	atomicAdd(index, 1);	
}

int main(int argc, char* argv[])

{
	int W = 16*16*16;
	int H = 4*4*4;

	int* d_index = 0;
	int h_index = 0;

	cudaMalloc((void**)&d_index, sizeof(int));
	cudaMemcpy(d_index, &h_index, sizeof(int), cudaMemcpyHostToDevice);
	dim3 grid = dim3(16, 16, 16);
	dim3 block = dim3(4, 4, 4);
	Sum << <grid,block>> > ( d_index);
	cudaMemcpy(&h_index, d_index, sizeof(int), cudaMemcpyDeviceToHost);
	fprintf(stderr, "%d\t %d\n", h_index,W*H);
	cudaFree(d_index);

	return 0;

}