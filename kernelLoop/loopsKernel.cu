
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

/*
// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif


//for __syncthreads()
#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif // !(__CUDACC_RTC__)
//for atomicAdd
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>

/*

//one loop
__global__ void single_loop() {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    printf("GPU - col = %d \n", col);
}
int main(void) {
    // Single loop in CPU
    for (int col = 0; col < 4; col++) {
        printf("CPU - col = %d \n", col);
    }
    // Single loop in GPU
    dim3 grid(1, 1, 1);
    dim3 block(4, 1, 1);   //or  dim3 grid(4, 1, 1); dim3 block(1, 1, 1);   or  dim3 grid(2, 1, 1); dim3 block(2, 1, 1);
    single_loop << <grid, block >> > ();
    cudaDeviceSynchronize();
    return 0;
}
*/



/*
//tow loops
__global__ void two_nested_loops() {
    int  row = threadIdx.y+ blockIdx.y * blockDim.y;
    int col = threadIdx.x+ blockIdx.x * blockDim.x;
   //+ blockIdx.y * blockDim.y;
    printf("GPU - row = %d - col = %d \n",  row, col);

}
int main(void) {
    // Two nested loops CPU
    for (int row = 0; row < 2; row++) {
        for (int col = 0; col < 4; col++) {
            printf("CPU - row = %d - col = %d \n", row, col);
        }
    }
    // Two nested loops GPU
    dim3 grid(1, 1, 1);
    dim3 block(4, 2, 1);
    two_nested_loops << <grid, block >> > (); // or dim3 grid(4, 2, 1); dim3 block(1, 1, 1);  or  dim3 grid(2, 2, 1); dim3 block(2, 1, 1);   or dim3 grid(2, 1, 1); dim3 block(2, 2, 1);
    cudaDeviceSynchronize();
    return 0;
}
*/

/*
__device__ const int size[3] = { 4,3,2 };
// Three loops
__global__ void triple_nested_loops(int* d_counterPoints) {
    int* compteur;
    int depth = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    int row = threadIdx.z + blockIdx.z * blockDim.z;

    printf("GPU - row = %d - j = %d - k = %d \n", row, col, depth );
}


int main(void) {
    // Triple nested loops in CPU
    int size[3] = {4,3,2 };
    int compteur=0;
    for (int row = 0; row <size[1]; row++) {
        for (int col = 0; col < size[0]; col++) {
            for (int depth = 0; depth <size[2]; depth++) {
                printf("CPU - row = %d - j = %d - k = %d  compteur =%d\n", row, col, depth, compteur++);
            }
        }
    }
    // Triple nested loops in GPU
    int* d_counterPoints;
    cudaMallocManaged(&d_counterPoints, sizeof(unsigned long));
    *d_counterPoints = 0;

    dim3 grid(1, 1, 30);
    dim3 block(1, 1, 1);

    triple_nested_loops << <grid, block >> > (d_counterPoints);
    cudaDeviceSynchronize();
    std::cout << *d_counterPoints-1 <<std:: endl;
    cudaFree(d_counterPoints);

    return 0;
}

*/



__device__ const int size[3] = { 4,4,4 };
__global__ void triple_nested_loops(unsigned int* d_counterPoints, unsigned int* x, unsigned int* y, unsigned int* z)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int index = row * gridDim.x * blockDim.x + col;

  
    
    if(index<(size[0]*size[1]*size[2]))
        for (int k = 0; k < size[2]; ++k)
        {
            printf("conterPoint = %d  %d  %d      %d\n", atomicAdd(x, 1), atomicAdd(y, 1), atomicAdd(z, 1), atomicAdd(d_counterPoints, 1));
            //atomicAdd(d_counterPoints, 1);
            
        }
   
}


int main(void) {
    
   
    unsigned  int* d_counterPoints(0);
    unsigned  int* x(0);
    unsigned  int* y(0);
    unsigned  int* z(0);
    cudaMallocManaged(&d_counterPoints, sizeof(unsigned  int));
    cudaMallocManaged(&x, sizeof(unsigned  int));
    cudaMallocManaged(&y, sizeof(unsigned  int));
    cudaMallocManaged(&z, sizeof(unsigned  int));
    


    triple_nested_loops << <dim3(size[0], size[1]), 1 >> > (d_counterPoints, x, y,z);
    cudaDeviceSynchronize();
    std::cout << *d_counterPoints  << std::endl;
    cudaFree(d_counterPoints);

    return 0;
}
