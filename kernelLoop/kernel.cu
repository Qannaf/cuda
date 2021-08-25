
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>


/*
//one loop
__global__ void single_loop_Kernel() {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    printf("GPU : col = %d \n", col);
}

void single_loop_GPU() 
{
    dim3 grid(1, 1, 1);
    dim3 block(4, 1, 1);   //or  dim3 grid(4, 1, 1); dim3 block(1, 1, 1);   or  dim3 grid(2, 1, 1); dim3 block(2, 1, 1);
    single_loop_Kernel << <grid, block >> > ();
    cudaDeviceSynchronize();
}

void single_loop_CPU()
{
    for (int col = 0; col < 4; col++) 
        std::cout<<"CPU : col = "<< col<<"\n";
}
int main(void) {
    // Single loop in CPU
    single_loop_CPU();
   
    // Single loop in GPU
    single_loop_GPU();
    return 0;
}
//*/


/*

//tow loops
__global__ void two_nested_loops_kernel() {
    int  row = threadIdx.y+ blockIdx.y * blockDim.y;
    int col = threadIdx.x+ blockIdx.x * blockDim.x;
   //+ blockIdx.y * blockDim.y;
    printf("GPU : row = %d \t col = %d \n",  row, col);

}

void  two_nested_loops_GPU()
{   dim3 grid(1, 1, 1);
    dim3 block(4, 2, 1);
    two_nested_loops_kernel << <grid, block >> > (); // or dim3 grid(4, 2, 1); dim3 block(1, 1, 1);  or  dim3 grid(2, 2, 1); dim3 block(2, 1, 1);   or dim3 grid(2, 1, 1); dim3 block(2, 2, 1);
    cudaDeviceSynchronize(); 
}
void  two_nested_loops_CPU()
{
    for (int row = 0; row < 2; row++)
        for (int col = 0; col < 4; col++)
            std::cout << "CPU : row = " << row << "\t col = " << col << "\n";
}
int main(void) {
    // Two nested loops CPU
        two_nested_loops_CPU();
    // Two nested loops GPU
        two_nested_loops_GPU();
    return 0;
}
//*/




// Three loops
__global__ void triple_nested_loops_kernel() {
    int depth = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    int  row = threadIdx.z + blockIdx.z * blockDim.z;
    printf("GPU : row = %d \t col = %d \t depth = %d \n", row, col, depth);
}
void triple_nested_loops(){
    dim3 grid(1, 1, 1);
    dim3 block(2, 3, 4); 
    triple_nested_loops_kernel << <grid, block >> > ();
    cudaDeviceSynchronize();
}
void triple_nested_loops_CPU()
{
    for (int row = 0; row < 4; row++) 
        for (int col = 0; col < 3; col++) 
            for (int depth = 0; depth < 2; depth++) 
                std::cout<<"CPU : row = "<<row<<"\t col = "<<col<<" \t depth = "<<depth<<" \n";  
}

int main(void) {
    // Triple nested loops in CPU
    triple_nested_loops_CPU();
    // Triple nested loops in GPU
    triple_nested_loops();
    return 0;
}

//*/

