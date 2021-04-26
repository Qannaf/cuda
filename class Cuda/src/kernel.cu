/**
 * @file kernel.cu 
 *
 * @brief class to compute the summe of 2 vectors
 *
 * @authors
 *          - Qannaf AL-SAHMI
 *
 * @version 1.0.0
 *
 * @date 08/04/2021
 *
 */

 #include <iostream>
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <cuda.h>
 #include "cuda_runtime.h"
 #include "device_launch_parameters.h"
 #include <memory>
 
 
 
 __global__ void kernel(int* v1, int* v2, int* out, unsigned int N)
 {
     const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < N)      out[i] = v1[i]+v2[i]; 
  
 }
 
 
 
 class MyClass 
 {
 
 public:
     // constructor
     __host__ MyClass(const int& _size = 1) 
     {
         size = _size;
         cudaMalloc((void**)&d_a, sizeof(int) * size);
         cudaMemset((void*)d_a, 0, sizeof(int) * size);
 
         cudaMalloc((void**)&d_b, sizeof(int) * size);
         cudaMemset((void*)d_b, 0, sizeof(int) * size);
 
         cudaMalloc((void**)&d_c, sizeof(int) * size);
         cudaMemset((void*)d_c, 0, sizeof(int) * size);
         h_c = new int[size];
     };
 
     // constructor
     __host__ MyClass(int* a, int* b, const size_t& _size)
     {
         size = _size;
         cudaMalloc((void**)&d_a, sizeof(int) * size);
         cudaMemcpy(d_a, a, sizeof(int) * size, cudaMemcpyHostToDevice);
 
         cudaMalloc((void**)&d_b, sizeof(int) * size);
         cudaMemcpy(d_b, b, sizeof(int) * size, cudaMemcpyHostToDevice);
 
 
         cudaMalloc((void**)&d_c, sizeof(int) * size);
         cudaMemset((void*)d_c, 0, sizeof(int) * size);
         h_c = new int[size];
     };
 
     // deconstructor
     __host__ virtual ~MyClass() 
     {
 
         cudaFree((void*)d_a);
         cudaFree((void*)d_b);
         cudaFree((void*)d_c);
         delete h_c;
     };
 
     // method of class
     __host__ void run(const dim3& _grid=1   ,const dim3& _block=1) 
     {
        
         dim3 grid(_grid);
         dim3 block(_block);
         kernel << <grid, block >> > (d_a,d_b, d_c,size);
     };
 
 
     __host__ int* get(void) 
     {
         cudaMemcpy(h_c, d_c, sizeof(int) * size, cudaMemcpyDeviceToHost);
         return h_c;
     };
 
     __host__ __device__  void show()
     {
 
         for (int i = 0; i < size; i++)
             printf(" %d     ", h_c[i]);
         printf("\n");
     }
 
 
 // attribute of class
 private:
     int* d_a;
     int* d_b;
     int* d_c;
     int* h_c;
     size_t size;
 };
 
 
 
 
 
 __host__ __device__ void show(int* data, unsigned int N) 
 {
     
     for (int i = 0; i < N; i++) 
         printf(" %d     ", data[i]);  
     printf("\n");
 }
 
 
 
 int main(void) 
 {
     size_t size = 10;
     int* v1 = new int[size];
     int* v2 = new int[size];
     int* s = new int[size];
 
     
     for (int i = 0; i < size; i++)  v1[i] = i;
     for (int i = 0; i < size; i++)  v2[i] = i*2;
     show(v1, size);
     show(v2, size);
 
 
 
     MyClass c(v1,v2,size);
     c.run((size),(1));
     s = c.get();
 
     // use class method
     c.show();
 
     // use normal method  
     show(s, size);
 
 
     system("pause");
     return 0;
 }