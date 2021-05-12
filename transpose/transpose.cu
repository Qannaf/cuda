#include <stdio.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>

#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

inline void __checkCudaErrors(cudaError err, const char* file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int)err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}





template<typename T>
__global__ void kernelTranspose(T* v, T* vT,int _row, int _col)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = j * gridDim.x * blockDim.x + i;
	vT[k] = v[j + i * _col];
}




/**
*@brief Transpose matrix
*@param  The data to be Transposed, the data widthand height
*@retval The Transposed data.
*/
template<typename T>
void transpose(const std::vector<T>& v, std::vector<T>& vT, const int& _rows, const int& _cols)
{
	// Initialize GPU
	T* d_v, * d_vT;

	// Allocate GPU buffers
	checkCudaErrors(cudaMalloc(&d_v, sizeof(T) * v.size()));
	checkCudaErrors(cudaMalloc(&d_vT, sizeof(T) * v.size()));


	// CPU 'host memory'  --->  GPU buffer.
	checkCudaErrors(cudaMemcpy(d_v, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vT, vT.data(), sizeof(T) * vT.size(), cudaMemcpyHostToDevice));


	// launch Kernel
	kernelTranspose << < 1,dim3( _rows, _cols, 1) >> > (d_v, d_vT,_rows,_cols);

	// Check for any errors launching the kernel
	checkCudaErrors(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, 
	// and returns any errors encountered during the launch.
	checkCudaErrors(cudaDeviceSynchronize());

	// GPU buffer ---> CPU 'host memory'
	checkCudaErrors(cudaMemcpy(vT.data(), d_vT, sizeof(T) * vT.size(), cudaMemcpyDeviceToHost));

	// freed the allocated memory
	checkCudaErrors(cudaFree(d_v));
	checkCudaErrors(cudaFree(d_vT));
}

template<typename T>
void transposecpu(const std::vector<T>& v, std::vector<T>& vT, const int& _rows, const int& _cols)
{
	vT.clear();
	
	for (int j = 0; j < _cols; ++j)
		for (int i = 0; i < _rows; ++i)
			std::cout << "CPU i = " << i << "\tj = " << j << "\n";
			//vT.push_back(v[j+i*_cols]);
		
}

int main()
{	
	std::vector<int> v1 = { 1,2,3,4,5,
							6,7,8,9,10,
							1,2,3,4,5,
							6,7,8,9,10 };
	std::vector<int> vt(v1.size()) ;

	transpose(v1, vt, 4, 5);
	//transposecpu(v1, vt, 2, 3);

	//for (const auto& e : v1) std::cout << e << "\t";
	for (const auto& e : vt) std::cout << e << "\t";
	system("pause");
	return EXIT_SUCCESS;
}