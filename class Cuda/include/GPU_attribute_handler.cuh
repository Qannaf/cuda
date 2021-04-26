
/**
 * @file functionsEigen.h
 *
 * @brief some useful functions which uses Eigen library 
 * for build this API (Eigen3) in microsoft windows ,Visual Studio 2019 with CMake 
 * You can find this tutorial in my page github :
 * https://github.com/Qannaf/Building-PCL-with-Visual-Studio-from-source-on-Windows/blob/main/Eigen3.3.8.md
 * 
 *
 * @authors
 *          - Qannaf AL-SAHMI
 *
 * @version 1.0.0
 *
 * @date 09/03/2021
 *
 */

#ifndef __GPU__attribute_handler_HPP__
#define __GPU__attribute_handler_HPP__

 //! Cuda libraries
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class gpu_attribute_handler {
private:
	size_t firstnode_id;
	size_t lastnode_id;

	char* attribute_ptr;

	bool boundtype;
	bool writefile;

public:
	__device__ gpu_attribute_handler() {
		firstnode_id = 0;
	}

	__device__ ~gpu_attribute_handler() {
	}

	__device__ void set(int* temp, size_t SIZE) {
		cudaMalloc((void**)&attribute_ptr, SIZE);
		cudaMemcpy(attribute_ptr, temp, SIZE, cudaMemcpyHostToDevice);
	}

	__device__ void get_result(int* temp, size_t SIZE) {
		cudaMemcpy(temp, attribute_ptr, SIZE, cudaMemcpyDeviceToHost);
	}

	template<typename valuetype>
	__device__ valuetype get_value(size_t nodenumber) {
		valuetype* value_ptr = (valuetype*)attribute_ptr;
		return value_ptr[nodenumber - firstnode_id];
	}

	template<typename valuetype>
	__device__ void set_value(size_t nodenumber, valuetype value) {
		valuetype* value_ptr = (valuetype*)attribute_ptr;
		value_ptr[nodenumber - firstnode_id] = value;
	}

	template<typename valuetype>
	__device__ void add_value(size_t nodenumber, valuetype value) {
		valuetype* value_ptr = (valuetype*)attribute_ptr;
		value_ptr[nodenumber - firstnode_id] += value;
	}

	__device__ void set_boundtype(bool _boundtype) {
		boundtype = _boundtype;
	}

	__device__ void set_writefile(bool _writefile) {
		writefile = _writefile;
	}
};

#endif
