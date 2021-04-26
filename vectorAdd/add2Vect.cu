#include <iostream>
#include <cuda.h>
#include <algorithm>
#include <cstdlib>
using namespace std;

__global__ void sum(float *x, float *y, float *z, int size) //  declaration kernel
{   int i = blockIdx.x*blockDim.x + threadIdx.x; 
	int j = blockIdx.y*blockDim.y + threadIdx.y; 
	int k = j*gridDim.x * blockDim.x + i;  

	if (k < size) z[k] = x[k] + y[k]; 
}

int main() {
	const int SIZE = 1024*1024; 
	const float n_byte = SIZE * sizeof(float);

	// 1) allocation et initialisation des données en mémoire centrale (cpu/host)
	float *h_x = new float[ SIZE ], *h_y = new float[ SIZE ], *h_z = new float[ SIZE ];
	fill(&h_x[0], &h_x[SIZE], 1);
	fill(&h_y[0], &h_y[SIZE], 2);

	//2) allocation dans la mémoire du GPU (device)
	float *d_x, *d_y, *d_z;
	cudaMalloc( (void**) &d_x, n_byte );
	cudaMalloc( (void**) &d_y, n_byte );
	cudaMalloc( (void**) &d_z, n_byte );

	// 3) copie des données en entrée de host (cpu) vers device (gpu) 
	cudaMemcpy(d_x, h_x, n_byte, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, n_byte, cudaMemcpyHostToDevice);
	
	// 4) exécution du kernel
	dim3 t_grid(1024,1,1);
	dim3 t_bloc(512,2,1);
	sum<<< t_grid, t_bloc >>>(d_x, d_y, d_z, SIZE);

	// 5) copie des données en sortie de device (gpu) vers host (cpu) 
	cudaMemcpy(h_z, d_z, n_byte, cudaMemcpyDeviceToHost);
	for(size_t i=0; i<SIZE; ++i) 
		cout<< h_x[i]<<" + "<<h_y[i]<<" = "<<h_z[i]<<endl;


	// 6) libération de la mémoire sur le GPU
	cudaFree( d_x );cudaFree( d_y );cudaFree( d_z );
    // 7) libération de la mémoire centrale
	delete [] h_x;delete [] h_y;delete [] h_z;
	exit(EXIT_FAILURE);
}