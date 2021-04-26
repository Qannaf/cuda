/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>
#include <chrono>
#include <random>

using namespace std;
//===========================       kernel    ========================================
__global__ void vectorAdd(int* a, int* b, int* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = j * gridDim.x * blockDim.x + i;

    if (k < N) c[k] = a[k] + b[k];
}

auto get_time() { return chrono::high_resolution_clock::now(); }



//===========================       fuction main      ===================================================
int main() {
    constexpr int N = 1000 << 16;
    size_t bytes = sizeof(int) * N;
    int NUM_THREADS = 1 << 10;
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    // CPU
    int* h_a, * h_b, * h_c;
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);

    for (int i = 0; i < N; i++) // initialisation les vacteurs a ,b
    {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }


    //GPU
    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // CPU ---> GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);


    auto start = get_time();

    vectorAdd << <NUM_BLOCKS, NUM_THREADS >> > (d_a, d_b, d_c, N);

    // GPU ---> CPU
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    auto finish = get_time();
    auto duration =
        chrono::duration_cast<std::chrono::milliseconds>(finish - start);

    cout << "temps ecoule en kernel = " << duration.count() << " ms\n";

    for (int i = 0; i < N; i++) {
        assert(h_c[i] == h_a[i] + h_b[i]);
    }


    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cout << "terminé avec succès" << endl;

    return 0;
}
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>
#include <chrono>
#include <random>
#include <cstdio>
#include <cstdlib>

#define gpuErrchk(ans) { gpuAssert( (ans), __FILE__, __LINE__ ); }

inline void
gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (cudaSuccess != code)
    {
        fprintf(stderr, "\nGPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }

    return;

} /* gpuAssert */

__global__ void Add(int N, int Offset, float* devA, float* devB, float* devC)
{

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x + Offset; idx < N; idx += blockDim.x * gridDim.x)

        devC[idx] = devA[idx] + devB[idx];

}

int main()
{

    int N = 400000000;

    int Threads = 256;

    const int NbStreams = 8;

    float* A, * B, * C;
    gpuErrchk(cudaHostAlloc((void**)&A, N * sizeof(*A), cudaHostAllocDefault));
    gpuErrchk(cudaHostAlloc((void**)&B, N * sizeof(*B), cudaHostAllocDefault));
    gpuErrchk(cudaHostAlloc((void**)&C, N * sizeof(*C), cudaHostAllocDefault));

    for (int i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = i + 1;
    }

    float* devA, * devB, * devC;
    gpuErrchk(cudaMalloc((void**)&devA, N * sizeof(*devA)));
    gpuErrchk(cudaMalloc((void**)&devB, N * sizeof(*devB)));
    gpuErrchk(cudaMalloc((void**)&devC, N * sizeof(*devC)));

    cudaEvent_t EventPre,
        EventPost;
    float PostPreTime;

    gpuErrchk(cudaEventCreate(&EventPre));
    gpuErrchk(cudaEventCreate(&EventPost));

    cudaStream_t Stream[NbStreams];
    for (int i = 0; i < NbStreams; i++)
        gpuErrchk(cudaStreamCreate(&Stream[i]));

#ifdef NOSTREAMS

    gpuErrchk(cudaEventRecord(EventPre));

    gpuErrchk(cudaMemcpy(devA, A, N * sizeof(*A), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(devB, B, N * sizeof(*B), cudaMemcpyHostToDevice));
    //        gpuErrchk( cudaMemcpy(devC, C, N * sizeof(*C), cudaMemcpyHostToDevice) );

    Add << < N / Threads, Threads >> > (N, 0, devA, devB, devC);

    //        gpuErrchk( cudaMemcpy(A, devA, N * sizeof(*A), cudaMemcpyDeviceToHost) );
    //        gpuErrchk( cudaMemcpy(B, devB, N * sizeof(*B), cudaMemcpyDeviceToHost) );
    gpuErrchk(cudaMemcpy(C, devC, N * sizeof(*C), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaEventRecord(EventPost));
    gpuErrchk(cudaEventSynchronize(EventPost));
    gpuErrchk(cudaEventElapsedTime(&PostPreTime, EventPre, EventPost));
    printf("\nTime not using streams: %f ms\n", PostPreTime);

#else

    const int StreamSize = N / NbStreams;
    gpuErrchk(cudaEventRecord(EventPre));
    for (int i = 0; i < NbStreams; i++)
    {
        int Offset = i * StreamSize;

        gpuErrchk(cudaMemcpyAsync(&devA[Offset], &A[Offset], StreamSize * sizeof(*A), cudaMemcpyHostToDevice, Stream[i]));
        gpuErrchk(cudaMemcpyAsync(&devB[Offset], &B[Offset], StreamSize * sizeof(*B), cudaMemcpyHostToDevice, Stream[i]));
        //                gpuErrchk( cudaMemcpyAsync(&devC[ Offset ], &C[ Offset ], StreamSize * sizeof(*C), cudaMemcpyHostToDevice, Stream[ i ]) );

        Add << < StreamSize / Threads, Threads, 0, Stream[i] >> > (Offset + StreamSize, Offset, devA, devB, devC);

        //                gpuErrchk( cudaMemcpyAsync(&A[ Offset ], &devA[ Offset ], StreamSize * sizeof(*devA), cudaMemcpyDeviceToHost, Stream[ i ]) );
        //                gpuErrchk( cudaMemcpyAsync(&B[ Offset ], &devB[ Offset ], StreamSize * sizeof(*devB), cudaMemcpyDeviceToHost, Stream[ i ]) );
        gpuErrchk(cudaMemcpyAsync(&C[Offset], &devC[Offset], StreamSize * sizeof(*devC), cudaMemcpyDeviceToHost, Stream[i]));

    }

    gpuErrchk(cudaEventRecord(EventPost));
    gpuErrchk(cudaEventSynchronize(EventPost));
    gpuErrchk(cudaEventElapsedTime(&PostPreTime, EventPre, EventPost));
    printf("\nTime using streams: %f ms\n", PostPreTime);

#endif /* ! USE_STREAMS */

    for (int i = 0; i < N; i++)
        if (C[i] != (A[i] + B[i])) { printf("mismatch at %d, was: %f, should be: %f\n", i, C[i], (A[i] + B[i])); return 1; }

    for (int i = 0; i < NbStreams; i++)
        gpuErrchk(cudaStreamDestroy(Stream[i]));

    gpuErrchk(cudaFree(devA));
    gpuErrchk(cudaFree(devB));
    gpuErrchk(cudaFree(devC));

    gpuErrchk(cudaFreeHost(A));
    gpuErrchk(cudaFreeHost(B));
    gpuErrchk(cudaFreeHost(C));

    gpuErrchk(cudaEventDestroy(EventPre));
    gpuErrchk(cudaEventDestroy(EventPost));

    printf("\n");

    return 0;

}