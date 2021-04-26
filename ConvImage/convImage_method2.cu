#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <opencv2/opencv.hpp>
using namespace std;

#define PGMHeaderSize           0x40
#define TILE_W      16
#define TILE_H      16
#define Rx          2                       // filter radius in x direction
#define Ry          2                       // filter radius in y direction
#define FILTER_W    (Rx*2+1)                // filter diameter in x direction
#define FILTER_H    (Ry*2+1)                // filter diameter in y direction
#define S           (FILTER_W*FILTER_H)     // filter size
#define BLOCK_W     (TILE_W+(2*Rx))        // 16+ 2*2 = 20
#define BLOCK_H     (TILE_H+(2*Ry))
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)



inline bool loadPPM(const char *file, unsigned char **data, unsigned int *w, unsigned int *h, unsigned int *channels);
__global__ void box_filter(const unsigned char *in, unsigned char *out, const unsigned int w, const unsigned int h);
inline bool savePPM(const char *file, unsigned char *data, unsigned int w, unsigned int h, unsigned int channels);
inline void __checkCudaErrors(cudaError err, const char *file, const int line);











int main()
{
	// CPU
	unsigned char *h_data=NULL;
	unsigned int w,h,channels;
	
    
	
	//load image
    if(! loadPPM("lena.ppm", &h_data, &w, &h, &channels)){
        cout<< "Failed to open File\n";
        exit(EXIT_FAILURE);
    }
    cout<<"------> Loaded file with :"<<w<<"*" << h << "  channels:"<<channels<<endl;

	//GPU
	unsigned char*d_idata=NULL;
	unsigned char *d_odata=NULL;
	size_t n_byte = w*h*channels * sizeof(unsigned char);

    // GPU ---> CPU
    cout<<"\n------> Allocate Devicememory for data"<<endl;
    checkCudaErrors(cudaMalloc((void **)&d_idata, n_byte));
    checkCudaErrors(cudaMalloc((void **)&d_odata, n_byte));

    // Copy to device
    cout<<"\n------> Copy h_data from the host memory to the CUDA device\n";
    checkCudaErrors(cudaMemcpy(d_idata, h_data, n_byte, cudaMemcpyHostToDevice));

    // kernel
    int GRID_W = w/TILE_W +1;                   // 512/16 +1 = 33
    int GRID_H = h/TILE_H +1;       
    dim3 threadsPerBlock(BLOCK_W, BLOCK_H);     
	dim3 blocksPerGrid(GRID_W,GRID_H);
    
    cout<<"\n------> CUDA kernel launch with [" <<blocksPerGrid.x<<" "<< blocksPerGrid.y <<"] blocks of [" <<threadsPerBlock.x<<" "<< threadsPerBlock.y<< "]threads"<<endl;
	
	box_filter<<<blocksPerGrid, threadsPerBlock>>>(d_idata, d_odata, w,h);
    checkCudaErrors(cudaGetLastError());

    // GPU --->  CPU
    cout<<"\n------> Copy odata from the CUDA device to the host memory"<<endl;
    checkCudaErrors(cudaMemcpy(h_data, d_odata, n_byte, cudaMemcpyDeviceToHost));

   
    cout<<"\n------> Free Device memory"<<endl;
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    cv::Mat TempMat = cv::Mat(h, w, CV_8UC1, h_data);
    cv::imshow("image output", TempMat);
    cv::waitKey();

    // Save Picture
	cout<<"\n------> Save Picture"<<endl;

    bool saved = false;
    if      (channels==1)    
        saved = savePPM("output.pgm", h_data, w,  h,  channels);
    else if (channels==3)
        saved = savePPM("output.ppm", h_data, w,  h,  channels);
	else 
		cout<<"\n------> ERROR: Unable to save file - wrong channel!\n";

	cout<<"\n------> Free Host memory\n";
	free(h_data);

	if (!saved){
        cout<<"\n------>  Failed to save File\n";
        exit(EXIT_FAILURE);
    }


	cout<<"\n------> Done\n";

	return 0;
}






__global__ void box_filter(const unsigned char *in, unsigned char *out, const unsigned int w, const unsigned int h)
{
    const int x = blockIdx.x * TILE_W + threadIdx.x - Rx;       
    const int y = blockIdx.y * TILE_H + threadIdx.y - Ry;       
	const int d = y * w+ x;       
	                             

	__shared__ float shMem[BLOCK_W][BLOCK_H];     // 20*20
	if(x<0 || y<0 || x>=w || y>=h)                // x et y ∈ [0,512]
	{            
        shMem[threadIdx.x][threadIdx.y] = 0;
        return; 
    }
    shMem[threadIdx.x][threadIdx.y] = in[d];
    __syncthreads();

    if ((threadIdx.x >= Rx) && (threadIdx.x < (BLOCK_W-Rx)) && (threadIdx.y >= Ry) && (threadIdx.y < (BLOCK_H-Ry))) {
        float sum = 0;
        for(int dx=-Rx; dx<=Rx; dx++) {
            for(int dy=-Ry; dy<=Ry; dy++) {
                sum += shMem[threadIdx.x+dx][threadIdx.y+dy];
            }
        }
    out[d] = sum / S;       
    }
}







inline bool loadPPM(const char *file, unsigned char **data, unsigned int *w, unsigned int *h, unsigned int *channels)
{
    FILE *fp = NULL;

    fp = fopen(file, "rb");
         if (!fp) {
              fprintf(stderr, "__LoadPPM() : unable to open file\n" );
                return false;
         }

    // check header
    char header[PGMHeaderSize];

    if (fgets(header, PGMHeaderSize, fp) == NULL)
    {
        fprintf(stderr,"__LoadPPM() : reading PGM header returned NULL\n" );
        return false;
    }

    if (strncmp(header, "P5", 2) == 0)
    {
        *channels = 1;
    }
    else if (strncmp(header, "P6", 2) == 0)
    {
        *channels = 3;
    }
    else
    {
        fprintf(stderr,"__LoadPPM() : File is not a PPM or PGM image\n" );
        *channels = 0;
        return false;
    }

    // parse header, read maxval, width and height
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int maxval = 0;
    unsigned int i = 0;

    while (i < 3)
    {
        if (fgets(header, PGMHeaderSize, fp) == NULL)
        {
            fprintf(stderr,"__LoadPPM() : reading PGM header returned NULL\n" );
            return false;
        }

        if (header[0] == '#')
        {
            continue;
        }

        if (i == 0)
        {
            i += sscanf(header, "%u %u %u", &width, &height, &maxval);
        }
        else if (i == 1)
        {
            i += sscanf(header, "%u %u", &height, &maxval);
        }
        else if (i == 2)
        {
            i += sscanf(header, "%u", &maxval);
        }
    }

    // check if given handle for the data is initialized
    if (NULL != *data)
    {
        if (*w != width || *h != height)
        {
            fprintf(stderr, "__LoadPPM() : Invalid image dimensions.\n" );
        }
    }
    else
    {
        *data = (unsigned char *) malloc(sizeof(unsigned char) * width * height * *channels);
        if (!data) {
         fprintf(stderr, "Unable to allocate hostmemory\n");
         return false;
        }
        *w = width;
        *h = height;
    }

    // read and close file
    if (fread(*data, sizeof(unsigned char), width * height * *channels, fp) == 0)
    {
        fprintf(stderr, "__LoadPPM() : read data returned error.\n" );
        fclose(fp);
        return false;
    }

    fclose(fp);

    return true;
}

inline bool savePPM(const char *file, unsigned char *data, unsigned int w, unsigned int h, unsigned int channels)
{
    assert(NULL != data);
    assert(w > 0);
    assert(h > 0);

    std::fstream fh(file, std::fstream::out | std::fstream::binary);

    if (fh.bad())
    {
        fprintf(stderr, "__savePPM() : Opening file failed.\n" );
        return false;
    }

    if (channels == 1)
    {
        fh << "P5\n";
    }
    else if (channels == 3)
    {
        fh << "P6\n";
    }
    else
    {
        fprintf(stderr, "__savePPM() : Invalid number of channels.\n" );
        return false;
    }

    fh << w << "\n" << h << "\n" << 0xff << std::endl;

    for (unsigned int i = 0; (i < (w*h*channels)) && fh.good(); ++i)
    {
        fh << data[i];
    }

    fh.flush();

    if (fh.bad())
    {
        fprintf(stderr,"__savePPM() : Writing data failed.\n" );
        return false;
    }

    fh.close();

    return true;
}




inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}