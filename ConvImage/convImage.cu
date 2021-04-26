#include "cuda_runtime.h"
#include "device_launch_parameters.h"
 
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

 
 
using namespace std;
using namespace cv;

 


#define TILE_W      1<<4
#define TILE_H      1<<4
#define R           1                   // filter radius
#define D           (R*2+1)             // filter diameter
#define S           (D*D)               // filter size
#define BLOCK_W     (TILE_W+(2*R))
#define BLOCK_H     (TILE_H+(2*R))
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)


__global__ void my_filter_origin(uchar *g_idata, uchar *g_odata, unsigned int width, unsigned int height, int* filter)
{
    __shared__ unsigned char smem[BLOCK_W*BLOCK_H];
    int x = blockIdx.x*TILE_W + threadIdx.x - 1;
    int y = blockIdx.y*TILE_H + threadIdx.y - 1;

    // clamp to edge of image
    x = max(0, x);
    x = min(x, width - 1);
    y = max(y, 0);
    y = min(y, height - 1);
    unsigned int index = y * width + x;
    unsigned int bindex = threadIdx.y*blockDim.y + threadIdx.x;
    // each thread copies its pixel of the block to shared memory
    smem[bindex] = g_idata[index];
    __syncthreads();

    // only threads inside the apron will write results
    if ((threadIdx.x >= 1) && (threadIdx.x <= TILE_W) && (threadIdx.y >= 1) && (threadIdx.y <= TILE_H)) {
        float sum = 0;
        for (int dy = 0; dy < 3; dy++) {
            for (int dx = 0; dx < 3; dx++) {
                float i = smem[bindex + ((dy - 1)*blockDim.x) + (dx - 1)] * filter[dy * 3 + dx];
                sum += i;
            
            }
        }     
        g_odata[index] = sum;
    }
}

__global__ void my_filter_not_for_loop(uchar *g_idata, uchar *g_odata, unsigned int width, unsigned int height, int* filter)
{
    __shared__ uchar smem[BLOCK_W*BLOCK_H];
    int x = blockIdx.x*TILE_W + threadIdx.x - 1;
    int y = blockIdx.y*TILE_H + threadIdx.y - 1;

    // clamp to edge of image
    x = max(0, x);
    x = min(x, width - 1);
    y = max(y, 0);
    y = min(y, height - 1);
    unsigned int index = y * width + x;
    unsigned int bindex = threadIdx.y*blockDim.y + threadIdx.x;
    // each thread copies its pixel of the block to shared memory
    smem[bindex] = g_idata[index];
    __syncthreads();

    // only threads inside the apron will write results
    if ((threadIdx.x >= 1) && (threadIdx.x <= TILE_W) && (threadIdx.y >= 1) && (threadIdx.y <= TILE_H)) {
        long int sum = 0;

        sum += smem[bindex + -1 * blockDim.x - 1] * filter[0];
        sum += smem[bindex + -1 * blockDim.x] * filter[1];
        sum += smem[bindex + -1 * blockDim.x + 1] * filter[2];
        sum += smem[bindex - 1] * filter[3];
        sum += smem[bindex] * filter[4];
        sum += smem[bindex + 1] * filter[5];
        sum += smem[bindex + blockDim.x - 1] * filter[6];
        sum += smem[bindex + blockDim.x] * filter[7];
        sum += smem[bindex + blockDim.x + 1] * filter[8];

        if (sum > 255) sum = 255;
        if (sum < 0) sum = 0;

        g_odata[index] = sum;
    }
}

__global__ void my_filter_no_if(uchar *g_idata, uchar *g_odata, unsigned int width, unsigned int height, int* filter)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = x + y*blockDim.x*gridDim.x;

    long int sum = 0;

    sum += g_idata[gid] * -1;
    sum += g_idata[gid + 1] * -1;
    sum += g_idata[gid + 2] * -1;
    sum += g_idata[gid + width] * -1;
    sum += g_idata[gid + width + 1] * 8;
    sum += g_idata[gid + width + 2] * -1;
    sum += g_idata[gid + 2 * width] * -1;
    sum += g_idata[gid + 2 * width + 1] * -1;
    sum += g_idata[gid + 2 * width + 2] * -1;

    if (sum > 255) sum = 255;
    if (sum < 0) sum = 0;

    g_odata[gid] = sum;
}


// filtring in CPU
void my_filter_cpu(const cv::Mat& img_input,int mask[3][3] ,cv::Mat& img_result_cpu)
{
    long int sum;
    double t1 = (double)getTickCount();
    for (int y = 0; y < img_input.rows; y++)
        for (int x = 0; x < img_input.cols; x++)
        {
            for (int c = 0; c < 3; c++) {
                sum = 0;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                    {
                        int center_y = y + i - 1;
                        int center_x = x + j - 1;

                        
                        if (center_y < 0) center_y = 0;
                        else if (center_y > img_input.rows - 1) center_y = img_input.rows - 1;

                        if (center_x < 0) center_x = 0;
                        else if (center_x > img_input.cols - 1) center_x = img_input.cols - 1;


                        //sum += img_input.at<Vec3b>(center_y, center_x)[c] * mask11[i][j];
                        sum += img_input.at<Vec3b>(center_y, center_x)[c] * mask[i][j];
                    }
                if (sum > 255) sum = 255;
                if (sum < 0) sum = 0;

                img_result_cpu.at<Vec3b>(y, x)[c] = sum;
            }
        }
    t1 = ((double)getTickCount() - t1) / getTickFrequency();
    cout << "CPU time at =  " << t1 << " sec" << endl << endl;

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

int main()
{
    
    cv::Mat img_input, img_result_cpu, img_result_gpu;
    img_input = cv::imread("lena.pgm", IMREAD_COLOR);
    if (img_input.empty())
    {
        cout << "image load error!" << endl;
        exit(1);
    }
 

    img_result_cpu = Mat(img_input.rows, img_input.cols, CV_8UC3);
    img_result_gpu = Mat(img_input.rows, img_input.cols, CV_8UC3);
    size_t sz = img_input.cols*img_input.rows * sizeof(uchar); 

    // declaration CPU
    uchar *h_input = new uchar[img_input.cols*img_input.rows];
    uchar *h_output = new uchar[img_input.cols*img_input.rows];
    int mask1[9] = { -1,-1,-1,  
                     -1, 8,-1,
                     -1,-1,-1 };
 
    int mask2[9] = { -1,-1,-1 , 
                     -1, 9,-1,
                     -1,-1,-1 };

    int mask3[9] = { -1, 0, 1, //sobel
                     -2, 0, 2,
                     -1, 0, 1 };
 
    int mask11[3][3] = { {-1,-1,-1},  
                    { -1,8,-1 },
                    { -1,-1,-1 } };
 
    int mask22[3][3] = { { -1,-1,-1 },
                        { -1,9,-1 },
                        { -1,-1,-1 } };

int mask33[3][3] = { {-1, 0, 1 },
                     {-2, 0, 2 },
                     {-1, 0, 1 } };
 
  
 
 
    // filtring in CPU
    double t1 = (double)getTickCount();
    my_filter_cpu(img_input,mask11,img_result_cpu);
    

    //               =====================================================================            //
    //                                           GPU 
    //               =====================================================================            //
    // declaraion GPU
    uchar *d_input = NULL, *d_output = NULL;
    int *d_filter;
   checkCudaErrors( cudaMalloc((void **)&d_input, sz) );
    checkCudaErrors( cudaMalloc((void **)&d_output, sz) );
    checkCudaErrors( cudaMalloc((void**)&d_filter, 9 * sizeof(int)) );
 
 
 
    for (int f = 2; f <= 16; f*=2) 
    {
 
        #ifdef TILE_W
        #undef TILE_W
        #endif
        #define TILE_W f
        
        #ifdef TILE_H
        #undef TILE_H
        #endif
        #define TILE_H f
 
        cout << "\n----------  filter orginal  -----------" << endl;

        //1. Red Green Bleu 
        for(int RGB = 0; RGB<3;++RGB)
        {
            for (int y = 0; y < img_input.rows; y++)
                for (int x = 0; x < img_input.cols; x++) {
                    h_input[y*img_input.cols + x] = img_input.at<Vec3b>(y, x)[RGB];
                }
            
    
                checkCudaErrors( cudaMemcpy(d_input, h_input, sz, cudaMemcpyHostToDevice) );
                checkCudaErrors( cudaMemcpy(d_filter, mask1, 9 * sizeof(int), cudaMemcpyHostToDevice) );
                checkCudaErrors( cudaMemcpy(d_output, h_input, sz, cudaMemcpyHostToDevice) );
    
            dim3 threadsPerBlock(TILE_W + 2, TILE_H + 2);
            dim3 blocksPerGrid(img_input.cols / TILE_W, img_input.rows / TILE_H);
    
            cout << "block size   x : " << threadsPerBlock.x << " y : " << threadsPerBlock.y << "     " << "Grid size   x :" << blocksPerGrid.x << " y : " << blocksPerGrid.y << endl;
    
            t1 = (double)getTickCount();
            my_filter_no_if<< <blocksPerGrid, threadsPerBlock >> > (d_input, d_output, img_input.cols, img_input.rows, d_filter);
            checkCudaErrors( cudaDeviceSynchronize() );
            t1 = ((double)getTickCount() - t1) / getTickFrequency();
            cout << "GPU "<< RGB <<"filter  time at =  " << t1 << " sec" << endl;
    
            
            checkCudaErrors(cudaMemcpy(h_output, d_output, sz, cudaMemcpyDeviceToHost));
            // uchar ---> Mat
            for (int y = 0; y < img_input.rows; y++)
                for (int x = 0; x < img_input.cols; x++) {
                    img_result_gpu.at<Vec3b>(y, x)[RGB] = h_output[y*img_input.cols + x];
                }
        }
    }
 
 
 
 
    Mat ret;
    hconcat(img_input, img_result_cpu, ret);        
    hconcat(ret, img_result_gpu, ret);
 
    namedWindow("result", WINDOW_AUTOSIZE);
    imshow("result", ret);
    waitKey(0);
 
    delete[] h_input;
    delete[] h_output;
    checkCudaErrors( cudaFree(d_input) );
    checkCudaErrors( cudaFree(d_output) );
    checkCudaErrors( cudaFree(d_filter) );
 
 
    //Save result to file
    imwrite("img_result.jpg", ret);
}
