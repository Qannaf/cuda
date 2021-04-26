#include <iostream>
#include <time.h>
#include "opencv2/highgui.hpp" // actually under /usr/include
#include "opencv2/opencv.hpp"

#define TILE_W      1<<4
#define TILE_H      1<<4
#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);

using namespace cv;
using namespace std;


__global__ void rgb2grayincuda(uchar3 * const d_in, unsigned char * const d_out, 
                                uint imgheight, uint imgwidth)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < imgwidth && idy < imgheight)
    {
        uchar3 rgb = d_in[idy * imgwidth + idx];
        d_out[idy * imgwidth + idx] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
    }
}

void rgb2grayincpu(unsigned char * const d_in, unsigned char * const d_out,
                                uint imgheight, uint imgwidth)
{
    for(int i = 0; i < imgheight; i++)
    {
        for(int j = 0; j < imgwidth; j++)
        {
            d_out[i * imgwidth + j] = 0.299f * d_in[(i * imgwidth + j)*3]
                                     + 0.587f * d_in[(i * imgwidth + j)*3 + 1]
                                     + 0.114f * d_in[(i * imgwidth + j)*3 + 2];
        }
    }
}




int main()
{
    // CPU
    Mat img_input, grayImage;
    img_input = imread("lena.png");

    if (img_input.empty())
    {
        cout << "image load error!" << endl;
        exit(1);
    }
    imshow("img_input", img_input);
    waitKey(0);
    
    grayImage = Mat (img_input.cols, img_input.rows, CV_8UC1, Scalar(0));
        



    //GPU
    uchar3 *d_in;
    unsigned char *d_out;
    float sz3 =img_input.cols*img_input.rows*sizeof(uchar3);
    float sz = img_input.cols*img_input.rows*sizeof(unsigned char);
    
    cudaMalloc((void**)&d_in, sz3);
    cudaMalloc((void**)&d_out, sz);

    // CPU -->  GPU
    cudaMemcpy(d_in, img_input.data, sz3, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(TILE_W , TILE_H );
    dim3 blocksPerGrid(img_input.cols / TILE_W, img_input.rows / TILE_H);

    // 1) gray in GPU
    clock_t start, end;
    start = clock();
    rgb2grayincuda<< <blocksPerGrid, threadsPerBlock>> >(d_in, d_out, img_input.cols,img_input.rows);
    cudaDeviceSynchronize();
    end = clock();

    printf("cuda exec time is %.8f\n", (double)(end-start)/CLOCKS_PER_SEC);
    cudaMemcpy(grayImage.data, d_out,sz, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    // 2) gray in CPU
    start = clock();
    rgb2grayincpu(img_input.data, grayImage.data, img_input.cols, img_input.rows);
    end = clock();

    printf("cpu exec time is %.8f\n", (double)(end-start)/CLOCKS_PER_SEC);

    // 3) gray with openCv
    start = clock();
    cvtColor(img_input, grayImage, CV_BGR2GRAY);
    end = clock();

    printf("opencv-cpu exec time is %.8f\n", (double)(end-start)/CLOCKS_PER_SEC);

    imshow("grayImage", grayImage);
    waitKey(0);
    imwrite("grayImage.pgm", grayImage);
    return 0;

}
