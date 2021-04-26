#include <stdio.h>
#include <stdlib.h>

#define IMW 407
#define IMH 887
#define IMAGE_BUFFER_SIZE (IMW*IMH*3)
#define BLOCKX 16
#define BLOCKY BLOCKX
#define BLUR_DEGREE 3

unsigned int width, height;

int hmask[3][3] = { 1, 2, 1,
2, 4, 2,
1, 2, 1
};


#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long prev){
  timeval tv1;
  gettimeofday(&tv1,0);
  return ((tv1.tv_sec * USECPSEC)+tv1.tv_usec) - prev;
}

int validate(unsigned char *d1, unsigned char *d2, int dsize){

  for (int i = 0; i < dsize; i++)
    if (d1[i] != d2[i]) {printf("validation mismatch at index %d, was %d, should be %d\n", i, d1[i], d2[i]); return 0;}
  return 1;
}

int h_getPixel(unsigned char *arr, int col, int row, int k)
{
    int sum = 0;
    int denom = 0;

    for (int j = -1; j <= 1; j++)
    {
        for (int i = -1; i <= 1; i++)
        {
            if ((row + j) >= 0 && (row + j) < height && (col + i) >= 0 && (col + i) < width)
            {
                int color = arr[(row + j) * 3 * width + (col + i) * 3 + k];
                sum += color * hmask[i + 1][j + 1];
                denom += hmask[i + 1][j + 1];
            }
        }
    }

    return sum / denom;
} // End getPixel

void h_blur(unsigned char *arr, unsigned char *result)
{
    for (unsigned int row = 0; row < height; row++)
    {
        for (unsigned int col = 0; col < width; col++)
        {
            for (int k = 0; k < 3; k++)
            {
                result[3 * row * width + 3 * col + k] = h_getPixel(arr, col, row, k);
            }
        }
    }
} // End h_blur

__global__ void d_blur(const unsigned char * __restrict__ arr, unsigned char *result, const int width, const int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int mask[3][3] = { 1, 2, 1,
        2, 4, 2,
        1, 2, 1
    };
    if ((row < height) && (col < width)){
      int sum = 0;
      int denom = 0;

      for (int k = 0; k < 3; k++)
      {
        for (int j = -1; j <= 1; j++)
        {
            for (int i = -1; i <= 1; i++)
            {
                if ((row + j) >= 0 && (row + j) < height && (col + i) >= 0 && (col + i) < width)
                {
                    int color = arr[(row + j) * 3 * width + (col + i) * 3 + k];
                    sum += color * mask[i + 1][j + 1];
                    denom += mask[i + 1][j + 1];
                }
            }
        }

        result[3 * row * width + 3 * col + k] = sum / denom;
        sum = 0;
        denom = 0;
      }
    }
}

int  main(int argc, char **argv)
{
/************ Setup work ***********************/
  unsigned char *d_resultPixels;
  unsigned char *h_resultPixels;
  unsigned char *h_devicePixels;

  unsigned char *h_pixels = NULL;
  unsigned char *d_pixels = NULL;

  int nBlurDegree;
  int imageSize = sizeof(unsigned char) * IMAGE_BUFFER_SIZE;

  h_pixels = (unsigned char *)malloc(imageSize);


  width  = IMW;
  height = IMH;


  h_resultPixels = (unsigned char *)malloc(imageSize);
  h_devicePixels = (unsigned char *)malloc(imageSize);

  for (int i = 0; i < imageSize; i++) h_pixels[i] = rand()%30;
  memcpy(h_devicePixels, h_pixels, imageSize);

/************************** Start host processing ************************/
  unsigned long long cputime = dtime_usec(0);
// Apply gaussian blur
  for (nBlurDegree = 0; nBlurDegree < BLUR_DEGREE; nBlurDegree++)
  {
    memset((void *)h_resultPixels, 0, imageSize);

    h_blur(h_pixels, h_resultPixels);

    memcpy((void *)h_pixels, (void *)h_resultPixels, imageSize);
  }
  cputime = dtime_usec(cputime);


/************************** End host processing **************************/

/************************** Start device processing **********************/


  cudaMalloc((void **)&d_pixels, imageSize);

  cudaMalloc((void **)&d_resultPixels, imageSize);

  cudaMemcpy(d_pixels, h_devicePixels, imageSize, cudaMemcpyHostToDevice);

  dim3 block(BLOCKX, BLOCKY);
  dim3 grid(IMW/block.x+1, IMH/block.y+1);

  unsigned long long gputime = dtime_usec(0);

  for (nBlurDegree = 0; nBlurDegree < BLUR_DEGREE; nBlurDegree++)
  {
    cudaMemset(d_resultPixels, 0, imageSize);

    d_blur << < grid, block >> >(d_pixels, d_resultPixels, width, height);

    cudaMemcpy(d_pixels, d_resultPixels, imageSize, cudaMemcpyDeviceToDevice);
  }
  cudaDeviceSynchronize();
  gputime = dtime_usec(gputime);
  cudaMemcpy(h_devicePixels, d_resultPixels, imageSize, cudaMemcpyDeviceToHost);

  printf("GPU time: %fs, CPU time: %fs\n", gputime/(float)USECPSEC, cputime/(float)USECPSEC);

  validate(h_pixels, h_devicePixels, imageSize);
  /************************** End device processing ************************/
  
  // Release resources
    cudaFree(d_pixels);
    cudaFree(d_resultPixels);
  
    free(h_devicePixels);
    free(h_pixels);
    free(h_resultPixels);
  
    return 0;
  } 