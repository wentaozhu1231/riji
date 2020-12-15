// Timothy 2020_12_15
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "cuda_texture_types.h"
#include "device_launch_parameters.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

// cuda binding security check
void checkCUDAError(const char* msg) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// declaration of the usage of the texture memory.Notice the dimension is 2
// here.
texture<uchar, 2, cudaReadModeElementType> texRef;

// core function ,actually I would like to test to use function in opencv
// afterwards
__global__ void meanfilter_kernel(uchar* dstcuda, int width) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  dstcuda[y * width + x] =
      (tex2D(texRef, x - 1, y - 1) + tex2D(texRef, x, y - 1) +
       tex2D(texRef, x + 1, y - 1) + tex2D(texRef, x - 1, y) +
       tex2D(texRef, x, y) + tex2D(texRef, x + 1, y) +
       tex2D(texRef, x - 1, y + 1) + tex2D(texRef, x, y + 1) +
       tex2D(texRef, x + 1, y + 1)) /
      9;
}

int main() {
  cv::Mat srcImg = cv::imread("F:\\workDairy\\2020_12_15_data\\scarlet.jpg",
                              cv::IMREAD_GRAYSCALE);
  // opencv security check
  if (srcImg.empty()) {
    std::cout << "Could not open or find the image,check your input path first."
              << std::endl;
    std::cin.get();  // wait for any key press
    return -1;
  }

  // Following code is the mannual way to set the cudaChannelFormatDesc
  // channelDesc

  /*cudaChannelFormatDesc channelDesc =
     cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
  cudaArray* srcArray;
 cudaMallocArray(&srcArray, &channelDesc, srcImg.cols, srcImg.rows);
 */

  // this is the better way to get channelDesc,remmember, here is stil a
  // reference &texRef.channelDesc
  cudaArray* srcArray;
  cudaMallocArray(&srcArray, &texRef.channelDesc, srcImg.cols, srcImg.rows);
  cudaMemcpyToArray(srcArray, 0, 0, srcImg.data, srcImg.cols * srcImg.rows,
                    cudaMemcpyHostToDevice);
  cudaBindTextureToArray(&texRef, srcArray, &texRef.channelDesc);
  // security check
  checkCUDAError("bind");

  // prepare the space for data after processing
  cv::Mat dstImg = cv::Mat(cv::Size(srcImg.cols, srcImg.rows), CV_8UC1);
  uchar* dstcuda;
  cudaMalloc((void**)&dstcuda, srcImg.cols * srcImg.rows * sizeof(uchar));

  // run kernel function in cuda texture memory
  dim3 dimBlock(32, 32);
  dim3 dimGrid((srcImg.cols + dimBlock.x - 1) / dimBlock.x,
               (srcImg.rows + dimBlock.y - 1) / dimBlock.y);
  meanfilter_kernel<<<dimGrid, dimBlock>>>(dstcuda, srcImg.cols);

  // cudaThreadSynchronize() is a host function that waits for all previous
  // async operations (i.e. kernel calls, async memory copies) to complete.
  cudaThreadSynchronize();

  // after processing, get the data back to host memory
  cudaMemcpy(dstImg.data, dstcuda, srcImg.cols * srcImg.rows * sizeof(uchar),
             cudaMemcpyDeviceToHost);

  // free all the memory both in cpu and gpu
  cudaUnbindTexture(&texRef);
  cudaFreeArray(srcArray);
  cudaFree(dstcuda);

  cv::imshow("Source Image", srcImg);
  cv::imshow("Result Image", dstImg);
  cv::waitKey();
  return 0;
}