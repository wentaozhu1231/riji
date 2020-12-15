#include <stdlib.h>

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

#define n 10
texture<float, 1, cudaReadModeElementType> tex_a;
texture<float, 1, cudaReadModeElementType> tex_b;

__global__ void test_texture(float *dev_test_a, float *dev_test_b,
                             float *dev_result_c) {
  int offset = threadIdx.x + blockIdx.x * blockDim.x;
  float a = tex1Dfetch(tex_a, offset);
  float b = tex1Dfetch(tex_b, offset);
  dev_result_c[offset] = a + b;
}

int main() {
  float *test_a = NULL, *test_b = NULL, *result_c = NULL;

  float *dev_test_a = NULL, *dev_test_b = NULL, *dev_result_c = NULL;

  cudaHostAlloc((void **)&test_a, n * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc((void **)&test_b, n * sizeof(float), cudaHostAllocMapped);
  cudaHostAlloc((void **)&result_c, n * sizeof(float), cudaHostAllocMapped);

  for (int i = 0; i < n; i++) {
    test_a[i] = i;
    test_b[i] = i * i;
  }
  cudaMalloc((void **)&dev_test_a, n * sizeof(float));
  cudaMalloc((void **)&dev_test_b, n * sizeof(float));
  cudaMalloc((void **)&dev_result_c, n * sizeof(float));

  cudaError_t cudaStatus;

  cudaStatus =
      cudaMemcpy(dev_test_a, test_a, n * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    cout << "cudaMemcpy dev_test_a failed!" << endl;
    exit(EXIT_FAILURE);
  }

  cudaStatus =
      cudaMemcpy(dev_test_b, test_b, n * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    cout << "cudaMemcpy dev_test_b failed!" << endl;
    exit(EXIT_FAILURE);
  }

  cudaBindTexture(0, &tex_a, dev_test_a, &tex_a.channelDesc, n * sizeof(float));
  checkCUDAError("binding  dex_test_a");
  cudaBindTexture(0, &tex_b, dev_test_b, &tex_b.channelDesc, n * sizeof(float));
  checkCUDAError("binding dex_test_b");

  test_texture<<<5, 10>>>(dev_test_a, dev_test_b, dev_result_c);

  cudaStatus = cudaMemcpy(result_c, dev_result_c, n * sizeof(float),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    cout << "cudaMemcpy result_c failed!" << endl;
    exit(EXIT_FAILURE);
  }

  cout << "Texture test result:" << endl;

  for (int i = 0; i < n; i++) {
    cout << test_a[i] << " " << test_b[i] << " " << result_c[i] << endl;
  }

  cudaUnbindTexture(tex_a);
  cudaUnbindTexture(tex_b);

  cudaFreeHost(test_a);
  cudaFreeHost(test_b);
  cudaFreeHost(result_c);
  cudaFree(dev_test_a);
  cudaFree(dev_test_b);
  cudaFree(dev_result_c);

  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!/n");
    return EXIT_FAILURE;
  }

  return 0;
}