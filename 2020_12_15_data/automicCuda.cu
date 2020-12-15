#include <iostream>

#include "cuda_runtime.h"
#include "time.h"

using namespace std;

#define num (256 * 1024 * 1024)


__global__ void hist(unsigned char* inputdata, int* outPutHist, long size) {
 
  __shared__ int temp[256];
  temp[threadIdx.x] = 0;
  __syncthreads();

  // 计算线程索引及线程偏移量
  int ids = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  while (ids < size) {
    //采用原子操作对一个block中的数据进行直方图统计
    atomicAdd(&temp[inputdata[ids]], 1);
    ids += offset;
  }

  // 等待统计完成，减去统计结果
  __syncthreads();
  atomicSub(&outPutHist[threadIdx.x], temp[threadIdx.x]);
}

int main() {
  // 生成随机数据 [0 255]
  unsigned char* cpudata = new unsigned char[num];
  for (size_t i = 0; i < num; i++)
    cpudata[i] = static_cast<unsigned char>(rand() % 256);

  // 声明数组用于记录统计结果
  int cpuhist[256];
  memset(cpuhist, 0, 256 * sizeof(int));

  /*******************************   CPU测试代码
   * *********************************/
  clock_t cpu_start, cpu_stop;
  cpu_start = clock();
  for (size_t i = 0; i < num; i++) cpuhist[cpudata[i]]++;
  cpu_stop = clock();
  cout << "CPU time: " << (cpu_stop - cpu_start) << "ms" << endl;

  /*******************************   GPU测试代码
   * *********************************/

  //定义事件用于计时
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //开辟显存并将数据copy进显存
  unsigned char* gpudata;
  cudaMalloc((void**)&gpudata, num * sizeof(unsigned char));
  cudaMemcpy(gpudata, cpudata, num * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  // 开辟显存用于存储输出数据,并将CPU的计算结果copy进去
  int* gpuhist;
  cudaMalloc((void**)&gpuhist, 256 * sizeof(int));
  cudaMemcpy(gpuhist, cpuhist, 256 * sizeof(int), cudaMemcpyHostToDevice);

  // 执行核函数并计时
  cudaEventRecord(start, 0);
  hist<<<1024, 256>>>(gpudata, gpuhist, num);
  cudaEventRecord(stop, 0);

  // 将结果copy回主机
  int histcpu[256];
  cudaMemcpy(cpuhist, gpuhist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

  // 销毁开辟的内存
  cudaFree(gpudata);
  cudaFree(gpuhist);
  delete cpudata;

  // 计算GPU花费时间并销毁计时事件
  cudaEventSynchronize(stop);
  float gputime;
  cudaEventElapsedTime(&gputime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cout << "GPU time: " << gputime << "ms" << endl;

  // 验证结果
  long result = 0;
  for (size_t i = 0; i < 256; i++) result += cpuhist[i];
  if (result == 0)
    cout << "GPU has the same result with CPU." << endl;
  else
    cout << "Error: GPU has a different result with CPU." << endl;

  system("pause");
  return 0;
}