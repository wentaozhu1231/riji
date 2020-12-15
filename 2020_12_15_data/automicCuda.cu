#include <iostream>

#include "cuda_runtime.h"
#include "time.h"

using namespace std;

#define num (256 * 1024 * 1024)


__global__ void hist(unsigned char* inputdata, int* outPutHist, long size) {
 
  __shared__ int temp[256];
  temp[threadIdx.x] = 0;
  __syncthreads();

  // �����߳��������߳�ƫ����
  int ids = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  while (ids < size) {
    //����ԭ�Ӳ�����һ��block�е����ݽ���ֱ��ͼͳ��
    atomicAdd(&temp[inputdata[ids]], 1);
    ids += offset;
  }

  // �ȴ�ͳ����ɣ���ȥͳ�ƽ��
  __syncthreads();
  atomicSub(&outPutHist[threadIdx.x], temp[threadIdx.x]);
}

int main() {
  // ����������� [0 255]
  unsigned char* cpudata = new unsigned char[num];
  for (size_t i = 0; i < num; i++)
    cpudata[i] = static_cast<unsigned char>(rand() % 256);

  // �����������ڼ�¼ͳ�ƽ��
  int cpuhist[256];
  memset(cpuhist, 0, 256 * sizeof(int));

  /*******************************   CPU���Դ���
   * *********************************/
  clock_t cpu_start, cpu_stop;
  cpu_start = clock();
  for (size_t i = 0; i < num; i++) cpuhist[cpudata[i]]++;
  cpu_stop = clock();
  cout << "CPU time: " << (cpu_stop - cpu_start) << "ms" << endl;

  /*******************************   GPU���Դ���
   * *********************************/

  //�����¼����ڼ�ʱ
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //�����Դ沢������copy���Դ�
  unsigned char* gpudata;
  cudaMalloc((void**)&gpudata, num * sizeof(unsigned char));
  cudaMemcpy(gpudata, cpudata, num * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  // �����Դ����ڴ洢�������,����CPU�ļ�����copy��ȥ
  int* gpuhist;
  cudaMalloc((void**)&gpuhist, 256 * sizeof(int));
  cudaMemcpy(gpuhist, cpuhist, 256 * sizeof(int), cudaMemcpyHostToDevice);

  // ִ�к˺�������ʱ
  cudaEventRecord(start, 0);
  hist<<<1024, 256>>>(gpudata, gpuhist, num);
  cudaEventRecord(stop, 0);

  // �����copy������
  int histcpu[256];
  cudaMemcpy(cpuhist, gpuhist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

  // ���ٿ��ٵ��ڴ�
  cudaFree(gpudata);
  cudaFree(gpuhist);
  delete cpudata;

  // ����GPU����ʱ�䲢���ټ�ʱ�¼�
  cudaEventSynchronize(stop);
  float gputime;
  cudaEventElapsedTime(&gputime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cout << "GPU time: " << gputime << "ms" << endl;

  // ��֤���
  long result = 0;
  for (size_t i = 0; i < 256; i++) result += cpuhist[i];
  if (result == 0)
    cout << "GPU has the same result with CPU." << endl;
  else
    cout << "Error: GPU has a different result with CPU." << endl;

  system("pause");
  return 0;
}