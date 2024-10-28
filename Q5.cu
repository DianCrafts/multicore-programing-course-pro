#include <device_functions.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>


#define TILE_WIDTH 16

void fillVector(float * v, size_t n);
void printVector(float * v, size_t n);


__global__ void matrixMulCUDA(float *c,  float *a, float *b , int n){
  __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;
  float Pvalue = 0;
  for (int p = 0; p < n/TILE_WIDTH; ++p) {

    ds_M[ty][tx] = a[Row * n + p * TILE_WIDTH + tx];
    ds_N[ty][tx] = b[(p * TILE_WIDTH + ty) * n + Col];
   __synchThreads();

    for (int i = 0; i < TILE_WIDTH; ++i) Pvalue += ds_M[ty][i] * ds_N[i][tx];
   __synchThreads();
  }	
  c[Row * n + Col] = Pvalue;
}


int main()
{
    const int vectorSize = 1024 * 16 * 4;
    float a[vectorSize], b[vectorSize], c[vectorSize];
    
    fillVector(a, vectorSize);
    fillVector(b, vectorSize);
 
 printVector(a ,vectorSize );
 float *dev_a = 0;
float *dev_b = 0;
float *dev_c = 0;
cudaError_t cudaStatus;



cudaStatus = cudaSetDevice(0);
if (cudaStatus != cudaSuccess) {
printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
}
 
 cudaStatus = cudaMalloc((void**)&dev_c, vectorSize * sizeof(float));
if (cudaStatus != cudaSuccess) {
printf("cudaMalloc failed!");
}
 
 cudaStatus = cudaMalloc((void**)&dev_a, vectorSize * sizeof(float));
if (cudaStatus != cudaSuccess) {
printf("cudaMalloc failed!");
}
cudaStatus = cudaMalloc((void**)&dev_b, vectorSize * sizeof(float));
if (cudaStatus != cudaSuccess) {
printf("cudaMalloc failed!");
}

cudaStatus = cudaMemcpy(dev_a, a, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {
printf("cudaMemcpy failed!");
}
cudaStatus = cudaMemcpy(dev_b, b, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {
printf("cudaMemcpy failed!");
}
 dim3 DimGrid( 1, 1, 1);
dim3 DimBlock(16,16, 1);
matrixMulCUDA <<<DimGrid, DimBlock>>>(dev_c, dev_a, dev_b , 256);
cudaStatus = cudaGetLastError();
if (cudaStatus != cudaSuccess) {
printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
}
cudaStatus = cudaDeviceSynchronize();
if (cudaStatus != cudaSuccess) {
printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n",
cudaStatus);
}
cudaStatus = cudaMemcpy(c, dev_c, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {
printf("cudaMemcpy failed!");
}
 
cudaFree(dev_c);
cudaFree(dev_a);
cudaFree(dev_b);
printVector(c, vectorSize);
return cudaStatus;

}


// Fills a vector with data
void fillVector(float * v, size_t n) {
    int i;
  //((float)rand()/RAND_MAX)*10;      
    for (i = 0; i < n; i++) {
        v[i] =  1.0f;                  
    }
}


// Prints a vector to the stdout.
void printVector(float * v, size_t n) {
    int i;
    printf("[-] Vector elements: ");
    for (i = 0; i < n; i++) {
        printf("%f, ", v[i]);
    }
    printf("\b\b  \n");
}
