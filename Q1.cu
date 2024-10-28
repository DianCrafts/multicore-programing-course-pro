/*
*   In His Exalted Name
*   Vector Addition - Sequential Code
*   Ahmad Siavashi, Email: siavashi@aut.ac.ir
*   21/05/2018
*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void fillVector(float * v, size_t n);
void printVector(float * v, size_t n);


__global__ void matrixMulCUDA(float *c,  float *a, float *b , int n){
    int k ;
 int row = threadIdx.y;
 int col = threadIdx.x;
 float sum = 0.0f ;
 for (k = 0 ; k < n ; ++k){
     sum += a[row* n +k] * b[k* n + col];
 }
 c[row* n + col] = sum;
}




int main()
{
    const int vectorSize = 1024;
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
// dim3 DimGrid( 1, 1, 1);
dim3 DimBlock(32, 32, 1);
matrixMulCUDA <<<1, DimBlock>>>(dev_c, dev_a, dev_b , 32);
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
        v[i] =  0.01f;                                //((float)rand()/RAND_MAX)*10;  
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
