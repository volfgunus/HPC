
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define BLOCK_SIZE 16

__global__ void gpuLUL(int *c, const int *a, const int *b, int size){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	if (col < size && row < size){
		for (int i = 0; i < size; i++){
			sum +=  a[row * size + i] * b[i * size + col];
		}
		c[row * size + col] = sum;
	}
}

__global__ void gpuLULs(int *C, const int *A, const int *B, int size) {
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int aBegin = size * BLOCK_SIZE * by;
	int aEnd = aBegin + size - 1;
	int aStep = BLOCK_SIZE;
	int bBegin = BLOCK_SIZE * bx;
	int bStep = BLOCK_SIZE * size;
	float Csub = 0;
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
		As[ty][tx] = A[a + size * ty + tx];
		Bs[ty][tx] = B[b + size * ty + tx];
		__syncthreads();
		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub += As[ty][k] * Bs[k][tx];
		__syncthreads();
	}
	int c = size * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + size * ty + tx] = Csub;
}

void matlul(int *c, const int *a, const int *b, int size) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time;
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;

	cudaMalloc((void**)&dev_a, size * size * sizeof(int));
	cudaMalloc((void**)&dev_b, size * size * sizeof(int));
	cudaMalloc((void**)&dev_c, size * size * sizeof(int));
	cudaMemcpy(dev_a, a, size * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size * size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	cudaEventRecord(start, 0);
	gpuLUL<<<grid, block >>>(dev_c, dev_a, dev_b, size);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaMemcpy(c, dev_c, size * size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << "Time for the GPU: " << time << " ms" << std::endl;
}


void matluls(int *c, const int *a, const int *b, int size) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time;
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;

	cudaMalloc((void**)&dev_a, size * size * sizeof(int));
	cudaMalloc((void**)&dev_b, size * size * sizeof(int));
	cudaMalloc((void**)&dev_c, size * size * sizeof(int));
	cudaMemcpy(dev_a, a, size * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size * size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 block1(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid1((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	cudaEventRecord(start, 0);
	gpuLULs << <grid1, block1 >> >(dev_c, dev_a, dev_b, size);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaMemcpy(c, dev_c, size * size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << "Time for the GPU shared: " << time << " ms" << std::endl;
}
