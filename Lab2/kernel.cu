
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 16
texture<unsigned char, 2, cudaReadModeElementType> inTexture;

__global__ void gpuCalculation(unsigned char* output, int width, int height) {
	int txIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int tyIndex = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned char res[9];
	unsigned char tmp;
	if ((txIndex < width) && (tyIndex < height)){
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				res[i * 3 + j] = tex2D(inTexture, txIndex + i-1, tyIndex + j-1);
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8 - i; j++) {
				if (res[j] > res[j + 1]) {
					tmp = res[j];
					res[j] = res[j + 1];
					res[j + 1] = tmp;
				}
			}
		}
		output[tyIndex*width + txIndex] = res[4];
	}
}

void pepper(const cv::Mat & input, cv::Mat & output) {
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int gray_size = input.step*input.rows;
	size_t pitch;
	unsigned char *d_input = NULL;
	unsigned char *d_output;
	cudaMallocPitch(&d_input, &pitch, sizeof(unsigned char)*input.step, input.rows);
	cudaMemcpy2D(d_input, pitch, input.ptr(), sizeof(unsigned char)*input.step, sizeof(unsigned char)*input.step, input.rows, cudaMemcpyHostToDevice);
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
	cudaBindTexture2D(0, inTexture, d_input, desc, input.step, input.rows, pitch);
	cudaMalloc<unsigned char>(&d_output, gray_size);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((input.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (input.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
	cudaEventRecord(start, 0);
	gpuCalculation <<<grid, block >>> (d_output, input.cols, input.rows);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaMemcpy(output.ptr(), d_output, gray_size, cudaMemcpyDeviceToHost);
	cudaUnbindTexture(inTexture);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << "Time for the GPU: " << time << " ms" << std::endl;
}