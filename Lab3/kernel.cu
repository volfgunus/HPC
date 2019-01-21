
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 16

texture<unsigned char, 2, cudaReadModeElementType> inTexture;


__global__ void gpuCalculation(unsigned char* output, int width, int height) {
	int txIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int tyIndex = threadIdx.y + blockIdx.y * blockDim.y;
	if ((txIndex < width) && (tyIndex < height)) {
		float txnorm = txIndex / height +0.5f;
		float tynorm = tyIndex / width +0.5f;
		output[tyIndex*width + txIndex] = tex2D(inTexture, txnorm, tynorm);
	}
}

void resize(const cv::Mat & input, cv::Mat & output) {
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int gray_size = output.step*output.rows;
	size_t pitch;
	unsigned char *d_input = NULL;
	unsigned char *d_output;
	cudaMallocPitch(&d_input, &pitch, sizeof(unsigned char)*input.step, input.rows);
	cudaMemcpy2D(d_input, pitch, input.ptr(), sizeof(unsigned char)*input.step, sizeof(unsigned char)*input.step, input.rows, cudaMemcpyHostToDevice);
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
	inTexture.addressMode[0] = cudaAddressModeClamp;
	inTexture.addressMode[1] = cudaAddressModeClamp;
	inTexture.filterMode = cudaFilterModeLinear;
	inTexture.normalized = true;
	cudaBindTexture2D(0, inTexture, d_input, desc, input.step, input.rows, pitch);
	cudaMalloc<unsigned char>(&d_output, gray_size);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((output.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (output.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
	cudaEventRecord(start, 0);
	gpuCalculation << <grid, block >> > (d_output, output.cols, output.rows);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaMemcpy(output.ptr(), d_output, gray_size, cudaMemcpyDeviceToHost);
	cudaUnbindTexture(inTexture);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << "Time for the GPU: " << time << " ms" << std::endl;
}