
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 16

__constant__ float cGaussian[64];
texture<unsigned char, 2, cudaReadModeElementType> inTexture;

void updateGaussian(int r, double sigma){
	float fGaussian[64];
	for (int i = 0; i < 2 * r + 1; i++){
		float x = i - r;
		fGaussian[i] = expf(-(x*x) / (2 * sigma*sigma));
	}
	cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2 * r + 1));
}

__global__ void gpuCalculation(unsigned char* output, int width, int height, int r, double sigma){
	int txIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int tyIndex = threadIdx.y + blockIdx.y * blockDim.y;
	if ((txIndex < width) && (tyIndex < height)){
		double iFiltered = 0;
		double wP = 0;
		unsigned char centrePx = tex2D(inTexture, txIndex, tyIndex);
		for (int dy = -r; dy <= r; dy++) {
			for (int dx = -r; dx <= r; dx++) {
				unsigned char currPx = tex2D(inTexture, txIndex + dx, tyIndex + dy);
				double w = (cGaussian[dy + r] * cGaussian[dx + r]) * expf(-(powf(centrePx - currPx, 2)) / (2 * powf(sigma, 2)));
				iFiltered += w * currPx;
				wP += w;
			}
		}
		output[tyIndex*width + txIndex] = iFiltered / wP;
	}
}

void bilateralFilter(const cv::Mat & input, cv::Mat & output, int r, double sigma){
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int gray_size = input.step*input.rows;
	size_t pitch; 
	unsigned char *d_input = NULL;
	unsigned char *d_output;
	updateGaussian(r, sigma);
	cudaMallocPitch(&d_input, &pitch, sizeof(unsigned char)*input.step, input.rows); 
	cudaMemcpy2D(d_input, pitch, input.ptr(), sizeof(unsigned char)*input.step, sizeof(unsigned char)*input.step, input.rows, cudaMemcpyHostToDevice); 
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
	cudaBindTexture2D(0, inTexture, d_input,desc, input.step, input.rows, pitch);
	cudaMalloc<unsigned char>(&d_output, gray_size); 
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((input.cols + BLOCK_SIZE-1) / BLOCK_SIZE, (input.rows + BLOCK_SIZE-1) / BLOCK_SIZE);
	cudaEventRecord(start, 0); 
	gpuCalculation <<<grid, block >>> (d_output, input.cols, input.rows, r, sigma);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaMemcpy(output.ptr(), d_output, gray_size, cudaMemcpyDeviceToHost);
	cudaUnbindTexture(inTexture);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << "Time for the GPU: " << time << " ms" << std::endl;
}