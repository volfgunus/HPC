#include <opencv2/opencv.hpp>
#include <time.h>
#include <iostream>
#define RADIUS 4

void bilateralFilter(const cv::Mat & input, cv::Mat & output, int r, double  sigma);
float Gaussian[64];
void updGaussian(int r, double sigma) {
	for (int i = 0; i < 2 * r + 1; i++) {
		float x = i - r;
		Gaussian[i] = expf(-(x*x) / (2 * sigma*sigma));
	}
}
void bilateralFilterCPU(const cv::Mat & input, cv::Mat & output, int r, double  sigma) {
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++) {
			double iFiltered = 0;
			double wP = 0;
			unsigned char centrePx = input.at<unsigned char>(i, j);
			for (int dy = -r; dy <= r; dy++) {
				int dyr = j + dy;
				if (dyr < 0)
					dyr = 0;
				if (dyr >= input.cols)
					dyr = input.cols - 1;
				for (int dx = -r; dx <= r; dx++) {
					int dxr = i + dy;
					if (dxr < 0)
						dxr = 0;
					if (dxr >= input.rows)
						dxr = input.rows - 1;
					unsigned char currPx = input.at<unsigned char>(dxr, dyr);
					double w = (Gaussian[dy + r] * Gaussian[dx + r]) * expf(-(powf(centrePx - currPx, 2)) / (2 * powf(sigma, 2)));
					iFiltered += w * currPx;
					wP += w;
				}
			}
			output.at<unsigned char>(i, j) = iFiltered / wP;
		}
}
int main(int argc, char **argv) {
	if (argc != 5) {
		std::cout << "Error! Usage "<< argv[0] <<" :input(png) outputGPU(png) outputCPU(png) sigma" << std::endl;
		return 1;
	}
	double sigma = atof(argv[4]);
	cv::Mat input = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
	std::vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	cv::Mat output_own(input.rows, input.cols, CV_8UC1);
	cv::Mat output_cv(input.rows, input.cols, CV_8UC1);
	updGaussian(RADIUS, sigma);
	bilateralFilter(input, output_own, RADIUS, sigma);
	clock_t start_s = clock();
	bilateralFilterCPU(input, output_cv, RADIUS, sigma);
	clock_t stop_s = clock();
	std::cout << "Time for the CPU: " << (stop_s - start_s)<< " ms" << std::endl;
	cv::imwrite(argv[2], output_own, compression_params);
	cv::imwrite(argv[3], output_cv, compression_params);
	return 0;
}