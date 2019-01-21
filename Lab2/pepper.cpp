#include <opencv2/opencv.hpp>
#include <time.h>
#include <iostream>

void pepper(const cv::Mat & input, cv::Mat & output);

int main(int argc, char **argv) {
	if (argc != 4) {
		std::cout << "Error! Usage " << argv[0] << " :input(png) outputGPU(png) outputCPU(png)" << std::endl;
		return 1;
	}
	cv::Mat input = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
	std::vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	cv::Mat output_own(input.rows, input.cols, CV_8UC1);
	cv::Mat output_cv;
	pepper(input, output_own);
	clock_t start_s = clock();
	cv::medianBlur(input, output_cv, 3);
	clock_t stop_s = clock();
	std::cout << "Time for the CPU: " << (stop_s - start_s) << " ms" << std::endl;
	cv::imwrite(argv[2], output_own, compression_params);
	cv::imwrite(argv[3], output_cv, compression_params);
	return 0;
}