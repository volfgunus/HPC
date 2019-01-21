#include <time.h>
#include <stdio.h>
#include <iostream>

void matlul(int *c, const int *a, const int *b, int size);
void matluls(int *c, const int *a, const int *b, int size);
void matLULcpu(int *c, const int *a, const int *b, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			int sum = 0;
			for (int k = 0; k < size; k++){
				sum += a[j * size + k] * b[k * size + i];
			}
			c[j * size + i] = sum;
		}
	}
}

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cout << "Error! Usage " << argv[0] << " :size of matrix" << std::endl;
		return 1;
	}
	int size = atoi(argv[1]);
	int *a = new int[size*size], *b = new int[size*size], *c1 = new int[size*size], *c2 = new int[size*size], *c3 = new int[size*size];
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			a[i*size + j] = 1;
			b[i*size + j] = 1;
		}
	}
	matlul(c1,a, b,size);
	matluls(c2, a, b, size);
	clock_t start_s = clock();
	matLULcpu(c3, a, b, size);
	clock_t stop_s = clock();
	std::cout << "Time for the CPU: " << (stop_s - start_s) << " ms" << std::endl;
	return 0;
}