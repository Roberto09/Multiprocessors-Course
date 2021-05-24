#include <stdio.h>
#include <math.h>
// Code can sum more than 1024 elements.

// CUDA kernel to add elements of two arrays
__global__ void add(int n, float* x, float* y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}

int main(void)
{
	//Variables
	int N = 100000000;
	float* h_x, * h_y;
	int size = sizeof(float) * N;
	printf("Number of elements in the array %d.\n",N);

	//Allocate Host Memory
	h_x = (float*)malloc(size);
	h_y = (float*)malloc(size);

	//Create Device Pointers
	float* d_x;
	float* d_y;

	//Allocate Device Memory
	cudaMalloc((void**)&d_x, size);
	cudaMalloc((void**)&d_y, size);

	//Initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		h_x[i] = 1.0;
		h_y[i] = 2.0;
	}

	//Memory copy Host to Device
	cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

	// Create Blocks
	int blockSize = 256;
	int numberBlocks = (N + blockSize - 1) / blockSize;

	//Launch kernel on N elements on the GPU
	add << <numberBlocks, blockSize >> > (N, d_x, d_y);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}

	//Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	//Memory copy Host to Device of the result
	cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

	//Print array
	//for (int i = 0; i < N; i++)
	//printf("%d: %f\n", i, h_y[i]);

	//Check for errors (all values should be 3.0)
	float maxError = 0.0;
	for (int i = 0; i < N; i++)
		maxError = (float)fmax(maxError, fabs(h_y[i] - 3.0));
	printf("Max error: %lf\n", maxError);

	//Free cuda memory
	cudaFree(d_x);
	cudaFree(d_y);

	//Free memory
	free(h_x);
	free(h_y);

	return 0;
}
