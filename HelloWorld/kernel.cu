
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

// The conventions for pointers is: if the value is stored in the device(GPU) it starts with d_, else if the value is stored
// in the host(CPU) the convention is to start with a h_

__global__ void kernel(float *d_out, float *d_in)
{
	
	// Create a function that returns pointers 
	int idx = threadIdx.x;
	float f = d_in[idx];
	// Output to a pointer from a float operation
	d_out[idx] = f * f * f;

}

int main()
{

	// Define the size of the array to utilize
	const int ARRAY_SIZE = 1000;
	// Calculate the size in bytes
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	
	// Create the input array on the host(CPU)
	float h_in[ARRAY_SIZE];
	for(int i = 0; i<ARRAY_SIZE; i++)
	{
		h_in[i] = float(i);
	}
	float h_out[ARRAY_SIZE];

	// Declare GPU memory pointers
	float *d_in;
	float *d_out;

	// Allocate GPU memory
	cudaMalloc((void **)&d_in, ARRAY_BYTES);
	cudaMalloc((void **)&d_out, ARRAY_BYTES);

	// Transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// Launch the kernel, each block(left input) defines how many times the threads(right input) will be executed
	kernel <<<1, ARRAY_SIZE>>>(d_out, d_in);

	// Copy back the array to the host
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// Print the resulting array
	for(int i = 0; i<ARRAY_SIZE; i++)
	{
	
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");

	}

	cudaFree(d_in);
	cudaFree(d_out);
	system("pause");
	return 0;

}
