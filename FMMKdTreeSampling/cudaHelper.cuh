#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <assert.h>

inline void getGridConfiguration(dim3& grid, dim3& block, unsigned int width, unsigned int height)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	unsigned int maxThreadsPerDim = 16;//std::sqrt(prop.maxThreadsPerBlock);
	//assert((maxThreadsPerDim*maxThreadsPerDim) == prop.maxThreadsPerBlock, "CUDA max. threads per block is not a square number.");

	block.x = std::min(width, maxThreadsPerDim);
	block.y = std::min(height, maxThreadsPerDim);
	grid.x = (width + block.x - 1) / block.x; //Integer ceil division
	grid.y = (height + block.y - 1) / block.y; //Integer ceil divison
	assert((grid.x * block.x) >= width && (grid.y * block.y >= height));
}