#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <stdexcept>

#include "transformRadixTree.cuh"


using StackPtr = int*;
//extern __constant__ char radixTreePtrsBuffer[sizeof(RadixTreeDevPtrs<float, 2>)];

template<typename real>
struct alignas(16) NodeState
{
	__host__ __device__
		NodeState() : index(0), used(0), capacity((real)1.0) {}
	unsigned int index, used;
	real capacity;
};

template<typename real>
struct alignas(8) NodeTrace
{
	unsigned int index;
	real probability;
};

__device__ 
inline unsigned int hash(unsigned int key, unsigned int mapSize)
{
	// Robert Jenkins' 32 bit Mix Function
	key += (key << 12);
	key ^= (key >> 22);
	key += (key << 4);
	key ^= (key >> 9);
	key += (key << 10);
	key ^= (key >> 2);
	key += (key << 7);
	key ^= (key >> 12);

	// Knuth's Multiplicative Method
	key = (key >> 3) * 2654435761;

	return key % mapSize;
}

template<typename real>
__device__ real getCapacity(unsigned int index, NodeState<real>* map, unsigned int mapSize)
{
	unsigned int mapIdx = hash(index, mapSize);

	// Linear probing to find current element or empty
	for (unsigned int i = 0; i < mapSize; i++)
	{
		if (map[mapIdx].index == index && map[mapIdx].used == 1)
		{
			return map[mapIdx].capacity;
		}
		else if (map[mapIdx].used == 0) // We don't remove elements from the list so this is okay
		{
			return 1.0f;
		}
		mapIdx = (mapIdx + 1) % mapSize;
	}
	return 1.0f;
}

template<typename real>
__device__ void setCapacity(unsigned int index, real capacity, NodeState<real>* map, unsigned int mapSize)
{
	unsigned int mapIdx = hash(index, mapSize);

	// Linear probing to find current element or empty spot
	for (unsigned int i = 0; i < mapSize; i++) //TODO: Vielleicht hier <=
	{
		if ((map[mapIdx].index == index) || (map[mapIdx].used == 0))
		{
			break;
		}
		mapIdx = (mapIdx + 1) % mapSize;
	}

	// Set/update capacity in the map
	map[mapIdx].index = index;
	map[mapIdx].capacity = capacity;
	map[mapIdx].used = 1;
}

template<typename real, unsigned int Size>
__device__ real normalize(real x[])
{
	real sum = 0.0;
	for (unsigned int i = 0; i < Size; i++)
		sum += x[i];
	for (unsigned int i = 0; i < Size; i++)
		x[i] = x[i] / sum;
	return sum;
}

template<typename real>
__device__ void computeCDF(real cdf[], const real x[], unsigned int size, real start, real end)
{
	cdf[0] = 0.0f;
	for (unsigned int i = 0; i < size; i++)
	{
		cdf[i + 1] = cdf[i] + x[i];
	}

	assert(cdf[2] > 0.9f);

	for (unsigned int i = 0; i <= size; i++)
	{
		cdf[i] = (end - start) * cdf[i] + start;
	}
}

template<typename real, unsigned int Dims, unsigned int HermiteDegree>
__global__
	void samplingKernel(
		int indices[],
		unsigned int width, unsigned int height,
		unsigned int numSamples, unsigned int sampleIndex,
		real radius, real precision,
		real h,
		const RadixTreeDevPtrs<real, Dims>* fmmTree,
		const real priors[],
		NodeState<real> map[], unsigned int mapSize
	)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int pixelOffset = j + height * i;//i * height + j;

	if ((i < width) && (j < height))
	{
		// Define query point
		real query[] = { static_cast<real>(i), static_cast<real>(j) };

		real start = 0.0f;
		real end = 1.0f;
		int nodeIdx = 0;
		Node* node = const_cast<Node*>(&(fmmTree->nodes[nodeIdx]));

		// Offset the base address to avoid overlap between maps for different pixels
		unsigned int mapOffset = pixelOffset * mapSize;
		NodeState<real>* localMap = &map[mapOffset];

		// If all points with a weight larger than the given precision are sampled 
		// return a default indicator
		unsigned int sampleOffset = sampleIndex + numSamples * (j + height * i); //i * height + j + k * width * height;
		indices[sampleOffset] = -1;
		if (getCapacity<real>(0, localMap, mapSize) < precision) // Capacity of the root
			return;

		// Store the path to the leaf (includes the root)
		NodeTrace<real> trace[STACKSIZE];
		unsigned int traceIdx = 1;
		trace[0].index = 0;
		trace[0].probability = 1.0f;

		real weights[2], stateWeights[2], cdf[3];
		real prior = priors[pixelOffset];
		assert((0.0f < prior) && (prior <= 1.0f));
		while (node->index == -1)
		{
			Node* left = const_cast<Node*>(&(fmmTree->nodes[node->left]));
			Node* right = const_cast<Node*>(&(fmmTree->nodes[node->right]));

			// Approximate the weight of the subtress
			weights[0] = max((real)0.0, getWeight<real, Dims, HermiteDegree>(fmmTree, node->left, query, radius, precision, h));
			weights[1] = max((real)0.0, getWeight<real, Dims, HermiteDegree>(fmmTree, node->right, query, radius, precision, h));
			// Verify if the approximate weights are relevant for the required accuarcy
			// and avoid duplicate sampling due to numerical instablity
			real totalWeight = 0.0f;
			for (unsigned int m = 0; m < 2; m++)
			{
				totalWeight += weights[m];
			}

			if (totalWeight < precision)
			{
				// Assume all point have equal probability
				weights[0] = static_cast<real>(fmmTree->sourceCounts[node->left]);
				weights[1] = static_cast<real>(fmmTree->sourceCounts[node->right]);
				totalWeight = weights[0] + weights[1];
			}
			// Normalize the weights
			for (unsigned int m = 0; m < 2; m++)
				weights[m] /= totalWeight;

			// Compute state weights based on the childs remaining capacity
			stateWeights[0] = weights[0] * getCapacity(node->left, localMap, mapSize);
			stateWeights[1] = weights[1] * getCapacity(node->right, localMap, mapSize);
			if (normalize<real, 2>(stateWeights) < precision)
			{
				stateWeights[0] = weights[0];
				stateWeights[1] = weights[1];
			}

			// Compute CDF of the weight function
			computeCDF(cdf, stateWeights, 2, start, end);

			/*if (cdf[0] < prior && prior <= cdf[1])
			{
				trace[traceIdx].index = node->left;
				trace[traceIdx].probability = weights[0];

				node = left;
				start = cdf[0];
				end = cdf[1];
			}
			else if (cdf[1] < prior && prior <= cdf[2])
			{
				trace[traceIdx].index = node->right;
				trace[traceIdx].probability = weights[1];

				node = right;
				start = cdf[1];
				end = cdf[2];
			}
			else
			{
				// Error due numerical stability
				assert(cdf[0] < prior);
				assert(prior <= cdf[2]);
			}*/
			if (prior <= cdf[1])
			{
				trace[traceIdx].index = node->left;
				trace[traceIdx].probability = weights[0];

				node = left;
				start = cdf[0];
				end = cdf[1];
			}
			else
			{
				trace[traceIdx].index = node->right;
				trace[traceIdx].probability = weights[1];

				node = right;
				start = cdf[1];
				end = cdf[2];
			}

			traceIdx++;
			assert(traceIdx < STACKSIZE);
		}
		indices[sampleOffset] = node->index;

		// Unroll stack and update capacities
		real probability = 1.0f;
		for (unsigned int m = traceIdx; m > 0; m--)
		{
			real capacity = max(0.0f, getCapacity(trace[m - 1].index, localMap, mapSize) - probability);
			setCapacity(trace[m - 1].index, capacity, localMap, mapSize);
			probability *= trace[m - 1].probability;
		}
	}

}

template<typename real, unsigned int Dims, unsigned int HermiteDegree>
void samplingRadixTree(int* devIndicesPtr, unsigned int width, unsigned int height, unsigned int numSamples, real precision, const RadixTree<real, Dims>& radixTree, real h)
{
	// Compute query radius
	real normalization = (real)(2.0 * M_PI * sqrt(h));
	real radius = std::sqrt(2.0f) * h * std::sqrt(-std::log(precision) - std::log(normalization));
	std::cout << "> samplingRadixTree: Query radius " << radius << std::endl;

	// Copy device tree pointers to gpu memory
	thrust::device_vector<RadixTreeDevPtrs<real, Dims>> treePtrs(1);
	treePtrs[0] = radixTree.getDevPtrsStruct();

	// Initialize curand generator used to generate a uniform prior and set its seed
	curandGenerator_t gen;
	if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
	{
		std::cerr << "> samplingRadixTree: curandCreateGenerator failed!" << std::endl;
		throw std::runtime_error("curandCreateGenerator failed");
	}
	if (curandSetPseudoRandomGeneratorSeed(gen, time(NULL)) != CURAND_STATUS_SUCCESS)
	{
		std::cerr << "> samplingRadixTree: curandSetPseudoRandomGeneratorSeed failed!" << std::endl;
		throw std::runtime_error("curandSetPseudoRandomGeneratorSeed failed");
	}

	// Allocate and initialize the node states used to avoid duplicates
	unsigned int mapSize = numSamples * radixTree.depth; // Map size per pixel: Tree depth times number of samples
	thrust::device_vector<NodeState<real>> nodeStates(width * height * mapSize);

	// Compute thread and grid layout
	dim3 blocks, threads;
	getGridConfiguration(blocks, threads, width, height);
	std::cout << "> samplingRadixTree: Block size " << blocks.x << "," << blocks.y << " with " << threads.x << "," << threads.y << " threads per block" << std::endl;

	//
	thrust::device_vector<real> priors(width * height);
	for (unsigned int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
	{
		// Generate a uniformly distributed prior for each pixel
		if (curandGenerateUniform(gen, thrust::raw_pointer_cast(priors.data()), width * height) != CURAND_STATUS_SUCCESS)
		{
			std::cerr << "> samplingRadixTree: curandGenerateUniform failed!" << std::endl;
			throw std::runtime_error("curandGenerateUniform failed");
		}

		// Launch sampling kernel
		samplingKernel<real, Dims, HermiteDegree> << <blocks, threads >> > (
			devIndicesPtr,
			width, height,
			numSamples, sampleIdx,
			radius, precision,
			h,
			thrust::raw_pointer_cast(treePtrs.data()),
			thrust::raw_pointer_cast(priors.data()),
			thrust::raw_pointer_cast(nodeStates.data()),
			mapSize
			);
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			std::cerr << "> samplingRadixTree: sample kernel launch failed:  " << cudaGetErrorString(cudaStatus) << std::endl;
			throw std::runtime_error("samplingKernel failed");
		}
	}
	
	// Sychronice device after kernel launch if in debug mode
#ifndef NDEBUG
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cerr << "> samplingRadixTree: cudaDeviceSynchronize returned error code" << cudaStatus << " after launching sampling kernel!\n";
		throw std::runtime_error("cudaDeviceSynchronize failed");
	}
#endif // NDEBUG
}
