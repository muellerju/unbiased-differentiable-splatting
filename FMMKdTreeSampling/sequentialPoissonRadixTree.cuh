#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <limits>

#include "sampleRadixTree.cuh"
#include "poissonRadixTree.cuh"
#include "logging.h"


template<typename real>
__device__ real insertSample(
	int index, real weight,
	int* indices, real* weights,
	unsigned int numSamples
)
{
	real max_xi = -1.0;
	for (unsigned int i = 0; i < numSamples; i++)
	{
		if (weight < weights[i])
		{
			unsigned int tmpIdx = indices[i];
			indices[i] = index;
			index = tmpIdx;

			real tmpWeight = weights[i];
			weights[i] = weight;
			weight = tmpWeight;
		}

		max_xi = max(max_xi, weights[i]);
		if (index == -1)
			break;
	}
	return max_xi;
}

template<typename real, unsigned int Dims, unsigned int HermiteDegree>
__global__
void sequentialPoissonKernel(
	int indices[],
	real poissonWeights[],
	unsigned int width, unsigned int height,
	unsigned int numSamples,
	real radius, real precision,
	real h,
	const RadixTreeDevPtrs<real, Dims>* fmmTree,
	NodeState<real> map[], unsigned int mapSize,
	curandState* states
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int pixelOffset = j + height * i;

	if ((i < width) && (j < height))
	{
		// Define query point
		real query[] = { static_cast<real>(i), static_cast<real>(j) };

		// Define starting node 
		int nodeIdx = 0;
		Node* node = const_cast<Node*>(&(fmmTree->nodes[nodeIdx]));

		// Offset the base address to avoid overlap between maps for different pixels
		unsigned int mapOffset = pixelOffset * mapSize;
		NodeState<real>* localMap = &map[mapOffset];

		// Compute the likelihood to obtain an empty sample
		real total = max(real(0), getWeight<real, Dims, HermiteDegree>(fmmTree, nodeIdx, query, radius, precision, h));
		if (total < precision) // Capacity of the root
			return;

		// Store the path to the leaf (includes the root)
		NodeTrace<real> trace[STACKSIZE];
		unsigned int traceIdx = 1;
		trace[0].index = 0;
		trace[0].probability = 1.0f;

		real weights[2], stateWeights[2];
		while (fmmTree->sourceCounts[nodeIdx] > numSamples)
		{
			Node* left = const_cast<Node*>(&(fmmTree->nodes[node->left]));
			Node* right = const_cast<Node*>(&(fmmTree->nodes[node->right]));

			// Approximate the weight of the subtress
			weights[0] = max(real(0), getWeight<real, Dims, HermiteDegree>(fmmTree, node->left, query, radius, precision, h));
			weights[1] = max(real(0), getWeight<real, Dims, HermiteDegree>(fmmTree, node->right, query, radius, precision, h));

			// Normalize the weights to use the ratio when up dating the remaining capacity
			normalize<real, 2>(weights);

			// Compute state weights based on the childs remaining capacity
			stateWeights[0] = weights[0] * getCapacity(node->left, localMap, mapSize);
			stateWeights[1] = weights[1] * getCapacity(node->right, localMap, mapSize);

			// Traverse into the subtree with the larger weight
			if (stateWeights[0] < stateWeights[1])
			{
				// Right subtree
				trace[traceIdx].index = node->right;
				trace[traceIdx].probability = weights[1];

				nodeIdx = node->right;
				node = right;
			}
			else
			{
				// Left subtree
				trace[traceIdx].index = node->left;
				trace[traceIdx].probability = weights[0];

				nodeIdx = node->left;
				node = left;
			}
			traceIdx++;
			assert(traceIdx < STACKSIZE);
		}

		// Unroll stack and update capacities
		real probability = 1.0f;
		for (unsigned int m = traceIdx; m > 0; m--)
		{
			real capacity = max(0.0f, getCapacity(trace[m - 1].index, localMap, mapSize) - probability);
			setCapacity(trace[m - 1].index, capacity, localMap, mapSize);
			probability *= trace[m - 1].probability;
		}

		// Get curand state
		curandState state = states[pixelOffset];

		unsigned int sampleOffset = numSamples * (j + height * i);
		int* localIndices = &indices[sampleOffset];
		real* localWeights = &poissonWeights[sampleOffset];

		// Perform bernoulli trail for the candidates within the subtree
		unsigned int leafCount = fmmTree->sourceCounts[nodeIdx];
		unsigned int subtreeParent = fmmTree->nodes[nodeIdx].parent;
		unsigned int leafIdx = nodeIdx;
		for (unsigned int k = 0; k < leafCount; k++)
		{
			leafIdx = nextLeaf<real, Dims>(fmmTree->nodes, subtreeParent, leafIdx);
			const Node& leaf = fmmTree->nodes[leafIdx];
			assert(leaf.index != -1);

			real prob = max(real(0), getWeight<real, Dims, HermiteDegree>(fmmTree, leafIdx, query, radius, precision, h));
			real x = curand_uniform(&state);
			real transformedRandomNumber = x / prob;

			insertSample(leaf.index, transformedRandomNumber, localIndices, localWeights, numSamples);
		}

		// Store update curand state
		states[pixelOffset] = state;
	}
}

template<typename real, unsigned int Dims, unsigned int HermiteDegree>
void sequentialPoissonRadixTree(
	int* devIndicesPtr,
	unsigned int width, unsigned int height,
	unsigned int numSamples,
	real precision,
	const RadixTree<real, Dims>& radixTree,
	real h
)
{
	// Compute query radius
	real normalization = (real)(2.0 * M_PI * sqrt(h));
	real radius = std::sqrt(2.0f) * h * std::sqrt(-std::log(precision) - std::log(normalization));
	logging::log<logLevel_t::LOG_DEBUG>("sequentialPoissonRadixTree", "Query radius", radius);

	// Copy device tree pointers to gpu memory
	thrust::device_vector<RadixTreeDevPtrs<real, Dims>> treePtrs(1);
	treePtrs[0] = radixTree.getDevPtrsStruct();

	// Compute thread and grid layout
	dim3 blocks, threads;
	getGridConfiguration(blocks, threads, width, height);
	logging::log<logLevel_t::LOG_DEBUG>("sequentialPoissonRadixTree", threads, blocks);

	// Initialize curand generator
	unsigned long long seed = unsigned(time(NULL));
	thrust::device_vector<curandState> curandStates(width * height);
	curandInitKernel<curandState> << <blocks, threads >> > (
		thrust::raw_pointer_cast(curandStates.data()),
		seed, width, height
		);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		logging::log<logLevel_t::LOG_CRITICAL>("sequentialPoissonRadixTree", cudaGetErrorString(cudaStatus));
		throw std::runtime_error("curandInitKernel failed");
	}

	// Allocate and initialize the node states used to avoid duplicates
	unsigned int mapSize = numSamples * radixTree.depth; // Map size per pixel: Tree depth times number of samples
	thrust::device_vector<NodeState<real>> nodeStates(width * height * mapSize);

	// Initialize the indices for non-allocated indices
	thrust::device_vector<real> poissonWeights(width * height * numSamples);
	thrust::fill(thrust::device, poissonWeights.begin(), poissonWeights.end(), std::numeric_limits<float>::infinity());
	thrust::fill(thrust::device, devIndicesPtr, devIndicesPtr + numSamples * width * height, -1);
	

	// Launch sampling kernel
	for (unsigned int i = 0; i < std::sqrt(numSamples); i++)
	{
		sequentialPoissonKernel<real, Dims, HermiteDegree> << <blocks, threads >> > (
			devIndicesPtr,
			thrust::raw_pointer_cast(poissonWeights.data()),
			width, height,
			numSamples,
			radius, precision,
			h,
			thrust::raw_pointer_cast(treePtrs.data()),
			thrust::raw_pointer_cast(nodeStates.data()),
			mapSize,
			thrust::raw_pointer_cast(curandStates.data())
			);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			logging::log<logLevel_t::LOG_CRITICAL>("sequentialPoissonRadixTree", cudaGetErrorString(cudaStatus));
			throw std::runtime_error("sequentialPoissonKernel failed");
		}
	}

	// Sychronice device after kernel launch if in debug mode
#ifndef NDEBUG
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		logging::log<logLevel_t::LOG_CRITICAL>("sequentialPoissonRadixTree", "cudaDeviceSynchronize returned error code", cudaStatus);
		throw std::runtime_error("cudaDeviceSynchronize failed");
	}
#endif // NDEBUG
}
