#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <limits>

#include "sampleRadixTree.cuh"
#include "poissonRadixTree.cuh"
#include "sequentialPoissonRadixTree.cuh"
#include "evaluateDynamicRadixTree.cuh"
#include "logging.h"

template<typename real, unsigned int Dims>
__device__ unsigned int nextLeaf(
	const DynamicNode<real>* nodes,
	unsigned int subtreeParent,
	unsigned int leafIdx
)
{
	// If current node is a leaf find the next right node
	if (nodes[leafIdx].index > -1)
	{
		while (nodes[leafIdx].parent != subtreeParent)
		{
			const DynamicNode<real>& node = nodes[leafIdx];
			if (leafIdx == nodes[node.parent].left && nodes[node.parent].right != -1)
			{
				const DynamicNode<real>& parent = nodes[node.parent];
				leafIdx = parent.right;
				break;
			}

			leafIdx = nodes[leafIdx].parent;
		}
		//assert(false);
	}

	// Traverse down to the left first if possible until a node is reached
	while (nodes[leafIdx].index == -1)
	{
		const DynamicNode<real>& node = nodes[leafIdx];

		if (node.left != -1)
			leafIdx = node.left;
		else if (node.right != -1)
			leafIdx = node.right;
	}

	return leafIdx;
}


template<typename real, unsigned int Dims>
__global__
void sequentialPoissonKernel(
	int indices[],
	real poissonWeights[],
	unsigned int width, unsigned int height,
	unsigned int numSamples,
	real precision, real h,
	const DynamicTreeDevPtrs<real, Dims>* fmmTree,
	NodeState<real> map[], unsigned int mapSize,
	curandState* states,
	unsigned int taylorDegree, unsigned int taylorExpansions
)
{
	extern __shared__ char shared[];
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		for(int i = 0; i < 8192; i++)
			shared[i] = 0;
	}
	__syncthreads();

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int pixelOffset = j + height * i;

	if ((i < width) && (j < height))
	{
		// Define query point
		real query[] = { static_cast<real>(i), static_cast<real>(j) };

		TaylorMap<real> taylorMap;
		taylorMap.degree = taylorDegree;
		taylorMap.nodes = reinterpret_cast<int*>(shared);
		taylorMap.coeffs = reinterpret_cast<real*>(shared + taylorExpansions * sizeof(int));

		// Per-compute taylor expansions
		searchTaylorNode<real, Dims>(taylorMap, fmmTree, query, h);
		__syncthreads();

		// Define starting node 
		int nodeIdx = 0;
		DynamicNode<real>* node = const_cast<DynamicNode<real>*>(&(fmmTree->nodes[nodeIdx]));

		// Offset the base address to avoid overlap between maps for different pixels
		unsigned int mapOffset = pixelOffset * mapSize;
		NodeState<real>* localMap = &map[mapOffset];

		// Compute the likelihood to obtain an empty sample
		//real total = max(real(0), getWeight<real, Dims>(fmmTree, nodeIdx, query, precision, h));
		real total = max(real(0), getWeightTaylor<real, Dims>(fmmTree, taylorMap, nodeIdx, query, precision, h));
		if (total < precision) // Capacity of the root
			return;

		// Store the path to the leaf (includes the root)
		NodeTrace<real> trace[STACKSIZE];
		unsigned int traceIdx = 1;
		trace[0].index = 0;
		trace[0].probability = 1.0f;

		real weights[2], stateWeights[2];
		while (fmmTree->nodes[nodeIdx].sourceCount > numSamples)
		{
			DynamicNode<real>* left = const_cast<DynamicNode<real>*>(&(fmmTree->nodes[node->left]));
			DynamicNode<real>* right = const_cast<DynamicNode<real>*>(&(fmmTree->nodes[node->right]));

			// Approximate the weight of the subtress
			//weights[0] = max(real(0), getWeight<real, Dims>(fmmTree, node->left, query, precision, h));
			//weights[1] = max(real(0), getWeight<real, Dims>(fmmTree, node->right, query, precision, h));
			weights[0] = max(real(0), getWeightTaylor<real, Dims>(fmmTree, taylorMap, node->left, query, precision, h));
			weights[1] = max(real(0), getWeightTaylor<real, Dims>(fmmTree, taylorMap, node->right, query, precision, h));

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
		unsigned int leafCount = fmmTree->nodes[nodeIdx].sourceCount;
		unsigned int subtreeParent = fmmTree->nodes[nodeIdx].parent;
		unsigned int leafIdx = nodeIdx;
		for (unsigned int k = 0; k < leafCount; k++)
		{
			leafIdx = nextLeaf<real, Dims>(fmmTree->nodes, subtreeParent, leafIdx);
			const DynamicNode<real>& leaf = fmmTree->nodes[leafIdx];
			assert(leaf.index != -1);

			//real prob = max(real(0), getWeight<real, Dims>(fmmTree, leafIdx, query, precision, h));
			real prob = max(real(0), getWeightTaylor<real, Dims>(fmmTree, taylorMap, leafIdx, query, precision, h));
			real x = curand_uniform(&state);
			real transformedRandomNumber = x / prob;

			real current_xi = insertSample(leaf.index, transformedRandomNumber, localIndices, localWeights, numSamples);
			//max_xi = max(max_xi, current_xi);
		}

		// Store update curand state
		states[pixelOffset] = state;
	}
}

template<typename real, unsigned int Dims>
void sequentialPoissonDynamicTree(
	int* devIndicesPtr,
	const DynamicTree<real, Dims>& radixTree,
	unsigned int width, unsigned int height,
	unsigned int numSamples,
	real precision, real h
)
{
	// Compute query radius
	real normalization = (real)(2.0 * M_PI * sqrt(h));
	real radius = std::sqrt(2.0f) * h * std::sqrt(-std::log(precision) - std::log(normalization));
	logging::log<logLevel_t::LOG_DEBUG>("sequentialPoissonDynamicTree", "Query radius", radius);

	// Copy device tree pointers to gpu memory
	thrust::device_vector<DynamicTreeDevPtrs<real, Dims>> treePtrs(1);
	treePtrs[0] = radixTree.getDevPtrsStruct();

	// Compute thread and grid layout
	dim3 blocks, threads;
	getGridConfiguration(blocks, threads, width, height);
	logging::log<logLevel_t::LOG_DEBUG>("sequentialPoissonDynamicTree", threads, blocks);

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
		logging::log<logLevel_t::LOG_CRITICAL>("sequentialPoissonDynamicTree", cudaGetErrorString(cudaStatus));
		throw std::runtime_error("curandInitKernel failed");
	}

	// Allocate and initialize the node states used to avoid duplicates
	unsigned int mapSize = numSamples * radixTree.depth; // Map size per pixel: Tree depth times number of samples
	thrust::device_vector<NodeState<real>> nodeStates(width * height * mapSize);

	// Initialize the indices for non-allocated indices
	thrust::device_vector<real> poissonWeights(width * height * numSamples);
	thrust::fill(thrust::device, poissonWeights.begin(), poissonWeights.end(), std::numeric_limits<float>::infinity());
	thrust::fill(thrust::device, devIndicesPtr, devIndicesPtr + numSamples * width * height, -1);

	// Compute required taylor expansion degree for each cell
	const unsigned int sharedBytes = 8192;
	const unsigned int taylorDegree = 4;
	const unsigned int numCoeffs = (taylorDegree + 1) * (taylorDegree + 1);
	const unsigned int taylorExpansions = sharedBytes / (sizeof(real) * taylorDegree + sizeof(int));

	// Launch sampling kernel
	for (unsigned int i = 0; i < std::sqrt(numSamples); i++)
	{
		sequentialPoissonKernel<real, Dims> << <blocks, threads, sharedBytes>> > (
			devIndicesPtr,
			thrust::raw_pointer_cast(poissonWeights.data()),
			width, height,
			numSamples,
			precision, h,
			thrust::raw_pointer_cast(treePtrs.data()),
			thrust::raw_pointer_cast(nodeStates.data()),
			mapSize,
			thrust::raw_pointer_cast(curandStates.data()),
			taylorDegree, taylorExpansions
			);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			logging::log<logLevel_t::LOG_CRITICAL>("sequentialPoissonDynamicTree", cudaGetErrorString(cudaStatus));
			throw std::runtime_error("sequentialPoissonKernel failed");
		}
	}

	// Sychronice device after kernel launch if in debug mode
#ifndef NDEBUG
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		logging::log<logLevel_t::LOG_CRITICAL>(
			"sequentialPoissonDynamicTree", 
			"cudaDeviceSynchronize returned error code", 
			cudaStatus
		);
		throw std::runtime_error("cudaDeviceSynchronize failed");
	}
#endif // NDEBUG
}