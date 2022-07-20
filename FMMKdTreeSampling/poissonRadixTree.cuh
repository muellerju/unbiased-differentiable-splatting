#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

//#include "transformRadixTree.cuh"
#include "logging.h"

template<typename randState_t>
__global__ void curandInitKernel(
	randState_t* states,
	unsigned long long seed,
	unsigned int width, unsigned int height
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned long long pixelOffset = j + height * i;
	
	if ((i < width) && (j < height))
	{
		curand_init(seed, pixelOffset, (unsigned)0, &states[pixelOffset]);
	}
	
}

template<typename real, unsigned int Dims>
__device__ unsigned int nextLeaf(
	const Node* nodes,
	unsigned int subtreeParent,
	unsigned int leafIdx
)
{
	// If current node is a leaf find the next right node
	if (nodes[leafIdx].index > -1)
	{
		while (nodes[leafIdx].parent != subtreeParent)
		{
			const Node& node = nodes[leafIdx];
			if (leafIdx == nodes[node.parent].left && nodes[node.parent].right != -1)
			{
				const Node& parent = nodes[node.parent];
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
		const Node& node = nodes[leafIdx];
		
		if (node.left != -1)
			leafIdx = node.left;
		else if (node.right != -1)
			leafIdx = node.right;
	}

	return leafIdx;
}


template<typename real, unsigned int Dims, unsigned int HermiteDegree>
__global__
void poissonKernel(
	int indices[],
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

		// Find the first empty element in sample
		unsigned int sampleOffset = numSamples * (j + height * i);
		unsigned int sampleIdx = 0;
		while (indices[sampleIdx + sampleOffset] > -1)
		{
			sampleIdx += 1;
		}
		if (sampleIdx >= numSamples)
			return;
		
		// Get curand state
		curandState state = states[pixelOffset];

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

			if (x < numSamples*prob)
			{
				unsigned int sampleOffset = sampleIdx + numSamples * (j + height * i);
				indices[sampleOffset] = leaf.index;
				sampleIdx += 1;
			}

			if (sampleIdx >= numSamples)
				break;
		}

		// Store update curand state
		states[pixelOffset] = state;
	}
}

template<typename real, unsigned int Dims, unsigned int HermiteDegree>
void poissonRadixTree(
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
	logging::log<logLevel_t::LOG_DEBUG>("poissonRadixTree", "Query radius", radius);

	// Copy device tree pointers to gpu memory
	thrust::device_vector<RadixTreeDevPtrs<real, Dims>> treePtrs(1);
	treePtrs[0] = radixTree.getDevPtrsStruct();

	// Compute thread and grid layout
	dim3 blocks, threads;
	getGridConfiguration(blocks, threads, width, height);
	logging::log<logLevel_t::LOG_DEBUG>("poissonRadixTree", threads, blocks);

	// Initialize curand generator
	unsigned long long seed = unsigned(time(NULL));
	thrust::device_vector<curandState> curandStates(width * height);
	curandInitKernel<curandState><<<blocks, threads>>>(
		thrust::raw_pointer_cast(curandStates.data()),
		seed, width, height
	);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		logging::log<logLevel_t::LOG_CRITICAL>("poissonRadixTree", cudaGetErrorString(cudaStatus));
		throw std::runtime_error("curandInitKernel failed");
	}
	
	// Allocate and initialize the node states used to avoid duplicates
	unsigned int mapSize = numSamples * radixTree.depth; // Map size per pixel: Tree depth times number of samples
	thrust::device_vector<NodeState<real>> nodeStates(width * height * mapSize);

	// Initialize the indices for non-allocated indices
	thrust::fill(thrust::device, devIndicesPtr, devIndicesPtr + numSamples * width * height, -1);

	// Launch sampling kernel
	for (unsigned int i = 0; i < std::sqrt(numSamples); i++)
	{
		poissonKernel<real, Dims, HermiteDegree> << <blocks, threads >> > (
			devIndicesPtr,
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
			logging::log<logLevel_t::LOG_CRITICAL>("poissonRadixTree", cudaGetErrorString(cudaStatus));
			throw std::runtime_error("poissonKernel failed");
		}
	}

	// Sychronice device after kernel launch if in debug mode
#ifndef NDEBUG
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) 
	{
		logging::log<logLevel_t::LOG_CRITICAL>("poissonRadixTree", "cudaDeviceSynchronize returned error code", cudaStatus);
		throw std::runtime_error("cudaDeviceSynchronize failed");
	}
#endif // NDEBUG
}