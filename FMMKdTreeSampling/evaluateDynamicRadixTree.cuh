#pragma once

#include <cuda_runtime.h>

#include <iostream>

#include "radixTree.cuh"
#include "hermitefmm.cuh"
#include "cudaHelper.cuh"
#include "mathHelper.cuh"

#include "logging.h"

template<typename real>
struct alignas(8) TaylorMap
{
	int * nodes; // 8 bytes
	real* coeffs; // 8 bytes
	unsigned int degree, mapsize; // 8 bytes
};

template<typename real, unsigned int Dims>
__device__ void searchTaylorNode(
	TaylorMap<real>& taylorMap,
	const DynamicTreeDevPtrs<real, Dims>* tree, const real* point, real h)
{
	DynamicNode<real> const* nodes = tree->nodes;

	unsigned int cellX = threadIdx.x / 4;
	unsigned int cellY = threadIdx.y / 4;
	unsigned int cellOffset = cellY + 4 * cellX;

	unsigned int pixelOffset = (threadIdx.y % 4) + 4 * (threadIdx.x % 4);
	if (pixelOffset > taylorMap.mapsize / 3)
		return;

	unsigned int nodeIdx = 0;
	unsigned int depth = 0;
	int taylorNode = -1;
	real taylorWeight = -INFINITY;

	do
	{
		// Traverse down the tree to a node with valid expansion
		while (depth < cellOffset)
		{
			if (nodes[nodeIdx].left > -1)
				nodeIdx = nodes[nodeIdx].left;
			else if (nodes[nodeIdx].right > -1)
				nodeIdx = nodes[nodeIdx].right;
			depth += 1;
		}

		// Convert taylor to hermite coefficient
		real const* expansionPoint = &(tree->expansionPoints[Dims * nodeIdx]);
		real distance = real(0);
		for (unsigned int i = 0; i < Dims; i++)
			distance += square(expansionPoint[i] - point[i]);
		real weight = exp(-0.5 * distance / square(h));
		if (weight > taylorWeight)
		{
			taylorWeight = taylorWeight;
			taylorNode = nodeIdx;
		}


		// Traverse upwards
		int parentIdx = nodes[nodeIdx].parent;
		while (parentIdx != -1 && (nodes[parentIdx].right == -1 || nodeIdx == nodes[parentIdx].right))
		{
			nodeIdx = parentIdx;
			parentIdx = nodes[parentIdx].parent;
		}
		if (parentIdx != -1)
			nodeIdx = nodes[parentIdx].right;

	} while (nodes[nodeIdx].parent != -1);

	unsigned int mapOffset = taylorMap.mapsize * cellOffset;
	//taylorMap.nodes[0 + 3 * (pixelOffset / 3) + mapOffset] = nodeIdx;
	//taylorMap.nodes[1 + 3 * (pixelOffset / 3) + mapOffset] = tree->nodes[nodeIdx].left;
	//taylorMap.nodes[2 + 3 * (pixelOffset / 3) + mapOffset] = tree->nodes[nodeIdx].right;
}

template<typename real>
__device__ int findTaylorExpansion(TaylorMap<real> const& taylorMap, unsigned int cellOffset, int nodeIdx)
{
	int mapIdx = -1;
	for (unsigned int i = taylorMap.mapsize*cellOffset; i < taylorMap.mapsize * (cellOffset + 1); i++)
	{
		if (taylorMap.nodes[i] == nodeIdx)
			mapIdx = i - taylorMap.mapsize*cellOffset;
	}
	return mapIdx;
}

template<typename real, unsigned int Dims>
__device__ real upperBound(const DynamicTreeDevPtrs<real, Dims>* fmmTree, int nodeIdx, const real* point, real h)
{
	const AlignedBox<real, Dims>& box = fmmTree->alignedBoxes[nodeIdx];
	// Check if points is in the box
	real xleft = point[0] - box.lower[0];
	real xright = box.upper[0] - point[0];
	real yleft = point[1] - box.lower[1];
	real yright = box.upper[1] - point[1];

	if (box.lower[0] <= point[0] && point[0] <= box.upper[0] && box.lower[1] <= point[1] && point[1] <= box.upper[1])
		return 1.0;

	real dx = min(abs(box.lower[0] - point[0]), abs(point[0] - box.upper[0]));
	real dy = min(abs(box.lower[1] - point[1]), abs(point[1] - box.upper[1]));

	real squareDist = 2*square(min(dx, dy));

	return fmmTree->nodes[nodeIdx].sourceCount * exp(-0.5 * squareDist / square(h));
}

template<typename real, unsigned int Dims>
__device__ bool truncate(DynamicTreeDevPtrs<real, Dims> const* fmmTree, int nodeIdx, const real* query, real precision, real h)
{
	unsigned int n = fmmTree->nodes[nodeIdx].sourceCount;
	real radius = sqrt(-2 * square(h) * log(precision / n ));

	AlignedBox<real, Dims> const& aabb = fmmTree->alignedBoxes[nodeIdx];
	return !intersectsCircle<real, 2>(aabb, query, radius);
}

template<typename real, unsigned int Dims>
__device__ real accumulateSubtree(
	const DynamicTreeDevPtrs<real, Dims>* tree,
	const real* query,
	int nodeIdx,
	real precision,
	real h
)
{
	DynamicNode<real> const* nodes = tree->nodes;
	int rootparent = nodes[nodeIdx].parent;
	real weight = real(0);

	do
	{
		// Traverse down the tree to a node with valid expansion
		while (nodes[nodeIdx].degree < 0 && !truncate<real, Dims>(tree, nodeIdx, query, precision, h)) //upperBound<real, Dims>(tree, nodeIdx, query, h) > precision
		{
			if (nodes[nodeIdx].left > -1)
				nodeIdx = nodes[nodeIdx].left;
			else if (nodes[nodeIdx].right > -1)
				nodeIdx = nodes[nodeIdx].right;
		}

		// Evaluate ther Hermite expansion
		if (nodes[nodeIdx].degree > -1)
		{
			const real* expansionPoint = &(tree->expansionPoints[Dims * nodeIdx]);
			const real* hermiteCoefficients = &(tree->hermiteCoefficients[nodes[nodeIdx].offset]);
			weight += evluateHermite<real, MAXDEGREE>(query, expansionPoint, hermiteCoefficients, nodes[nodeIdx].degree, h);
		}
		

		// Traverse upwards
		int parentIdx = nodes[nodeIdx].parent;
		while ( parentIdx != rootparent && (nodes[parentIdx].right == -1 || nodeIdx == nodes[parentIdx].right) )
		{
			nodeIdx = parentIdx;
			parentIdx = nodes[parentIdx].parent;
		}
		if (parentIdx != rootparent)
			nodeIdx = nodes[parentIdx].right;

	} while (nodes[nodeIdx].parent != rootparent);

	return weight;
}

template<typename real, unsigned int Dims>
__device__
real getWeightTaylor(
	const DynamicTreeDevPtrs<real, Dims>* tree,
	TaylorMap<real> const& taylorMap,
	int entryNodeIdx,
	const real query[2],
	real precision, real h)
{
	real weight = real(0);
	int nodeIdx = entryNodeIdx;
	DynamicNode<real> const& entryNode = tree->nodes[nodeIdx];
	int subtreeParent = entryNode.parent;

	unsigned int cellX = floor(query[0] / 4);
	unsigned int cellY = floor(query[1] / 4);
	unsigned int cellOffset = cellY + 4 * cellX;

	int taylorMapIdx = findTaylorExpansion(taylorMap, cellOffset, entryNodeIdx);
	if (taylorMapIdx > -1)
	{
		real expansionPoint[] = { cellX * 4 + 1.5, cellY * 4 + 1.5 };
		unsigned int numCoeffs = (taylorMap.degree + 1) * (taylorMap.degree + 1);
		unsigned int taylorOffset = numCoeffs * (taylorMapIdx + taylorMap.mapsize * cellOffset);
		const real* taylorCoefficients = &(taylorMap.coeffs[taylorOffset]);
		return evaluateTaylor<real>(query, expansionPoint, taylorCoefficients, taylorMap.degree, h);
	}
	else if (entryNode.degree > -1)
	{
		const real* expansionPoint = &(tree->expansionPoints[Dims * nodeIdx]);
		const real* hermiteCoefficients = &(tree->hermiteCoefficients[entryNode.offset]);
		return evluateHermite<real, MAXDEGREE>(query, expansionPoint, hermiteCoefficients, entryNode.degree, h);
	}
	else
	{
		return accumulateSubtree<real, Dims>(tree, query, entryNodeIdx, precision, h);
	}

	return weight;
}

template<typename real, unsigned int Dims>
__device__
real getWeight(
	const DynamicTreeDevPtrs<real, Dims>* tree,
	int entryNodeIdx,
	const real query[2],
	real precision, real h)
{
	real weight = real(0);
	int nodeIdx = entryNodeIdx;
	DynamicNode<real> const& entryNode = tree->nodes[nodeIdx];
	int subtreeParent = entryNode.parent;

	if (entryNode.degree > -1)
	{
		const real* expansionPoint = &(tree->expansionPoints[Dims * nodeIdx]);
		const real* hermiteCoefficients = &(tree->hermiteCoefficients[entryNode.offset]);
		return evluateHermite<real, MAXDEGREE>(query, expansionPoint, hermiteCoefficients, entryNode.degree, h);
	}
	else
	{
		return accumulateSubtree<real, Dims>(tree, query, entryNodeIdx, precision, h);
	}

	return weight;
}

template<typename real, unsigned int Dims>
__global__
void dynamicTreeDgtKernel(
	real* values, 
	const DynamicTreeDevPtrs<real, Dims>* tree,
	unsigned int width, unsigned int height,
	unsigned int taylorDegree, unsigned int taylorExpansions,
	real precision, real h
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = j + width * i; //i * width + j;

	extern __shared__ char shared[];


	if ((i < width) && (j < height))
	{
		values[index] = real(0);
		float query[2];
		query[0] = __int2float_rn(i);
		query[1] = __int2float_rn(j);

		/*TaylorMap<real> taylorMap;
		taylorMap.degree = taylorDegree;
		taylorMap.mapsize = taylorExpansions;
		taylorMap.nodes = reinterpret_cast<int*>(shared);
		taylorMap.coeffs = reinterpret_cast<real*>(shared + taylorExpansions * sizeof(int));
		// Per-compute taylor expansions
		searchTaylorNode<real, Dims>(taylorMap, tree, query, h);
		//precomputeTaylor();

		__syncthreads();*/

		// Evaluate hermite and taylor expansions
		values[index] = getWeight<real, Dims>(
			tree,
			0,
			query,
			precision, h
		);
	}
}

template<typename real, unsigned int Dims>
void evaluateDgtDynamicTree(
	real* devValuesPtr, 
	const DynamicTree<real, Dims>& radixTree, 
	unsigned int width, unsigned int height,
	real precision, real h
)
{
	// Copy device tree pointers to gpu memory
	thrust::device_vector<DynamicTreeDevPtrs<real, Dims>> treePtrs(1);
	treePtrs[0] = radixTree.getDevPtrsStruct();

	// Compute thread and grid layout
	dim3 blocks, threads;
	getGridConfiguration(blocks, threads, width, height);
	logging::log<logLevel_t::LOG_DEBUG>("evaluateDgtDynamicTree", threads, blocks);

	// Compute required taylor expansion degree for each cell
	const unsigned int sharedBytes = 8192;
	const unsigned int taylorDegree = 4;
	const unsigned int numCoeffs = (taylorDegree + 1) * (taylorDegree + 1);
	const unsigned int taylorExpansions = sharedBytes / (sizeof(real) * taylorDegree + sizeof(int));

	// Evaluate gauss transform
	dynamicTreeDgtKernel<real, Dims> << <blocks, threads, sharedBytes >> > (
		devValuesPtr,
		thrust::raw_pointer_cast(treePtrs.data()),
		width, height,
		taylorDegree, taylorExpansions / 16,
		precision, h
	);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		logging::log<logLevel_t::LOG_CRITICAL, cudaError_t>("evaluateDgtDynamicTree", "Kernel launch failed with", cudaStatus);
		throw std::runtime_error("dynamicTreeDgtKernel launch failed");
	}
}
