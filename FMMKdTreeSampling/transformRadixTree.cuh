#pragma once

#include <cuda_runtime.h>

#include <iostream>

#include "radixTree.cuh"
#include "hermitefmm.cuh"
#include "cudaHelper.cuh"
#include "mathHelper.cuh"

#define STACKSIZE 64
using StackPtr = int*;

template<typename real, unsigned int Dims, unsigned int HermiteDegree>
__device__
real getWeight(const RadixTreeDevPtrs<real, 2>* fmmTree, unsigned int entryNodeIdx, real query[2], real radius, real precision, real h)
{
	int stack[STACKSIZE + 1];
	StackPtr stackPtr = &stack[0];
	*stackPtr++ = -1; // push empty indicator
	*stackPtr++ = entryNodeIdx; // push root
	const unsigned int numCoefficients = (HermiteDegree + 1) * (HermiteDegree + 1);

	float weight = 0.0f;
	while (*(stackPtr - 1) != -1)
	{
		int nodeIdx = *--stackPtr; // pop current element
		const Node& node(fmmTree->nodes[nodeIdx]);

		// Check if current node has a small enough approximation error
		if (fmmTree->hermiteErrors[nodeIdx] <= precision || node.index != -1)
		{
			const float* expansionPoint = &(fmmTree->expansionPoints[Dims * nodeIdx]);
			const float* hermiteCoefficients = &(fmmTree->hermiteCoefficients[numCoefficients * nodeIdx]);
			weight += evluateHermite<real, HermiteDegree>(query, expansionPoint, hermiteCoefficients, HermiteDegree, h);
			continue;
		}

		int leftIdx = node.left;
		const AlignedBox<real, 2>& leftAlignedBox(fmmTree->alignedBoxes[leftIdx]);
		if (intersectsCircle<real, 2>(leftAlignedBox, query, radius) == true)
		{
			*stackPtr++ = leftIdx;
		}

		int rightIdx = node.right;
		const AlignedBox<real, 2>& rightAlignedBox(fmmTree->alignedBoxes[rightIdx]);
		if (intersectsCircle<real, 2>(rightAlignedBox, query, radius) == true)
		{
			*stackPtr++ = rightIdx;
		}

		/*if (nodeCircleIntersection(query, radius, node) == true)
		{
			*stackPtr++ = left;
			*stackPtr++ = right;
		}
		else if (query[node->axis] < node->cut)
		{
			*stackPtr++ = left;
		}
		else
		{
			*stackPtr++ = right;
		}*/
	}
	return weight;
}

template<typename real, unsigned int Dims>
__device__ real upperBound(const RadixTreeDevPtrs<real, Dims>* fmmTree, int nodeIdx, const real* point, real h)
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

	return fmmTree->sourceCounts[nodeIdx] * exp(-0.5 * squareDist / square(h));
}

template<typename real, unsigned int Dims, unsigned int HermiteDegree>
__device__ real nextExpansionNode(
	const RadixTreeDevPtrs<real, Dims>* tree,
	const real* query,
	int nodeIdx,
	real precision,
	real h
)
{
	Node const* nodes = tree->nodes;
	real const* fmmErrors = tree->hermiteErrors;
	int rootparent = nodes[nodeIdx].parent;
	const unsigned int numCoefficients = (HermiteDegree + 1) * (HermiteDegree + 1);
	real weight = real(0);

	do
	{
		// Traverse down the tree to a node with valid expansion
		while (fmmErrors[nodeIdx] > precision && upperBound<real, Dims>(tree, nodeIdx, query, h) > precision)
		{
			if (nodes[nodeIdx].left > -1)
				nodeIdx = nodes[nodeIdx].left;
			else if (nodes[nodeIdx].right > -1)
				nodeIdx = nodes[nodeIdx].right;
		}

		// Evaluate ther Hermite expansion
		if (fmmErrors[nodeIdx] <= precision)
		{
			const real* expansionPoint = &(tree->expansionPoints[Dims * nodeIdx]);
			const real* hermiteCoefficients = &(tree->hermiteCoefficients[numCoefficients * nodeIdx]);
			weight += evluateHermite<real, HermiteDegree>(query, expansionPoint, hermiteCoefficients, HermiteDegree, h);
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


template<typename real, unsigned int Dims, unsigned int HermiteDegree>
__device__
real getWeightStackless(const RadixTreeDevPtrs<real, 2>* fmmTree, int entryNodeIdx, real query[2], real radius, real precision, real h)
{
	real weight = real(0);
	int nodeIdx = entryNodeIdx;
	Node const& entryNode = fmmTree->nodes[nodeIdx];
	int subtreeParent = entryNode.parent;
	const unsigned int numCoefficients = (HermiteDegree + 1) * (HermiteDegree + 1);

	if (fmmTree->hermiteErrors[nodeIdx] <= precision)
	{
		const real* expansionPoint = &(fmmTree->expansionPoints[Dims * nodeIdx]);
		const real* hermiteCoefficients = &(fmmTree->hermiteCoefficients[numCoefficients * nodeIdx]);
		weight += evluateHermite<real, HermiteDegree>(query, expansionPoint, hermiteCoefficients, HermiteDegree, h);
	}
	else
	{
		weight += nextExpansionNode<real, Dims, HermiteDegree>(fmmTree, query, entryNodeIdx, precision, h);
	}

	return weight;
}

template<typename real, unsigned int Dims, unsigned int HermiteDegree>
__global__
void gaussTransformKernel(real* values, unsigned int width, unsigned int height, real radius, real precision, const RadixTreeDevPtrs<real, Dims>* fmmTree, real h)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = j + width * i; //i * width + j;

	if ((i < width) && (j < height))
	{
		values[index] = 0.0f;
		float query[2];
		query[0] = __int2float_rn(i);
		query[1] = __int2float_rn(j);

		values[index] = getWeight<real, Dims, HermiteDegree>(
			fmmTree,
			0,
			query,
			radius,
			precision,
			h
		);

		/*values[index] = getWeightStackless<real, Dims, HermiteDegree>(
			fmmTree, 0, query, radius, precision, h
		);*/
	}
}

template<typename real, unsigned int Dims, unsigned int HermiteDegree>
void gaussTransformRadixTree(real* devGaussValuesPtr, unsigned int width, unsigned int height, real precision, const RadixTree<real, Dims>& radixTree, real h)
{
	// Compute query radius
	real normalization = real(2.0 * M_PI) * std::sqrt(h);
	real radius = std::sqrt(2.0f) * h * std::sqrt(-std::log(precision) - std::log(normalization));
	std::cout << "> gaussTransformRadixTree: Query radius " << radius << std::endl;

	// Copy device tree pointers to gpu memory
	thrust::device_vector<RadixTreeDevPtrs<real, Dims>> treePtrs(1);
	treePtrs[0] = radixTree.getDevPtrsStruct();

	// Compute thread and grid layout
	dim3 blocks, threads;
	getGridConfiguration(blocks, threads, width, height);
	std::cout << "> gaussTransformRadixTree: Block size " << blocks.x << "," << blocks.y << " with " << threads.x << "," << threads.y << " threads per block" << std::endl;


	// Evaluate gauss transform
	gaussTransformKernel<real, Dims, HermiteDegree> << <blocks, threads >> > (
		devGaussValuesPtr,
		width, height,
		radius, precision,
		thrust::raw_pointer_cast(treePtrs.data()),
		h
		);
}