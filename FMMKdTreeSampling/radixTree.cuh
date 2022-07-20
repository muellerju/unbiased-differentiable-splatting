#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "alignedBox.cuh"

#define MAXDEGREE 16

struct alignas(16) Node
{
	int left, right, parent; // general attributes (12 bytes)
	int index; // child attributes(4 bytes)
};

template<typename real, unsigned int Dims>
struct alignas(16) RadixTreeDevPtrs
{
	const Node* nodes; 
	const AlignedBox<real, Dims>* alignedBoxes;
	const unsigned int* sourceCounts;
	const real* expansionPoints;
	const real* hermiteErrors;
	const real* hermiteCoefficients;
};

template<typename real, unsigned int Dims>
class RadixTree
{
public:
	RadixTree(size_t _numSources, unsigned int hermiteDegree) :
		nodes(2*_numSources - 1),
		boundingBoxes(2*_numSources - 1),
		sourceCounts(2*_numSources-1),
		expansionPoints(Dims*2*_numSources - 1),
		hermiteErrors(2*_numSources - 1),
		hermiteCoefficients((hermiteDegree+1)*(hermiteDegree+1)*(2*_numSources-1))
	{}

	RadixTreeDevPtrs<real, Dims> getDevPtrsStruct() const
	{
		RadixTreeDevPtrs<real, Dims> treeDevPtrs = {
			thrust::raw_pointer_cast(nodes.data()),
			thrust::raw_pointer_cast(boundingBoxes.data()),
			thrust::raw_pointer_cast(sourceCounts.data()),
			thrust::raw_pointer_cast(expansionPoints.data()),
			thrust::raw_pointer_cast(hermiteErrors.data()),
			thrust::raw_pointer_cast(hermiteCoefficients.data())
		};
		return treeDevPtrs;
	}


	unsigned int depth;
	thrust::device_vector<Node> nodes;
	thrust::device_vector<AlignedBox<real, Dims>> boundingBoxes;
	thrust::device_vector<unsigned int> sourceCounts;
	thrust::device_vector<real> expansionPoints;
	thrust::device_vector<real> hermiteErrors;
	thrust::device_vector<real> hermiteCoefficients;
};

template<typename real>
struct alignas(16) DynamicNode
{
	int left, right, parent; // 12 bytes
	int index, sourceCount; // 8 bytes
	int degree, offset; // 8 bytes
	real radius;
};

template<typename real, unsigned int Dims>
struct alignas(16) DynamicTreeDevPtrs
{
	const DynamicNode<real>* nodes;
	const AlignedBox<real, Dims>* alignedBoxes;
	const real* expansionPoints;
	const real* hermiteCoefficients;
};


template<typename real, unsigned int Dims>
class DynamicTree
{
public:
	DynamicTree(size_t _numSources) :
		nodes(2 * _numSources - 1),
		boundingBoxes(2 * _numSources - 1),
		expansionPoints(Dims * (2 * _numSources - 1))
	{}

	DynamicTreeDevPtrs<real, Dims> getDevPtrsStruct() const
	{
		DynamicTreeDevPtrs<real, Dims> treeDevPtrs = {
			thrust::raw_pointer_cast(nodes.data()),
			thrust::raw_pointer_cast(boundingBoxes.data()),
			thrust::raw_pointer_cast(expansionPoints.data()),
			thrust::raw_pointer_cast(hermiteCoefficients.data())
		};
		return treeDevPtrs;
	}
	
	unsigned int depth;
	thrust::device_vector<DynamicNode<real>> nodes;
	thrust::device_vector<AlignedBox<real, Dims>> boundingBoxes;
	thrust::device_vector<real> expansionPoints;
	thrust::device_vector<real> hermiteCoefficients;
};

