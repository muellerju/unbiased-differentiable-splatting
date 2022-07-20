#pragma once

#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
#include <thrust/fill.h>

#include "logging.h"
#include "radixTree.cuh"
#include "hermitefmm.cuh"
#include "linalg.cuh"
#include "transformation.cuh"

#include "buildRadixTree.cuh"

template<typename real, unsigned int Dims>
struct LeafNodeFunctor : public thrust::unary_function<thrust::tuple<unsigned int, unsigned int>, void>
{
    DynamicNode<real>* leafs;
    AlignedBox<real, 2>* boxes;
    real* centers;
    real width, height, focallength;
    real const* devPointsPtr;

    LeafNodeFunctor(
        DynamicTree<real, Dims> & tree,
        unsigned int _width, 
        unsigned int _height,
        real _focallength,
        real const* _devPointPtr,
        unsigned int numSources
    )
    {
        leafs = thrust::raw_pointer_cast(tree.nodes.data() + numSources - 1);
        boxes = thrust::raw_pointer_cast(tree.boundingBoxes.data() + numSources - 1);
        centers = thrust::raw_pointer_cast(tree.expansionPoints.data() + Dims*(numSources -1));

        width = static_cast<real>(_width);
        height = static_cast<real>(_height);
        focallength = _focallength;
        devPointsPtr = _devPointPtr;
    }

    __device__
        void operator()(const thrust::tuple<unsigned int, unsigned int>& tp)
    {
        unsigned int idx = thrust::get<0>(tp);
        unsigned int sourceId = thrust::get<1>(tp);
        DynamicNode<real> & leaf = leafs[idx];

        leaf.left = -1;
        leaf.right = -1;
        leaf.index = sourceId;
        leaf.sourceCount = 1;
        leaf.degree = 0;

        // Project to camera space
        const ConstVector3<real> camPoint(&devPointsPtr[3 * sourceId]);
        Vector2<real> screenPoint = toScreenspace<real>(width, height, focallength, camPoint);

        // Initialize leaf bounding box and its expansion point
        AlignedBox<real, 2>& leafBox(boxes[idx]);
        real* leafExpansionPoint = &centers[Dims * idx];
        for (unsigned int i = 0; i < Dims; i++)
        {
            real coord = screenPoint[i];
            leafBox.lower[i] = coord;
            leafBox.upper[i] = coord;
            leafExpansionPoint[i] = coord;
        }
    }
};

template<typename real, unsigned int Dims>
struct HierarchyFunctor : public thrust::unary_function<unsigned int, void>
{
    const size_t* mortonKeys;
    unsigned int numSources, leafOffset;
    DynamicNode<real>* nodes;

    HierarchyFunctor(thrust::device_vector<DynamicNode<real>>& _nodes, const thrust::device_vector<size_t>& mortonKeys64, unsigned int _numSources)
    {
        numSources = _numSources;
        leafOffset = numSources - 1;
        nodes = thrust::raw_pointer_cast(_nodes.data());
        mortonKeys = thrust::raw_pointer_cast(mortonKeys64.data());
    }

    __device__
        void operator()(unsigned int idx)
    {
        if (idx == 0) // Root
            nodes[idx].parent = -1;
        nodes[idx].index = -1; //  internal node

        const uint2 ij = determineRange(mortonKeys, numSources, idx);
        const int gamma = findSplit(mortonKeys, ij.x, ij.y);

        // Compute cutting plane
        //int prefix = commonUpperBits(mortonKeys[gamma], mortonKeys[gamma + 1]);
        //nodes[idx].axis = prefix % Dims;
        //nodes[idx].cut = planePosition<size_t, real, 2>(mortonKeys[gamma], min(30, prefix));

        nodes[idx].left = gamma;
        nodes[idx].right = gamma + 1;
        if (thrust::min(ij.x, ij.y) == gamma)
        {
            // Left child is a leaf
            nodes[idx].left += numSources - 1;
            //nodes[nodes[idx].left].axis = (nodes[idx].axis+1) % Dims;
        }
        if (thrust::max(ij.x, ij.y) == gamma + 1)
        {
            // Right child is a leaf
            nodes[idx].right += numSources - 1;
            //nodes[nodes[idx].right].axis = (nodes[idx].axis + 1) % Dims;
        }
        nodes[nodes[idx].left].parent = idx;
        nodes[nodes[idx].right].parent = idx;
        return;
    }
};

template<typename real, unsigned int Dims>
struct InternalNodeFunctor : public thrust::unary_function<unsigned int, void>
{
    DynamicNode<real> * nodes;
    AlignedBox<real, 2> * boxes;
    real* expansionPoints;
    unsigned int * nodeLocks, * treeDepth;
    unsigned int numSources;
    real h, precision;

    InternalNodeFunctor(
        DynamicTree<real, Dims>& radixTree,
        thrust::device_vector<unsigned int> & nodeLocks,
        thrust::device_vector<unsigned int> & treeDepth,
        unsigned int _numSources,
        real stdDev, 
        real _precision) :
        numSources(_numSources),
        h(stdDev),
        precision(_precision)
    {
        nodes = thrust::raw_pointer_cast(radixTree.nodes.data());
        boxes = thrust::raw_pointer_cast(radixTree.boundingBoxes.data());
        expansionPoints = thrust::raw_pointer_cast(radixTree.expansionPoints.data());

        this->nodeLocks = thrust::raw_pointer_cast(nodeLocks.data());
        this->treeDepth = thrust::raw_pointer_cast(treeDepth.data());
    }

    __device__
        void operator()(unsigned int idx)
    {
        unsigned int leafOffset = numSources - 1;
        unsigned int leafIdx = leafOffset + idx;
        DynamicNode<real>* leaf = &nodes[leafIdx];
        unsigned int depth = 1;

        // Traverse tree to the root
        unsigned int nodeIdx = leafIdx;
        DynamicNode<real>* node = leaf;
        do
        {
            // Proceed with the parent node
            nodeIdx = node->parent;
            node = &nodes[node->parent];

            real* nodeExpansionPoint = &expansionPoints[2 * nodeIdx];
            // Critical section begin
            // Only perform the merging by on thread
            unsigned int order = orderFanIn(&nodeLocks[nodeIdx]);
            if (order > 0)
            {
                DynamicNode<real>& left = nodes[node->left];
                DynamicNode<real>& right = nodes[node->right];

                // Use the second thread to merge the bounding box
                mergeBoxes<real, Dims>(boxes[nodeIdx], boxes[node->left], boxes[node->right]);

                // to compute the the expansion point
                unsigned int leftCount = left.sourceCount;
                real* leftPoint = &expansionPoints[node->left];
                unsigned int rightCount = right.sourceCount;
                real* rightPoint = &expansionPoints[node->right];
                unsigned int nodeCount = leftCount + rightCount;
                for (unsigned int i = 0; i < Dims; i++)
                {
                    nodeExpansionPoint[i] = (leftCount * leftPoint[i] + rightCount * rightPoint[i]) / nodeCount;
                }
                node->sourceCount = nodeCount;

                // Compute required degree to obtain
                //real maxNorm = longestSidelengthBBox(boxes[nodeIdx]);
                real maxNorm = maxNormFromBbox<real, Dims>(boxes[nodeIdx], nodeExpansionPoint);
                real r = maxNorm /h;
                node->radius = r;
                unsigned int degree = max(nodes[node->left].degree, nodes[node->right].degree);

                while (hermiteError<real>(r, nodeCount, degree) > precision && r < 1.0)
                    degree += 1;

                //node->degree = r < 1.0 ? degree : -1;
                node->degree = -1;
                if (r < 1.0 && degree <= MAXDEGREE)
                    node->degree = degree;

                // to increase the depth counter 
                depth += 1;
            }
            else
            {
                // All threads which don't merge the node can return;
                return;
            }
        } while (node->parent != -1);
        *treeDepth = depth;
    }

};

template<typename real>
struct HermiteSizeFunctor
{
    __device__ int operator()(DynamicNode<real> const& node)
    {
        int size = node.degree > -1 ? (node.degree + 1) * (node.degree + 1) : 0;
        return size;
    }
};

template<typename real>
struct HermiteOffsetFunctor
{
    DynamicNode<real>* nodes;
    int* offsets;

    HermiteOffsetFunctor(
        thrust::device_vector<DynamicNode<real>>& _nodes, 
        thrust::device_vector<int>& _offsets)
    {
        nodes = thrust::raw_pointer_cast(_nodes.data());
        offsets = thrust::raw_pointer_cast(_offsets.data());
    }
    __device__ void operator()(unsigned int idx)
    {
        nodes[idx].offset = offsets[idx];
    }
};

template<typename real, unsigned int Dims>
struct HermiteCoefficientFunctor : public thrust::unary_function<unsigned int, void>
{
    DynamicNode<real>* nodes;
    unsigned int* nodeLocks;
    real* expansionPoints, * hermiteCoefficients;
    real const* attributeValues;
    unsigned int numSources;
    real h;

    HermiteCoefficientFunctor(
        DynamicTree<real, Dims>& radixTree,
        thrust::device_vector<unsigned int>& _nodeLocks,
        real const* _attributeValues,
        unsigned int _numSources,
        real stdDev) :
        attributeValues(_attributeValues), 
        numSources(_numSources), 
        h(stdDev)
    {
        nodes = thrust::raw_pointer_cast(radixTree.nodes.data());
        nodeLocks = thrust::raw_pointer_cast(_nodeLocks.data());
        expansionPoints = thrust::raw_pointer_cast(radixTree.expansionPoints.data());
        hermiteCoefficients = thrust::raw_pointer_cast(radixTree.hermiteCoefficients.data());
    }

    __device__
        void operator()(unsigned int idx)
    {
        unsigned int leafIdx = numSources + idx - 1;
        DynamicNode<real>* leaf = &nodes[leafIdx];

        // Intialize leaf hermite coefficients
        ConstVector3<real> attribute(&attributeValues[3 * leaf->index]);
        real leafCoefficient = 1.0; //(attribute[0]+attribute[1]+attribute[2])/3.0; 
        hermiteCoefficients[leaf->offset] = leafCoefficient;

        real* leafCoeffs = &hermiteCoefficients[leaf->degree];
        real* leafExpansionPoint = &expansionPoints[Dims * leafIdx];

        unsigned int nodeIdx = leafIdx;
        DynamicNode<real>* node = leaf;
        do
        {
            // Proceed with the parent node
            nodeIdx = node->parent;
            node = &nodes[node->parent];

            // Shift the leaf coefficient to its ancestor
            if (node->degree >= 0)
            {
                real* nodeCoeffs = &hermiteCoefficients[node->offset];
                real* nodeExpansionPoint = &expansionPoints[Dims * nodeIdx];
                shiftLeafHermiteExpansion(
                    nodeCoeffs, nodeExpansionPoint, leafExpansionPoint,
                    leafCoefficient, node->degree, h
                );
            }            
        } while (node->parent != -1);
    }
};

template<typename real, unsigned int Dims>
void buildDynamicTree(
    DynamicTree<real, Dims>& radixTree,
    IndexIter indicesBegin, IndexIter indicesEnd,
    real const* devBoundsPtr,
    real width, real height, real focallength,
    real const* devPointPtr,
    real const* devColorPtr,
    real precision,
    real h
)
{
    size_t numElements = std::distance(indicesBegin, indicesEnd);

    // Compute and sort morton indices
    thrust::device_vector<unsigned int> mortonKeys(numElements);
    MortonTransformFunctor<real> mortonOp(devBoundsPtr, width, height, focallength, devPointPtr);
    thrust::transform(
        indicesBegin,
        indicesEnd,
        mortonKeys.begin(),
        mortonOp
    );
    thrust::sort_by_key(mortonKeys.begin(), mortonKeys.end(), indicesBegin);

    // Extend the sorted keys by their index to ensure that the keys are unique.
    thrust::device_vector<size_t> mortonKeys64(numElements);
    thrust::transform(mortonKeys.begin(), mortonKeys.end(),
        thrust::counting_iterator<unsigned int>(0),
        mortonKeys64.begin(),
        Morton64Functor()
    );

    // Initialize leaf nodes
    auto leafIndexFirst = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::counting_iterator<unsigned int>(0),
            indicesBegin
        )
    );
    auto leafIndexLast = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::counting_iterator<unsigned int>(numElements),
            indicesEnd
        )
    );
    LeafNodeFunctor<real, Dims> leafOp(
        radixTree,
        width, height, focallength,
        devPointPtr, numElements
    );
    thrust::for_each(leafIndexFirst, leafIndexLast, leafOp);

    // Create branches
    thrust::counting_iterator<unsigned int> branchIndicesBegin(0);
    thrust::counting_iterator<unsigned int> branchIndicesEnd(numElements - 1);
    HierarchyFunctor<real, 2> branchOp(radixTree.nodes, mortonKeys64, numElements);
    thrust::for_each(
        branchIndicesBegin,
        branchIndicesEnd,
        branchOp
    );

    // Bottom-up computation of the boundig boxes, expansion points
    // and hermite degree of internal nodes
    thrust::device_vector<unsigned int> treeDepth(1);
    thrust::device_vector<unsigned int> nodeLocks(numElements - 1, 0);
    InternalNodeFunctor<real, Dims> internalOp(
        radixTree,
        nodeLocks,
        treeDepth,
        numElements,
        h, 
        precision
    );
    thrust::counting_iterator<unsigned int> leafIndicesBegin(0);
    thrust::counting_iterator<unsigned int> leafIndicesEnd(numElements);
    thrust::for_each(leafIndicesBegin, leafIndicesEnd, internalOp);

    // Copy tree depth to host
    radixTree.depth = treeDepth[0];
    logging::log<logLevel_t::LOG_DEBUG, unsigned int>("buildDynamicTree", "radix tree depth is", radixTree.depth);
    
    // Compute the total amount of bytes to store hermite coefficients and
    // the Hermite offsets for each node.
    thrust::device_vector<int> hermiteOffset(radixTree.nodes.size());
    thrust::transform(radixTree.nodes.begin(), radixTree.nodes.end(), hermiteOffset.begin(), HermiteSizeFunctor<real>());
    unsigned int hermiteBytes = thrust::reduce(hermiteOffset.begin(), hermiteOffset.end(), 0);
    thrust::exclusive_scan(hermiteOffset.begin(), hermiteOffset.end(), hermiteOffset.begin(), 0, thrust::plus<int>());
    thrust::counting_iterator<unsigned int> nodeIndicesBegin(0);
    thrust::counting_iterator<unsigned int> nodeIndicesEnd(radixTree.nodes.size());
    HermiteOffsetFunctor<real> hermiteOffsetOp(radixTree.nodes, hermiteOffset);
    thrust::for_each(nodeIndicesBegin, nodeIndicesEnd, hermiteOffsetOp);
    logging::log<logLevel_t::LOG_DEBUG, unsigned int>("buildDynamicTree", "hermite coefficient bytes", hermiteBytes);
    
    // Compute Hermite coefficients for all nodes in a bottom-up fashion
    radixTree.hermiteCoefficients.resize(hermiteBytes);
    thrust::fill(
        radixTree.hermiteCoefficients.begin(),
        radixTree.hermiteCoefficients.end(),
        real(0)
    );
    HermiteCoefficientFunctor<real, Dims> coefficientOp(radixTree, nodeLocks, devColorPtr, numElements, h);
    thrust::for_each(leafIndicesBegin, leafIndicesEnd, coefficientOp);
}