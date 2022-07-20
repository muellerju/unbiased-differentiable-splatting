#pragma once

#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
#include <thrust/fill.h>

#include "radixTree.cuh"
#include "hermitefmm.cuh"
#include "linalg.cuh"
#include "transformation.cuh"
//#include "splatting.cuh"

//extern __device__ unsigned int devTreeDepth;

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__
inline unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given ND point located within the unit cube [0,1].
template<typename real, unsigned int Dims>
__device__
unsigned int mortonND(const real point[])
{
    unsigned int key = 0;
    for (unsigned int i = 0; i < Dims; i++)
    {
        real coord = point[i];
        coord = min(max(coord * (real)1024, (real)0.0), (real)1023.0);

        unsigned int offset = 1 << (Dims - 1 - i);
        unsigned int bits = expandBits(static_cast<unsigned int>(coord));
        key += offset * bits;
    }
    return key;
    /*real x = min(max(point[0] * 1024.0f, 0.0f), 1023.0f);
    real y = min(max(point[1] * 1024.0f, 0.0f), 1023.0f);
    real z = min(max(1.0 * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;*/
}

__device__
inline int commonUpperBits(unsigned int a, unsigned int b)
{
#ifdef __CUDA_ARCH__
    return __clz(a ^ b);
#else
    return 0;
#endif // __CUDA_ARCH__
}

__device__
inline int commonUpperBits(size_t a, size_t b)
{
#ifdef __CUDA_ARCH__
    return  __clzll(a ^ b);
#else
    return 0;
#endif // __CUDA_ARCH__
}

template<typename key_t>
__device__
uint2 determineRange(const key_t* mortonKeys, unsigned int numSources, unsigned int idx)
{
    if (idx == 0)
    {
        return make_uint2(0, numSources - 1);
    }

    // determine direction of the range
    const key_t self_code = mortonKeys[idx];
    const int L_delta = commonUpperBits(self_code, mortonKeys[idx - 1]);
    const int R_delta = commonUpperBits(self_code, mortonKeys[idx + 1]);
    const int d = (R_delta > L_delta) ? 1 : -1;

    // Compute upper bound for the length of the range

    const int delta_min = thrust::min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = idx + d * l_max;
    if (0 <= i_tmp && i_tmp < numSources)
    {
        delta = commonUpperBits(self_code, mortonKeys[i_tmp]);
    }
    while (delta > delta_min)
    {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if (0 <= i_tmp && i_tmp < numSources)
        {
            delta = commonUpperBits(self_code, mortonKeys[i_tmp]);
        }
    }

    // Find the other end by binary search
    int l = 0;
    int t = l_max >> 1;
    while (t > 0)
    {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if (0 <= i_tmp && i_tmp < numSources)
        {
            delta = commonUpperBits(self_code, mortonKeys[i_tmp]);
        }
        if (delta > delta_min)
        {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * d;
    if (d < 0)
    {
        thrust::swap(idx, jdx); // make it sure that idx < jdx
    }
    return make_uint2(idx, jdx);
}

template<typename key_t>
__device__
int findSplit(const key_t* mortonKeys, const unsigned int first, const unsigned int last)
{
    const key_t first_code = mortonKeys[first];
    const key_t last_code = mortonKeys[last];
    if (first_code == last_code)
    {
        return (first + last) >> 1;
    }
    const int delta_node = commonUpperBits(first_code, last_code);

    // binary search...
    int split = first;
    int stride = last - first;
    do
    {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last)
        {
            const int delta = commonUpperBits(first_code, mortonKeys[middle]);
            if (delta > delta_node)
            {
                split = middle;
            }
        }
    } while (stride > 1);

    return split;
}

template<typename key_t, typename real, unsigned int Dims>
__device__
real planePosition(key_t key, int prefixLength)
{
    unsigned int bits = 0;
    unsigned int offset = prefixLength % Dims;
    unsigned int mask = 1 << offset;
    unsigned int bitMask = 1;
    for (unsigned int i = offset; i <= prefixLength - 3; i += 3)
    {
        if (mask & key)
            bits |= bitMask;

        mask <<= 3;
        bitMask <<= 1;
    }
    //bitMask |= bitMask;
    return static_cast<real>(bits);
}

template<typename real, unsigned int Dims>
struct MortonFunctor : public thrust::unary_function<unsigned int, unsigned int>
{
    const real* devBoundsPtr;
    const real* devSourcePtr;
    MortonFunctor(const real* _devBoundsPtr, const real* _devSourcePtr) : devBoundsPtr(_devBoundsPtr), devSourcePtr(_devSourcePtr) {}

    __device__
        unsigned int operator()(unsigned int srcIdx)
    {
        // Scale source to [0,1]
        real source[Dims];
        for (unsigned int i = 0; i < Dims; i++)
        {
            real start = devBoundsPtr[2*i];
            real end = devBoundsPtr[2*i + 1];
            source[i] = (devSourcePtr[Dims * srcIdx + i] - start) / (end - start);
        }
        return mortonND<real, Dims>(source);
    }
};


template<typename real>
struct MortonTransformFunctor : public thrust::unary_function<unsigned int, unsigned int>
{
    const real* devBoundsPtr;
    const real* devPointsPtr;
    real width, height, focallength;

    MortonTransformFunctor(
        const real* _devBoundsPtr, 
        real _width, real _height, real _focallength, 
        const real* _devPointsPtr) : 
        devBoundsPtr(_devBoundsPtr), 
        devPointsPtr(_devPointsPtr)
    {
        width = _width;
        height = _height;
        focallength = _focallength;
    }

    __device__
        unsigned int operator()(unsigned int srcIdx)
    {
        const ConstVector3<real> camPoint(&devPointsPtr[3 * srcIdx]);
        Vector2<real> screenPoint = toScreenspace<real>(width, height, focallength, camPoint);

        // Scale source to [0,1]
        real source[2];
        for (unsigned int i = 0; i < 2; i++)
        {
            real start = devBoundsPtr[2 * i];
            real end = devBoundsPtr[2 * i + 1];
            source[i] = (screenPoint[i] - start) / (end - start);
        }
        return mortonND<real, 2>(source);
    }
};

struct Morton64Functor : public thrust::binary_function<unsigned int, unsigned int, size_t>
{
    __device__
        size_t operator()(unsigned int key, unsigned int idx)
    {
        size_t key64 = key;
        key64 <<= 32;
        key64 |= idx;
        return key64;
    }
};

template<typename real, unsigned int Dims>
struct LeafFunctor : public thrust::unary_function<thrust::tuple<unsigned int, unsigned int>, void>
{
    Node* leafs;
    LeafFunctor(thrust::device_vector<Node>& nodes, unsigned int numSources)
    {
        leafs = thrust::raw_pointer_cast(nodes.data() + numSources - 1);
    }

    __device__
        void operator()(const thrust::tuple<unsigned int, unsigned int>& tp)
    {
        unsigned int idx = thrust::get<0>(tp);
        unsigned int sourceId = thrust::get<1>(tp);
        Node& leaf = leafs[idx];

        leaf.index = sourceId;
        leaf.left = -1;
        leaf.right = -1;
    }
};

template<typename real, unsigned int Dims>
struct BranchFunctor : public thrust::unary_function<unsigned int, void>
{
    const size_t* mortonKeys;
    unsigned int numSources, leafOffset;
    Node* nodes;

    BranchFunctor(thrust::device_vector<Node>& _nodes, const thrust::device_vector<size_t>& mortonKeys64, unsigned int _numSources)
    {
        numSources = _numSources;
        leafOffset = numSources - 1;
        nodes = thrust::raw_pointer_cast(_nodes.data());
        mortonKeys = thrust::raw_pointer_cast(mortonKeys64.data());
        //devBranchesPtr = thrust::raw_pointer_cast(tree.nodes.data());
        //devLeafsPtr = thrust::raw_pointer_cast(tree.nodes.data()) + leafOffset;

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

__device__
inline int orderFanIn(volatile unsigned int* lock)
{
#ifdef  __CUDA_ARCH__
    return atomicCAS((unsigned int*)lock, 0, 1); // Only on thread gets 0 order (1,2 for the others)
#else
    return 0;
#endif
}

enum branchDir_t
{
    left = 1,
    right = 2
};

template<typename real, unsigned int Dims, unsigned int HermiteDegree>
struct MergeAlignedBoxesFunctor : public thrust::unary_function<unsigned int, void>
{
    Node* nodes;
    AlignedBox<real, Dims>* boxes;
    real* expansionPoints, * hermiteErrors;
    const real* sources;
    unsigned int* pointCounts, * nodeLocks, * treeDepth;
    unsigned int numSources;
    real h;

    MergeAlignedBoxesFunctor(
        RadixTree<real, 2>& radixTree,
        thrust::device_vector<unsigned int>& nodeLocks,
        thrust::device_vector<unsigned int>& treeDepth,
        const real* _devSourcePtr,
        unsigned int _numSources,
        real stdDev) :
        sources(_devSourcePtr), numSources(_numSources), h(stdDev)
    {
        nodes = thrust::raw_pointer_cast(radixTree.nodes.data());
        boxes = thrust::raw_pointer_cast(radixTree.boundingBoxes.data());
        pointCounts = thrust::raw_pointer_cast(radixTree.sourceCounts.data());
        expansionPoints = thrust::raw_pointer_cast(radixTree.expansionPoints.data());
        hermiteErrors = thrust::raw_pointer_cast(radixTree.hermiteErrors.data());
        this->nodeLocks = thrust::raw_pointer_cast(nodeLocks.data());
        this->treeDepth = thrust::raw_pointer_cast(treeDepth.data());
    }

    __device__
        void operator()(unsigned int idx)
    {
        unsigned int leafOffset = numSources - 1;
        unsigned int leafIdx = leafOffset + idx;
        // Initialize leaf point count
        pointCounts[leafIdx] = 1;
        // Initialize leaf bounding box
        Node* leaf = &nodes[leafIdx];
        AlignedBox<real, Dims>& leafBox(boxes[leafIdx]);
        real* leafExpansionPoint = &expansionPoints[Dims * leafIdx];
        for (unsigned int i = 0; i < Dims; i++)
        {
            real coord = sources[Dims * leaf->index + i];
            leafBox.lower[i] = coord;
            leafBox.upper[i] = coord;
            leafExpansionPoint[i] = coord;
        }
        // Set the approximation error to 0
        hermiteErrors[leafIdx] = (real)0.0;
        unsigned int depth = 1;

        // Traverse tree to the root
        unsigned int nodeIdx = leafIdx;
        Node* node = leaf;
        do
        {
            // Proceed with the parent node
            nodeIdx = node->parent;
            node = &nodes[node->parent];

            real* nodeExpansionPoint = &expansionPoints[Dims * nodeIdx];
            // Critical section begin
            // Only perform the merging by on thread
            unsigned int order = orderFanIn(&nodeLocks[nodeIdx]);
            if (order > 0)
            {
                // Use the second thread to merge the bounding box
                mergeBoxes<real, Dims>(boxes[nodeIdx], boxes[node->left], boxes[node->right]);
                // to compute the the expansion point
                unsigned int leftCount = pointCounts[node->left];
                real* leftPoint = &expansionPoints[node->left];
                unsigned int rightCount = pointCounts[node->right];
                real* rightPoint = &expansionPoints[node->right];
                unsigned int nodeCount = leftCount + rightCount;
                for (unsigned int i = 0; i < Dims; i++)
                {
                    nodeExpansionPoint[i] = (leftCount * leftPoint[i] + rightCount * rightPoint[i]) / nodeCount;
                }
                pointCounts[nodeIdx] = nodeCount;
                // to compute the hermite approx. error
                real maxNorm = maxNormFromBbox<real, Dims>(boxes[nodeIdx], nodeExpansionPoint);
                real r = maxNorm / h;
                hermiteErrors[nodeIdx] = hermiteError<real>(r, nodeCount, HermiteDegree);
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

template<typename real, unsigned int HermiteDegree>
struct MergeAlignedBoxesTransformFunctor : public thrust::unary_function<unsigned int, void>
{
    Node* nodes;
    AlignedBox<real, 2>* boxes;
    real* expansionPoints, * hermiteErrors;
    real width, height, focallength;
    const real* devPointsPtr;
    unsigned int* pointCounts, * nodeLocks, * treeDepth;
    unsigned int numSources;
    real h;

    MergeAlignedBoxesTransformFunctor(
        RadixTree<real, 2>& radixTree,
        thrust::device_vector<unsigned int>& nodeLocks,
        thrust::device_vector<unsigned int>& treeDepth,
        real _width, real _height, real _focallength,
        const real* _devPointerSrc,
        unsigned int _numSources,
        real stdDev) :
        devPointsPtr(_devPointerSrc), 
        numSources(_numSources), 
        h(stdDev)
    {
        nodes = thrust::raw_pointer_cast(radixTree.nodes.data());
        boxes = thrust::raw_pointer_cast(radixTree.boundingBoxes.data());
        pointCounts = thrust::raw_pointer_cast(radixTree.sourceCounts.data());
        expansionPoints = thrust::raw_pointer_cast(radixTree.expansionPoints.data());
        hermiteErrors = thrust::raw_pointer_cast(radixTree.hermiteErrors.data());

        width = _width;
        height = _height;
        focallength = _focallength;

        this->nodeLocks = thrust::raw_pointer_cast(nodeLocks.data());
        this->treeDepth = thrust::raw_pointer_cast(treeDepth.data());
    }

    __device__
        void operator()(unsigned int idx)
    {
        unsigned int leafOffset = numSources - 1;
        unsigned int leafIdx = leafOffset + idx;
        Node* leaf = &nodes[leafIdx];
        // Initialize leaf point count
        pointCounts[leafIdx] = 1;
        // Project to camera space
        const ConstVector3<real> camPoint(&devPointsPtr[3 * leaf->index]);
        Vector2<real> screenPoint = toScreenspace<real>(width, height, focallength, camPoint);
        // Initialize leaf bounding box and its expansion point
        AlignedBox<real, 2>& leafBox(boxes[leafIdx]);
        real* leafExpansionPoint = &expansionPoints[2 * leafIdx];
        for (unsigned int i = 0; i < 2; i++)
        {
            real coord = screenPoint[i];
            leafBox.lower[i] = coord;
            leafBox.upper[i] = coord;
            leafExpansionPoint[i] = coord;
        }
        // Set the approximation error to 0
        hermiteErrors[leafIdx] = (real)0.0;
        unsigned int depth = 1;

        // Traverse tree to the root
        unsigned int nodeIdx = leafIdx;
        Node* node = leaf;
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
                // Use the second thread to merge the bounding box
                mergeBoxes<real, 2>(boxes[nodeIdx], boxes[node->left], boxes[node->right]);
                // to compute the the expansion point
                unsigned int leftCount = pointCounts[node->left];
                real* leftPoint = &expansionPoints[node->left];
                unsigned int rightCount = pointCounts[node->right];
                real* rightPoint = &expansionPoints[node->right];
                unsigned int nodeCount = leftCount + rightCount;
                for (unsigned int i = 0; i < 2; i++)
                {
                    nodeExpansionPoint[i] = (leftCount * leftPoint[i] + rightCount * rightPoint[i]) / nodeCount;
                }
                pointCounts[nodeIdx] = nodeCount;
                // to compute the hermite approx. error
                real maxNorm = maxNormFromBbox<real, 2>(boxes[nodeIdx], nodeExpansionPoint);
                real r = maxNorm / h;
                hermiteErrors[nodeIdx] = hermiteError<float>(r, nodeCount, HermiteDegree);
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

template<typename real, unsigned int Dims, unsigned int HermiteDegree>
struct ComputeHermiteFunctor : public thrust::unary_function<unsigned int, void>
{
    Node* nodes;
    real* expansionPoints, * hermiteCoefficients;
    unsigned int numSources;
    real h;

    ComputeHermiteFunctor(
        RadixTree<real, Dims>& radixTree,
        unsigned int _numSources,
        real stdDev) :
        numSources(_numSources), h(stdDev)
    {
        nodes = thrust::raw_pointer_cast(radixTree.nodes.data());
        expansionPoints = thrust::raw_pointer_cast(radixTree.expansionPoints.data());
        hermiteCoefficients = thrust::raw_pointer_cast(radixTree.hermiteCoefficients.data());
    }

    __device__
        void operator()(unsigned int idx)
    {
        unsigned int leafIdx = numSources + idx - 1;
        Node* leaf = &nodes[leafIdx];
        // Intialize leaf hermite coefficients
        const unsigned int numCoeffs = (HermiteDegree + 1) * (HermiteDegree + 1);

        real* leafCoeffs = &hermiteCoefficients[numCoeffs * leafIdx];
        real* leafExpansionPoint = &expansionPoints[Dims * leafIdx];

        real leafCoefficient = (real)1.0 / ((real)2.0 * M_PI * sqrt(h));
        leafCoeffs[0] = leafCoefficient;

        unsigned int nodeIdx = leafIdx;
        Node* node = leaf;
        do
        {
            // Proceed with the parent node
            nodeIdx = node->parent;
            node = &nodes[node->parent];

            // Shift the leaf coefficient to its ancestor
            real* nodeCoeffs = &hermiteCoefficients[numCoeffs * nodeIdx];
            real* nodeExpansionPoint = &expansionPoints[Dims * nodeIdx];
            //printf("Thread %i shifts leaf hermite coeffs. to node %i", idx, nodeIdx);
            shiftLeafHermiteExpansion(nodeCoeffs, nodeExpansionPoint, leafExpansionPoint, leafCoefficient, HermiteDegree, h);
        } while (node->parent != -1);
    }
};

template<typename real>
bool verifyTree(const std::vector<Node>& nodes)
{
    bool structureOk = true;
    for (size_t idx = 0; idx < nodes.size(); idx++)
    {
        const Node& node(nodes[idx]);
        // Verify child/parent relation ships
        if (node.left != -1)
        {
            const Node& left(nodes[node.left]);
            if (left.parent != idx)
                structureOk = false;
        }
        if (node.right != -1)
        {
            const Node& right(nodes[node.right]);
            if (right.parent != idx)
                structureOk = false;
        }
    }
    return structureOk;
}

template<typename real, unsigned int Dims, unsigned int HermiteDegree>
void buildRadixTree(RadixTree<real, Dims>& radixTree, real* devBoundsPtr, real* devSourcePtr, unsigned int numSources, float h)
{
    // Compute and sort morton indices
    thrust::device_vector<unsigned int> mortonKeys(numSources);
    MortonFunctor<real, Dims> mortonOp(devBoundsPtr, devSourcePtr);
    thrust::transform(
        thrust::counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(numSources),
        mortonKeys.begin(),
        mortonOp
    );
    thrust::device_vector<unsigned int> sortedIndices(numSources);
    thrust::sequence(sortedIndices.begin(), sortedIndices.end());
    thrust::sort_by_key(mortonKeys.begin(), mortonKeys.end(), sortedIndices.begin());

    // Extend the sorted keys by their index to ensure that the keys are unique.
    thrust::device_vector<size_t> mortonKeys64(numSources);
    thrust::transform(mortonKeys.begin(), mortonKeys.end(),
        thrust::counting_iterator<unsigned int>(0), //radixTree.sortedIndices.begin(),
        mortonKeys64.begin(),
        Morton64Functor()
    );

    // Verify that the key sequence is unique
    /*thrust::device_vector<unsigned int> uniqueKeys(radixTree.sortedIndices.size());
    auto uniqueKeysEnd = thrust::unique_copy(radixTree.mortonKeys.begin(), radixTree.mortonKeys.end(), uniqueKeys.begin());
    size_t uniqueCount = std::distance(uniqueKeys.begin(), uniqueKeysEnd);
    if (uniqueCount != radixTree.mortonKeys.size())
    {

    }*/

    // Create leafs
    auto leafIndexFirst = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::counting_iterator<unsigned int>(0),
            sortedIndices.begin()
        )
    );
    auto leafIndexLast = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::counting_iterator<unsigned int>(numSources),
            sortedIndices.end()
        )
    );
    LeafFunctor<real, Dims> leafOp(radixTree.nodes, numSources);
    thrust::for_each(leafIndexFirst, leafIndexLast, leafOp);

    // Create branches
    BranchFunctor<real, Dims> branchOp(radixTree.nodes, mortonKeys64, numSources);
    thrust::for_each(
        thrust::counting_iterator<unsigned int>(0),
        thrust::counting_iterator<unsigned int>(numSources - 1),
        branchOp
    );


    // Bottom-up construction of the boundig boxes and expansion points 
    thrust::device_vector<unsigned int> treeDepth(1);
    thrust::device_vector<unsigned int> nodeLocks(numSources - 1, 0);
    MergeAlignedBoxesFunctor<real, Dims, HermiteDegree> mergeOp(
        radixTree,
        nodeLocks,
        treeDepth,
        devSourcePtr,
        numSources,
        h
    );
    thrust::for_each(thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(numSources), mergeOp);

    // Copy tree depth to host
    radixTree.depth = treeDepth[0];
    std::cout << "> buildRadixTree: radix tree depth is " << radixTree.depth << "\n";

    // Bottom-up computation of the hermite coeffcieitns
    ComputeHermiteFunctor<real, Dims, HermiteDegree> hermiteOp(
        radixTree, numSources, h
    );
    thrust::for_each(thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(numSources), hermiteOp);

    // Verify child/parent relationships
    /*thrust::host_vector<Node> hostNodes(radixTree.nodes);
    std::vector<Node> tmp(hostNodes.begin(), hostNodes.end());
    bool treeOk = verifyTree<real>(tmp);
    assert(treeOk);*/

    return;
}

using IndexIter = thrust::device_vector<unsigned int>::iterator;
template<typename real, unsigned int HermiteDegree>
void buildRadixTreeTransform(
    RadixTree<real, 2>& radixTree, 
    IndexIter indicesBegin, IndexIter indicesEnd, 
    const real* devBoundsPtr, 
    real width, real height, real focallength, 
    const real* devPointPtr, 
    real h)
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

    // Create leafs
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
    LeafFunctor<real, 2> leafOp(radixTree.nodes, numElements);
    thrust::for_each(leafIndexFirst, leafIndexLast, leafOp);

    // Create branches
    thrust::counting_iterator<unsigned int> branchIndicesBegin(0);
    thrust::counting_iterator<unsigned int> branchIndicesEnd(numElements - 1);
    BranchFunctor<real, 2> branchOp(radixTree.nodes, mortonKeys64, numElements);
    thrust::for_each(
        branchIndicesBegin,
        branchIndicesEnd,
        branchOp
    );


    // Bottom-up construction of the boundig boxes and expansion points 
    thrust::device_vector<unsigned int> treeDepth(1);
    thrust::device_vector<unsigned int> nodeLocks(numElements - 1, 0);
    MergeAlignedBoxesTransformFunctor<real, HermiteDegree> mergeOp(
        radixTree,
        nodeLocks,
        treeDepth,
        width, height, focallength,
        devPointPtr,
        numElements,
        h
    );
    thrust::counting_iterator<unsigned int> leafIndicesBegin(0);
    thrust::counting_iterator<unsigned int> leafIndicesEnd(numElements);
    thrust::for_each(leafIndicesBegin, leafIndicesEnd, mergeOp);

    // Copy tree depth to host
    radixTree.depth = treeDepth[0];
    std::cout << "> buildRadixTree: radix tree depth is " << radixTree.depth << "\n";

    // Bottom-up computation of the hermite coeffcieitns
    ComputeHermiteFunctor<real, 2, HermiteDegree> hermiteOp(
        radixTree, numElements, h
    );
    thrust::for_each(leafIndicesBegin, leafIndicesEnd, hermiteOp);

    return;
}