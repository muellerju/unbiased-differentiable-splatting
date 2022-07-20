#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

#include "logging.h"
#include "linalg.cuh"

#include "alphacompositing.cuh"
#include "alphacompositingGrad.cuh"

template<typename scalar_t, typename indice_t>
__global__
void alphaBlendForwardKernel(
    scalar_t* filteredPtr, 
    unsigned int numSamples,
    const indice_t* indicesPtr,
    const scalar_t* weightsPtr,
    unsigned int width, unsigned int height, scalar_t focallength,
    unsigned int numPointBatches, unsigned int numSources,
    const scalar_t* pointsPtr,
    const scalar_t* normalsPtr
)
{
    extern __shared__ indice_t sharedIndices[];
    unsigned int b = blockIdx.x * blockDim.x;
    unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int y = blockIdx.z * blockDim.z + threadIdx.z;

    scalar_t pixelBuffer[] = { static_cast<scalar_t>(x), static_cast<scalar_t>(y) };
    const ConstVector2<scalar_t> pixel(pixelBuffer);

    unsigned int sharedOffset = numSamples*(threadIdx.z + blockDim.z * threadIdx.y);
    indice_t* sortedIndices = &sharedIndices[sharedOffset];
    for (unsigned int i = 0; i < numSamples; i++)
        sortedIndices[i] = i;

    unsigned int entryOffset = numSamples * (y + (height * (x + width * b)));
    scalar_t* filtered = &filteredPtr[entryOffset];
    const indice_t* indices = &indicesPtr[entryOffset];
    const scalar_t* weights = &weightsPtr[entryOffset];

    unsigned int pointOffset = numPointBatches > 1 ? 3 * numSources * b : 0;
    const scalar_t* points = &pointsPtr[pointOffset];
    const scalar_t* normals = &normalsPtr[pointOffset];

    if ((x < width) && (y < height))
    {
        // Transform to camera space and compute transformations
        signedDistances(filtered, 
            pixel,
            static_cast<scalar_t>(width), static_cast<scalar_t>(height), focallength,
            numSamples,
            indices,
            points, normals);

        // Sort the depth values
        argSort(sortedIndices, filtered, numSamples);

        // Perform depth blending
        blending(filtered, sortedIndices, weights, numSamples);
    }
}

at::Tensor cudaAlphaBlendForward(
    const at::Tensor& indices, 
    const at::Tensor& weights,
    unsigned int width, unsigned int height, double focallength,
    const at::Tensor& points, 
    const at::Tensor& normals
)
{
    assert(indices.is_contiguous());
    assert(weights.is_contiguous());
    assert(points.is_contiguous());
    assert(normals.is_contiguous());

    assert(indices.device().type() == torch::kCUDA);
    assert(weights.device().type() == torch::kCUDA);
     assert(points.device().type() == torch::kCUDA);
    assert(normals.device().type() == torch::kCUDA);

    assert(indices.get_device() == weights.get_device()
        && weights.get_device() == points.get_device()
        && points.get_device() == normals.get_device());

    // Allocate tensor for the filtered weights
    unsigned int batchSize = indices.sizes()[0];
    unsigned int numSamples = indices.sizes()[3];
    unsigned int numPointBatches = points.sizes()[0];
    unsigned int numSources = points.sizes()[1];
    at::Tensor filtered = at::empty(weights.sizes(), weights.options());

    // Invoke the depth filtering forward kernel
    dim3 threads(1, 16, 16);
    dim3 grid(batchSize, (width + threads.y - 1) / threads.y, (height + threads.z - 1) / threads.z);

    unsigned int sharedBufferSize = sizeof(int) * numSamples * threads.x * threads.y * threads.z;
    logging::log<LOG_DEBUG>("cudaAlphaBlendForward", "shared buffer size", sharedBufferSize);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.scalar_type(), "alphaBlendForwardKernel", ([&] {
            alphaBlendForwardKernel<scalar_t, int>
                << <grid, threads, sharedBufferSize>> > (
                    filtered.data_ptr<scalar_t>(),
                    numSamples,
                    indices.data_ptr<int>(),
                    weights.data_ptr<scalar_t>(),
                    width, height, focallength,
                    numPointBatches, numSources,
                    points.data_ptr<scalar_t>(),
                    normals.data_ptr<scalar_t>()
                    );
            })
    );

#ifndef NDEBUG
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
        logging::log<LOG_CRITICAL>("cudaAlphaBlendForward", "alphaBlendForwardKernel failed with status", cudaStatus);
#endif // !NDEBUG

    return filtered;
}


template<typename scalar_t, typename indice_t>
__global__
void alphaBlendBackwardKernel(
    scalar_t* gradOutPtr, 
    unsigned int numSamples,
    const indice_t* indicesPtr,
    const scalar_t* weightsPtr,
    const scalar_t* gradInPtr,
    unsigned int width, unsigned int height, scalar_t focallength,
    unsigned int numPointBatches, unsigned int numSources,
    const scalar_t* pointsPtr,
    const scalar_t* normalsPtr
)
{
    extern __shared__ indice_t sharedIndices[];
    unsigned int b = blockIdx.x * blockDim.x;
    unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int y = blockIdx.z * blockDim.z + threadIdx.z;

    scalar_t pixelBuffer[] = { static_cast<scalar_t>(x), static_cast<scalar_t>(y) };
    const ConstVector2<scalar_t> pixel(pixelBuffer);

    unsigned int sharedOffset = numSamples*(threadIdx.z + blockDim.z * threadIdx.y);
    indice_t* sortedIndices = &sharedIndices[sharedOffset];
    for (unsigned int i = 0; i < numSamples; i++)
        sortedIndices[i] = i;

    unsigned int entryOffset = numSamples * (y + (height * (x + width * b)));
    scalar_t* gradOut = &gradOutPtr[entryOffset];
    const indice_t* indices = &indicesPtr[entryOffset];
    const scalar_t* weights = &weightsPtr[entryOffset];
    const scalar_t* gradIn = &gradInPtr[entryOffset];
    
    unsigned int pointOffset = numPointBatches > 1 ? 3 * numSources * b : 0;
    const scalar_t* points = &pointsPtr[pointOffset];
    const scalar_t* normals = &normalsPtr[pointOffset];

    if ((x < width) && (y < height))
    {
        // Re-compute parts of the forward pass
        signedDistances(gradOut, 
            pixel,
            static_cast<scalar_t>(width), static_cast<scalar_t>(height), focallength,
            numSamples,
            indices,
            points, normals);

        argSort(sortedIndices, gradOut, numSamples);

        // Compute gradient
        gradBlending(gradOut, sortedIndices, weights, gradIn, numSamples);
    }
}

at::Tensor cudaAlphaBlendBackward(
    const at::Tensor& indices, 
    const at::Tensor& weights,
    const at::Tensor& gradIn,
    unsigned int width, unsigned int height, double focallength,
    const at::Tensor& points, 
    const at::Tensor& normals    
)
{
    assert(indices.is_contiguous());
    assert(weights.is_contiguous());
    assert(gradIn.is_contiguous());
    assert(points.is_contiguous());
    assert(normals.is_contiguous());

    assert(indices.device().type() == torch::kCUDA);
    assert(weights.device().type() == torch::kCUDA);
    assert(gradIn.device().type() == torch::kCUDA);
    assert(points.device().type() == torch::kCUDA);
    assert(normals.device().type() == torch::kCUDA);

    assert(indices.get_device() == weights.get_device()
        && weights.get_device() == gradIn.get_device()
        && gradIn.get_device() == points.get_device()
        && points.get_device() == normals.get_device());

    // Allocate tensor for the output gradient
    unsigned int batchSize = indices.sizes()[0];
    unsigned int numSamples = indices.sizes()[3];
    unsigned int numPointBatches = points.sizes()[0];
    unsigned int numSources = points.sizes()[1];
    at::Tensor gradOut = at::zeros(gradIn.sizes(), gradIn.options());

     // Invoke the backward kernel
     dim3 threads(1, 16, 16);
     dim3 grid(batchSize, (width + threads.y - 1) / threads.y, (height + threads.z - 1) / threads.z);
 
     unsigned int sharedBufferSize = sizeof(int) * numSamples * threads.x * threads.y * threads.z;
     logging::log<LOG_DEBUG>("cudaAlphaBlendForward", "shared buffer size", sharedBufferSize);
 
     AT_DISPATCH_FLOATING_TYPES_AND_HALF(
         weights.scalar_type(), "alphaBlendBackwardKernel", ([&] {
            alphaBlendBackwardKernel<scalar_t, int>
                 << <grid, threads, sharedBufferSize>> > (
                     gradOut.data_ptr<scalar_t>(),
                     numSamples,
                     indices.data_ptr<int>(),
                     weights.data_ptr<scalar_t>(),
                     gradIn.data_ptr<scalar_t>(),
                     width, height, focallength,
                     numPointBatches, numSources,
                     points.data_ptr<scalar_t>(),
                     normals.data_ptr<scalar_t>()
                     );
             })
     );
 
#ifndef NDEBUG
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
        logging::log<LOG_CRITICAL>("cudaAlphaBlendBackward", "alphaBlendBackwardKernel failed with status", cudaStatus);
#endif // !NDEBUG

    return gradOut;
}