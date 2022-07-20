#include <torch/extension.h>
#include <cuda_runtime.h>

//#include "hashmap.cuh"
//#define MAPSIZE 1024

template<typename scalar_t, typename indice_t>
__global__
void scatterKernel(
    scalar_t* out,
    unsigned int width, unsigned int height,
    unsigned int numSamples, unsigned int numSources, unsigned int numChannels,
    const indice_t* indices, const scalar_t* values
)
{
    unsigned int b = blockIdx.x * blockDim.x;
    unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int y = blockIdx.z * blockDim.z + threadIdx.z;

    if ((x < width) && (y < height))
    {
        unsigned int pixelOffset = y + height * (x + width * b);
        for (unsigned int i = 0; i < numSamples; i++)
        {
            unsigned int sampleOffset = i + numSamples * pixelOffset;
            indice_t index = indices[sampleOffset];
            if (index < 0)
                continue;

            for (unsigned int c = 0; c < numChannels; c++)
            {
                unsigned int outOffset = c + numChannels * sampleOffset;
                out[outOffset] = values[c + numChannels * (index + numSources * b)];
            }
        }
    }
}

at::Tensor scatterCuda(const at::Tensor& indices, const at::Tensor& values)
{
    assert(indices.is_contiguous());
    assert(indices.device().type() == torch::kCUDA);
    assert(values.is_contiguous());
    assert(values.device().type() == torch::kCUDA);

    at::IntArrayRef valuesShape(values.sizes().begin(), values.sizes().end());
    unsigned int batchSize = valuesShape[0];
    unsigned int numSources = valuesShape[1];
    unsigned int numChannels = valuesShape[2];

    at::IntArrayRef indicesShape(indices.sizes().begin(), indices.sizes().end());
    unsigned int width = indicesShape[1];
    unsigned int height = indicesShape[2];
    unsigned int numSamples = indicesShape[3];

    // Create output tensor for the sampled indices
    std::array<int64_t, 5>  outSizeBuffer = { batchSize, width, height, numSamples, numChannels };
    at::IntArrayRef outSize(outSizeBuffer.data(), 5);
    at::Tensor gathered = at::zeros(outSize, values.options());

    dim3 threads(1, 32, 32);
    dim3 grid(batchSize, (width + threads.y - 1) / threads.y, (height + threads.z - 1) / threads.z);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        values.scalar_type(), "scatterKernel", ([&] {
            scatterKernel<scalar_t, int>
            << <grid, threads >> > (
                gathered.data_ptr<scalar_t>(),
                width, height,
                numSamples, numSources, numChannels,
                indices.data_ptr<int>(),
                values.data_ptr<scalar_t>()
                );
        })
    );


    return gathered;
}

template<typename scalar_t, typename indice_t>
__global__
void gatherKernel(
    scalar_t* out, 
    unsigned int width, unsigned int height, 
    unsigned int numSamples, unsigned int numSources, unsigned int numChannels, 
    const indice_t* indices, const scalar_t* values
)
{
    unsigned int b = blockIdx.x * blockDim.x;
    unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int y = blockIdx.z * blockDim.z + threadIdx.z;

    // Accumulate gradient in the shared memory map
    if ((x < width) && (y < height))
    {
        unsigned int pixelOffset = y + height * (x + width * b);
        for (unsigned int i = 0; i < numSamples; i++)
        {
            unsigned int sampleOffset = i + numSamples * pixelOffset;
            indice_t index = indices[sampleOffset];
            if (index < 0)
                continue;

            for (unsigned int c = 0; c < numChannels; c++)
            {
                unsigned int inOffset = c + numChannels * sampleOffset;
                unsigned int outOffset = c + numChannels * (index + numSources * b);
                atomicAdd(&out[outOffset], values[inOffset]);
            }
        }
    }
}

at::Tensor gatherCuda(const at::Tensor& indices, const at::Tensor& values, unsigned int numSources)
{
    assert(indices.is_contiguous());
    assert(indices.device().type() == torch::kCUDA);
    assert(values.is_contiguous());
    assert(values.device().type() == torch::kCUDA);


    at::IntArrayRef valuesInShape(values.sizes().begin(), values.sizes().end());
    unsigned int batchSize = valuesInShape[0];
    unsigned int width = valuesInShape[1];
    unsigned int height = valuesInShape[2];
    unsigned int numSamples = valuesInShape[3];
    unsigned int numChannels = valuesInShape[4];

    // Create output tensor for the sampled indices
    std::array<int64_t, 3> outSizeBuffer = { batchSize, numSources, numChannels };
    at::IntArrayRef outSize(outSizeBuffer.data(), 3);
    at::Tensor accumulated = at::zeros(outSize, values.options());

    dim3 threads(1, 32, 32);
    dim3 grid(batchSize, (width + threads.y - 1) / threads.y, (height + threads.z - 1) / threads.z);

    //AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    AT_DISPATCH_FLOATING_TYPES(
        values.scalar_type(), "gatherKernel", ([&] {
            gatherKernel<scalar_t, int>
                << <grid, threads >> > (
                    accumulated.data_ptr<scalar_t>(),
                    width, height,
                    numSamples, numSources, numChannels,
                    indices.data_ptr<int>(),
                    values.data_ptr<scalar_t>()
                    );
            }));

    return accumulated;
}

template<typename scalar_t, typename indice_t>
__global__
void gatherBackwardKernel(scalar_t* gradOut, unsigned int width, unsigned int height, unsigned int numSamples, unsigned int numSources, unsigned int numChannels, const scalar_t* gradIn, const indice_t* indices)
{
    unsigned int b = blockIdx.x * blockDim.x;
    unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int y = blockIdx.z * blockDim.z + threadIdx.z;

    // Initialize shared memory hash map
    /*__shared__ MapEntry<scalar_t> map[MAPSIZE];
    unsigned int mapIdx = threadIdx.z + blockDim.z * threadIdx.y;
    map[mapIdx].index = -1;
    mapIdx[mapIdx].value = 0.0;
    __syncthreads();*/

    // Accumulate gradient in the shared memory map
    if ((x < width) && (y < height))
    {
        unsigned int pixelOffset = y + height * (x + width * b);
        for (unsigned int i = 0; i < numSamples; i++)
        {
            unsigned int sampleOffset = i + numSamples * pixelOffset;
            indice_t index = indices[sampleOffset];
            if (index < 0)
                continue;
            
            for (unsigned int c = 0; c < numChannels; c++)
            {
                unsigned int inOffset = c + numChannels * sampleOffset;
                unsigned int outOffset = c + numChannels * (index + numSources * b);
                atomicAdd(&gradOut[outOffset], gradIn[inOffset]);
                //out[outOffset] = gradIn[c + numChannels * (index + numSources * b)];
            }
        }
    }

    // Add results to global memory
    /*__syncthreads();
    unsigned int sourceIdx = map[mapIdx].index;
    atomicAdd()*/
}

at::Tensor gatherBackwardCuda(const at::Tensor& gradIn, const at::Tensor& indices, unsigned int numSources)
{
    assert(gradIn.is_contiguous());
    assert(gradIn.device().type() == torch::kCUDA);
    assert(indices.is_contiguous());
    assert(indices.device().type() == torch::kCUDA);

    at::IntArrayRef gradInShape(gradIn.sizes().begin(), gradIn.sizes().end());
    unsigned int batchSize = gradInShape[0];
    unsigned int width = gradInShape[1];
    unsigned int height = gradInShape[2];
    unsigned int numSamples = gradInShape[3];
    unsigned int numChannels = gradInShape[4];

    // Create output tensor for the sampled indices
    std::array<int64_t, 3> outSizeBuffer = { batchSize, numSources, numChannels };
    at::IntArrayRef outSize(outSizeBuffer.data(), 3);
    at::Tensor accumulated = at::zeros(outSize, gradIn.options());

    dim3 threads(1, 32, 32);
    dim3 grid(batchSize, (width + threads.y - 1) / threads.y, (height + threads.z - 1) / threads.z);
    //unsigned int sharedMemSize = 1024 * sizeof();

    //AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    AT_DISPATCH_FLOATING_TYPES(
        gradIn.scalar_type(), "gatherBackwardKernel", ([&] {
            gatherBackwardKernel<scalar_t, int>
                << <grid, threads>> > (
                    accumulated.data_ptr<scalar_t>(),
                    width,
                    height,
                    numSamples,
                    numSources,
                    numChannels,
                    gradIn.data_ptr<scalar_t>(),
                    indices.data_ptr<int>()
                    );
            }));

    return accumulated;
}
