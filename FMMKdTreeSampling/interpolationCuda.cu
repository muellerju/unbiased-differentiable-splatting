#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t, typename indice_t>
__global__
void interpolationForwardKernel(
    scalar_t* interpolatedPtr,
    const indice_t* indicesPtr,
    const scalar_t* weightsPtr,
    unsigned int batchSize,
    unsigned int width,
    unsigned int height,
    unsigned int numSamples,
    const scalar_t* attributePtr,
    unsigned int attributeBatch,
    unsigned int numSources,
    unsigned int numChannel
)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int b = blockIdx.z * blockDim.z + threadIdx.z;

    if ((b < batchSize) && (x < width) && (y < height))
	{
        unsigned int pixelOffset = y + (height * (x + width * b));
        for(unsigned int s = 0; s < numSamples; s++)
        {
            unsigned int sampleOffset = s + numSamples * pixelOffset;
            indice_t index = indicesPtr[sampleOffset];
            scalar_t weight = weightsPtr[sampleOffset];
            if (index < 0)
                continue;

            for(unsigned int c = 0; c < numChannel; c++)
            {
                unsigned int attributeOffset = (attributeBatch > 1) ? c + numChannel * (index + numSources * b) : c + numChannel * index;
                scalar_t attribute = attributePtr[attributeOffset];
                
                unsigned int interpolatedOffset = c + numChannel * pixelOffset;
                atomicAdd(&interpolatedPtr[interpolatedOffset], weight*attribute);
            }
        }
    }
}

void cudaInterpolationForward(
	at::Tensor& interpolated,
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& attribute
)
{
    unsigned int batchSize = indices.sizes()[0];
	unsigned int width = indices.sizes()[1];
	unsigned int height = indices.sizes()[2];
    unsigned int numSamples = indices.sizes()[3];

    unsigned int attributeBatch = attribute.sizes()[0];
    unsigned int numSources = attribute.sizes()[1];
    unsigned int numChannel = attribute.sizes()[2];

    // Invoke the depth filtering forward kernel
    dim3 threads(16, 16, 1);
    dim3 grid((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, batchSize);

    AT_DISPATCH_FLOATING_TYPES(
        attribute.scalar_type(), "interpolationForwardKernel", ([&] {
            interpolationForwardKernel<scalar_t, int>
                <<<grid, threads>>> (
                    interpolated.data_ptr<scalar_t>(),
                    indices.data_ptr<int>(),
                    weights.data_ptr<scalar_t>(),
                    batchSize, width, height, numSamples,
                    attribute.data_ptr<scalar_t>(),
                    attributeBatch, numSources, numChannel
            );
        })
    );
}

template<typename scalar_t, typename indice_t>
__global__
void interpolationBackwardKernel(
    scalar_t* gradWeightsPtr,
    scalar_t* gradAttributePtr,
    const scalar_t* gradInPtr,
    const indice_t* indicesPtr,
    const scalar_t* weightsPtr,
    unsigned int batchSize,
    unsigned int width,
    unsigned int height,
    unsigned int numSamples,
    const scalar_t* attributePtr,
    unsigned int attributeBatch,
    unsigned int numSources,
    unsigned int numChannel
)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int b = blockIdx.z * blockDim.z + threadIdx.z;

    if ((b < batchSize) && (x < width) && (y < height))
	{
        unsigned int pixelOffset = y + (height * (x + width * b));
        for(unsigned int s = 0; s < numSamples; s++)
        {
            unsigned int sampleOffset = s + numSamples * pixelOffset;
            indice_t index = indicesPtr[sampleOffset];
            scalar_t weight = weightsPtr[sampleOffset];
            if (index < 0)
                continue;

            for(unsigned int c = 0; c < numChannel; c++)
            {
                unsigned int gradInOffset = c + numChannel * pixelOffset;
                scalar_t gradIn = gradInPtr[gradInOffset];

                unsigned int attributeOffset = (attributeBatch > 1) ? c + numChannel * (index + numSources * b) : c + numChannel * index;
                scalar_t attribute = attributePtr[attributeOffset];
                atomicAdd(&gradAttributePtr[attributeOffset], gradIn*weight);

                gradWeightsPtr[sampleOffset] += gradIn*attribute;
            }
        }
    }
}

void cudaInterpolationBackward(
    at::Tensor& gradWeights,
	at::Tensor& gradAttribute,
    const at::Tensor& gradIn,
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& attribute
)
{
    unsigned int batchSize = indices.sizes()[0];
	unsigned int width = indices.sizes()[1];
	unsigned int height = indices.sizes()[2];
    unsigned int numSamples = indices.sizes()[3];

    unsigned int attributeBatch = attribute.sizes()[0];
    unsigned int numSources = attribute.sizes()[1];
    unsigned int channel = attribute.sizes()[2];

    // Invoke the depth filtering forward kernel
    dim3 threads(16, 16, 1);
    dim3 grid((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, batchSize);

    AT_DISPATCH_FLOATING_TYPES(
        attribute.scalar_type(), "interpolationBackwardKernel", ([&] {
            interpolationBackwardKernel<scalar_t, int>
                <<<grid, threads>>> (
                    gradWeights.data_ptr<scalar_t>(),
                    gradAttribute.data_ptr<scalar_t>(),
                    gradIn.data_ptr<scalar_t>(),
                    indices.data_ptr<int>(),
                    weights.data_ptr<scalar_t>(),
                    batchSize, width, height, numSamples,
                    attribute.data_ptr<scalar_t>(),
                    attributeBatch, numSources, channel
            );
        })
    );
}