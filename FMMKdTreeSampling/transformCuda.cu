#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

#include "transformation.h"
#include "transformation.cuh"
#include "transformationGrad.cuh"
#include "linalg.cuh"
#include "logging.h"

template<typename scalar_t, transform_t mode>
__global__ void transformForwardKernel(
	scalar_t* transformedPtr,
	const scalar_t* transformationPtr,
	unsigned int batchSize,
	const scalar_t* vectorPtr,
	unsigned int numVectorBatches, 
	unsigned int numSources
)
{
	unsigned int b = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (b < batchSize && x < numSources)
	{
		unsigned int transformedOffset = 3 * (x + numSources * b);
		PointerVector3<scalar_t> transformed(&transformedPtr[transformedOffset]);

		unsigned int transformationOffset = 16 * b;
		const ConstMatrix44<scalar_t> transformation(&transformationPtr[transformationOffset]);

		unsigned int vectorOffset = numVectorBatches > 1 ? 3 * (x + numSources * b) : 3 * x;
		const ConstVector3<scalar_t> vector(&vectorPtr[vectorOffset]);

		transformed = toCameraspace<scalar_t, mode>(transformation, vector);
	}
}

template<transform_t mode>
void cudaTransformForward(
	at::Tensor& transformedVector,
	const at::Tensor& transformation,
	const at::Tensor& vector
)
{
	unsigned int batchSize = transformation.sizes()[0];
	unsigned int batchVectorSize = vector.sizes()[0];
	unsigned int numSources = vector.sizes()[1];

	dim3 threads(1024, 1, 1);
	dim3 grid((numSources + threads.x - 1)/ threads.x, batchSize, 1);

	AT_DISPATCH_FLOATING_TYPES(
		vector.scalar_type(), "transformForwardKernel", ([&] {
			transformForwardKernel<scalar_t, mode>
				<< <grid, threads>> > (
					transformedVector.data_ptr<scalar_t>(),
					transformation.data_ptr<scalar_t>(),
					batchSize,
					vector.data_ptr<scalar_t>(),
					batchVectorSize,
					numSources
					);
			})
	);

#ifndef NDEBUG
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		logging::log<LOG_CRITICAL>("cudaTransformForward", "transformForwardKernel failed with status", cudaStatus);
#endif // !NDEBUG
}

template<typename scalar_t, transform_t mode>
__global__ void transformBackwardKernel(
	scalar_t* gradTransformationPtr,
	scalar_t* gradVectorPtr,
	const scalar_t* gradInPtr,
	const scalar_t* transformationPtr,
	unsigned int batchSize,
	const scalar_t* vectorPtr,
	unsigned int numVectorBatches,
	unsigned int numSources
)
{
	unsigned int b = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (b < batchSize && x < numSources)
	{
		// Compute offset and pointers
		unsigned int transformedOffset = 3 * (x + numSources * b);
		const ConstVector3<scalar_t> gradIn(&gradInPtr[transformedOffset]);

		unsigned int transformationOffset = 16 * b;
		scalar_t* gradOutTransformation = &gradTransformationPtr[transformationOffset];
		const ConstMatrix44<scalar_t> transformation(&transformationPtr[transformationOffset]);

		unsigned int vectorOffset = numVectorBatches > 1 ? 3 * (x + numSources * b) : 3 * x;
		scalar_t* gradOutVector = &gradVectorPtr[vectorOffset];
		const ConstVector3<scalar_t> vector(&vectorPtr[vectorOffset]);

		// Compute transformation grad 
		Tensor443<scalar_t> gradTransformation;
		Matrix33<scalar_t> gradVector;
		gradCameraspace<scalar_t, mode>(gradTransformation, gradVector, transformation, vector);
		Matrix44<scalar_t> gradChainTransformation = gradIn * gradTransformation;
		Vector3<scalar_t> gradChainVector = gradIn * gradVector;

		// Accumulate gradient over points
		for (unsigned int i = 0; i < 16; i++)
			atomicAdd(&gradOutTransformation[i], gradChainTransformation[i]);

		for (unsigned int i = 0; i < 3; i++)
			atomicAdd(&gradOutVector[i], gradChainVector[i]);
	}
}

template<transform_t mode>
void cudaTransformBackward(
	at::Tensor& gradTransformation,
	at::Tensor& gradVector,
	const at::Tensor& gradIn,
	const at::Tensor& transformation,
	const at::Tensor& vector
)
{
	unsigned int batchSize = transformation.sizes()[0];
	unsigned int batchVectorSize = vector.sizes()[0];
	unsigned int numElements = vector.sizes()[1];

	dim3 threads(1024, 1, 1);
	dim3 grid((numElements + threads.x - 1) / threads.x, batchSize, 1);

	AT_DISPATCH_FLOATING_TYPES(
		vector.scalar_type(), "transformBackwardKernel", ([&] {
			transformBackwardKernel<scalar_t, mode>
				<< <grid, threads >> > (
					gradTransformation.data_ptr<scalar_t>(),
					gradVector.data_ptr<scalar_t>(),
					gradIn.data_ptr<scalar_t>(),
					transformation.data_ptr<scalar_t>(),
					batchSize,
					vector.data_ptr<scalar_t>(),
					batchVectorSize,
					numElements
					);
			})
	);

#ifndef NDEBUG
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		logging::log<LOG_CRITICAL>("cudaTransformBackward", "transformBackwardKernel failed with status", cudaStatus);
#endif // !NDEBUG
}

///
// Explicit template instantiation
///

template void cudaTransformForward<affine>(
	at::Tensor& transformedVector,
	const at::Tensor& transformation,
	const at::Tensor& vector
);

template void cudaTransformBackward<affine>(
	at::Tensor& gradTransformation,
	at::Tensor& gradVector,
	const at::Tensor& gradIn,
	const at::Tensor& transformation,
	const at::Tensor& vector
);

template void cudaTransformForward<linear>(
	at::Tensor& transformedVector,
	const at::Tensor& transformation,
	const at::Tensor& vector
);

template void cudaTransformBackward<linear>(
	at::Tensor& gradTransformation,
	at::Tensor& gradVector,
	const at::Tensor& gradIn,
	const at::Tensor& transformation,
	const at::Tensor& vector
);
