#include <torch/extension.h>
#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include "radixTree.cuh"
#include "buildRadixTree.cuh"
#include "sampleRadixTree.cuh"
//#include "maxRadixTree.cuh"

#include "ewaSplatting.cuh"
#include "ewaSplattingGrad.cuh"

#include "logging.h"

template<typename scalar_t, typename indice_t>
__global__
void ewaKernel(
	scalar_t* weights,
	unsigned int width, unsigned int height,
	scalar_t focallength,
	const indice_t* indices,
	unsigned int batchSize,
	unsigned int numSamples,
	const scalar_t* points,
	const scalar_t* normals,
	const scalar_t* stdDevs,
	unsigned int stdDevBatch,
	unsigned int numSources,
	scalar_t smoothing
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int b = blockIdx.z * blockDim.z + threadIdx.z;

	scalar_t pixelBuffer[] = { static_cast<scalar_t>(i), static_cast<scalar_t>(j) };
	const ConstVector2<scalar_t> pixel(pixelBuffer);

	if ((b < batchSize) &&(i < width) && (j < height))
	{
		for (unsigned int s = 0; s < numSamples; s++)
		{
			// Get the point index processed by the current thread 
			// and verify if it is a valid index
			unsigned int sampleOffset = s + numSamples * (j + height * (i + width * b));
			indice_t index = indices[sampleOffset];
			if (index < 0)
			{
				weights[sampleOffset] = 0.0;
				continue; // It might be fast to initialize the tensor with zero and to return here. This assumes that no valid index appears after the first invalid index.
			}

			unsigned int pointOffset = 3 * (index + numSources * b);
			const ConstVector3<scalar_t> point(&points[pointOffset]);
			const ConstVector3<scalar_t> normal(&normals[pointOffset]);

			unsigned int stdDevOffset = stdDevBatch > 1 ? 2 * (index + numSources * b) : 2 * index;
			const ConstVector2<scalar_t> stdDev(&stdDevs[stdDevOffset]);

			weights[sampleOffset] = evaluateEWAWeight(pixel, static_cast<scalar_t>(width), static_cast<scalar_t>(height), focallength, point, normal, stdDev, smoothing);
		}
	}
}


void cudaSplattingForward(
	at::Tensor weights,
	unsigned int width, unsigned int height,
	double focallength,
	const at::Tensor& indices,
	const at::Tensor& points,
	const at::Tensor& normals,
	const at::Tensor& stdDevs,
	double smoothing
)
{
	// Compute ewa gradients
	dim3 threads(16, 16, 1);
	dim3 grid((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, (indices.sizes()[0] + threads.z - 1) / threads.z);

	AT_DISPATCH_FLOATING_TYPES(
		points.scalar_type(), "ewaKernel", ([&] {
			ewaKernel<scalar_t, int> << <grid, threads >> > (
				weights.data_ptr<scalar_t>(),
				width, height, focallength,
				indices.data_ptr<int>(),
				indices.sizes()[0],
				indices.sizes()[3],
				points.data_ptr<scalar_t>(),
				normals.data_ptr<scalar_t>(),
				stdDevs.data_ptr<scalar_t>(),
				stdDevs.sizes()[0],
				stdDevs.sizes()[1],
				smoothing
				);
			})
	);
}

template<typename real, unsigned int size>
__device__
inline void atomicMultiplyAdd(real* target, const real* source, real value)
{
	for (unsigned int i = 0; i < size; i++)
	{
		//assert(isfinite(target[i]));
		atomicAdd(&target[i], source[i] * value);
	}
}

template<typename scalar_t, typename indice_t>
__global__
void gradEwaKernel(
	scalar_t* gradPoints,
	scalar_t* gradNormals,
	scalar_t* gradStdDevs,
	unsigned int width, unsigned int height,
	scalar_t focallength,
	const scalar_t* gradIn,
	const indice_t* indices,
	unsigned int batchSize,
	unsigned int numSamples,
	const scalar_t* points,
	const scalar_t* normals,
	const scalar_t* stdDevs,
	unsigned int stdDevBatch,
	unsigned int numSources,
	scalar_t smoothing
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int b = blockIdx.z * blockDim.z + threadIdx.z;
	
	scalar_t pixelBuffer[] = { static_cast<scalar_t>(i), static_cast<scalar_t>(j) };
	const ConstVector2<scalar_t> pixel(pixelBuffer);

	if ((b < batchSize) && (i < width) && (j < height))
	{
		for (unsigned int s = 0; s < numSamples; s++)
		{
			// Get the point index processed by the current thread 
			// and verify if it is a valid index
			unsigned int sampleOffset = s + numSamples * (j + height * (i + width * b));
			indice_t index = indices[sampleOffset];
			if (index < 0)
				continue; // It might be fast to initialize the tensor with zero and to return here. This assumes that no valid index appears after the first invalid index.


			unsigned int pointOffset = 3 * (index + numSources * b);
			const ConstVector3<scalar_t> point(&points[pointOffset]);
			const ConstVector3<scalar_t> normal(&normals[pointOffset]);
			scalar_t* gradOutPoint = &gradPoints[pointOffset];
			scalar_t* gradOutNormal = &gradNormals[pointOffset];

			unsigned int stdDevOffset = stdDevBatch > 1 ? 2 * (index + numSources * b) : 2 * index;
			const ConstVector2<scalar_t> stdDev(&stdDevs[stdDevOffset]);
			scalar_t* gradOutStdDev = &gradStdDevs[stdDevOffset];

			scalar_t widthf = static_cast<scalar_t>(width);
			scalar_t heightf = static_cast<scalar_t>(height);

			// Re-compute the ewa weight
			Vector2<scalar_t> screenpoint = toScreenspace(widthf, heightf, focallength, point);
			Matrix22<scalar_t> invJk = buildInverseJacobian<scalar_t>(widthf, heightf, focallength, screenpoint, point, normal);
			Matrix22<scalar_t> Vk = buildCovariance<scalar_t>(invJk, stdDev, smoothing);
			scalar_t weight = evaluteGaussian<scalar_t>(pixel, screenpoint, invJk, Vk);

			// Compute the gradient if the weight is not clamped
			if (weight <= 1.0)
			{
				Vector3<scalar_t> gradPoint, gradNormal;
				Vector2<scalar_t> gradStdDev;
				gradEWAWeight<scalar_t>(gradPoint, gradNormal, gradStdDev, pixel, widthf, heightf, focallength, point, normal, stdDev, smoothing);

				// Mulitply results with the gradient from the previous operations and store the results in global memory
				scalar_t gradWeight = gradIn[sampleOffset];
				atomicMultiplyAdd<scalar_t, 3>(gradOutPoint, gradPoint.data_ptr(), gradWeight);
				atomicMultiplyAdd<scalar_t, 3>(gradOutNormal, gradNormal.data_ptr(), gradWeight);
				atomicMultiplyAdd<scalar_t, 2>(gradOutStdDev, gradStdDev.data_ptr(), gradWeight);

				/*Vector3<real> gradPoint;
				gradEWAWeightSimplified<real>(gradPoint, pixel, widthf, heightf, focallength, point, normal, stdDev, smoothing);

				real gradWeight = gradWeights[sampleOffset];
				atomicMultiplyAdd<real, 3>(&gradPoints[index * 3], gradPoint.data_ptr(), gradWeight);*/
			}
		}
	}
}


void cudaSplattingBackward(
	at::Tensor& gradPoints,
	at::Tensor& gradNormals,
	at::Tensor& gradStdDevs,
	unsigned int width, unsigned int height,
	double focallength,
	const at::Tensor& gradIn,
	const at::Tensor& indices,
	const at::Tensor& points,
	const at::Tensor& normals,
	const at::Tensor& stdDevs,
	double smoothing
)
{
	// Compute ewa gradients
	dim3 threads(16, 16, 1);
	dim3 grid((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, (indices.sizes()[0] + threads.z - 1) / threads.z);

	AT_DISPATCH_FLOATING_TYPES(
		gradIn.scalar_type(), "gradEwaKernel", ([&] {
			gradEwaKernel<scalar_t, int> << <grid, threads >> > (
					gradPoints.data_ptr<scalar_t>(),
					gradNormals.data_ptr<scalar_t>(),
					gradStdDevs.data_ptr<scalar_t>(),
					width, height, focallength,
					gradIn.data_ptr<scalar_t>(),
					indices.data_ptr<int>(),
					indices.sizes()[0],
					indices.sizes()[3],
					points.data_ptr<scalar_t>(),
					normals.data_ptr<scalar_t>(),
					stdDevs.data_ptr<scalar_t>(),
					stdDevs.sizes()[0],
					stdDevs.sizes()[1],
					smoothing
				);
			})
	);
}