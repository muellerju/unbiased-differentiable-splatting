#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

#include "diffuseShading.cuh"
#include "diffuseShadingGrad.cuh"

///
// Forward pass
///

template<typename scalar_t>
__global__
void shadingForwardKernel(
	scalar_t* images,
	unsigned int batchSize, unsigned int witdh, unsigned int height,
	const scalar_t* directionsPtr,
	const scalar_t* intensitiesPtr,
	unsigned int intensitiesBatch, unsigned int numLights,
	const scalar_t* normalsPtr,
	const scalar_t* diffusePtr
)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int b = blockIdx.z * blockDim.z + threadIdx.z;

	if ((x < witdh) && (y < height))
	{
		unsigned int pixelOffset = 3 * (y + (height * (x + witdh * b)));
		PointerVector3<scalar_t> pixel(&images[pixelOffset]);
		const ConstVector3<scalar_t> normals(&normalsPtr[pixelOffset]);
		const ConstVector3<scalar_t> diffuse(&diffusePtr[pixelOffset]);
		
		unsigned int directionsOffset = 3 * numLights * b;
		const scalar_t* directions = &directionsPtr[directionsOffset];
		unsigned int intensitiesOffset = intensitiesBatch > 1 ? 3 * numLights * b : 0;
		const scalar_t* intensities = &intensitiesPtr[intensitiesOffset];

		pixel = shading(directions, intensities, numLights, normals, diffuse);
	}
}

void cudaShadingForward(
	at::Tensor& images,
	const at::Tensor& directions,
	const at::Tensor& intensities,
	const at::Tensor& normals,
	const at::Tensor& diffuse
)
{
	unsigned int batchSize = images.sizes()[0];
	unsigned int width = images.sizes()[1];
	unsigned int height = images.sizes()[2];

	unsigned int intensitiesBatch = intensities.sizes()[0];
	unsigned int numLights = intensities.sizes()[1];

	// Invoke the depth filtering forward kernel
	dim3 threads(16, 16, 1);
	dim3 grid((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, batchSize);

    AT_DISPATCH_FLOATING_TYPES(
		images.scalar_type(), "shadingForwardKernel", ([&] {
			shadingForwardKernel<scalar_t>
                << <grid, threads >> > (
					images.data_ptr<scalar_t>(),
					batchSize, width, height,
					directions.data_ptr<scalar_t>(),
					intensities.data_ptr<scalar_t>(),
					intensitiesBatch, numLights,
					normals.data_ptr<scalar_t>(),
					diffuse.data_ptr<scalar_t>()
            );
        })
    );
}

///
// Backward pass
///

template<typename scalar_t>
__global__
void shadingBackwardKernel(
	scalar_t* gradOutDirectionsPtr, // Shape (bn, l ,3)
	scalar_t* gradOutIntensitiesPtr, // Shape (bn, l , 3) or (1, l, 3)
	scalar_t* gradOutNormalsPtr, // Shape (bn, w, h, 3)
	scalar_t* gradOutDiffusePtr, // Shape (bn, w, h, 3)
	const scalar_t* gradInPtr, // Shape (bn, w, h, 3)
	unsigned int batchSize, unsigned int witdh, unsigned int height,
	const scalar_t* directionsPtr, // Shape (bn, l ,3)
	const scalar_t* intensitiesPtr, // Shape (bn, l , 3) or (1, l, 3)
	unsigned int intensitiesBatch, unsigned int numLights,
	const scalar_t* normalsPtr, // Shape (bn, w, h, 3)
	const scalar_t* diffusePtr // Shape (bn, w, h, 3)
)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int b = blockIdx.z * blockDim.z + threadIdx.z;

	if ((x < witdh) && (y < height))
	{
		unsigned int pixelOffset = 3 * (y + (height * (x + witdh * b)));
		const ConstVector3<scalar_t> gradIn(&gradInPtr[pixelOffset]);

		// Compute gradient point offsets
		PointerVector3<scalar_t> gradOutNormal(&gradOutNormalsPtr[pixelOffset]);
		PointerVector3<scalar_t> gradOutDiffuse(&gradOutDiffusePtr[pixelOffset]);
		const ConstVector3<scalar_t> normals(&normalsPtr[pixelOffset]);
		const ConstVector3<scalar_t> diffuse(&diffusePtr[pixelOffset]);


		// Compute input pointer offsets
		unsigned int directionsOffset = 3 * numLights * b;
		scalar_t* gradOutDirections = &gradOutDirectionsPtr[directionsOffset];
		const scalar_t* directions = &directionsPtr[directionsOffset];

		unsigned int intensitiesOffset = intensitiesBatch > 1 ? 3 * numLights * b : 0;
		scalar_t* gradOutIntensities = &gradOutIntensitiesPtr[intensitiesOffset];
		const scalar_t* intensities = &intensitiesPtr[intensitiesOffset];

		gradShading(
			gradOutDirections, gradOutIntensities, gradOutNormal, gradOutDiffuse,
			gradIn, directions, intensities, numLights, normals, diffuse);
	}
}

void cudaShadingBackward(
	at::Tensor& gradOutDirections,
	at::Tensor& gradOutIntensities,
	at::Tensor& gradOutNormals,
	at::Tensor& gradOutDiffuse,
	const at::Tensor& gradIn,
	const at::Tensor& directions,
	const at::Tensor& intensities,
	const at::Tensor& normals,
	const at::Tensor& diffuse
)
{
	unsigned int batchSize = normals.sizes()[0];
	unsigned int width = normals.sizes()[1];
	unsigned int height = normals.sizes()[2];

	unsigned int intensitiesBatch = intensities.sizes()[0];
	unsigned int numLights = intensities.sizes()[1];

	// Invoke the depth filtering forward kernel
	dim3 threads(16, 16, 1);
	dim3 grid((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, batchSize);

	AT_DISPATCH_FLOATING_TYPES(
		gradIn.scalar_type(), "shadingBackwardKernel", ([&] {
			shadingBackwardKernel<scalar_t>
				<< <grid, threads >> > (
					gradOutDirections.data_ptr<scalar_t>(),
					gradOutIntensities.data_ptr<scalar_t>(),
					gradOutNormals.data_ptr<scalar_t>(),
					gradOutDiffuse.data_ptr<scalar_t>(),
					gradIn.data_ptr<scalar_t>(),
					batchSize, width, height,
					directions.data_ptr<scalar_t>(),
					intensities.data_ptr<scalar_t>(),
					intensitiesBatch, numLights,
					normals.data_ptr<scalar_t>(),
					diffuse.data_ptr<scalar_t>()
			);
		})
	);
}

///
// SH forward pass
///

template<typename scalar_t>
__global__
void shadingShForwardKernel(
	scalar_t* images,
	unsigned int batchSize, unsigned int witdh, unsigned int height,
	const scalar_t* shCoefficients,
	unsigned int coefficientsBatch,
	const scalar_t* normalsPtr,
	const scalar_t* diffusePtr
)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int b = blockIdx.z * blockDim.z + threadIdx.z;

	if ((x < witdh) && (y < height))
	{
		unsigned int pixelOffset = 3 * (y + (height * (x + witdh * b)));
		PointerVector3<scalar_t> pixel(&images[pixelOffset]);
		const ConstVector3<scalar_t> normal(&normalsPtr[pixelOffset]);
		const ConstVector3<scalar_t> diffuse(&diffusePtr[pixelOffset]);

		for (unsigned int i = 0; i < 3; i++)
		{
			unsigned int coefficientsOffset = coefficientsBatch > 1 ? 9 * (i + 3 * b) : 9 * i;
			const scalar_t* coefficients = &shCoefficients[coefficientsOffset];
			pixel[i] = diffuse[i] * shadingSh(coefficients, normal) / M_PI;
		}
	}
}

void cudaShadingShForward(
	at::Tensor& images,
	const at::Tensor& shCoefficients,
	const at::Tensor& normals,
	const at::Tensor& diffuse
)
{
	unsigned int batchSize = images.sizes()[0];
	unsigned int width = images.sizes()[1];
	unsigned int height = images.sizes()[2];

	unsigned int coefficientsBatch = shCoefficients.sizes()[0];

	// Invoke the depth filtering forward kernel
	dim3 threads(16, 16, 1);
	dim3 grid((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, batchSize);

	AT_DISPATCH_FLOATING_TYPES(
		images.scalar_type(), "shadingShForwardKernel", ([&] {
			shadingShForwardKernel<scalar_t>
				<< <grid, threads >> > (
					images.data_ptr<scalar_t>(),
					batchSize, width, height,
					shCoefficients.data_ptr<scalar_t>(),
					coefficientsBatch,
					normals.data_ptr<scalar_t>(),
					diffuse.data_ptr<scalar_t>()
					);
			})
	);
}

///
// Sh backward pass
///

template<typename scalar_t>
__global__
void shadingShBackwardKernel(
	scalar_t* gradOutShCoefficients, // Shape (bn, l , 3) or (1, l, 3)
	scalar_t* gradOutNormalsPtr, // Shape (bn, w, h, 3)
	scalar_t* gradOutDiffusePtr, // Shape (bn, w, h, 3)
	const scalar_t* gradInPtr, // Shape (bn, w, h, 3)
	unsigned int batchSize, unsigned int witdh, unsigned int height,
	const scalar_t* shCoefficients,
	unsigned int coefficientsBatch,
	const scalar_t* normalsPtr, // Shape (bn, w, h, 3)
	const scalar_t* diffusePtr // Shape (bn, w, h, 3)
)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int b = blockIdx.z * blockDim.z + threadIdx.z;

	if ((x < witdh) && (y < height))
	{
		unsigned int pixelOffset = 3 * (y + (height * (x + witdh * b)));
		const ConstVector3<scalar_t> gradIn(&gradInPtr[pixelOffset]);

		// Compute gradient point offsets
		PointerVector3<scalar_t> gradOutNormal(&gradOutNormalsPtr[pixelOffset]);
		PointerVector3<scalar_t> gradOutDiffuse(&gradOutDiffusePtr[pixelOffset]);
		const ConstVector3<scalar_t> normal(&normalsPtr[pixelOffset]);
		const ConstVector3<scalar_t> diffuse(&diffusePtr[pixelOffset]);

		for (unsigned int i = 0; i < 3; i++)
		{
			// Compute input pointer offsets
			unsigned int coefficientsOffset = coefficientsBatch > 1 ? 9 * (i + 3 * b) : 9 * i;
			const scalar_t* coefficients = &shCoefficients[coefficientsOffset];
			scalar_t* gradOutCoefficients = &gradOutShCoefficients[coefficientsOffset];

			scalar_t irradiance = shadingSh(coefficients, normal);

			scalar_t gradShCoefficients[9];
			Vector3<scalar_t> gradNormal;
			gradShadingSh(gradShCoefficients, gradNormal, coefficients, normal);

			gradOutDiffuse[i] = gradIn[i] * irradiance / M_PI;
			gradOutNormal[i] = gradIn[i] * diffuse[i] * gradNormal[i] / M_PI;
			for (unsigned int j = 0; j < 9; j++)
				atomicAdd(&gradOutCoefficients[j], gradIn[i] * diffuse[i] * gradShCoefficients[j] / M_PI);
		}

	}
}

void cudaShadingShBackward(
	at::Tensor& gradOutShCoefficients,
	at::Tensor& gradOutNormals,
	at::Tensor& gradOutDiffuse,
	const at::Tensor& gradIn,
	const at::Tensor& shCoefficients,
	const at::Tensor& normals,
	const at::Tensor& diffuse
)
{
	unsigned int batchSize = normals.sizes()[0];
	unsigned int width = normals.sizes()[1];
	unsigned int height = normals.sizes()[2];

	unsigned int coefficientsBatch = shCoefficients.sizes()[0];

	// Invoke the depth filtering forward kernel
	dim3 threads(16, 16, 1);
	dim3 grid((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, batchSize);

	AT_DISPATCH_FLOATING_TYPES(
		gradIn.scalar_type(), "shadingShBackwardKernel", ([&] {
			shadingShBackwardKernel<scalar_t>
				<< <grid, threads >> > (
					gradOutShCoefficients.data_ptr<scalar_t>(),
					gradOutNormals.data_ptr<scalar_t>(),
					gradOutDiffuse.data_ptr<scalar_t>(),
					gradIn.data_ptr<scalar_t>(),
					batchSize, width, height,
					shCoefficients.data_ptr<scalar_t>(),
					coefficientsBatch,
					normals.data_ptr<scalar_t>(),
					diffuse.data_ptr<scalar_t>()
					);
			})
	);
}