#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include<vector>

#include "transformation.h"
#include "logging.h"

///
// CUDA forward declarations
///

template<transform_t mode>
void cudaTransformForward(
	at::Tensor& transformedVector,
	const at::Tensor& transformation,
	const at::Tensor& vector
);

template<transform_t mode>
void cudaTransformBackward(
	at::Tensor& gradTransformation,
	at::Tensor& gradVector,
	const at::Tensor& gradIn,
	const at::Tensor& transformation,
	const at::Tensor& vector
);

///
// C++ functions
///


template<transform_t mode>
at::Tensor transformForward(
	const at::Tensor& transformation, // Shape (bn, 4, 4)
	const at::Tensor& vector // Shape (bn, n, 3) or (1, n, 3)
)
{
	// Verify that every input tensor has a contiguous memory layout.
	assert(transformation.is_contiguous());
	assert(vector.is_contiguous());
	
	// Verify that every input tensor is a GPU tensor and on the same device
	assert(transformation.device().type() == torch::kCUDA);
	assert(vector.device().type() == torch::kCUDA);
	assert(transformation.get_device() == vector.get_device());

	// Create output tensors for the transformed vector
	unsigned int batchSize = transformation.sizes()[0];
	unsigned int numElements = vector.sizes()[1];
	unsigned int dimension = vector.sizes()[2];

	std::array<int64_t, 3> outSizeBuffer = { batchSize, numElements, dimension };
	at::IntArrayRef outSize(outSizeBuffer.data(), 3);
	at::Tensor transformedVector = at::empty(outSize, vector.options());

	cudaTransformForward<mode>(transformedVector, transformation, vector);

	return transformedVector;
}

template<transform_t mode>
std::vector<at::Tensor> transformBackward(
	const at::Tensor& gradIn, // Shape (bn, n, 3)
	const at::Tensor& transformation, // Shape (bn, 4, 4)
	const at::Tensor& vector // Shape (bn, n, 3) or (1, n, 3)
)
{
	// Verify that every input tensor has a contiguous memory layout.
	assert(gradIn.is_contiguous());
	assert(transformation.is_contiguous());
	assert(vector.is_contiguous());

	// Verify that every input tensor is a GPU tensor and on the same device
	assert(transformation.device().type() == torch::kCUDA);
	assert(vector.device().type() == torch::kCUDA);
	assert(gradIn.get_device() == transformation.get_device() 
		&& transformation.get_device() == vector.get_device());

	at::Tensor gradTransformation = at::zeros(transformation.sizes(), transformation.options());
	at::Tensor gradVector = at::zeros(vector.sizes(), vector.options());

	cudaTransformBackward<mode>(gradTransformation, gradVector, gradIn, transformation, vector);

	return { gradTransformation, gradVector };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	// module docstring
	m.doc() = "pybind11 shading binding";
	m.def("affineForward", &transformForward<affine>, "");
	m.def("affineBackward", &transformBackward<affine>, "");
	m.def("linearForward", &transformForward<linear>, "");
	m.def("linearBackward", &transformBackward<linear>, "");
}