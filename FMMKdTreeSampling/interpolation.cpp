#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

#include "logging.h"

///
// CUDA forward declarations
///

void cudaInterpolationForward(
	at::Tensor& interpolated,
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& attribute
);

void cudaInterpolationBackward(
    at::Tensor& gradWeights,
	at::Tensor& gradAttribute,
    const at::Tensor& gradIn,
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& attribute
);

///
// C++ functions
///

at::Tensor interpolationForward(
    const at::Tensor& indices, // Shape (bn, w, h, k)
    const at::Tensor& weights, // Shape (bn, w, h, k)
    const at::Tensor& attribute // Shape (bn, n, c) or (1, n, c)
)
{
    // Verify that every input tensor has a contiguous memory layout.
	assert(indices.is_contiguous());
	assert(weights.is_contiguous());
	assert(attribute.is_contiguous());
	
	// Verify that every input tensor is a GPU tensor and on the same device
	assert(indices.device().type() == torch::kCUDA);
	assert(weights.device().type() == torch::kCUDA);
	assert(attribute.device().type() == torch::kCUDA);
	assert(indices.get_device() == weights.get_device()
		&& weights.get_device() ==  attribute.get_device());

    // Create output tensor for the interpolated attribute
	unsigned int batchSize = indices.sizes()[0];
	unsigned int width = indices.sizes()[1];
	unsigned int height = indices.sizes()[2];
    unsigned int channel = attribute.sizes()[2];
	std::array<int64_t, 4> outSizeBuffer = { batchSize, width, height, channel };
	at::IntArrayRef outSize(outSizeBuffer.data(), 4);
    at::Tensor interpolated = at::zeros(outSize, attribute.options());

    cudaInterpolationForward(interpolated, indices, weights, attribute);

    return interpolated;
}

std::vector<at::Tensor> interpolationBackward(
	const at::Tensor& gradIn, // Shape (bn, w, h, c)
	const at::Tensor& indices, // Shape (bn, w, h, k)
    const at::Tensor& weights, // Shape (bn, w, h, k)
    const at::Tensor& attribute // Shape (bn, n, c) or (1, n, c)
)
{
    // Verify that every input tensor has a contiguous memory layout.
    assert(gradIn.is_contiguous());
	assert(indices.is_contiguous());
	assert(weights.is_contiguous());
	assert(attribute.is_contiguous());
	
	// Verify that every input tensor is a GPU tensor and on the same device
    assert(gradIn.device().type() == torch::kCUDA);
	assert(indices.device().type() == torch::kCUDA);
	assert(weights.device().type() == torch::kCUDA);
	assert(attribute.device().type() == torch::kCUDA);
	assert( gradIn.get_device() == indices.get_device()
        && indices.get_device() == weights.get_device()
		&& weights.get_device() ==  attribute.get_device());

    // Create output tensor for the weight and attribute gradients
    at::Tensor gradWeights = at::zeros(weights.sizes(), weights.options());
    at::Tensor gradAttribute = at::zeros(attribute.sizes(), attribute.options());

    cudaInterpolationBackward(
        gradWeights, gradAttribute,
        gradIn, indices, weights, attribute
    );

    return {gradWeights, gradAttribute};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	// module docstring
	m.doc() = "pybind11 interpolation binding";
	m.def("forward", &interpolationForward, "");
	m.def("backward", &interpolationBackward, "");
}