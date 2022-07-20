#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include<vector>

#include "logging.h"

///
// CUDA forward declarations
///
void cudaNearestSamples(
	at::Tensor& nearest,
	const at::Tensor& indices,
	const at::Tensor& weights,
	const at::Tensor& points,
	unsigned int numNearest
);

void cudaFastNearestSamples(
	at::Tensor& nearest,
	const at::Tensor& indices,
	const at::Tensor& weights,
	const at::Tensor& points,
	unsigned int numNearest
);

///
// C++ functions
///

at::Tensor nearestSamples(
	const at::Tensor& indices, // Shape (bn, w, h, k)
	const at::Tensor& weights, // Shape (bn, w, h, k)
	const at::Tensor& points, // Shape (B, N, 3)
	unsigned int numNearest
)
{
	// Verify that every input tensor has a contiguous memory layout.
	assert(indices.is_contiguous());
	assert(weights.is_contiguous());
	assert(points.is_contiguous());

	// Verify that every input tensor is a GPU tensor and on the same device
	assert(indices.device().type() == torch::kCUDA);
	assert(weights.device().type() == torch::kCUDA);
	assert(points.device().type() == torch::kCUDA);
	assert(indices.get_device() == weights.get_device()
		&& weights.get_device() == points.get_device());

	unsigned int batchSize = points.sizes()[0];
	unsigned int numPoints = points.sizes()[1];
	std::array<int64_t, 4> outBufferSize = { batchSize, numPoints, numNearest};

	// Create output tensors for nearest sample indices
	at::IntArrayRef outSize(outBufferSize.data(), 3);
	at::Tensor nearestSamples = at::empty(outSize, indices.options());
	nearestSamples.fill_(-1);

	/*cudaNearestSamples(
		nearestSamples,
		indices,
		weights,
		points,
		numNearest
	);*/

	cudaFastNearestSamples(
		nearestSamples,
		indices,
		weights,
		points,
		numNearest
	);

	return nearestSamples;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	// module docstring
	m.doc() = "pybind11 nearest samples binding";
	m.def("nearest", &nearestSamples, "");
}
