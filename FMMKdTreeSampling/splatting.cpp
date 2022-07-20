#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include<vector>

#include "logging.h"

#define DIMS 2
#define HERMITEDEGREE 6

///
// CUDA forward declarations
///

void cudaSplattingForward(
	at::Tensor weight,
	unsigned int width, unsigned int height,
	double focallength,
	const at::Tensor& indices,
	const at::Tensor& points,
	const at::Tensor& normals,
	const at::Tensor& stdDevs,
	double smoothing
);

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
);

///
// C++ functions
///

at::Tensor splattingForward(
	unsigned int width, unsigned int height,
	double focallength,
	const at::Tensor& indices, // Shape (bn, w, h, k)
	const at::Tensor& points, // Shape (bn, n, 3)
	const at::Tensor& normals, // Shape (bn, n, 3)
	const at::Tensor& stdDevs, // Shape (bn, n, 2) or Shape (1, n, 2)
	double smoothing
)
{
	// Verify that every input tensor has a contiguous memory layout.
	assert(indices.is_contiguous());
	assert(points.is_contiguous());
	assert(normals.is_contiguous());
	assert(stdDevs.is_contiguous());

	// Verify that every input tensor is a GPU tensor and on the same device
	assert(indices.device().type() == torch::kCUDA);
	assert(points.device().type() == torch::kCUDA);
	assert(normals.device().type() == torch::kCUDA);
	assert(stdDevs.device().type() == torch::kCUDA);
	assert( indices.get_device() == points.get_device()
		&& points.get_device() == normals.get_device()
		&& normals.get_device() == stdDevs.get_device());

	auto deviceIdx = points.get_device();
	auto weightOptions = torch::TensorOptions().dtype(points.scalar_type()).device(torch::kCUDA, deviceIdx);
	at::Tensor weights = at::empty(indices.sizes(), weightOptions);

	cudaSplattingForward(
		weights,
		width, height,
		focallength,
		indices,
		points, normals, stdDevs,
		smoothing
	);

	return weights;
}

std::vector<at::Tensor> splattingBackward(
	unsigned int width, unsigned int height,
	double focallength,
	const at::Tensor& gradIn, // Shape (bn, w, h, k)
	const at::Tensor& indices, // Shape (bn, w, h, k)
	const at::Tensor& points, // Shape (B, N, 3)
	const at::Tensor& normals, // Shape (B, N, 3)
	const at::Tensor& stdDevs, // Shape (B, N, 2) or (1, N, 2)
	double smoothing
)
{
	// Verify that every input tensor has a contiguous memory layout.
	assert(gradIn.is_contiguous());
	assert(indices.is_contiguous());
	assert(points.is_contiguous());
	assert(normals.is_contiguous());
	assert(stdDevs.is_contiguous());

	// Verify that every input tensor is a GPU tensor and on the same device
	assert(gradIn.device().type() == torch::kCUDA);
	assert(indices.device().type() == torch::kCUDA);
	assert(points.device().type() == torch::kCUDA);
	assert(normals.device().type() == torch::kCUDA);
	assert(stdDevs.device().type() == torch::kCUDA);
	assert( gradIn.get_device() == indices.get_device()
		&& indices.get_device() == points.get_device()
		&& points.get_device() == normals.get_device()
		&& normals.get_device() == stdDevs.get_device()
	);

	// Create output tensors for the gradients
	at::Tensor gradPoints = at::zeros(points.sizes(), points.options());
	at::Tensor gradNormals = at::zeros(normals.sizes(), normals.options());
	at::Tensor gradStdDevs = at::zeros(stdDevs.sizes(), stdDevs.options());

	cudaSplattingBackward(
		gradPoints, gradNormals, gradStdDevs,
		width, height, focallength,
		gradIn,
		indices,
		points, normals, stdDevs,
		smoothing
	);
	
	return { gradPoints, gradNormals, gradStdDevs };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	// module docstring
	m.doc() = "pybind11 splatting binding";
	m.def("forward", &splattingForward, "");
	m.def("backward", &splattingBackward, "");
}
