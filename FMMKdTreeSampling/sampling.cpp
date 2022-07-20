#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "logging.h"

///
// CUDA forward declarations
///

void cudaSample(
	at::Tensor& indices,
	unsigned int width, unsigned int height,
	double focallength,
	const at::Tensor& points, // Shape (bn, n, 3)
	const at::Tensor& normals, // Shape (bn, n, 3)
	const at::Tensor& colors, // Shape (bn, n, 2) or Shape (1, n, 2)
	const at::Tensor& stdDevs, // Shape (bn, n, 2) or Shape (1, n, 2)
	double smoothing,
	double precision
);

void cudaCovariance(
	at::Tensor& covariances,
	unsigned int width, unsigned height,
	double focallength,
	at::Tensor const& points,
	at::Tensor const& normals,
	at::Tensor const& stdDevs,
	double smoothing
);

void cudaApproxWeights(
	at::Tensor& weights,
	unsigned int width, unsigned int height,
	double focallength,
	at::Tensor const& points, // Shape (bn, n, 3)
	at::Tensor const& normals, // Shape (bn, n, 3)
	at::Tensor const& colors, // Shape (bn, n, 2) or Shape (1, n, 2)
	at::Tensor const& stdDevs, // Shape (bn, n, 2) or Shape (1, n, 2)
	double smoothing,
	double precision
);

/// 
// Host wrapper functions
/// 

at::Tensor sample(
	unsigned int width, unsigned int height,
	double focallength,
	const at::Tensor& points, // Shape (bn, n, 3)
	const at::Tensor& normals, // Shape (bn, n, 3)
	const at::Tensor& colors, // Shape (bn, n, 3) or Shape (1, n, 3)
	const at::Tensor& stdDevs, // Shape (bn, n, 2) or Shape (1, n, 2)
	unsigned int numSamples,
	double smoothing,
	double precision)
{
	// Verify that every input tensor has a contiguous memory layout.
	assert(points.is_contiguous());
	assert(normals.is_contiguous());
	assert(stdDevs.is_contiguous());

	// Verify that every input tensor is a GPU tensor and on the same device
	assert(points.device().type() == torch::kCUDA);
	assert(normals.device().type() == torch::kCUDA);
	assert(stdDevs.device().type() == torch::kCUDA);
	assert(points.get_device() == normals.get_device()
		&& normals.get_device() == stdDevs.get_device());

	// Create output tensors for the sampled indices and ewa weights
	unsigned int batchSize = points.sizes()[0];
	auto deviceIdx = points.get_device();
	std::array<int64_t, 4> outSizeBuffer = { batchSize, width, height, numSamples };
	at::IntArrayRef outSize(outSizeBuffer.data(), 4);
	logging::log<LOG_DEBUG>("fmmSampling", "output size", outSize.begin(), outSize.end());

	auto indicesOptions = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, deviceIdx);
	at::Tensor indices = at::empty(outSize, indicesOptions);

	cudaSample(
		indices,
		width, height,
		focallength,
		points, // Shape (bn, n, 3)
		normals, // Shape (bn, n, 3)
		colors,
		stdDevs, // Shape (bn, n, 2) or Shape (1, n, 2)
		smoothing,
		precision
	);

	return indices;
}

at::Tensor covariance(
	unsigned int width, unsigned int height,
	double focallength,
	at::Tensor const& points,
	at::Tensor const& normals,
	at::Tensor const& stdDevs,
	double smoothing
)
{
	// Verify that every input tensor has a contiguous memory layout.
	assert(points.is_contiguous());
	assert(normals.is_contiguous());
	assert(stdDevs.is_contiguous());

	// Verify that every input tensor is a GPU tensor and on the same device
	assert(points.device().type() == torch::kCUDA);
	assert(normals.device().type() == torch::kCUDA);
	assert(stdDevs.device().type() == torch::kCUDA);
	assert(points.get_device() == normals.get_device()
		&& normals.get_device() == stdDevs.get_device());

	// Create output tensors for the covariance
	unsigned int batchSize = points.sizes()[0];
	unsigned int numPoints = points.sizes()[1];

	std::array<int64_t, 4> outSizeBuffer = { batchSize, numPoints, 2, 2 };
	at::IntArrayRef outSize(outSizeBuffer.data(), 4);
	at::Tensor covariances = at::zeros(outSize, points.options());
	logging::log<LOG_DEBUG>("covariance", "output size", outSize.begin(), outSize.end());

	cudaCovariance(
		covariances,
		width, height, focallength,
		points, normals, stdDevs,
		smoothing
	);

	return covariances;
}

at::Tensor approximateFMMWeights(
	unsigned int width, unsigned int height,
	double focallength,
	const at::Tensor& points, // Shape (bn, n, 3)
	const at::Tensor& normals, // Shape (bn, n, 3)
	const at::Tensor& colors, // Shape (bn, n, 3) or Shape (1, n, 3)
	const at::Tensor& stdDevs, // Shape (bn, n, 2) or Shape (1, n, 2)
	double smoothing,
	double precision
)
{
	// Verify that every input tensor has a contiguous memory layout.
	assert(points.is_contiguous());
	assert(normals.is_contiguous());
	assert(stdDevs.is_contiguous());

	// Verify that every input tensor is a GPU tensor and on the same device
	assert(points.device().type() == torch::kCUDA);
	assert(normals.device().type() == torch::kCUDA);
	assert(stdDevs.device().type() == torch::kCUDA);
	assert(points.get_device() == normals.get_device()
		&& normals.get_device() == stdDevs.get_device());

	// Create output tensors for the sampled indices and ewa weights
	unsigned int batchSize = points.sizes()[0];
	auto deviceIdx = points.get_device();
	std::array<int64_t, 3> outSizeBuffer = { batchSize, width, height };
	at::IntArrayRef outSize(outSizeBuffer.data(), 3);
	at::Tensor fmmWeights = at::empty(outSize, points.options());
	logging::log<LOG_DEBUG>("approximateFMMWeights", "output size", outSize.begin(), outSize.end());

	cudaApproxWeights(
		fmmWeights,
		width, height, focallength,
		points,  normals, colors,  stdDevs,
		smoothing, precision
	);

	return fmmWeights;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	// module docstring
	m.doc() = "python binding for the fmm sampling procedure";
	m.def("sample", &sample, "");
	m.def("covariance", &covariance, "");
	m.def("approximateFMMWeights", &approximateFMMWeights, "");
}
