#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include<vector>

#include "logging.h"

///
// CUDA forward declarations
///
void cudaShadingForward(
	at::Tensor& images,
	const at::Tensor& directions,
	const at::Tensor& intensities,
	const at::Tensor& normals,
	const at::Tensor& diffuse
);

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
);

void cudaShadingShForward(
	at::Tensor& images,
	const at::Tensor& shCoefficients,
	const at::Tensor& normals,
	const at::Tensor& diffuse
);

void cudaShadingShBackward(
	at::Tensor& gradShCoefficients,
	at::Tensor& gradOutNormals,
	at::Tensor& gradOutDiffuse,
	const at::Tensor& gradIn,
	const at::Tensor& shCoefficients,
	const at::Tensor& normals,
	const at::Tensor& diffuse
);

///
// C++ functions
///

at::Tensor shadingForward(
	const at::Tensor& directions, // Shape (bn, l, 3)
	const at::Tensor& intensities, // Shape (bn, l, 3) or (1, l, 3)
	const at::Tensor& normals, // Shape (bn, w, h, 3)
	const at::Tensor& diffuse // Shape (bn, w, h, 3)
)
{
	// Verify that every input tensor has a contiguous memory layout.
	assert(directions.is_contiguous());
	assert(intensities.is_contiguous());
	assert(normals.is_contiguous());
	assert(diffuse.is_contiguous());

	// Verify that every input tensor is a GPU tensor and on the same device
	assert(directions.device().type() == torch::kCUDA);
	assert(intensities.device().type() == torch::kCUDA);
	assert(normals.device().type() == torch::kCUDA);
	assert(diffuse.device().type() == torch::kCUDA);
	assert(directions.get_device() == intensities.get_device()
		&& intensities.get_device() ==  normals.get_device()
		&& normals.get_device() == diffuse.get_device());

	// Create output tensors for the sampled indices and ewa weights
	unsigned int batchSize = normals.sizes()[0];
	unsigned int width = normals.sizes()[1];
	unsigned int height = normals.sizes()[2];

	std::array<int64_t, 4> outSizeBuffer = { batchSize, width, height, 3 };
	at::IntArrayRef outSize(outSizeBuffer.data(), 4);
	logging::log<LOG_DEBUG>("shadingForward", "output size", outSize.begin(), outSize.end());
	auto deviceIdx = directions.get_device();
	auto imageOptions = torch::TensorOptions().dtype(normals.scalar_type()).device(torch::kCUDA, deviceIdx);
	at::Tensor images = at::empty(outSize, imageOptions);

	cudaShadingForward(images, directions, intensities, normals, diffuse);
	
	return images;
}

std::vector<at::Tensor> shadingBackward(
	const at::Tensor& gradIn, // Shape (bn, w, h, 3)
	const at::Tensor& directions, // Shape (bn, l, 3)
	const at::Tensor& intensities, // Shape (bn, l, 3) or (1, l, 3)
	const at::Tensor& normals, // Shape (bn, w, h, 3)
	const at::Tensor& diffuse // Shape (bn, w, h, 3)
)
{
	// Verify that every input tensor has a contiguous memory layout.
	assert(gradIn.is_contiguous());
	assert(directions.is_contiguous());
	assert(intensities.is_contiguous());
	assert(normals.is_contiguous());
	assert(diffuse.is_contiguous());

	// Verify that every input tensor is a GPU tensor and on the same device
	assert(gradIn.device().type() == torch::kCUDA);
	assert(directions.device().type() == torch::kCUDA);
	assert(intensities.device().type() == torch::kCUDA);
	assert(normals.device().type() == torch::kCUDA);
	assert(diffuse.device().type() == torch::kCUDA);
	assert(gradIn.get_device() == directions.get_device()
		&& directions.get_device() == intensities.get_device()
		&& intensities.get_device() == normals.get_device()
		&& normals.get_device() == diffuse.get_device());

	// Create output tensors for the sampled indices and ewa weights
	unsigned int batchSize = normals.sizes()[0];
	unsigned int width = normals.sizes()[1];
	unsigned int height = normals.sizes()[2];

	auto deviceIdx = directions.get_device();
	auto gradientOptions = torch::TensorOptions().dtype(normals.scalar_type()).device(torch::kCUDA, deviceIdx);
	at::Tensor gradOutDirections = at::zeros(directions.sizes(), gradientOptions);
	at::Tensor gradOutIntensities = at::zeros(intensities.sizes(), gradientOptions);
	at::Tensor gradOutNormals = at::zeros(normals.sizes(), gradientOptions);
	at::Tensor gradOutDiffuse = at::zeros(diffuse.sizes(), gradientOptions);

	cudaShadingBackward(gradOutDirections, gradOutIntensities, gradOutNormals, gradOutDiffuse, gradIn, directions, intensities, normals, diffuse);

	return {gradOutDirections, gradOutIntensities, gradOutNormals, gradOutDiffuse};
}

at::Tensor shadingShForward(
	const at::Tensor& shCoefficients, // Shape (bn, 3, 9) or (1, 3, 9)
	const at::Tensor& normals, // Shape (bn, w, h, 3)
	const at::Tensor& diffuse // Shape (bn, w, h, 3)
)
{
	// Verify that every input tensor has a contiguous memory layout.
	assert(shCoefficients.is_contiguous());
	assert(normals.is_contiguous());
	assert(diffuse.is_contiguous());

	// Verify that every input tensor is a GPU tensor and on the same device
	assert(shCoefficients.device().type() == torch::kCUDA);
	assert(normals.device().type() == torch::kCUDA);
	assert(diffuse.device().type() == torch::kCUDA);
	assert(shCoefficients.get_device() == normals.get_device()
		&& normals.get_device() == diffuse.get_device());

	// Create output tensors for the sampled indices and ewa weights
	unsigned int batchSize = normals.sizes()[0];
	unsigned int width = normals.sizes()[1];
	unsigned int height = normals.sizes()[2];

	std::array<int64_t, 4> outSizeBuffer = { batchSize, width, height, 3 };
	at::IntArrayRef outSize(outSizeBuffer.data(), 4);
	auto deviceIdx = shCoefficients.get_device();
	auto imageOptions = torch::TensorOptions().dtype(normals.scalar_type()).device(torch::kCUDA, deviceIdx);
	at::Tensor images = at::empty(outSize, imageOptions);

	cudaShadingShForward(images, shCoefficients, normals, diffuse);

	return images;
}

std::vector<at::Tensor> shadingShBackward(
	const at::Tensor& gradIn,
	const at::Tensor& shCoefficients,
	const at::Tensor& normals,
	const at::Tensor& diffuse
)
{
	// Verify that every input tensor has a contiguous memory layout.
	assert(gradIn.is_contiguous());
	assert(shCoefficients.is_contiguous());
	assert(normals.is_contiguous());
	assert(diffuse.is_contiguous());

	// Verify that every input tensor is a GPU tensor and on the same device
	assert(gradIn.device().type() == torch::kCUDA);
	assert(shCoefficients.device().type() == torch::kCUDA);
	assert(normals.device().type() == torch::kCUDA);
	assert(diffuse.device().type() == torch::kCUDA);
	assert(gradIn.get_device() == shCoefficients.get_device()
		&& shCoefficients.get_device() == normals.get_device()
		&& normals.get_device() == diffuse.get_device());

	// Create output tensors for the gradient
	auto deviceIdx = shCoefficients.get_device();
	auto gradientOptions = torch::TensorOptions().dtype(normals.scalar_type()).device(torch::kCUDA, deviceIdx);
	at::Tensor gradOutShCoefficients = at::zeros(shCoefficients.sizes(), gradientOptions);
	at::Tensor gradOutNormals = at::zeros(normals.sizes(), gradientOptions);
	at::Tensor gradOutDiffuse = at::zeros(diffuse.sizes(), gradientOptions);

	cudaShadingShBackward(gradOutShCoefficients, gradOutNormals, gradOutDiffuse, gradIn, shCoefficients, normals, diffuse);

	return { gradOutShCoefficients, gradOutNormals, gradOutDiffuse };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	// module docstring
	m.doc() = "pybind11 shading binding";
	m.def("forward", &shadingForward, "");
	m.def("backward", &shadingBackward, "");
	m.def("forwardSh", &shadingShForward, "");
	m.def("backwardSh", &shadingShBackward, "");
}