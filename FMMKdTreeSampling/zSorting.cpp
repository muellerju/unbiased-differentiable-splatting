#include <pybind11/pybind11.h>
#include <torch/extension.h>

// Forward CUDA declaration
at::Tensor cudaAlphaBlendForward(
	const at::Tensor& indices, 
	const at::Tensor& weights,
	unsigned int width, unsigned int height, double focallength,
	const at::Tensor& points, 
	const at::Tensor& normals
);

at::Tensor cudaAlphaBlendBackward(
    const at::Tensor& indices, 
    const at::Tensor& weights,
    const at::Tensor& gradIn,
    unsigned int width, unsigned int height, double focallength,
    const at::Tensor& points, 
    const at::Tensor& normals    
);


// Cpp wrapper functions
at::Tensor forward(
	const at::Tensor& indices, 
	const at::Tensor& weights,
	unsigned int width, unsigned int height, double focallength,
	const at::Tensor& points, 
	const at::Tensor& normals)
{
	return cudaAlphaBlendForward(indices, weights, width, height, focallength, points, normals);
}

at::Tensor backward(
	const at::Tensor& indices, 
	const at::Tensor& weights,
	const at::Tensor& gradIn,
	unsigned int width, unsigned int height, double focallength,
	const at::Tensor& points, 
	const at::Tensor& normals)
{
	return cudaAlphaBlendBackward(indices, weights, gradIn, width, height, focallength, points, normals);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.doc() = "pybind11 filtering binding";
	m.def("forward", &forward, "per pixel depth alpha blending forward");
	m.def("backward", &backward, "per pixel depth alpha blending backward");
}