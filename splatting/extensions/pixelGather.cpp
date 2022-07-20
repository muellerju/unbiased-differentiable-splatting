#include <pybind11/pybind11.h>
#include <torch/extension.h>

// Forward CUDA declaration
at::Tensor scatterCuda(const at::Tensor& indices, const at::Tensor& values);
at::Tensor gatherCuda(const at::Tensor& indices, const at::Tensor& values, unsigned int numSources);

//at::Tensor gatherBackwardCuda(const at::Tensor& gradIn, const at::Tensor& indices, unsigned int numSources);

at::Tensor scatter(const at::Tensor& indices, const at::Tensor& values)
{
	return scatterCuda(indices, values);
}

at::Tensor gather(const at::Tensor& indices, const at::Tensor& values, unsigned int numSources)
{
	return gatherCuda(indices, values, numSources);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("scatter", &scatter, "per pixel scatter forward");
	m.def("gather", &gather, "per pixel gather backward");
}
