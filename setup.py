from setuptools import setup, find_packages
import unittest
from typing import List

from torch.utils.cpp_extension import BuildExtension, CUDAExtension
#from cpp_extension import BuildExtension, CUDAExtension
print(find_packages())
CUDA_FLAGS = []  # type: List[str]

# The modules are as small as possible to reduce incremental building time during development
modules = [
    CUDAExtension('cudaExtensions.transform',
        sources=[
            'FMMKdTreeSampling/transformCuda.cu',
            'FMMKdTreeSampling/transform.cpp'
        ],
        extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
    ),
    CUDAExtension('cudaExtensions.sampling', 
        sources=[
            './FMMKdTreeSampling/samplingCuda.cu',
            './FMMKdTreeSampling/sampling.cpp'
        ],
        extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        #extra_compile_args={'cxx': ['-g'], 'nvcc': ['-G']}
    ),
    CUDAExtension('cudaExtensions.pixelgather',
    sources = [
            'splatting/extensions/pixelGatherCuda.cu',
            'splatting/extensions/pixelGather.cpp'
        ],
        extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
    ),
    CUDAExtension('cudaExtensions.interpolation',
    sources = [
            'FMMKdTreeSampling/interpolationCuda.cu',
            'FMMKdTreeSampling/interpolation.cpp'
        ],
        extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
    ),
    CUDAExtension('cudaExtensions.splatting',
        sources=[
            'FMMKdTreeSampling/splattingCuda.cu',
            'FMMKdTreeSampling/splatting.cpp'
        ],
        extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
    ),
    CUDAExtension('cudaExtensions.filtering',
        sources=[
            'FMMKdTreeSampling/zSortingCuda.cu',
            'FMMKdTreeSampling/zSorting.cpp'
        ],
        extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
    ),
    CUDAExtension('cudaExtensions.shading',
        sources=[
            'FMMKdTreeSampling/shadingCuda.cu',
            'FMMKdTreeSampling/shading.cpp'
        ],
        extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
    ),
    CUDAExtension('cudaExtensions.nearestsamples',
        sources=[
            'FMMKdTreeSampling/nearestSamplesCuda.cu',
            'FMMKdTreeSampling/nearestSamples.cpp'
        ],
        extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        #extra_compile_args={'cxx': ['-g'], 'nvcc': ['-G']}
    )
]

setup(name='cudaExtensions',
    description='Unbiased Differentiable Splatting',
    author='Jan Mueller',
    version='1.0',
    install_requires = [],
    ext_modules= modules,
    cmdclass={'build_ext': BuildExtension}
)
