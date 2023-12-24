from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='skc',
    ext_modules=[
        CUDAExtension('skc_cuda', [
            'skc_cuda.cpp',
            'skc_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })