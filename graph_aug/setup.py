from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='graph_aug_cuda',
    ext_modules=[
        CUDAExtension(
            name='graph_aug_cuda',
            sources=[
                'cuda_kernels/drop_nodes_kernel.cu',
                'cuda_kernels/mask_nodes_kernel.cu',
                'cuda_kernels/permute_edges_kernel.cu',
                'cpp/graph_aug.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--expt-relaxed-constexpr']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
