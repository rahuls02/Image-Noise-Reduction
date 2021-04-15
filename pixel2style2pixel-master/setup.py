from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='stylegan_extension',
    ext_modules=[
        CUDAExtension(
            name='upfirdn2d',
            sources=['models/stylegan2/op/upfirdn2d_kernel.cu', 'models/stylegan2/op/upfirdn2d.cpp'],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']}),
        CUDAExtension(
            name='fused',
            sources=['models/stylegan2/op/fused_bias_act_kernel.cu', 'models/stylegan2/op/fused_bias_act.cpp'],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']})
        ],
    cmdclass={
        'build_ext': BuildExtension
        }
)

