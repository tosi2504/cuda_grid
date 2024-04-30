from pyBenchmark import *

# res = run_benchmark('stencil_blas')
# writePickle(res, filepath='stencil_blas.pickle')


# benchmarks using precompiled binaries
compile_binaries('stencil_blas')
# res = run_benchmark_on_precompiled_binaries('stencil_blas', useSrun=False)
# writePickle(res, filepath='stencil_blas.pickle')








