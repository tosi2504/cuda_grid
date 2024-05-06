from pyBenchmark import *

# res = run_benchmark('stencil_blas')
# writePickle(res, filepath='stencil_blas.pickle')


# benchmarks using precompiled binaries
compile_binaries('copy_noMalloc')
res = run_benchmark_on_precompiled_binaries('copy_noMalloc', useSrun=False)
writePickle(res, filepath='copy_noMalloc.pickle')

# target = 'mrhs_lanes'
# T = 'realD'
# N = 32
# numRHS = 12
# blkSize = 9999
# set_meson_bench_params(T=T, N=N, numRHS=numRHS, blkSize=blkSize)
# try:
#     reconfigure()
#     compile_target(target, force_recompile=True)
# except CommandFailedError as err:
#     print("PANIC: shell command failed:")
#     print(err.stdout)
#     print(err.stderr)
