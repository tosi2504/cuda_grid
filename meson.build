project('cugrid'
  , 'cpp'
  , 'cuda'
  , default_options : ['buildtype=release', 'cpp_std=c++20'])

# load dependency
cuda_dep = dependency('cuda', version : '>=12', modules : ['cublas'])

# build cugrid tests
test_files = {'bench_stencil_mrhs' : 'test/bench_stencil_mrhs.cu'
                , 'bench_matmul_mrhs' : 'test/bench_matmul_mrhs.cu'}
foreach key, value : test_files
  executable(key, value, dependencies: cuda_dep)
endforeach

# build cugrid2 tests
test2_files = {'test_tensor'  : 'test2/test_tensor.cu'
                , 'main2'     : 'test2/main2.cu'
                , 'stencil2'     : 'test2/stencil2.cu'
                , 'main2mrhs' : 'test2/main2mrhs.cu'}
foreach key, value : test2_files
  executable(key, value, dependencies: cuda_dep)
endforeach

# build benchmark_stencil tests
bench_params = ['-DBENCH_PARAM_T=complexD', '-DBENCH_PARAM_N=128', '-DBENCH_PARAM_numRHS=60', '-DBENCH_PARAM_blkSize=256']
bench_files = {'stencil_blas'  : 'benchmark/stencil_blas.cu'
                , 'mrhs_blas' : 'benchmark/mrhs_blas.cu'
                , 'mrhs_sharedmem' : 'benchmark/mrhs_sharedmem.cu'
                , 'stencil_lanes' : 'benchmark/stencil_lanes.cu'
                , 'mrhs_lanes' : 'benchmark/mrhs_lanes.cu'
                , 'copy_noMalloc' : 'benchmark/copy_noMalloc.cu'
                , 'copy_withMalloc' : 'benchmark/copy_withMalloc.cu'}
foreach key, value : bench_files
  executable(key, value, dependencies: cuda_dep, cuda_args: bench_params)
endforeach

