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
                , 'complexStencil'     : 'test2/complexStencil.cu'
                , 'main2mrhs' : 'test2/main2mrhs.cu'}
foreach key, value : test2_files
  executable(key, value, dependencies: cuda_dep)
endforeach

# build benchmark_stencil tests
bench_files = {'matmul_blas' : 'benchmark/matmul_blas.cu'
                , 'matmul_shmem' : 'benchmark/matmul_shmem.cu'
                , 'matmul_lanes' : 'benchmark/matmul_lanes.cu'
                , 'stencil_blas' : 'benchmark/stencil_blas.cu'
                , 'stencil_lanes' : 'benchmark/stencil_lanes.cu'
                , '9points_blas' : 'benchmark/9points_blas.cu'}
foreach key, value : bench_files
  executable(key, value, dependencies: cuda_dep)
endforeach

