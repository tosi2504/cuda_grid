project('cugrid'
  , 'cpp'
  , 'cuda'
  , default_options : ['buildtype=release'])

# load dependency
cuda_dep = dependency('cuda', version : '>=12', modules : ['cublas'])

# build cugrid tests
test_files = {'bench_stencil_mrhs' : 'test/bench_stencil_mrhs.cu'}
foreach key, value : test_files
  executable(key, value, dependencies: cuda_dep)
endforeach

# build cugrid2 tests
test2_files = {'test_tensor' : 'test2/test_tensor.cu'
                , 'main2' : 'test2/main2.cu'}
foreach key, value : test2_files
  executable(key, value, dependencies: cuda_dep)
endforeach
