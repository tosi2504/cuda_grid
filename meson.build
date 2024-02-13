project('cugrid'
  , 'cpp'
  , 'cuda'
  , default_options : ['buildtype=release'])

# build cugrid tests
test_files = {'bench_stencil_mrhs' : 'test/bench_stencil_mrhs.cu'}
foreach key, value : test_files
  executable(key, value)
endforeach

# build cugrid2 tests
test2_files = {'test_tensor' : 'test2/test_tensor.cu'}
foreach key, value : test2_files
  executable(key, value)
endforeach
