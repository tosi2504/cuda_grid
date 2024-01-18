project('cugrid'
  , 'cpp'
  , 'cuda'
  , default_options : ['buildtype=release'])


test_files = {'bench_stencil_mrhs' : 'test/bench_stencil_mrhs.cu'}


foreach key, value : test_files
  executable(key, value)
endforeach
