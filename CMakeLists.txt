cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(cuda_grid LANGUAGES CXX CUDA)

add_library(cugrid STATIC 
	include/lattice.h include/grid.h include/tensor.h include/simd.h
	src/lattice.cu src/grid.cu src/tensor.cu src/simd.cu
	)


add_executable(test1 test/test1.cu)
target_link_libraries(test1 cugrid)
add_executable(test2 test/test2.cu)
target_link_libraries(test2 cugrid)

target_compile_features(cugrid PUBLIC cxx_std_11)
target_compile_features(test1 PUBLIC cxx_std_11)
target_compile_features(test2 PUBLIC cxx_std_11)
