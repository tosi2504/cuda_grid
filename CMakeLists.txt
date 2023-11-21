cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(cuda_grid LANGUAGES CXX CUDA)

add_library(cugrid INTERFACE 
	cugrid/lane.h
	cugrid/tensor.h
	cugrid/errorcheck.h
	cugrid/grid.h
	cugrid/lattice.h
    cugrid/random.h
	)

add_compile_options(-fopenmp)

add_executable(vector_add test/vector_add.cu)
target_link_libraries(vector_add cugrid)
target_link_libraries(vector_add gomp)
target_compile_features(vector_add PUBLIC cxx_std_11)

add_executable(matmul test/matmul.cu)
target_link_libraries(matmul cugrid)
target_link_libraries(matmul gomp)
target_compile_features(matmul PUBLIC cxx_std_11)

add_executable(lattice_matmul test/lattice_matmul.cu)
target_link_libraries(lattice_matmul cugrid)
target_link_libraries(lattice_matmul gomp)
target_compile_features(lattice_matmul PUBLIC cxx_std_11)

add_executable(bench_matmul test/bench_matmul.cu)
target_link_libraries(bench_matmul cugrid)
target_link_libraries(bench_matmul gomp)
target_compile_features(bench_matmul PUBLIC cxx_std_11)
