cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(cuda_grid LANGUAGES CXX CUDA)

add_library(cugrid INTERFACE 
	cugrid/lane.h
	)

add_executable(test1 test/test1.cu)
target_link_libraries(test1 cugrid)
add_executable(test2 test/test2.cu)
target_link_libraries(test2 cugrid)

target_compile_features(test1 PUBLIC cxx_std_11)
target_compile_features(test2 PUBLIC cxx_std_11)
