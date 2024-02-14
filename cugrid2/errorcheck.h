#pragma once

#include <iostream>
#include <map>
#include <string>
#include <cublas_v2.h>


#define CCE(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CLCE() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

const std::map<cublasStatus_t, std::string> cublasErrorStringMap = {
	{CUBLAS_STATUS_SUCCESS, "CUBLAS_STATUS_SUCCESS"},
	{CUBLAS_STATUS_NOT_INITIALIZED, "CUBLAS_STATUS_NOT_INITIALIZED"},
	{CUBLAS_STATUS_ALLOC_FAILED, "CUBLAS_STATUS_ALLOC_FAILED"},
	{CUBLAS_STATUS_INVALID_VALUE, "CUBLAS_STATUS_INVALID_VALUE"},
	{CUBLAS_STATUS_ARCH_MISMATCH, "CUBLAS_STATUS_ARCH_MISMATCH"},
	{CUBLAS_STATUS_MAPPING_ERROR, "CUBLAS_STATUS_MAPPING_ERROR"},
	{CUBLAS_STATUS_EXECUTION_FAILED, "CUBLAS_STATUS_EXECUTION_FAILED"},
	{CUBLAS_STATUS_INTERNAL_ERROR, "CUBLAS_STATUS_INTERNAL_ERROR"},
	{CUBLAS_STATUS_NOT_SUPPORTED, "CUBLAS_STATUS_NOT_SUPPORTED"},
	{CUBLAS_STATUS_LICENSE_ERROR, "CUBLAS_STATUS_LICENSE_ERROR"}
};

#define cublasCCE(val) cublasCheck((val), #val, __FILE__, __LINE__)
void cublasCheck(const cublasStatus_t & cublasStat, const char* const func_name, const char* const file, const int line) 
{
	if (cublasStat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cublasErrorStringMap.at(cublasStat) << " " << func_name << std::endl;
	}
}
