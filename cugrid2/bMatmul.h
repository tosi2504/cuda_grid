#pragma once

#include <stdexcept>
#include <type_traits>

#include "/usr/local/cuda/include/cublas_v2.h"
// #include "cublas_v2.h"
#include <cuda_runtime.h>

// this should be the batched matmul operation
template<class T, unsigned N>
cublasStatus_t matmul(cublasHandle_t & handle 
		, bVectorField<T,N> & y
		, const bMatrixField<T,N> & A
		, const bVectorField<T,N> & x ) {
	// disable ints and complex numbers (for now)
	static_assert(std::is_same_v<float,T>, "matmul only for float atm");

	// check compatibility of grids
	if ( not (y.grid.isCompatible(A.grid) and y.grid.isCompatible(x.grid)) ) {
		throw std::invalid_argument("Grids are not compatible");
	}

	// assume data already on devc mem
	// call cublas function for batched matmul, i.e., cuBLAS
	const float alpha = 1;
	const float beta = 0;
	if constexpr (std::is_same_v<float,T>) {
		return cublasSgemvStridedBatched(handle
								, CUBLAS_OP_T
								, N, N // m, n
								, &alpha 
								, &A.d_data[0].data[0][0], N // ld of A
								, N*N // stride of batch of A
								, &x.d_data[0].data[0], 1 // incx
								, N // stride of batch of x
								, &beta 
								, &y.d_data[0].data[0], 1 // incy
								, N // stride of batch of y
								, y.grid.numSites); // batchCount
	} else {
		throw std::runtime_error("Only FLOATs atm ffs!");
	}
}

