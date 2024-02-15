#pragma once

#include <stdexcept>
#include <type_traits>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "errorcheck.h"
#include "datatypes.h"

namespace gemvStridedBatched {
	template<class T>
	cublasStatus_t call(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const T *alpha, const T *A, int lda, long long int strideA, const T *x, int incx, long long int stridex, const T *beta, T *y, int incy, long long int stridey, int batchCount);

	template<>
	cublasStatus_t call<realF>(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const realF *alpha, const realF *A, int lda, long long int strideA, const realF *x, int incx, long long int stridex, const realF *beta, realF *y, int incy, long long int stridey, int batchCount)
	{
		return cublasSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey,batchCount);
	}

	template<>
	cublasStatus_t call<realD>(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const realD *alpha, const realD *A, int lda, long long int strideA, const realD *x, int incx, long long int stridex, const realD *beta, realD *y, int incy, long long int stridey, int batchCount)
	{
		return cublasDgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey,batchCount);
	}

	template<>
	cublasStatus_t call<complexF>(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const complexF *alpha, const complexF *A, int lda, long long int strideA, const complexF *x, int incx, long long int stridex, const complexF *beta, complexF *y, int incy, long long int stridey, int batchCount)
	{
		return cublasCgemvStridedBatched(handle, trans, m, n, (cuComplex*)alpha, (cuComplex*)A, lda, strideA, (cuComplex*)x, incx, stridex, (cuComplex*)beta, (cuComplex*)y, incy, stridey, batchCount);
	}

	template<>
	cublasStatus_t call<complexD>(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const complexD *alpha, const complexD *A, int lda, long long int strideA, const complexD *x, int incx, long long int stridex, const complexD *beta, complexD *y, int incy, long long int stridey, int batchCount)
	{
		return cublasZgemvStridedBatched(handle, trans, m, n, (cuDoubleComplex*)alpha, (cuDoubleComplex*)A, lda, strideA, (cuDoubleComplex*)x, incx, stridex, (cuDoubleComplex*)beta, (cuDoubleComplex*)y, incy, stridey, batchCount);
	}
}

template<class T, unsigned N>
void matmul(cublasHandle_t & handle 
		, bVectorField<T,N> & y
		, const bMatrixField<T,N> & A
		, const bVectorField<T,N> & x ) {
	// check compatibility of grids
	if ( not (y.grid.isCompatible(A.grid) and y.grid.isCompatible(x.grid)) ) {
		throw std::invalid_argument("Grids are not compatible");
	}

	// assume data already on devc mem
	// call cublas function for batched matmul, i.e., cuBLAS
	const T alpha = 1;
	const T beta = 0;
	cublasStatus_t status = gemvStridedBatched::call<T>(handle
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
	cublasCCE(status);
	CCE(  cudaDeviceSynchronize()  );
}

namespace mrhs_helper {
	template<class bTensor, unsigned numRHS>
	bool areGridsInBatchCompatible(const bLattice<bTensor> * const * fields) {
		for (unsigned iRHS = 1; iRHS < numRHS; iRHS++)
			if (not fields[0]->grid.isCompatible(fields[iRHS]->grid)) return false;
		return true;
	}
	template<class T, unsigned N, unsigned numRHS>
	bool areGridsCompatible(bVectorField<T,N> * const * ys
			, const bMatrixField<T,N> & A
			, const bVectorField<T,N> * const * xs) {
		if ( A.grid.isCompatible(ys[0]->grid) and A.grid.isCompatible(xs[0]->grid) )
			if ( areGridsInBatchCompatible<bVector<T,N>,numRHS>(ys) and areGridsInBatchCompatible<bVector<T,N>,numRHS>(xs) )
				return true;
		return false;
	}
}
 
namespace matmul_mrhs {
	template<class T, unsigned N, unsigned numRHS>
	void naive(cublasHandle_t & handle
			, bVectorField<T,N> * const ys[numRHS]
			, const bMatrixField<T,N> & A
			, const bVectorField<T,N> * const xs[numRHS]) 
	{
		// check compatibility of grids
		if (not mrhs_helper::areGridsCompatible<T,N,numRHS>(ys, A, xs)) throw std::invalid_argument("Grids are not compatible");

		// call cublas function
		const T alpha = 1;
		const T beta = 0;
		for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) {
			cublasStatus_t status = gemvStridedBatched::call<T>(handle
								, CUBLAS_OP_T
								, N, N // m, n
								, &alpha 
								, &A.d_data[0].data[0][0], N // ld of A
								, N*N // stride of batch of A
								, &xs[iRHS]->d_data[0].data[0], 1 // incx
								, N // stride of batch of x
								, &beta 
								, &ys[iRHS]->d_data[0].data[0], 1 // incy
								, N // stride of batch of y
								, A.grid.numSites); // batchCount
			cublasCCE(status);
		}
		CCE(  cudaDeviceSynchronize()  );
	}
}
