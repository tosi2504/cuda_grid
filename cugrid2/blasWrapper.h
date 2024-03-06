#pragma once

#include <cublas_v2.h>

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

namespace gemmStridedBatched {
	template<class T>
	cublasStatus_t call(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const T *alpha, const T *A, int lda, long long int strideA, const T *B, int ldb, long long int strideB, const T *beta, T *C, int ldc, long long int strideC, int batchCount);

	template<>
	cublasStatus_t call<realF>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const realF *alpha, const realF *A, int lda, long long int strideA, const realF *B, int ldb, long long int strideB, const realF *beta, realF *C, int ldc, long long int strideC, int batchCount) {
		return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
	}

	template<>
	cublasStatus_t call<realD>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const realD *alpha, const realD *A, int lda, long long int strideA, const realD *B, int ldb, long long int strideB, const realD *beta, realD *C, int ldc, long long int strideC, int batchCount) {
		return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
	}

	template<>
	cublasStatus_t call<complexF>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const complexF *alpha, const complexF *A, int lda, long long int strideA, const complexF *B, int ldb, long long int strideB, const complexF *beta, complexF *C, int ldc, long long int strideC, int batchCount) {
		return cublasCgemmStridedBatched(handle, transa, transb, m, n, k, (cuComplex*)alpha, (cuComplex*)A, lda, strideA, (cuComplex*)B, ldb, strideB, (cuComplex*)beta, (cuComplex*)C, ldc, strideC, batchCount);
	}

	template<>
	cublasStatus_t call<complexD>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const complexD *alpha, const complexD *A, int lda, long long int strideA, const complexD *B, int ldb, long long int strideB, const complexD *beta, complexD *C, int ldc, long long int strideC, int batchCount) {
		return cublasZgemmStridedBatched(handle, transa, transb, m, n, k, (cuDoubleComplex*)alpha, (cuDoubleComplex*)A, lda, strideA, (cuDoubleComplex*)B, ldb, strideB, (cuDoubleComplex*)beta, (cuDoubleComplex*)C, ldc, strideC, batchCount);
	}
}

namespace gemmBatched{
	template<class T>
	cublasStatus_t call(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const T *alpha, const T * const * const A, int lda, const T * const * const B, int ldb, const T *beta, T * const * const C, int ldc, int batchCount);

	template<>
	cublasStatus_t call<realF>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const realF *alpha, const realF * const * const A, int lda, const realF * const * const B, int ldb, const realF *beta, realF * const * const C, int ldc, int batchCount) {
		return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
	}

	template<>
	cublasStatus_t call<realD>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const realD *alpha, const realD * const * const A, int lda, const realD * const * const B, int ldb, const realD *beta, realD * const * const C, int ldc, int batchCount) {
		return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
	}

	template<>
	cublasStatus_t call<complexF>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const complexF *alpha, const complexF * const * const A, int lda, const complexF * const * const B, int ldb, const complexF *beta, complexF * const * const C, int ldc, int batchCount) {
		return cublasCgemmBatched(handle, transa, transb, m, n, k, (const cuComplex*)alpha, (const cuComplex*const*)A, lda, (const cuComplex*const*)B, ldb, (const cuComplex*)beta, (cuComplex*const*)C, ldc, batchCount);
	}

	template<>
	cublasStatus_t call<complexD>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const complexD *alpha, const complexD * const * const A, int lda, const complexD * const * const B, int ldb, const complexD *beta, complexD * const * const C, int ldc, int batchCount) {
		return cublasZgemmBatched(handle, transa, transb, m, n, k, (const cuDoubleComplex*)alpha, (const cuDoubleComplex*const*)A, lda, (const cuDoubleComplex*const*)B, ldb, (const cuDoubleComplex*)beta, (cuDoubleComplex*const*)C, ldc, batchCount);
	}
}


