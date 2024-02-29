#pragma once

#include <stdexcept>
#include <type_traits>
#include <cassert>

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

namespace matmul_srhs{
	template<class T, unsigned N>
	void cublas(cublasHandle_t & handle 
			, bVectorField<T,N> & y
			, const bMatrixField<T,N> & A
			, const bVectorField<T,N> & x ) 
	{
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
	
	template<class T, unsigned len>
	__device__ void reduce(T * sdata, unsigned n) {
		T v = sdata[n];
		if constexpr (len >= 64) { v += sdata[n+32]; __syncwarp(); }
		if constexpr (len >= 64) { sdata[n] = v;     __syncwarp(); }
		if constexpr (len >= 32) { v += sdata[n+16]; __syncwarp(); }
		if constexpr (len >= 32) { sdata[n] = v;     __syncwarp(); }
		if constexpr (len >= 16) { v += sdata[n+8];  __syncwarp(); }
		if constexpr (len >= 16) { sdata[n] = v;     __syncwarp(); }
		if constexpr (len >= 8)	 { v += sdata[n+4];  __syncwarp(); }
		if constexpr (len >= 8)  { sdata[n] = v;     __syncwarp(); }
		if constexpr (len >= 4)  { v += sdata[n+2];  __syncwarp(); }
		if constexpr (len >= 4)  { sdata[n] = v;     __syncwarp(); }
		if constexpr (len >= 2)  { v += sdata[n+1];  __syncwarp(); }
		if constexpr (len >= 2)  { sdata[n] = v;	 __syncwarp(); }
	}

	template<class T, unsigned N, unsigned blkSize>
	__global__ void ker_cacheMatrix(T * const d_y
			, const T * const d_A
			, const T * const d_x ) 
	{
		const unsigned tIdx = threadIdx.x;
		const unsigned siteIdx = blockIdx.x;
		constexpr unsigned numThreadsPerResult = blkSize / N;

		// prepare shared memory
		__shared__ T tempA[N*N];
		for (unsigned i_flat = tIdx; i_flat < N*N; i_flat += blkSize) {
			tempA[i_flat] = d_A[siteIdx*N*N + i_flat];
		}
		__shared__ T tempX[N];
		if (tIdx < N) tempX[tIdx] = d_x[siteIdx*N + tIdx];
		__syncthreads();

		// perform dot product (in-warp reduction? -> no and after that yes)
		const unsigned i = tIdx / numThreadsPerResult;
		const unsigned n = tIdx % numThreadsPerResult;
		const unsigned dK_max = N/numThreadsPerResult;
		const unsigned K = n * dK_max;
		T temp = 0;
		for (unsigned k = K; k < K+dK_max; k++) {
			temp += tempA[i*N+k] * tempX[k];
		}
		__shared__ T tempResults[N*numThreadsPerResult];
		tempResults[i*numThreadsPerResult + n] = temp;
		__syncthreads();

		// reduce the tempResults array
		reduce<T,numThreadsPerResult>(tempResults+i*numThreadsPerResult, n);
		__syncthreads();

		// write out results
		if (n == 0) d_y[siteIdx*N + i] = tempResults[i*numThreadsPerResult];
	}

	template<class T, unsigned N, unsigned blkSize>
	void cacheMatrix(bVectorField<T,N> & y
			, const bMatrixField<T,N> & A
			, const bVectorField<T,N> & x ) 
	{
		// check validity of template parameters
		static_assert(blkSize % N == 0);
		if (not std::ceil(std::log2(blkSize/N)) == std::floor(std::log2(blkSize/N))) throw std::invalid_argument("N/blkSize must be a power of two");

		// check compatibility of grids
		if ( not (y.grid.isCompatible(A.grid) and y.grid.isCompatible(x.grid)) ) {
			throw std::invalid_argument("Grids are not compatible");
		}

		// call kernel
		ker_cacheMatrix <T,N,blkSize> <<<A.grid.numSites,blkSize>>> ((T*)y.d_data, (T*)A.d_data, (T*)x.d_data);
		CLCE();
		CCE(  cudaDeviceSynchronize()  );
	}

	template<class T, unsigned N, unsigned blkSize>
	__global__ void ker_cacheMatrixWarpReduce(T * const d_y
			, const T * const d_A
			, const T * const d_x )
	{
		const unsigned tIdx = threadIdx.x;
		const unsigned siteIdx = blockIdx.x;
		constexpr unsigned numParallelDotProducts = blkSize / N;
		// constexpr unsigned numWarpsPerDotProduct = N / 32;

		// prepare shared memory
		__shared__ T tempA[N*N];
		for (unsigned i_flat = tIdx; i_flat < N*N; i_flat += blkSize) {
			tempA[i_flat] = d_A[siteIdx*N*N + i_flat];
		}
		__shared__ T tempX[N];
		if (tIdx < N) tempX[tIdx] = d_x[siteIdx*N + tIdx];
		__syncthreads();
		
		// perform dot product
		__shared__ T tempRes[blkSize];
		const unsigned k = tIdx % N;
		const unsigned di = tIdx / N;
		for (unsigned I = 0; I < N; I += numParallelDotProducts) {
			// perform multiplication
			tempRes[di*N + k] = tempA[(I + di)*N + k] * tempX[k];
			__syncthreads();
			// perform reduction on warps (Assumes N <= 64)
			if (tIdx < blkSize/2) reduce<T,N>(tempRes, tIdx);
			// write results to gmem
			if (k == 0) d_y[siteIdx*N + I + di] = tempRes[di*N];
			__syncthreads();
		}
	}

	template<class T, unsigned N, unsigned blkSize>
	void cacheMatrixWarpReduce(bVectorField<T,N> & y
			, const bMatrixField<T,N> & A
			, const bVectorField<T,N> & x) 
	{
		// check validity of template parameters
		static_assert(N%32 == 0);
		static_assert(N==32 or N==64);
		if (not std::ceil(std::log2(blkSize/N)) == std::floor(std::log2(blkSize/N))) throw std::invalid_argument("N/blkSize must be a power of two");

		// check compatibility of grids
		if ( not (y.grid.isCompatible(A.grid) and y.grid.isCompatible(x.grid)) ) {
			throw std::invalid_argument("Grids are not compatible");
		}

		// call kernel
		ker_cacheMatrixWarpReduce <T,N,blkSize> <<<A.grid.numSites,blkSize>>> ((T*)y.d_data, (T*)A.d_data, (T*)x.d_data);
		CLCE();
		CCE(  cudaDeviceSynchronize()  );
	}
}

namespace mrhs_helper {
	// grid compatibility checks
	template<class tensor, unsigned numRHS>
	bool areGridsInBatchCompatible(const bLattice<tensor> * const * fields) {
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

	// batch device pointer management functions
	template<class tensor, unsigned numRHS>
	typename tensor::_T ** createBatchDvcPtr(const bLattice<tensor> * const * const fields) {
		using T = typename tensor::_T;
		T ** d_res;
		CCE(  cudaMalloc(&d_res, sizeof(T*)*numRHS)  );
		for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) {
			CCE(  cudaMemcpy(d_res+iRHS, (T**)&(fields[iRHS]->d_data)
						, sizeof(T*)
						, cudaMemcpyHostToDevice)  );
		}
		return d_res;
	}

	// non contiguous vector fields to matrix field
	template<class T, unsigned N, unsigned numRHS>
	__global__ void ker_fillMatrixfieldFromBatch(T * d_res, const T * const * const d_fields, const unsigned numSites) {
		const unsigned gIdx = blockIdx.x*blockDim.x + threadIdx.x;
		const unsigned iRhs = gIdx/(numSites*N);
		const unsigned iSite = (gIdx%(numSites*N))/N;
		const unsigned iTnsr = gIdx%N;
		if (gIdx < numSites*N*numRHS) {
			d_res[iSite*N*numRHS + iRhs*N + iTnsr] = d_fields[iRhs][iSite*N + iTnsr];
		}
	}
	template<class T, unsigned N, unsigned numRHS, unsigned blkSize>
	void fillMatrixfieldFromBatch(T * const d_matfield, const bVectorField<T,N> * const * const vecfields) {
		// assume grids of vecfields to be compatible
		const bGrid & grid = vecfields[0]->grid;

		// prepare batch pointers for GPU
		T ** d_vecfields = createBatchDvcPtr<bVector<T,N>,numRHS>(vecfields);

		// call the copy kernel 
		const unsigned numBlocks = (numRHS*grid.numSites*N + blkSize - 1)/blkSize;
		ker_fillMatrixfieldFromBatch <T,N,numRHS> <<<numBlocks,blkSize>>> (d_matfield, d_vecfields, grid.numSites);
		
		// free batch pointers on GPU
		CCE(  cudaFree(d_vecfields)  );
	}

	template<class T, unsigned N, unsigned numRHS>
	__global__ void ker_fillBatchFromMatrixfield(T * const * const d_vecfields, const T * const d_matfield, const unsigned numSites) {
		const unsigned gIdx = blockIdx.x*blockDim.x + threadIdx.x;
		const unsigned iRhs = gIdx/(numSites*N);
		const unsigned iSite = (gIdx%(numSites*N))/N;
		const unsigned iTnsr = gIdx%N;
		if (gIdx < numSites*N*numRHS) {
			d_vecfields[iRhs][iSite*N + iTnsr] = d_matfield[iSite*N*numRHS + iRhs*N + iTnsr];
		}
	}
	template<class T, unsigned N, unsigned numRHS, unsigned blkSize>
	void fillBatchFromMatrixfield(bVectorField<T,N> * const * const vecfields, const T * const d_matfield) {
		// assume grids of vecfields to be compatible
		const bGrid & grid = vecfields[0]->grid;

		// prepare batch pointers for GPU
		T ** d_vecfields = createBatchDvcPtr<bVector<T,N>,numRHS>(vecfields);

		// call the copy kernel
		const unsigned numBlocks = (numRHS*grid.numSites*N + blkSize - 1)/blkSize;
		ker_fillBatchFromMatrixfield <T,N,numRHS> <<<numBlocks,blkSize>>> (d_vecfields, d_matfield, grid.numSites);

		// free batch array on GPU
		CCE(  cudaFree(d_vecfields)  );
	}
}
 
namespace matmul_mrhs {
	template<class T, unsigned N, unsigned numRHS>
	void naive(cublasHandle_t & handle
			, bVectorField<T,N> * const * const ys
			, const bMatrixField<T,N> & A
			, const bVectorField<T,N> * const * xs) 
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

	template<class T, unsigned N, unsigned numRHS, unsigned blkSize>
	__global__ void ker_cacheMatrix(T * const * const d_ys
			, const T * const d_A
			, const T * const * const d_xs)
	{
		const unsigned tIdx = threadIdx.x;
		const unsigned siteIdx = blockIdx.x;
		constexpr unsigned sizeA = N*N;

		// need to load d_A into shared memory
		__shared__ T tempA[sizeA];
		for (unsigned i = tIdx; i < sizeA; i+=blkSize) {
			tempA[i] = d_A[siteIdx*sizeA + i];
		}
		__syncthreads();

		// do the matmul for every entry in the resulting vectors
		// also cache those vectors manually
		__shared__ T tempXs[blkSize];
		constexpr unsigned numRhsPerBlk = blkSize/N;
		for (unsigned iRHS = 0; iRHS < numRHS; iRHS += numRhsPerBlk) {
			const unsigned i = tIdx%N;
			const unsigned delta_iRHS = tIdx/N;

			// fill cache
			tempXs[delta_iRHS*N + i] = d_xs[iRHS + delta_iRHS][siteIdx*N + i];
			__syncthreads();

			// perform dotProduct
			T temp = 0;
			for (unsigned k = 0; k < N; k++) {
				temp += tempA[i*N + k] * tempXs[delta_iRHS*N + k];
			}		

			// write out results
			d_ys[iRHS + delta_iRHS][siteIdx*N+i] = temp;
		}
	}

	template<class T, unsigned N, unsigned numRHS, unsigned blkSize>
	void cacheMatrix(bVectorField<T,N> * const * const ys
			, const bMatrixField<T,N> & A
			, const bVectorField<T,N> * const * const xs) 
	{
		if (not (blkSize%N==0)) throw std::invalid_argument("blkSize%N==0 not satisfied");
		if (not (numRHS%(blkSize/N)==0)) throw std::invalid_argument("numRHS%(blkSize/N)==0 not satisfied");
		
		// check compatibility of grids
		if (not mrhs_helper::areGridsCompatible<T,N,numRHS>(ys, A, xs)) throw std::invalid_argument("Grids are not compatible");

		// prepare pointer to array of pointers on device
		T ** d_ys = mrhs_helper::createBatchDvcPtr<bVector<T,N>,numRHS>(ys);
		T ** d_xs = mrhs_helper::createBatchDvcPtr<bVector<T,N>,numRHS>(xs);

		// call the kernel -> a block for every lattice site
		matmul_mrhs::ker_cacheMatrix <T,N,numRHS,blkSize> <<<A.grid.numSites,blkSize>>> (d_ys, &A.d_data->data[0][0], d_xs);
		CLCE();
		CCE(  cudaDeviceSynchronize()  );


		// free the device pointers to pointers of data
		CCE(  cudaFree(d_ys)  );
		CCE(  cudaFree(d_xs)  );
	}

	template<class T, unsigned N, unsigned numRHS, unsigned blkSize>
	void gemm(cublasHandle_t & handle
			, bVectorField<T,N> * const * const ys
			, const bMatrixField<T,N> & A
			, const bVectorField<T,N> * const * xs
			, T * d_Y_buff, T * d_X_buff)
	{
		// check compatibility of grids
		if (not mrhs_helper::areGridsCompatible<T,N,numRHS>(ys, A, xs)) throw std::invalid_argument("Grids are not compatible");
		const bGrid & grid = A.grid;

		// create intermediate matfields
		// T * d_X, * d_Y;
		// CCE(  cudaMalloc(&d_X, sizeof(T)*numRHS*grid.numSites*N)  );
		// CCE(  cudaMalloc(&d_Y, sizeof(T)*numRHS*grid.numSites*N)  );

		// copy inputs to matrixfield X
		mrhs_helper::fillMatrixfieldFromBatch<T,N,numRHS,blkSize>(d_X_buff, xs);
		
		// call gemm on d_X, d_Y and A.d_data
		T alpha = 1;
		T beta = 0;
		cublasCCE(  gemmStridedBatched::call<T>(handle
												, CUBLAS_OP_T
												, CUBLAS_OP_N
												, N, numRHS, N
												, &alpha
												, (T*)A.d_data, N, N*N
												, d_X_buff, N, N*numRHS
												, &beta
												, d_Y_buff, N, N*numRHS
												, grid.numSites)  );
		CCE(  cudaDeviceSynchronize()  );

		// copy result to vectorfields ys
		mrhs_helper::fillBatchFromMatrixfield<T,N,numRHS,blkSize>(ys, d_Y_buff);

		// free temporary matfields
		// CCE(  cudaFree(d_X)  );
		// CCE(  cudaFree(d_Y)  );
	}
}
