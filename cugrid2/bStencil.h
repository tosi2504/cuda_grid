#pragma once

#include <stdexcept>
#include <vector>



struct bMuStencil {
	const bGrid grid;
	const unsigned mu;
	const bool isForward;
	std::vector<unsigned> targetmap;
	
	bMuStencil(const bGrid & grid, const unsigned mu, const bool isForward) : grid(grid), mu(mu), isForward(isForward) {
		// setup pointer map
		targetmap = grid.calcTargetMap(mu, isForward);
	}
	~bMuStencil() {}

	template<class T, unsigned stride>
	T** createDevicePointerArray(const T * const d_field, const bool doPermute) const {
		const T ** const h_d_field = new const T*[grid.numSites];
		for (unsigned site = 0; site < grid.numSites; site++) {
			if (doPermute) h_d_field[site] = d_field + targetmap[site]*stride;
			else h_d_field[site] = d_field + site*stride;
		}
		T ** d_d_field;
		CCE(  cudaMalloc(&d_d_field, sizeof(T*)*grid.numSites)  );
		CCE(  cudaMemcpy(d_d_field, h_d_field, sizeof(T*)*grid.numSites, cudaMemcpyHostToDevice)  );
		delete[] h_d_field;
		return d_d_field;
	}

	// execute functions
	template<class T, unsigned N, unsigned numRHS, unsigned blkSize>
	void execute(cublasHandle_t handle
				, bVectorField<T,N> * const * const ys
				, const bMatrixField<T,N> & A
				, const bVectorField<T,N> * const * const xs) const {

		// check grid compatibility
		if( not bLatticeHelpers::areGridsCompatible<T,N,numRHS>(ys,A,xs) ) throw std::invalid_argument("Grids not compatible");
		if( not grid.isCompatible(A.grid) ) throw std::invalid_argument("Grids not compatible");
		
		// prepare device pointers and layout changes
		T * d_X, * d_Y;
		CCE(  cudaMalloc(&d_X, sizeof(T)*numRHS*grid.numSites*N)  );
		CCE(  cudaMalloc(&d_Y, sizeof(T)*numRHS*grid.numSites*N)  );

		// copy inputs to matrixfield X
		mrhs_helper::fillMatrixfieldFromBatch<T,N,numRHS,blkSize>(d_X, xs);
		
		// create permuted pointer array for X
		T ** d_d_X = createDevicePointerArray<T,N*numRHS>(d_X, true);
		// create unpermuted pointer array for Y and A
		T ** d_d_A = createDevicePointerArray<T,N*N>((T*)A.d_data, false);
		T ** d_d_Y = createDevicePointerArray<T,N*numRHS>(d_Y, false);

		// call gemmBatched on d_d_X, d_d_Y and A.d_data
		const T alpha = 1;
		const T beta = 0;
		cublasCCE(
			gemmBatched::call<T>(handle
								, CUBLAS_OP_T
								, CUBLAS_OP_N
								, N, numRHS, N
								, &alpha
								, d_d_A, N
								, d_d_X, N
								, &beta
								, d_d_Y, N
								, grid.numSites)
		);
		CCE(  cudaDeviceSynchronize()  );

		// copy result to vectorfields ys
		mrhs_helper::fillBatchFromMatrixfield<T,N,numRHS,blkSize>(ys, d_Y);

		// free temporary device arrays
		CCE(  cudaFree(d_d_X)  );
		CCE(  cudaFree(d_d_A)  );
		CCE(  cudaFree(d_d_Y)  );
		CCE(  cudaFree(d_X)  );
		CCE(  cudaFree(d_Y)  );
	}
};
