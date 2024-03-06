#pragma once

#include "bTensor.h"
#include <random>
#include "errorcheck.h"
#include <vector>



template<class tensor>
struct bLattice {
	using _tensor = tensor;
	using T = typename tensor::_T;

	tensor * h_data;
	tensor * d_data;
	const bGrid grid;

	// constructors and destructors
	bLattice(const bGrid & grid) : grid(grid) {
		// construct on host mem
		h_data = new tensor[grid.numSites];

		// construct on devc mem
		CCE(  cudaMalloc(&d_data, sizeof(tensor)*grid.numSites)  );
	}
	~bLattice() {
		// free host mem
		delete[] h_data;

		// free devc mem
		CCE(  cudaFree(d_data)  );
	}

	// upload and download
	void upload() const {
		CCE(  cudaMemcpy(d_data, h_data, sizeof(tensor)*grid.numSites, cudaMemcpyHostToDevice)  );
	}
	void download() const {
		CCE(  cudaMemcpy(h_data, d_data, sizeof(tensor)*grid.numSites, cudaMemcpyDeviceToHost)  );
	}

	void fill_random(std::mt19937 & gen, T min, T max) {
		for (unsigned site = 0; site < grid.numSites; site++) {
			h_data[site].fill_random(gen, min, max);
		}
	}
}; 

template<class T, unsigned N> using bVectorField = bLattice<bVector<T,N>>;
template<class T, unsigned N> using bMatrixField = bLattice<bMatrix<T,N>>;

namespace bLatticeHelpers {
	// grid compatibility checks
	template<class tensor, unsigned numRHS>
	bool areGridsInBatchCompatible(const bLattice<tensor> * const * fields) {
		for (unsigned iRHS = 1; iRHS < numRHS; iRHS++)
			if (not fields[0]->grid.isCompatible(fields[iRHS]->grid)) return false;
		return true;
	}
	template<class T, unsigned N, unsigned numRHS>
	bool areGridsCompatible(const bVectorField<T,N> * const * const ys
			, const bMatrixField<T,N> & A
			, const bVectorField<T,N> * const * const xs) {
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

	// batch of vecfields from and to matfield
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





