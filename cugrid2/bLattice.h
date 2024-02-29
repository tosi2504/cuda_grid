#pragma once

#include "bTensor.h"
#include <random>
#include "errorcheck.h"

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

