#pragma once

#include "lattice.h"

#include <array>
#include <stdexcept>


// optimized arithmetic operations
template<class lobj, unsigned N>
__global__ void ker_matmul(iVector<lobj, N> * d_res, const iMatrix<lobj, N> * d_lhs, const iVector<lobj, N> * d_rhs, unsigned sizeVNodes) {
	warpInfo w;
	unsigned n = w.warpIdxGlobal / N; // vNode index
	unsigned i = w.warpIdxGlobal % N; // iTensor index
	if (n < sizeVNodes) {
		lobj::mul(w, &d_res[n][i], &d_lhs[n][i][0], &d_rhs[n][0]);	
		for (unsigned j = 1; j < N; j++) {
			lobj::mac(w, &d_res[n][i], &d_lhs[n][i][j], &d_rhs[n][j]);	
		}
	}
}
template<class lobj, unsigned N, unsigned blocksize = 256>
void matmul_opt(Lattice<iVector<lobj, N>> & res, const Lattice<iMatrix<lobj, N>> & lhs, const Lattice<iVector<lobj, N>> & rhs) {
	static_assert(blocksize % lobj::_lenLane == 0, "Length of lane does not divide the blocksize. Change blocksize or lane length!");
	if (res.grid != lhs.grid or res.grid != rhs.grid) throw std::logic_error("Grids do not match!");
	unsigned lanes_per_block = blocksize / lobj::_lenLane;
	unsigned blocks = (res.sizeVNodes*N + lanes_per_block - 1)/lanes_per_block;
	std::cout << "calling ker_matmul with:" << std::endl;
	std::cout << "    blocks  : " << blocks << std::endl;
	std::cout << "    threads : " << blocksize << std::endl;
	std::cout << "    #lpb    : " << lanes_per_block << std::endl;
	ker_matmul<lobj, N><<< blocks , blocksize >>>(res.d_data, lhs.d_data, rhs.d_data, res.sizeVNodes);
	CCE(cudaDeviceSynchronize());
}


template<class lobj, unsigned N, unsigned batchsize>
using VectorBatch = std::array< Lattice<iVector<lobj,N>> * , batchsize >

template<class lobj, unsigned N, unsigned batchsize>
bool check_grid_compatible(const VectorBatch<lobj,N,batchsize> & batch_res, const Grid<lobj::_lenLane> & grid) {
	for (unsigned b = 0; b < batchsize; b++) if (grid != batch_res[b]->grid) return false;
	return true;
}

// okay now multiple right hand sides
// so how would the call signature of such a function look like
template<class lobj, unsigned N, unsigned batchsize>
void matmul_mrhs(VectorBatch<lobj,N,batchsize> & batch_res, const Lattice< iMatrix<lobj,N> > & lhs, const VectorBatch<lobj,N,batchsize> & batch_rhs) {
	// we should check whether all grids are compatible
	// Tensor sizes, lenLane and arithmetic type should be compatible per template definition
	if ((not check_grid_compatible(batch_res, lhs.grid)) or (not check_grid_compatible(batch_rhs, lhs.grid))) {
		throw std::logic_error("Grids not compatible");
	}


}
