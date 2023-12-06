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
using VectorBatch = std::array< Lattice<iVector<lobj,N>> * , batchsize >;

template<class lobj, unsigned N, unsigned batchsize>
bool check_grid_compatible(const VectorBatch<lobj,N,batchsize> & batch_res, const Grid<lobj::_lenLane> & grid) {
	for (unsigned b = 0; b < batchsize; b++) if (grid != batch_res[b]->grid) return false;
	return true;
}

template<class lobj, unsigned N, unsigned batchsize>
__global__ void ker_matmul_mrhs(iVector<lobj, N> ** d_batch_res
		, const iMatrix<lobj, N> * d_lhs
		, const iVector<lobj, N> ** d_batch_rhs
		, unsigned sizeVNodes) 
{
	warpInfo w;
	unsigned n = w.warpIdxGlobal / N; // vNode index
	unsigned i = w.warpIdxGlobal % N; // iTensor index
	if (n < sizeVNodes) {
		for (unsigned b = 0; b < batchsize; b++) {
			lobj::mul(w, &d_batch_res[b][n][i], &d_lhs[n][i][0], &d_batch_rhs[b][n][0]);	
			for (unsigned j = 1; j < N; j++) {
				lobj::mac(w, &d_batch_res[b][n][i], &d_lhs[n][i][j], &d_batch_rhs[b][n][j]);	
			}
		}
	}
}

template<class lobj, unsigned N, unsigned batchsize, unsigned blocksize = 256>
void matmul_mrhs(VectorBatch<lobj,N,batchsize> & batch_res
		, const Lattice< iMatrix<lobj,N> > & lhs
		, const VectorBatch<lobj,N,batchsize> & batch_rhs) 
{
	static_assert(blocksize % lobj::_lenLane == 0, "Length of lane does not divide the blocksize. Change blocksize or lane length!");
	if ((not check_grid_compatible<lobj, N, batchsize>(batch_res, lhs.grid)) or (not check_grid_compatible<lobj, N, batchsize>(batch_rhs, lhs.grid))) {
		throw std::logic_error("Grids not compatible");
	}

	// prepare arrays of pointers to batch data
	iVector<lobj, N> * h_batch_res[batchsize], * h_batch_rhs[batchsize];
	iVector<lobj, N> ** d_batch_res, ** d_batch_rhs;
	for (unsigned b = 0; b < batchsize; b++) {
		h_batch_res[b] = batch_res[b]->d_data;
		h_batch_rhs[b] = batch_rhs[b]->d_data;
	}

	// copy those arrays to device memory
	CCE(  cudaMalloc(&d_batch_res, sizeof(iVector<lobj,N>*)*batchsize)  );
	CCE(  cudaMalloc(&d_batch_rhs, sizeof(iVector<lobj,N>*)*batchsize)  );
	CCE(  cudaMemcpy(d_batch_res, h_batch_res, sizeof(iVector<lobj,N>*)*batchsize, cudaMemcpyHostToDevice)  );
	CCE(  cudaMemcpy(d_batch_rhs, h_batch_rhs, sizeof(iVector<lobj,N>*)*batchsize, cudaMemcpyHostToDevice)  );

	// kernel call!
	unsigned lanes_per_block = blocksize / lobj::_lenLane;
	unsigned blocks = (lhs.sizeVNodes*N + lanes_per_block - 1)/lanes_per_block;
	std::cout << "calling ker_matmul_mrhs with:" << std::endl;
	std::cout << "    blocks  : " << blocks << std::endl;
	std::cout << "    threads : " << blocksize << std::endl;
	std::cout << "    #lpb    : " << lanes_per_block << std::endl;
	ker_matmul_mrhs<lobj, N, batchsize><<< blocks , blocksize >>>(d_batch_res, lhs.d_data, d_batch_rhs, lhs.sizeVNodes);
	CCE(cudaDeviceSynchronize());

	cudaFree(d_batch_res);
	cudaFree(d_batch_rhs);
}
