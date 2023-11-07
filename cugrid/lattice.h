#pragma once

#include <cuda.h>
#include <type_traits>
#include <stdexcept>

#include "errorcheck.h"
#include "grid.h"
#include "tensor.h"


// needs to contain some memory management --> rip
// maybe two buffers, one for host, one for device?

template<class tobj>
class Lattice {
	private:
	constexpr static unsigned lenLane = tobj::_lobj::_lenLane; 
	const Grid<lenLane> grid;

	unsigned lenBuffer;
	tobj * d_data, * h_data;

	public:
	using _tobj = tobj;
	Lattice(const Grid<lenLane> & grid):
		grid(grid)
	{
		lenBuffer = grid.calcLatticeBufferSize();
		h_data = new tobj[lenBuffer];
		CCE(cudaMalloc(&d_data, lenBuffer*sizeof(tobj)));
	}
	~Lattice() {
		delete[] h_data;
		CCE(cudaFree(d_data));
	}

	// getters and setters

	// memory management
	void upload() {
		CCE(cudaMemcpy(d_data, h_data, lenBuffer*sizeof(tobj), cudaMemcpyHostToDevice));
	}
	void download() {
		CCE(cudaMemcpy(h_data, d_data, lenBuffer*sizeof(tobj), cudaMemcpyDeviceToHost));
	}

	//arithmetic operations
	template<class lobj, unsigned N>
	friend void add(Lattice<iVector<lobj, N>> * res, const Lattice<iVector<lobj, N>> * lhs, const Lattice<iVector<lobj, N>> * rhs);

	template<class lobj, unsigned N>
	friend void matmul(Lattice<iVector<lobj, N>> * res, const Lattice<iMatrix<lobj, N>> * lhs, const Lattice<iVector<lobj, N>> * rhs);
};

// arithmetic operations
template<class lobj, unsigned N>
void add(Lattice<iVector<lobj, N>> * res, const Lattice<iVector<lobj, N>> * lhs, const Lattice<iVector<lobj, N>> * rhs) {
	if (res->grid != lhs->grid or res->grid != rhs->grid) throw std::logic_error("Grids do not match!");
	using prms = typename iVector<lobj, N>::add_prms<512>;
	for (unsigned int x = 0; x < res->lenBuffer; x++) {
		run_add<<<prms::numBlocks, prms::blocksize>>>(&res->d_data[x], &lhs->d_data[x], &rhs->d_data[x]);
	}
	CLCE();
	CCE(cudaDeviceSynchronize());
}
template<class lobj, unsigned N>
void matmul(Lattice<iVector<lobj, N>> * res, const Lattice<iMatrix<lobj, N>> * lhs, const Lattice<iVector<lobj, N>> * rhs) {
	if (res->grid != lhs->grid or res->grid != rhs->grid) throw std::logic_error("Grids do not match!");
	using prms = typename iMatrix<lobj, N>::matmul_prms<512>;
	for (unsigned int x = 0; x < res->lenBuffer; x++) {
		run_matmul<<<prms::numBlocks, prms::blocksize>>>(&res->d_data[x], &lhs->d_data[x], &rhs->d_data[x]);
	}
	CLCE();
	CCE(cudaDeviceSynchronize());
}

