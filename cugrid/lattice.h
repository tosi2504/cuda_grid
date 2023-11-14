#pragma once

#include <cuda.h>
#include <type_traits>
#include <stdexcept>

#include "errorcheck.h"
#include "grid.h"
#include "tensor.h"
#include "random.h"


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
    using _T = typename tobj::_lobj::_T;
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

	// getters and setters ... for now only on host
    const tobj & operator [] (unsigned idx) const {
        return h_data[idx];
    }
    tobj & operator [] (unsigned idx) {
        return h_data[idx];
    }
    

	// memory management
	void upload() {
		CCE(cudaMemcpy(d_data, h_data, lenBuffer*sizeof(tobj), cudaMemcpyHostToDevice));
	}
	void download() {
		CCE(cudaMemcpy(h_data, d_data, lenBuffer*sizeof(tobj), cudaMemcpyDeviceToHost));
	}

	// arithmetic operations
	template<class lobj, unsigned N, unsigned blocksize>
	friend void add(Lattice<iVector<lobj, N>> * res, const Lattice<iVector<lobj, N>> * lhs, const Lattice<iVector<lobj, N>> * rhs);

	template<class lobj, unsigned N, unsigned blocksize>
	friend void matmul(Lattice<iVector<lobj, N>> * res, const Lattice<iMatrix<lobj, N>> * lhs, const Lattice<iVector<lobj, N>> * rhs);


    // fill random
    void fill_random(unsigned seed, _T min, _T max) {
        // okaaaaay so
        // the problem is that lobj could have any of the fundamental data types int, float, double, complex<>
        // so I need a random number generator for each of these types
        std::mt19937 gen(seed);
        for (unsigned x = 0; x < lenBuffer; x++) {
            h_data[x].fill_random(gen, min, max);
        } 
    }
};

// arithmetic operations
template<class lobj, unsigned N, unsigned blocksize = 256>
void add(Lattice<iVector<lobj, N>> * res, const Lattice<iVector<lobj, N>> * lhs, const Lattice<iVector<lobj, N>> * rhs) {
	if (res->grid != lhs->grid or res->grid != rhs->grid) throw std::logic_error("Grids do not match!");
	using prms = typename iVector<lobj, N>::add_prms<blocksize>;
	for (unsigned int x = 0; x < res->lenBuffer; x++) {
		run_add<<<prms::numBlocks, prms::blocksize>>>(&res->d_data[x], &lhs->d_data[x], &rhs->d_data[x]);
	}
	CLCE();
	CCE(cudaDeviceSynchronize());
}
template<class lobj, unsigned N, unsigned blocksize = 256>
void matmul(Lattice<iVector<lobj, N>> * res, const Lattice<iMatrix<lobj, N>> * lhs, const Lattice<iVector<lobj, N>> * rhs) {
	if (res->grid != lhs->grid or res->grid != rhs->grid) throw std::logic_error("Grids do not match!");
	using prms = typename iMatrix<lobj, N>::matmul_prms<blocksize>;
	for (unsigned int x = 0; x < res->lenBuffer; x++) {
		run_matmul<<<prms::numBlocks, prms::blocksize>>>(&res->d_data[x], &lhs->d_data[x], &rhs->d_data[x]);
	}
	CLCE();
	CCE(cudaDeviceSynchronize());
}

