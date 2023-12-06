#pragma once

#include <cuda.h>
#include <type_traits>
#include <stdexcept>
#include <omp.h>

#include "errorcheck.h"
#include "grid.h"
#include "tensor.h"
#include "random.h"


template<class tobj>
class Lattice {
	static_assert(is_Tensor<tobj>::value, "Template parameter must be of tensor type!");

	public:
	constexpr static unsigned lenLane = tobj::_lobj::_lenLane; 
	constexpr static unsigned N = tobj::_N; 
	const Grid<lenLane> grid;

	unsigned sizeVNodes;
	tobj * d_data, * h_data;
	
	using _tobj = tobj;
    using _T = typename tobj::_lobj::_T;

	Lattice(const Grid<lenLane> & grid):
		grid(grid)
	{
		sizeVNodes = grid.calcSizeVNodes();
		h_data = new tobj[sizeVNodes];
		CCE(cudaMalloc(&d_data, sizeVNodes*sizeof(tobj)));
	}
	~Lattice() {
		delete[] h_data;
		CCE(cudaFree(d_data));
	}
	Lattice(const Lattice<tobj> & other):
		grid(other.grid)
	{
		sizeVNodes = grid.calcSizeVNodes();
		h_data = new tobj[sizeVNodes];
		CCE(cudaMalloc(&d_data, sizeVNodes*sizeof(tobj)));
	}

	// getters and setters ... for now only on host
    const tobj & operator [] (unsigned idx) const {
        return h_data[idx];
    }
    tobj & operator [] (unsigned idx) {
        return h_data[idx];
    }
	// and for specific tensor entries
	const _T & get (const cart & coords, unsigned i) const {
	 	flat f = grid.toFlat(coords);
	 	return h_data[f.n][i][f.l];
	}
	_T & get(const cart & coords, unsigned i) {
		flat f = grid.toFlat(coords);
		return h_data[f.n][i][f.l];
	}
	const _T & get(const cart & coords, unsigned i, unsigned j) const {
		flat f = grid.toFlat(coords);
		return h_data[f.n][i][j][f.l];
	}
	_T & get(const cart & coords, unsigned i, unsigned j) {
		flat f = grid.toFlat(coords);
		return h_data[f.n][i][j][f.l];
	}
    

	// memory management
	void upload() {
		CCE(cudaMemcpy(d_data, h_data, sizeVNodes*sizeof(tobj), cudaMemcpyHostToDevice));
	}
	void download() {
		CCE(cudaMemcpy(h_data, d_data, sizeVNodes*sizeof(tobj), cudaMemcpyDeviceToHost));
	}

    // fill random
    void fill_random(unsigned seed, _T min, _T max) {
        std::mt19937 gen(seed);
        for (unsigned x = 0; x < sizeVNodes; x++) {
            h_data[x].fill_random(gen, min, max);
        } 
	}

	void fill_benchmark(unsigned seed, _T min, _T max) {
        std::mt19937 gen(seed);

		// create array with 1000 random numbers
		unsigned lenRandBuffer = 1000;
		_T rn[lenRandBuffer];
		for (unsigned i = 0; i < lenRandBuffer; i++) {
			rn[i] = get_random_value(gen, min, max);		
		}

		// copy the data into the buffers
		if constexpr (is_Matrix<tobj>::value) {
			#pragma omp parallel for
			for (unsigned x = 0; x < sizeVNodes; x++) {
				for (unsigned i = 0; i < N; i++) {
					for (unsigned j = 0; j < N; j++) {
						for (unsigned l = 0; l < lenLane; l++) {
							h_data[x][i][j][l] = rn[(x*N*N*lenLane + i*N*lenLane + j*lenLane + l) % lenRandBuffer];
						}
					}
				}
			}
		} else if constexpr (is_Vector<tobj>::value) {
			#pragma omp parallel for
			for (unsigned x = 0; x < sizeVNodes; x++) {
				for (unsigned i = 0; i < N; i++) {
					for (unsigned l = 0; l < lenLane; l++) {
						h_data[x][i][l] = rn[(x*N*lenLane + i*lenLane + l) % lenRandBuffer];
					}
				}
			}
		}
	}
};

// arithmetic operations
template<class lobj, unsigned N, unsigned blocksize = 256>
void add(Lattice<iVector<lobj, N>> * res, const Lattice<iVector<lobj, N>> * lhs, const Lattice<iVector<lobj, N>> * rhs) {
	if (res->grid != lhs->grid or res->grid != rhs->grid) throw std::logic_error("Grids do not match!");
	using prms = typename iVector<lobj, N>::add_prms<blocksize>;
	for (unsigned int x = 0; x < res->sizeVNodes; x++) {
		run_add<<<prms::numBlocks, prms::blocksize>>>(&res->d_data[x], &lhs->d_data[x], &rhs->d_data[x]);
	}
	CLCE();
	CCE(cudaDeviceSynchronize());
}

