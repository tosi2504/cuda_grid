#pragma once

#include "lane.h"

#include <iostream>

// implements scalar, vector and matrix objects
// focus for now is on matrix * vector operations

// implements all operations on Lanes

template<class lobj, unsigned N>
class iVector {
	public:
	lobj data[N];

	using _lobj = lobj;
	constexpr static unsigned _N = N;

	// constructor
	__host__ __device__ iVector() {}

	// getter and setter and repr
	__host__ __device__ lobj & operator [] (unsigned index) {
		return data[index]; 
	}
	__host__ __device__ const lobj & operator [] (unsigned index) const {
		return data[index];
	}
	__host__ void print() const {
		std::cout << "Lanes expand to the right" << std::endl;
		for (unsigned i = 0; i < N; i++) {
			for (unsigned j = 0; j < lobj::_lenLane; j++) {
				std::cout << data[i][j] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "End of vector" << std::endl;
	}
	__host__ void print(unsigned laneIdx) const {
		for (unsigned i = 0; i < N; i++) {
			std::cout << data[i][laneIdx] << " ";
		}
		std::cout << std::endl;
	}

	// lin alg operations
	// this is kernel parameter critial -> i.e. it needs to be called correctly
	// the correct call would be one kernel call per LinAlg operation
	// since iVector effectively contains 32 vectors, this is efficient
	// blocks * numWarps = blocks * (threads_per_block % lenLane) needs to be equal to N
	// here a metafunction to calculate the number of blocks required
	// do we really need a metafunction though xDDDDDDD
	// fuck yeah, we do! (no, we don't)
	template<unsigned _blocksize = 512>
	struct add_prms {
		static_assert(_blocksize % lobj::_lenLane == 0, "blocksize not divisible by warp size");
		static constexpr unsigned lanes_per_block = _blocksize / lobj::_lenLane;
		static_assert(N % lanes_per_block == 0, "N not divisible ");
		static constexpr unsigned blocksize = _blocksize;
		static constexpr unsigned numBlocks = N / lanes_per_block;
	};
	static __device__ void add(const warpInfo<lobj::_lenLane> & w, iVector *res, const iVector * lhs, const iVector * rhs) {
		// Do we need explicit template instantiation or is the type inferred from the arguments?
		const unsigned i = w.warpIdxGlobal;
		lobj::add(w, &res->data[i], &lhs->data[i], &rhs->data[i]);
	} 
};

template<class lobj, unsigned N>
__global__ void run_add(iVector<lobj, N> * res, const iVector<lobj, N> * lhs, const iVector<lobj, N> * rhs) {
	warpInfo w;
	iVector<lobj, N>::add(w, res, lhs, rhs);
}

template<class lobj, unsigned N>
class iMatrix {
	public:
	lobj data[N][N];

	using _lobj = lobj;
	constexpr static unsigned _N = N;

	// constructor
	__host__ __device__ iMatrix() {}

	// getter and setter
	__host__ __device__  lobj * operator [] (unsigned index) { return &data[index][0]; }
	__host__ __device__  const lobj * operator [] (unsigned index) const { return &data[index][0]; }

	// lin alg
	template<unsigned _blocksize>
	struct matmul_prms {
		static_assert(_blocksize % lobj::_lenLane == 0, "blocksize not divisible by warp size");
		static constexpr unsigned lanes_per_block = _blocksize / lobj::_lenLane;
		static_assert(N % lanes_per_block == 0, "N not divisible by number of lanes per block");
		static constexpr unsigned blocksize = _blocksize;
		static constexpr unsigned numBlocks = N / lanes_per_block;
	};
	static __device__ void matmul(const warpInfo<lobj::_lenLane> & w, iVector<lobj, N> * res, const iMatrix<lobj, N> * lhs, const iVector<lobj, N> * rhs) {
		// first iteration: use add
		const unsigned i = w.warpIdxGlobal;
		lobj::mul(w, &res->data[i], &lhs->data[i][0], &rhs->data[0]);
		for (unsigned j = 1; j < N; j++) {
			lobj::mac(w, &res->data[i], &lhs->data[i][j], &rhs->data[j]);
		}
	}
};

template<class lobj, unsigned N>
__global__ void run_matmul(iVector<lobj, N> * res, const iMatrix<lobj, N> * lhs, const iVector<lobj, N> * rhs) {
	warpInfo w;
	iMatrix<lobj, N>::matmul(w, res, lhs, rhs);
}
