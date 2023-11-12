#pragma once

#include "lane.h"
#include "random.h"

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
	template<unsigned _blocksize = 512>
	struct add_prms {
		static_assert(_blocksize % lobj::_lenLane == 0, "blocksize not divisible by warp size");
		static constexpr unsigned lanes_per_block = _blocksize / lobj::_lenLane;
		static_assert(N % lanes_per_block == 0, "N not divisible ");
		static constexpr unsigned blocksize = _blocksize;
		static constexpr unsigned numBlocks = N / lanes_per_block;
	};
	static __device__ inline void add(const warpInfo<lobj::_lenLane> & w, iVector *res, const iVector * lhs, const iVector * rhs) {
		// Do we need explicit template instantiation or is the type inferred from the arguments?
		const unsigned i = w.warpIdxGlobal;
		lobj::add(w, &res->data[i], &lhs->data[i], &rhs->data[i]);
	} 

    // random number function for host
    __host__ void fill_random(std::mt19937 & gen, typename lobj::_T min, typename lobj::_T max) {
        for (unsigned i = 0; i < N; i++) {
            data[i].fill_random(gen, min, max);
        }
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

	// lin alg operations
	template<unsigned _blocksize>
	struct matmul_prms {
		static_assert(_blocksize % lobj::_lenLane == 0, "blocksize not divisible by warp size");
		static constexpr unsigned lanes_per_block = _blocksize / lobj::_lenLane;
		static_assert(N % lanes_per_block == 0, "N not divisible by number of lanes per block");
		static constexpr unsigned blocksize = _blocksize;
		static constexpr unsigned numBlocks = N / lanes_per_block;
	};
	static __device__ inline void matmul(const warpInfo<lobj::_lenLane> & w, iVector<lobj, N> * res, const iMatrix<lobj, N> * lhs, const iVector<lobj, N> * rhs) {
		// first iteration: use add
		const unsigned i = w.warpIdxGlobal;
		lobj::mul(w, &res->data[i], &lhs->data[i][0], &rhs->data[0]);
		for (unsigned j = 1; j < N; j++) {
			lobj::mac(w, &res->data[i], &lhs->data[i][j], &rhs->data[j]);
		}
	}

    // random number function for host
    __host__ void fill_random(std::mt19937 & gen, typename lobj::_T min, typename lobj::_T max) {
        for (unsigned i = 0; i < N; i++) {
            for (unsigned j = 0; j < N; j++) {
                data[i][j].fill_random(gen, min, max);
            }
        }
    }
};

template<class lobj, unsigned N>
__global__ void run_matmul(iVector<lobj, N> * res, const iMatrix<lobj, N> * lhs, const iVector<lobj, N> * rhs) {
	warpInfo w;
	iMatrix<lobj, N>::matmul(w, res, lhs, rhs);
}
