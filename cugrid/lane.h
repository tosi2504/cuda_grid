#pragma once

#include "random.h"

// okay, so how do we do this?
// we want a simd object with operator overloads
// how to we get the simd lane length though??

// also: all data always lies on the device memory
// might need to change that later

// maybe something like this
// not sure how the functions are then executed

// basetype T should be one of int, float, double or their respective complex types 
// need to have the arithmetic operations implemented

template<unsigned lenLane = 32>
struct warpInfo {
	const unsigned warpIdx;
	const unsigned laneIdx;
	const unsigned warpNum;
	const unsigned warpIdxGlobal;

	__device__ inline warpInfo(): 
		warpIdx(threadIdx.x / lenLane),
		laneIdx(threadIdx.x % lenLane),
		warpNum(blockDim.x  / lenLane),
		warpIdxGlobal(warpNum*blockIdx.x + warpIdx)
	{}
};

template<class T, unsigned lenLane = 32>
class Lane {
	private:
	T data[lenLane];

	public:
	static constexpr unsigned _lenLane = lenLane;
    using _T = T;

	// contructor
	__host__ __device__ Lane() {}

	// getter and setter
	__host__ __device__ T& operator [] (unsigned index) { return data[index]; }
	__host__ __device__ const T& operator [] (unsigned index) const { return data[index]; }
	__device__ void setByThread(const warpInfo<lenLane> & w, const T & val) { data[w.laneIdx] = val; }
	__device__ T & getByThread(const warpInfo<lenLane> & w) const { return data[w.laneIdx]; }

	// arithmetic operations
	static __device__ inline void add(const warpInfo<lenLane> & w, Lane * res, const Lane * lhs, const Lane * rhs) {
		res->data[w.laneIdx] = lhs->data[w.laneIdx] + rhs->data[w.laneIdx];
	}
	static __device__ inline void sub(const warpInfo<lenLane> & w, Lane * res, const Lane * lhs, const Lane * rhs) {
		res->data[w.laneIdx] = lhs->data[w.laneIdx] - rhs->data[w.laneIdx];
	}
	static __device__ inline void mul(const warpInfo<lenLane> & w, Lane * res, const Lane * lhs, const Lane * rhs) {
		res->data[w.laneIdx] = lhs->data[w.laneIdx] * rhs->data[w.laneIdx];
	}
	static __device__ inline void mul(const warpInfo<lenLane> & w, Lane * res, const Lane * lhs, const Lane * rhs, const unsigned * laneIdxMap) {
		res->data[w.laneIdx] = lhs->data[laneIdxMap[w.laneIdx]] * rhs->data[w.laneIdx];
	}
	static __device__ inline void mac(const warpInfo<lenLane> & w, Lane * res, const Lane * lhs, const Lane * rhs) {
		res->data[w.laneIdx] += lhs->data[w.laneIdx] * rhs->data[w.laneIdx];
	}
	static __device__ inline void mac(const warpInfo<lenLane> & w, Lane * res, const Lane * lhs, const Lane * rhs, const unsigned * laneIdxMap) {
		res->data[w.laneIdx] += lhs->data[laneIdxMap[w.laneIdx]] * rhs->data[w.laneIdx];
	}
	static __device__ inline void div(const warpInfo<lenLane> & w, Lane * res, const Lane * lhs, const Lane * rhs) {
		res->data[w.laneIdx] = lhs->data[w.laneIdx] / rhs->data[w.laneIdx];
	}

    // random number filling
    __host__ void fill_random(std::mt19937 & gen, T min, T max) {
        for (unsigned i = 0; i < lenLane; i++) {
            data[i] = get_random_value<T>(gen, min, max);
        }
    }
};
