#pragma once

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
	unsigned warpIdx;
	unsigned laneIdx;

	__device__ warpInfo() {
		warpIdx = threadIdx.x / lenLane;
		laneIdx = threadIdx.x % lenLane;
	}
};

template<class T, unsigned lenLane = 32>
class Lane {
	private:
	T data[lenLane];

	public:
	// contructor
	__host__ __device__ Lane() {}
	
	// getter and setter
	__host__ __device__ T& operator [] (unsigned index) { return data[index]; }
	__host__ __device__ T& operator [] (unsigned index) const { return data[index]; }
	__device__ void setByThread(const T & val) { warpInfo w; data[w.laneIdx] = val; }
	__device__ T & getByThread() { warpInfo w; return data[w.laneIdx]; }

	// arithmetic operations
	static __device__ void add(Lane * res, const Lane * lhs, const Lane * rhs) {
		warpInfo w;
		res->data[w.laneIdx] = lhs->data[w.laneIdx] + rhs->data[w.laneIdx];
	}
	static __device__ void sub(Lane * res, const Lane * lhs, const Lane * rhs) {
		warpInfo w;
		res->data[w.laneIdx] = lhs->data[w.laneIdx] - rhs->data[w.laneIdx];
	}
	static __device__ void mul(Lane * res, const Lane * lhs, const Lane * rhs) {
		warpInfo w;
		res->data[w.laneIdx] = lhs->data[w.laneIdx] * rhs->data[w.laneIdx];
	}
	static __device__ void mac(Lane * res, const Lane * lhs, const Lane * rhs) {
		warpInfo w;
		res->data[w.laneIdx] += lhs->data[w.laneIdx] * rhs->data[w.laneIdx];
	}
	static __device__ void div(Lane * res, const Lane * lhs, const Lane * rhs) {
		warpInfo w;
		res->data[w.laneIdx] = lhs->data[w.laneIdx] / rhs->data[w.laneIdx];
	}
};
