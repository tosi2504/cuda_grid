#pragma once

#include "lane.h"

// implements scalar, vector and matrix objects
// focus for now is on matrix * vector operations



// implements all operations on Lanes
template<class Lane<T, lenLane>, unsigned N>
class Vector {
	private:
	Lane<T> data[N];

	public:
	// constructor
	__host__ __device__ Vector() {}

	// getter and setter
	__host__ __device__ Lane<T>& operator [] (unsigned index) {return data[index]; }
	__host__ __device__ Lane<T>& operator [] (unsigned index) const {return data[index]; }

	// lin alg operations
	static __device__ void add(const warpInfo<Lane::lenLane> & w,  CONTINUE HERE!!!!!!!!!!); 
};
