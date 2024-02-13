#pragma once

#include <iostream>
#include "bRandom.h"

template<class T, unsigned N>
struct bVector{
	// template parameters
	using _T = T;
	constexpr static unsigned _N = N;

	// data
	T data[N];

	//constructor
	__host__ __device__ bVector() {}

	// getter, setter, print
	__host__ __device__ T & operator[] (unsigned index) {return data[index];}
	__host__ __device__ const T & operator[] (unsigned index) const {return data[index];}
	void print() const { 
		if constexpr (is_complex_v<T>) {
			std::cout << "[  ( " << data[0].real() << " , " << data[0].imag() << " )";
			for (unsigned i = 1; i < N; i++) {
				std::cout << std::endl << "   ( " << data[i].real() << " , " << data[i].imag() << " )";
			}
			std::cout << "  ]" << std::endl;
		} else {
			std::cout << "[  " << data[0];
			for (unsigned i = 1; i < N; i++) {
				std::cout << std::endl << "   " << data[i];
			}
			std::cout << "  ]" << std::endl;
		}
	}
	
	// random number functions
	void fill_random(std::mt19937 & gen, T min, T max) {
		fill_buffer_random<T, N>(gen, data, min, max);	
	}
};


template<class T, unsigned N>
struct bMatrix{
	// template parameters
	using _T = T;
	constexpr static unsigned _N = N;

	// data
	T data[N][N];

	//constructor
	__host__ __device__ bMatrix() {}

	// getter, setter, print
	__host__ __device__ T & operator[] (unsigned index) {return data[index];}
	__host__ __device__ const T & operator[] (unsigned index) const {return data[index];}
	void print() const { 
		if constexpr (is_complex_v<T>) {
			std::cout << "[  ";
			for (unsigned j = 0; j < N-1; j++)
				std::cout << " (" << data[0][j].real() << "," << data[0][j].imag() << ") |";
			std::cout << " (" << data[0][N-1].real() << "," << data[0][N-1].imag() << ")" << std::endl;
			for (unsigned i = 1; i < N-1; i++) {
				std::cout << "   ";
				for (unsigned j = 0; j < N-1; j++)
					std::cout << " (" << data[i][j].real() << "," << data[i][j].imag() << ") |";
				std::cout << " (" << data[i][N-1].real() << "," << data[i][N-1].imag() << ")" << std::endl;
			}
			std::cout << "   ";
			for (unsigned j = 0; j < N-1; j++)
				std::cout << " (" << data[0][j].real() << "," << data[0][j].imag() << ") |";
			std::cout << " (" << data[0][N-1].real() << "," << data[0][N-1].imag() << ")   ]" << std::endl;
		} else {
			std::cout << "[  ";
			for (unsigned j = 0; j < N-1; j++) 
				std::cout << " " << data[0][j] << " |";
			std::cout << " " << data[0][N-1] << std::endl;
			for (unsigned i = 1; i < N-1; i++) {
				std::cout << "   ";
				for (unsigned j = 0; j < N-1; j++)
					std::cout << " " << data[i][j] << " |";
				std::cout << " " << data[i][N-1] << std::endl;
			}
			std::cout << "   ";
			for (unsigned j = 0; j < N-1; j++)
				std::cout << " " << data[0][j] << " |";
			std::cout << " " << data[0][N-1] << "   ]" << std::endl;
		}
	}
	
	// random number functions
	void fill_random(std::mt19937 & gen, T min, T max) {
		fill_buffer_random<T, N*N>(gen, (T*)data, min, max);	
	}
};
