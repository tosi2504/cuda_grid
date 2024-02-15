#pragma once


#include <chrono>
#include <type_traits>


#define BENCHMARK(resTime, repetitions, func, ...) \
{ \
	static_assert(std::is_floating_point_v<decltype(resTime)>); \
	static_assert(std::is_integral_v<decltype(repetitions)>); \
	auto start_time = std::chrono::high_resolution_clock::now(); \
	for(unsigned rep = 0; rep < (repetitions); rep++) func(__VA_ARGS__); \
	auto end_time = std::chrono::high_resolution_clock::now(); \
	resTime = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / (double)(repetitions); \
}

double calcBandwidthInGBs_matmul(const double resTime
									, const unsigned long numSites
									, const unsigned long N
									, const unsigned long numBytes) {
	return (numSites*numBytes*(N*N + 2*N)/resTime)/1000;
}

double calcBandwidthInGBs_matmul_mrhs(const double resTime
									, const unsigned long numSites
									, const unsigned long N
									, const unsigned long numBytes
									, const unsigned long numRHS) {
	return (numSites*numBytes*(N*N + 2*N)*numRHS/resTime)/1000;
}
