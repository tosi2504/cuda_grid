#pragma once

#include <chrono>
#include <type_traits>

#define COMMA ,

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

// template<class T, unsigned N>
// bVectorField<T,N> ** createBatchVecFields(const unsigned numRHS, const bGrid & grid) {
// 	bVectorField<T,N> ** res = new bVectorField<T,N>*[numRHS];
// 	for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) res[iRHS] = new bVectorField<T,N>(grid);
// 	return res;
// }
// template<class T, unsigned N>
// bVectorField<T,N> ** createAndFillAndUploadBatchVecFields(const unsigned numRHS
// 					, const bGrid & grid
// 					, std::mt19937 & gen
// 					, T min, T max) {
// 	bVectorField<T,N> ** res = new bVectorField<T,N>*[numRHS];
// 	for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) { 
// 		res[iRHS] = new bVectorField<T,N>(grid);
// 		res[iRHS]->fill_random(gen, min, max);
// 		res[iRHS]->upload();
// 	}
// 	return res;
// }
template<class T, unsigned lenLane>
void print_results(const char * task, double resTime, unsigned N, unsigned numRHS, unsigned blkSize, const Grid<lenLane> & grid, unsigned mu, bool isForward) {
	std::cout << "========= BENCHMARK RESULTS =========" << std::endl;
	std::cout << "  Task              : " << task << std::endl;
	std::cout << "  mu                : " << mu << std::endl;
	std::cout << "  isForward         : " << (isForward ? "true" : "false") << std::endl;
	std::cout << "  Arithmetic type   : " << type_as_string<T>::value << std::endl;
	std::cout << "     --> bytes      : " << sizeof(T) << std::endl;
	std::cout << "  Tensor size       : " << N << std::endl;
	std::cout << "  numRHS            : " << numRHS << std::endl;
	std::cout << "  Grid config       : (" << grid.Lx << "," << grid.Ly << "," << grid.Lz << "," << grid.Lt << ")" << std::endl;
	std::cout << "     --> numSites   : " << grid.vol << std::endl;
	std::cout << "     --> lenLane    : " << lenLane << std::endl;
	std::cout << "  Block size        : " << blkSize << std::endl;
	std::cout << "  One cycle took    : " << resTime << "us (on average)" << std::endl;
	std::cout << "  srhs-Bandw. GB/s  : " << calcBandwidthInGBs_matmul_mrhs(resTime, grid.numSites, N, sizeof(T), numRHS) << std::endl;
	std::cout << "  mrhs-Bandw. GB/s  : " << ((N*N + 2*N*numRHS)*(long)grid.numSites*sizeof(T))/(resTime*1000) << std::endl;
	std::cout << "=====================================" << std::endl;
}