#include "../cugrid2/cugrid2.h"

#include <random>
#include "cublas_v2.h"

template<class T, unsigned N>
bVectorField<T,N> ** createBatchVecFields(const unsigned numRHS, const bGrid & grid) {
	bVectorField<T,N> ** res = new bVectorField<T,N>*[numRHS];
	for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) res[iRHS] = new bVectorField<T,N>(grid);
	return res;
}
template<class T, unsigned N>
bVectorField<T,N> ** createAndFillAndUploadBatchVecFields(const unsigned numRHS
					, const bGrid & grid
					, std::mt19937 & gen
					, T min, T max) {
	bVectorField<T,N> ** res = new bVectorField<T,N>*[numRHS];
	for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) { 
		res[iRHS] = new bVectorField<T,N>(grid);
		res[iRHS]->fill_random(gen, min, max);
		res[iRHS]->upload();
	}
	return res;
}

using T = realF;
constexpr unsigned N = 16;
constexpr unsigned numRHS = 8;

int main () {
	bGrid grid(8,8,16,16);
	std::mt19937 gen(0);

	// prepare fields
	bVectorField<T,N> ** xs = createAndFillAndUploadBatchVecFields<T,N>(numRHS, grid, gen, 0, 1);
	bVectorField<T,N> ** ys = createBatchVecFields<T,N>(numRHS, grid);
	bMatrixField<T,N> A(grid);
	A.fill_random(gen, 0, 1);
	A.upload();
	std::cout << "Fields allocated and randomly filled" << std::endl;

	// run benchmark
	cublasHandle_t handle;
	cublasCCE(  cublasCreate(&handle)  );
	double resTime = 0;
	// auto func = matmul_mrhs::naive<T,N,numRHS>;
	// BENCHMARK(resTime, 1000, func, handle, ys, A, xs);
	auto func = matmul_mrhs::cacheMatrix<T,N,numRHS>;
	BENCHMARK(resTime, 1000, func, ys, A, xs, 8*N);
	std::cout << "T has numBytes: " << sizeof(T) << std::endl;
	std::cout << "One cycle took " << resTime << "us on average" << std::endl;
	std::cout << "BANDWIDTH in GB/s: " << calcBandwidthInGBs_matmul_mrhs(resTime, grid.numSites, N, sizeof(T), numRHS) << std::endl;
	cublasCCE(  cublasDestroy(handle)  );
    
	// check results
	for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) ys[iRHS]->download();
	unsigned long site = grid.numSites-1;
	unsigned long i = N-1;
	unsigned iRHS = numRHS-1;
	std::cout << ys[iRHS]->h_data[site].data[i] << std::endl;
	std::cout << debugMatmul(A.h_data[site], xs[iRHS]->h_data[site]).data[i] << std::endl;
}
