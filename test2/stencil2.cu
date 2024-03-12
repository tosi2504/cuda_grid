#include "../cugrid2/cugrid2.h"

#include <random>
#include <string>
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
template<class T>
void print_results(double resTime, unsigned N, unsigned numRHS, unsigned blkSize, const bGrid & grid) {
	std::cout << "========= BENCHMARK RESULTS =========" << std::endl;
	std::cout << "  Arithmetic type   : " << type_as_string<T>::value << std::endl;
	std::cout << "     --> bytes      : " << sizeof(T) << std::endl;
	std::cout << "  Tensor size       : " << N << std::endl;
	std::cout << "  numRHS            : " << numRHS << std::endl;
	std::cout << "  Grid config       : (" << grid.Lx << "," << grid.Ly << "," << grid.Lz << "," << grid.Lt << ")" << std::endl;
	std::cout << "     --> numSites   : " << grid.numSites << std::endl;
	std::cout << "  Block size        : " << blkSize << std::endl;
	std::cout << "  One cycle took    : " << resTime << "us (on average)" << std::endl;
	std::cout << "  srhs-Bandw. GB/s  : " << calcBandwidthInGBs_matmul_mrhs(resTime, grid.numSites, N, sizeof(T), numRHS) << std::endl;
	std::cout << "  mrhs-Bandw. GB/s  : " << ((N*N + 2*N*numRHS)*(long)grid.numSites*sizeof(T))/(resTime*1000) << std::endl;
	std::cout << "=====================================" << std::endl;
}

using T = realF;
constexpr unsigned N = 128;
constexpr unsigned numRHS = 8;
constexpr unsigned blkSize = 4*N;
constexpr unsigned mu = 0;
constexpr bool isForward = true;

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
	bMuStencil stencil(grid, mu, isForward);

	// run benchmark
	cublasHandle_t handle;
	cublasCCE(  cublasCreate(&handle)  );
	double resTime = 0;
	BENCHMARK(resTime, 100, stencil.execute<T COMMA N COMMA numRHS COMMA blkSize>, handle, ys, A, xs);
	cublasCCE(  cublasDestroy(handle)  );

	// print out the results
	print_results<T>(resTime, N, numRHS, blkSize, grid);
    
	// check results
	for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) ys[iRHS]->download();
	unsigned long site = 0;//grid.numSites-1;
	unsigned long i = 0;//N-1;
	unsigned iRHS = 0;//numRHS-1;
	std::cout << ys[iRHS]->h_data[site].data[i] << std::endl;
	std::cout << debugMatmul(A.h_data[site], xs[iRHS]->h_data[site]).data[i] << std::endl;
}
