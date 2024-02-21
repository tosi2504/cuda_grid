#include "../cugrid2/cugrid2.h"

#include <random>
#include "cublas_v2.h"

using T = complexF;
constexpr unsigned N = 32;
constexpr unsigned blkSize = 16*N;

int main () {
	bGrid grid(8,8,8,8);
	bVectorField<T,N> x(grid), y(grid);
	bMatrixField<T,N> A(grid);

	std::mt19937 gen(0);

	A.fill_random(gen, 0, 1);
	x.fill_random(gen, 0, 1);
	A.upload();
	x.upload();

	cublasHandle_t handle;
	cublasCCE(  cublasCreate(&handle)  );

	double resTime = 0;

	BENCHMARK(resTime, 100, matmul_srhs::cublas, handle, y, A, x);
	// auto func = matmul_srhs::cacheMatrix<T,N,blkSize>;
	// BENCHMARK(resTime, 100, func, y, A, x);

	// benchmark results
	y.download();
	std::cout << "One cycle took " << resTime << "us on average" << std::endl;
	std::cout << "BANDWIDTH in GB/s: " << calcBandwidthInGBs_matmul(resTime, grid.numSites, N, sizeof(T)) << std::endl;

	cublasCCE(  cublasDestroy(handle)  );


	// check results
	unsigned long site = 0;
	unsigned long i = 0;
	std::cout << "numBytes: " << sizeof(T) << std::endl;
	std::cout << y.h_data[site].data[i] << std::endl;
	std::cout << debugMatmul(A.h_data[site], x.h_data[site]).data[i] << std::endl;
}
