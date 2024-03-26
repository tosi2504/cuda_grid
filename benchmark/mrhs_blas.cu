#include "../cugrid2/cugrid2.h"
#include "bench_params.h"
#include <random>
#include <string>
#include "cublas_v2.h"


using T = BENCH_PARAM_T;
constexpr unsigned N = BENCH_PARAM_N;
constexpr unsigned numRHS = BENCH_PARAM_numRHS;
constexpr unsigned blkSize = BENCH_PARAM_blkSize;


int main (int argc, char * argv[]) {
	unsigned Lx, Ly, Lz, Lt;
	unsigned mu;
	bool isForward;
	parseArgs(argc, argv, &Lx, &Ly, &Lz, &Lt, &mu, &isForward);
	bGrid grid(Lx,Ly,Lz,Lt);

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

	// auto func = matmul_mrhs::cacheMatrix<T,N,numRHS,blkSize>;
	// BENCHMARK(resTime, 1000, func, ys, A, xs);

	T * d_X, * d_Y;
	CCE(  cudaMalloc(&d_X, sizeof(T)*numRHS*grid.numSites*N)  );
	CCE(  cudaMalloc(&d_Y, sizeof(T)*numRHS*grid.numSites*N)  );
	auto func = matmul_mrhs::gemm<T,N,numRHS,blkSize>;
	BENCHMARK(resTime, 1000, func, handle, ys, A, xs, d_Y, d_X);
	CCE(  cudaFree(d_X)  );
	CCE(  cudaFree(d_Y)  );

	cublasCCE(  cublasDestroy(handle)  );

	// print out the results
	print_results<T>("matmul", resTime, N, numRHS, blkSize, grid, 999, true);
}
