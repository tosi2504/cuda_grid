#include "../cugrid2/cugrid2.h"
#include "bench_params.h"
#include <random>
#include <string>
#include <cublas_v2.h>


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
	bMuStencil stencil(grid, mu, isForward);

	// run benchmark
	cublasHandle_t handle;
	cublasCCE(  cublasCreate(&handle)  );
	double resTime = 0;
	BENCHMARK(resTime, 100, stencil.execute<T COMMA N COMMA numRHS COMMA blkSize>, handle, ys, A, xs);
	cublasCCE(  cublasDestroy(handle)  );

	// print out the results
	print_results<T>("stencil_blas", resTime, N, numRHS, blkSize, grid, mu, isForward);
}
