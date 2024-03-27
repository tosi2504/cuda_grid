#include "../cugrid2/cugrid2.h"
#include "bench_params.h"
#include <random>
#include <string>
#include "cublas_v2.h"


using T = BENCH_PARAM_T;
constexpr unsigned N = BENCH_PARAM_N;
constexpr unsigned numRHS = BENCH_PARAM_numRHS;
constexpr unsigned blkSize = BENCH_PARAM_blkSize;




template<class T, unsigned N, unsigned numRHS, unsigned blkSize>
void copy_with_malloc(bVectorField<T,N> * const * const ys
		, const bMatrixField<T,N> & A
		, const bVectorField<T,N> * const * xs)
{
	// check compatibility of grids
	if (not mrhs_helper::areGridsCompatible<T,N,numRHS>(ys, A, xs)) throw std::invalid_argument("Grids are not compatible");
	const bGrid & grid = A.grid;

	// create intermediate matfields
	T * d_X, * d_Y;
	CCE(  cudaMalloc(&d_X, sizeof(T)*numRHS*grid.numSites*N)  );
	CCE(  cudaMalloc(&d_Y, sizeof(T)*numRHS*grid.numSites*N)  );

	// copy inputs to matrixfield X
	mrhs_helper::fillMatrixfieldFromBatch<T,N,numRHS,blkSize>(d_X, xs);

	// copy result to vectorfields ys
	mrhs_helper::fillBatchFromMatrixfield<T,N,numRHS,blkSize>(ys, d_Y);

	// free temporary matfields
	CCE(  cudaFree(d_X)  );
	CCE(  cudaFree(d_Y)  );
}



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
	double resTime = 0;

	auto func = copy_with_malloc<T,N,numRHS,blkSize>;
	BENCHMARK(resTime, 1000, func, ys, A, xs);

	// print out the results
	print_results<T>("copy_withMalloc", resTime, N, numRHS, blkSize, grid, 999, true);
}
