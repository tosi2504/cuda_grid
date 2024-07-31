#include "../cugrid2/cugrid2.h"

#include <random>
#include "cublas_v2.h"
#include "cugrid2/bLattice.h"
#include "cugrid2/bStencil.h"
#include "cugrid2/errorcheck.h"
#include "cugrid2/stopwatch.h"


constexpr unsigned reps = 100; 

const bGrid grids[] = {bGrid(4,4,4,4)
                    , bGrid(4,4,8,8)
                    , bGrid(8,8,8,8)};
                    // , bGrid(16,16,16,16)};

constexpr unsigned get_blkSize(unsigned N, unsigned numRHS) {
    switch(numRHS) {
        case 1:
            return N;
            break;
        case 12:
            return (N >= 64) ? 256 : 128;
            break;
        case 24:
            return 256;
            break;
        case 36:
            return  (N >= 64) ? 256 : 128;
            break;
        case 48:
            return 256;
            break;
        case 60:
            return (N >= 64) ? 256 : 128;
            break;
        default:
            return -1;
    }
}

template<class T, unsigned N, unsigned numRHS>
void runBenchmark(
        cublasHandle_t & handle,
        bVectorField<T, 128> ** ys,
        const bMatrixField<T, 128> & A,
        bVectorField<T, 128> ** xs,
        T * d_Y, T * d_X
) {
    constexpr unsigned blkSize = get_blkSize(N, numRHS);
    for (unsigned i_grid = 0; i_grid < sizeof(grids)/sizeof(bGrid); ++i_grid) {
        bVectorField<T,N> ** ys_temp = new bVectorField<T,N>*[numRHS];
        bVectorField<T,N> ** xs_temp = new bVectorField<T,N>*[numRHS];
        for (unsigned i_rhs = 0; i_rhs < numRHS; ++i_rhs) {
            ys_temp[i_rhs] = new bVectorField<T,N>(grids[i_grid], *(ys[i_rhs]));
            xs_temp[i_rhs] = new bVectorField<T,N>(grids[i_grid], *(xs[i_rhs]));
        }
        bMatrixField<T,N> A_temp(grids[i_grid], A);
        
        bFullStencil stencil(grids[i_grid]);

        for (unsigned i = 0; i<reps; i++) {
            stopwatch.reset();
            // perform the call 
            stencil.execute_shmem<T,N,numRHS,blkSize>(ys_temp
                                                  , A_temp
                                                  , xs_temp);
            // read out stopwatch
            std::cout << i << ",";
            std::cout << grids[i_grid].Lx << ".";
            std::cout << grids[i_grid].Ly << ".";
            std::cout << grids[i_grid].Lz << ".";
            std::cout << grids[i_grid].Lt << ",";
            std::cout << N << ",";
            std::cout << numRHS << ",";
            std::cout << blkSize << ",";
            std::cout << stopwatch.getdiff(1) << std::endl;
        }
        for (unsigned i_rhs = 0; i_rhs < numRHS; ++i_rhs) {
            delete ys_temp[i_rhs];
            delete xs_temp[i_rhs];
        }
        delete[] ys_temp;
        delete[] xs_temp;
    }
}

template<class T, unsigned N>
void iterate_over_numRHS(
        cublasHandle_t & handle,
        bVectorField<T, 128> ** ys,
        const bMatrixField<T,128> & A,
        bVectorField<T, 128> ** xs,
        T * d_Y, T * d_X
) {
    runBenchmark<T, N, 1>(handle,ys,A,xs,d_Y,d_X);
    runBenchmark<T, N, 12>(handle,ys,A,xs,d_Y,d_X);
    runBenchmark<T, N, 24>(handle,ys,A,xs,d_Y,d_X);
    runBenchmark<T, N, 36>(handle,ys,A,xs,d_Y,d_X);
    runBenchmark<T, N, 48>(handle,ys,A,xs,d_Y,d_X);
    runBenchmark<T, N, 60>(handle,ys,A,xs,d_Y,d_X);
}

template<class T>
void iterate_over_N(
        cublasHandle_t & handle,
        bVectorField<T, 128> ** ys,
        const bMatrixField<T,128> & A,
        bVectorField<T, 128> ** xs,
        T * d_Y, T * d_X
) {
    iterate_over_numRHS<T, 32>(handle,ys,A,xs,d_Y,d_X);
    iterate_over_numRHS<T, 64>(handle,ys,A,xs,d_Y,d_X);
    // iterate_over_numRHS<T, 128>(handle,ys,A,xs,d_Y,d_X);
}



using T = realF;
constexpr unsigned N = 128;
constexpr unsigned numRHS = 60;
const bGrid grid = grids[sizeof(grids)/sizeof(bGrid)-1];

int main () {
    // first setup the largest fields for this benchmark
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

	T * d_X, * d_Y;
	CCE(  cudaMalloc(&d_X, sizeof(T)*numRHS*grid.numSites*N)  );
	CCE(  cudaMalloc(&d_Y, sizeof(T)*numRHS*grid.numSites*N)  );

    iterate_over_N<T>(handle,ys,A,xs,d_Y,d_X);

	CCE(  cudaFree(d_X)  );
	CCE(  cudaFree(d_Y)  );

	cublasCCE(  cublasDestroy(handle)  );
}
