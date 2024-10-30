#include "../cugrid2/cugrid2.h"

#include <random>
#include "cublas_v2.h"
#include "cugrid2/bLattice.h"
#include "cugrid2/bStencil.h"
#include "cugrid2/errorcheck.h"
#include "cugrid2/stopwatch.h"


constexpr unsigned reps = 100; 

// const bGrid grids[] = {bGrid(4,4,4,4)
//                     , bGrid(4,4,8,8)
//                     , bGrid(8,8,8,8)};
//                     // , bGrid(16,16,16,16)};

const bGrid grids[] = {bGrid(8,8,8,8)};

template<class T, unsigned N, unsigned numRHS>
void runBenchmark(
        cublasHandle_t & handle,
        bVectorField<T, 128> ** ys,
        const bMatrixField<T, 128> * const * As,
        bVectorField<T, 128> ** xs,
        T * d_Y, T * d_X
) {
    for (unsigned i_grid = 0; i_grid < sizeof(grids)/sizeof(bGrid); ++i_grid) {
        bVectorField<T,N> ** ys_temp = new bVectorField<T,N>*[numRHS];
        bVectorField<T,N> ** xs_temp = new bVectorField<T,N>*[numRHS];
        for (unsigned i_rhs = 0; i_rhs < numRHS; ++i_rhs) {
            ys_temp[i_rhs] = new bVectorField<T,N>(grids[i_grid], *(ys[i_rhs]));
            xs_temp[i_rhs] = new bVectorField<T,N>(grids[i_grid], *(xs[i_rhs]));
        }
        bMatrixField<T,N> ** As_temp = new bMatrixField<T,N>*[9];
        for (unsigned iDir = 0; iDir < 9; iDir++) {
            As_temp[iDir] = new bMatrixField<T, N>(grids[i_grid], *(As[iDir]));
        }
        
        bFullStencil stencil(grids[i_grid]);

        for (unsigned i = 0; i<reps; i++) {
            stopwatch.reset();

            // perform the call 
            unsigned blkSize;
            if constexpr (numRHS == 1) {
                stencil.execute_2DBTV2
                    <T,N,numRHS, N/2, 1, 2, 1>
                    (ys_temp , As_temp , xs_temp);
                blkSize = (N/2);
            } else {
                stencil.execute_2DBTV2
                    <T,N,numRHS, N/2, numRHS/2, 2, 2>
                    (ys_temp , As_temp , xs_temp);
                blkSize = (N/2) * (numRHS/2);
            }

            // read out stopwatch
            std::cout << i << ",";
            std::cout << grids[i_grid].Lx << ".";
            std::cout << grids[i_grid].Ly << ".";
            std::cout << grids[i_grid].Lz << ".";
            std::cout << grids[i_grid].Lt << ",";
            std::cout << N << ",";
            std::cout << numRHS << ",";
            std::cout << blkSize << ",";
            std::cout << 0 << ","; // spacer for compatibility
            std::cout << 0 << ","; // spacer for compatibility
            std::cout << stopwatch.getdiff(1) << ",";
            std::cout << 0 << std::endl; // spacer for compatibility
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
        const bMatrixField<T,128> * const * As,
        bVectorField<T, 128> ** xs,
        T * d_Y, T * d_X
) {
    runBenchmark<T, N, 8>(handle,ys,As,xs,d_Y,d_X);
    runBenchmark<T, N, 16>(handle,ys,As,xs,d_Y,d_X);
    runBenchmark<T, N, 24>(handle,ys,As,xs,d_Y,d_X);
    runBenchmark<T, N, 32>(handle,ys,As,xs,d_Y,d_X);
    runBenchmark<T, N, 40>(handle,ys,As,xs,d_Y,d_X);
    runBenchmark<T, N, 48>(handle,ys,As,xs,d_Y,d_X);
    runBenchmark<T, N, 56>(handle,ys,As,xs,d_Y,d_X);
    runBenchmark<T, N, 64>(handle,ys,As,xs,d_Y,d_X);
}

template<class T>
void iterate_over_N(
        cublasHandle_t & handle,
        bVectorField<T, 128> ** ys,
        const bMatrixField<T,128> * const * As,
        bVectorField<T, 128> ** xs,
        T * d_Y, T * d_X
) {
    iterate_over_numRHS<T, 32>(handle,ys,As,xs,d_Y,d_X);
    iterate_over_numRHS<T, 64>(handle,ys,As,xs,d_Y,d_X);
    // iterate_over_numRHS<T, 128>(handle,ys,A,xs,d_Y,d_X);
}



using T = realF;
constexpr unsigned N = 128;
constexpr unsigned numRHS = 64;
const bGrid grid = grids[sizeof(grids)/sizeof(bGrid)-1];

int main () {
    // first setup the largest fields for this benchmark
	std::mt19937 gen(0);

    // prepare fields
	bVectorField<T,N> ** xs = createAndFillAndUploadBatchVecFields<T,N>(numRHS, grid, gen, 0, 1);
	bVectorField<T,N> ** ys = createBatchVecFields<T,N>(numRHS, grid);
	bMatrixField<T,N> ** As = (bMatrixField<T,N> **) malloc (sizeof(bMatrixField<T,N>*)*9);
    for (unsigned iDir = 0; iDir < 9; iDir++) {
        As[iDir] = new bMatrixField<T, N>(grid);
        As[iDir]->fill_random(gen, 0, 1);
        As[iDir]->upload();
    }

	// run benchmark
	cublasHandle_t handle;
	cublasCCE(  cublasCreate(&handle)  );

	T * d_X, * d_Y;
	CCE(  cudaMalloc(&d_X, sizeof(T)*numRHS*grid.numSites*N)  );
	CCE(  cudaMalloc(&d_Y, sizeof(T)*numRHS*grid.numSites*N)  );

    iterate_over_N<T>(handle,ys,As,xs,d_Y,d_X);

	CCE(  cudaFree(d_X)  );
	CCE(  cudaFree(d_Y)  );

	cublasCCE(  cublasDestroy(handle)  );
}
