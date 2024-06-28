#include "../cugrid2/cugrid2.h"
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>

using T = realF;
constexpr unsigned N = 64;//128;
constexpr unsigned numRHS = 60;
constexpr unsigned blkSize = 4*N;
const bGrid grid = bGrid(8,8,8,8);

int main() {
    std::mt19937 gen(0);
    bVectorField<T, N> **xs =
        createAndFillAndUploadBatchVecFields<T, N>(numRHS, grid, gen, 0, 1);
    bVectorField<T, N> **ys = createBatchVecFields<T, N>(numRHS, grid);
    bMatrixField<T, N> A(grid);
    A.fill_random(gen, 0, 1);
    A.upload();
    std::cout << "Fields filled and uploaded" << std::endl;

    // stencil stuff
    bFullStencil stencil(grid);
   
    // run it!
    cublasHandle_t handle;
    cublasCCE(  cublasCreate(&handle)  );
    // stencil.execute_blas<T, N, numRHS>(handle, ys, A, xs);
    stencil.execute_shmem<T, N, numRHS, blkSize>(ys, A, xs);
    cublasCCE(  cublasDestroy(handle)  );

    // download it
    downloadBatchVecFields<T, N>(numRHS, ys);

    std::cout << "Fields downloaded" << std::endl;

    // check the results
    for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) {
        for (unsigned site = 0; site < grid.numSites; site++) {
            // unsigned iRHS = 68;
            // unsigned site = grid.toFlat({1,1,1,1});
            bVector<T,N> y = debugMatmul(A.h_data[site], xs[iRHS]->h_data[site]);
            debugMatmulAccumulate(y, A.h_data[grid.shift(site, 0, true)], xs[iRHS]->h_data[site]);
            debugMatmulAccumulate(y, A.h_data[grid.shift(site, 0, false)], xs[iRHS]->h_data[site]);
            debugMatmulAccumulate(y, A.h_data[grid.shift(site, 1, true)], xs[iRHS]->h_data[site]);
            debugMatmulAccumulate(y, A.h_data[grid.shift(site, 1, false)], xs[iRHS]->h_data[site]);
            debugMatmulAccumulate(y, A.h_data[grid.shift(site, 2, true)], xs[iRHS]->h_data[site]);
            debugMatmulAccumulate(y, A.h_data[grid.shift(site, 2, false)], xs[iRHS]->h_data[site]);
            debugMatmulAccumulate(y, A.h_data[grid.shift(site, 3, true)], xs[iRHS]->h_data[site]);
            debugMatmulAccumulate(y, A.h_data[grid.shift(site, 3, false)], xs[iRHS]->h_data[site]);


            // std::cout << "Comparison: " << iRHS << std::endl;
            for (unsigned n = 0; n < N; n++) {
                T diff = y.data[n] - ys[iRHS]->h_data[site].data[n];
                if (std::abs(diff) > 0.001f) {
                    std::cout << n << ": " << std::abs(diff) << std::endl;
                }
            }
        }
    }
    // y.print();
    // ys[iRHS]->h_data[site].print();
}


