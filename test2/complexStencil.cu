#include "../cugrid2/cugrid2.h"
#include "cugrid2/bTensor.h"
#include "cugrid2/stopwatch.h"
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>

using T = realF;
constexpr unsigned N = 32;
constexpr unsigned numRHS = 60;
// constexpr unsigned blkSize = 8*N;
const bGrid grid = bGrid(8,8,8,8);

int main() {
    std::mt19937 gen(0);
    bVectorField<T, N> **xs =
        createAndFillAndUploadBatchVecFields<T, N>(numRHS, grid, gen, 0, 1);
    bVectorField<T, N> **ys = createBatchVecFields<T, N>(numRHS, grid);
    bMatrixField<T, N> A(grid);
    // xs[0]->h_data[grid.toFlat({0,0,0,0})] = bVector<T, N>::Unit();
    // xs[0]->upload();
    // A.fill_zero();
    // A.h_data[grid.toFlat({0,0,0,0})].fill_random(gen, 0, 1);// = bMatrix<T, N>::Unit();
    A.fill_random(gen, 0, 1);
    A.upload();
    std::cout << "Fields filled and uploaded" << std::endl;

    // stencil stuff
    bFullStencil stencil(grid);
   
    // run it!
    cublasHandle_t handle;
    cublasCCE(  cublasCreate(&handle)  );
    double execTime = 0;
    const unsigned reps = 100; 
    for (uint8_t i = 0; i < reps; i++) { 
        stopwatch.reset();

        stencil.execute_2DBT<T, N, numRHS, 8, 16, 4, 4>(ys, A, xs);
        // stencil.execute_1DBT<T, N, numRHS, 8, 4>(ys, A, xs);
        // stencil.execute_shmem<T, N, numRHS, blkSize>(ys, A, xs);
        execTime += stopwatch.getdiff(1);

        // stencil.execute_blas<T, N, numRHS>(handle, ys, A, xs);
        // execTime += stopwatch.getdiff(1);
        // execTime += stopwatch.getdiff(2);
        // execTime += stopwatch.getdiff(3);
        // execTime += stopwatch.getdiff(4);
        // std::cout << stopwatch.getdiff(1) << ",";
        // std::cout << stopwatch.getdiff(2) << ",";
        // std::cout << stopwatch.getdiff(3) << ",";
        // std::cout << stopwatch.getdiff(4) << ",";
    }
    cublasCCE(  cublasDestroy(handle)  );
    std::cout << "Kernel-Stats:\n    Bandwidth(MB/s): ";
    std::cout << (long) reps * (long) grid.numSites * sizeof(T) * (long)(9 * N*N + 2*N*numRHS) / execTime << std::endl;
    std::cout << "    Flops(MFlops): ";
    std::cout << (long)grid.numSites * (long)(18*N*N*numRHS) * reps / execTime << std::endl;

    // download it
    downloadBatchVecFields<T, N>(numRHS, ys);

    std::cout << "Fields downloaded" << std::endl;

     // check the results
     for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) {
     // for (unsigned iRHS = 0; iRHS < 1; iRHS++) {
         for (unsigned site = 0; site < grid.numSites; site++) {
         // for (unsigned site = 0; site < 1; site++) {
             // unsigned iRHS = 0;
             // unsigned site = grid.toFlat({0,0,0,0});
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
                     std::cout << "site: " << site;
                     std::cout << "    iRHS: " << iRHS;
                     std::cout << "    n: " << n;
                     std::cout << "    diff: " << std::abs(diff) << std::endl;
                 }
             }
         }
     }
}


