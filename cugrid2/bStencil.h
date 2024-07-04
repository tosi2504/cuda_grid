#pragma once

#include "bLattice.h"
#include "bMatmul.h"
#include "blasWrapper.h"
#include "errorcheck.h"
#include "stopwatch.h"
#include <cstring>
#include <stdexcept>
#include <vector>

struct bMuStencil {
    const bGrid grid;
    const unsigned mu;
    const bool isForward;
    std::vector<unsigned> targetmap;
    
    bMuStencil(const bGrid &grid, const unsigned mu, const bool isForward)
                       : grid(grid), mu(mu), isForward(isForward) {
        targetmap = grid.calcTargetMap(mu, isForward);
    }
    ~bMuStencil() {}
    
    template <class T, unsigned stride>
    T **createDevicePointerArray(const T *const d_field,
                                   const bool doPermute) const {
        const T **const h_d_field = new const T *[grid.numSites];
        for (unsigned site = 0; site < grid.numSites; site++) {
            if (doPermute)
                h_d_field[site] = d_field + targetmap[site] * stride;
            else
                h_d_field[site] = d_field + site * stride;
        }
        T **d_d_field;
        CCE(cudaMalloc(&d_d_field, sizeof(T *) * grid.numSites));
        CCE(cudaMemcpy(d_d_field, h_d_field, sizeof(T *) * grid.numSites,
                       cudaMemcpyHostToDevice));
        delete[] h_d_field;
        return d_d_field;
    }
    
    // execute functions
    template <class T, unsigned N, unsigned numRHS, unsigned blkSize>
    void execute(cublasHandle_t handle, bVectorField<T, N> *const *const ys,
                   const bMatrixField<T, N> &A,
                   const bVectorField<T, N> *const *const xs) const {
        
        // check grid compatibility
        if (not bLatticeHelpers::areGridsCompatible<T, N, numRHS>(ys, A, xs))
            throw std::invalid_argument("Grids not compatible");
        if (not grid.isCompatible(A.grid))
            throw std::invalid_argument("Grids not compatible");
        
        // prepare device pointers and layout changes
        stopwatch.press();
        T *d_X, *d_Y;
        CCE(cudaMalloc(&d_X, sizeof(T) * numRHS * grid.numSites * N));
        CCE(cudaMalloc(&d_Y, sizeof(T) * numRHS * grid.numSites * N));
        
        stopwatch.press();
        // copy inputs to matrixfield X
        mrhs_helper::fillMatrixfieldFromBatch<T, N, numRHS, blkSize>(d_X, xs);
        
        // create permuted pointer array for X
        T **d_d_X = createDevicePointerArray<T, N * numRHS>(d_X, false);
        // create unpermuted pointer array for Y and A
        T **d_d_A = createDevicePointerArray<T, N * N>((T *)A.d_data, true);
        T **d_d_Y = createDevicePointerArray<T, N * numRHS>(d_Y, false);
        
        // call gemmBatched on d_d_X, d_d_Y and A.d_data
        stopwatch.press();
        const T alpha = 1;
        const T beta = 0;
        cublasCCE(gemmBatched::call<T>(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, numRHS,
                                       N, &alpha, d_d_A, N, d_d_X, N, &beta, d_d_Y,
                                       N, grid.numSites));
        CCE(cudaDeviceSynchronize());
        stopwatch.press();
        // copy result to vectorfields ys
        mrhs_helper::fillBatchFromMatrixfield<T, N, numRHS, blkSize>(ys, d_Y);
        stopwatch.press();
        
        // free temporary device arrays
        CCE(cudaFree(d_d_X));
        CCE(cudaFree(d_d_A));
        CCE(cudaFree(d_d_Y));
        CCE(cudaFree(d_X));
        CCE(cudaFree(d_Y));
    }
};


__device__ inline unsigned rowm(unsigned iRow, unsigned iCol, unsigned lenRows, unsigned lenCols) {
    return iRow * lenRows + iCol;
}
__device__ inline unsigned colm(unsigned iRow, unsigned iCol, unsigned lenRows, unsigned lenCols) {
    return iCol * lenCols + iRow;
}

template<class T, unsigned N, unsigned numRHS, unsigned blkSize>
__global__ void ker_stencilShmem(T * const * const g_d_ys
                            , const T * const d_A
                            , const T * const * const g_d_xs
                            , const unsigned * const g_indexmap
                            , unsigned numSites) {
    const unsigned site = blockIdx.x;
    const unsigned tIdx = threadIdx.x;
    
    const unsigned dRhs = tIdx % N;
    const unsigned dRow = tIdx / N;
    const unsigned iCol = tIdx % N; // for matrix loading only!
    
    constexpr unsigned rowStride = blkSize/N;
    constexpr unsigned rhsStride = N;
    
    __shared__ T shmemX[N*numRHS]; // column major (lol)
    // __shared__ T shmemY[N*numRHS]; // column major (lol)
    __shared__ T shmemY[numRHS*N]; // row major (lol)
    __shared__ T shmemA[blkSize];  // row major (no lol) 
    
    for (unsigned i = tIdx; i < N*numRHS; i+=blkSize) {
        const unsigned iRhs = i / N;
        const unsigned n    = i % N;
        shmemX[colm(n, iRhs, numRHS, N)] = g_d_xs[iRhs][N*site + n];
    }
    
    for (unsigned iDir = 0; iDir < 9; iDir++) {
        const unsigned targetSite = g_indexmap[iDir*numSites + site];
        for (unsigned iiRow = 0; iiRow < N; iiRow += rowStride) {
            const unsigned iRow = iiRow + dRow;
            shmemA[rowm(dRow,iCol,N,rowStride)] =
                d_A[targetSite*N*N + rowm(iRow, iCol, N, N)];
            __syncthreads();
            for (unsigned iiRhs = 0; iiRhs < numRHS; iiRhs += rhsStride) {
                const unsigned iRhs = iiRhs + dRhs;
                if (iRhs >= numRHS) continue; // access guard
                T tempRes = 0;
                for (unsigned k = 0; k < N; k++) {
                    const unsigned _k = (iRhs + k) % N;
                    tempRes += shmemA[rowm(dRow, _k, N, N)] * shmemX[colm(_k, iRhs, numRHS, N)]; 
                    // tempRes += shmemA[rowm(dRow, k, N, N)] * shmemX[colm(k, iRhs, numRHS, N)]; 
                }
                if (iDir == 0) shmemY[rowm(iRow, iRhs, numRHS, N)]  = tempRes;
                else           shmemY[rowm(iRow, iRhs, numRHS, N)] += tempRes;
            }
            __syncthreads();
        }
    }
    
    for (unsigned i = tIdx; i < N*numRHS; i+=blkSize) {
        const unsigned iRhs = i / N;
        const unsigned n    = i % N;
        g_d_ys[iRhs][N*site + n] = shmemY[rowm(n, iRhs, numRHS, N)];
    }
}

// assert (N % (tilesize*rowStride) == 0)
template<class T, unsigned N, unsigned numRHS, unsigned rowStride, unsigned tileLen>
__global__ void ker_stencil1DBlocktiling(T * const * const g_d_ys
                            , const T * const d_A
                            , const T * const * const g_d_xs
                            , const unsigned * const g_indexmap
                            , unsigned numSites) {
    const unsigned site = blockIdx.x;
    const unsigned tIdx = threadIdx.x;
    
    const unsigned dRhs = tIdx % N;
    const unsigned dRow = (tIdx / N) * tileLen; // mul and div not commutative LOLOLOL
    const unsigned iCol = tIdx % N; // for matrix loading only!
    
    constexpr unsigned rhsStride = N;
    
    __shared__ T shmemX[N*numRHS]; // column major (lol)
    __shared__ T shmemY[numRHS*N]; // row major (lol)
    __shared__ T shmemA[N * rowStride * tileLen];  // row major (no lol) 
    
    for (unsigned i = tIdx; i < N*numRHS; i+=N*rowStride) {
        const unsigned iRhs = i / N;
        const unsigned n    = i % N;
        shmemX[colm(n, iRhs, numRHS, N)] = g_d_xs[iRhs][N*site + n];
    }
    
    for (unsigned iDir = 0; iDir < 9; iDir++) {
        const unsigned targetSite = g_indexmap[iDir*numSites + site];
        for (unsigned iiRow = 0; iiRow < N; iiRow += rowStride*tileLen) {

            for (unsigned iTile = 0; iTile < tileLen; iTile++) {
                const unsigned iRow = iiRow + dRow + iTile;
                shmemA[rowm(dRow+iTile,iCol,N,rowStride*tileLen)] =
                    d_A[targetSite*N*N + rowm(iRow, iCol, N, N)];
            }
            __syncthreads();

            for (unsigned iiRhs = 0; iiRhs < numRHS; iiRhs += rhsStride) {
                const unsigned iRhs = iiRhs + dRhs;
                if (iRhs >= numRHS) continue; // access guard
                T regRes[tileLen] = {0.0};
                for (unsigned k = 0; k < N; k++) {
                    const unsigned _k = (iRhs + k) % N;
                    T regX = shmemX[colm(_k, iRhs, numRHS, N)];
                    for (unsigned iTile = 0; iTile < tileLen; iTile++) {
                        regRes[iTile] += shmemA[rowm(dRow+iTile, _k, N, rowStride*tileLen)] * regX; 
                    }
                }

                if (iDir == 0) {
                    for (unsigned iTile = 0; iTile < tileLen; iTile++) {
                        const unsigned iRow = iiRow + dRow + iTile;
                        shmemY[rowm(iRow, iRhs, numRHS, N)]  = regRes[iTile];
                    }
                } else { 
                    for (unsigned iTile = 0; iTile < tileLen; iTile++) {
                        const unsigned iRow = iiRow + dRow + iTile;
                        shmemY[rowm(iRow, iRhs, numRHS, N)] += regRes[iTile]; 
                    }
                }
            }
            __syncthreads();
        }
    }
    
    for (unsigned i = tIdx; i < N*numRHS; i+=N*rowStride) {
        const unsigned iRhs = i / N;
        const unsigned n    = i % N;
        g_d_ys[iRhs][N*site + n] = shmemY[rowm(n, iRhs, numRHS, N)];
    }
}


template<class T>
__device__ void printMatrixRowMajor(const T * mat, unsigned N, unsigned M, const char * indenter) {
    if (blockIdx.x != 0) return;
    for (unsigned n = 0; n < N; n++) {
        printf("%s", indenter);
        for (unsigned m = 0; m < M; m++) {
            printf("%f ", mat[n*M + m]);
        }
        printf("\n");
    }
}
template<class T>
__device__ void printMatrixColMajor(const T * mat, unsigned N, unsigned M, const char * indenter) {
    if (blockIdx.x != 0) return;
    for (unsigned n = 0; n < N; n++) {
        printf("%s", indenter);
        for (unsigned m = 0; m < M; m++) {
            printf("%f ", mat[n + m*N]);
        }
        printf("\n");
    }
}
// assert (N % (tileHeight*rowStride) == 0)
// assert (numRHS % (tileWidth) == 0)
// numThreads = rowStride * rhsStride
template<class T, unsigned N, unsigned numRHS
                , unsigned rowStride
                , unsigned rhsStride
                , unsigned tileHeight
                , unsigned tileWidth>
__global__ void ker_stencil2DBlocktiling(T * const * const g_d_ys
                            , const T * const d_A
                            , const T * const * const g_d_xs
                            , const unsigned * const g_indexmap
                            , unsigned numSites) {
    const unsigned site = blockIdx.x;
    const unsigned tIdx = threadIdx.x;
    const unsigned numThreads = rowStride * rhsStride;
    
    const unsigned dRhs = (tIdx % rhsStride) * tileWidth;
    const unsigned dRow = (tIdx / rhsStride) * tileHeight; // mul and div not commutative LOLOLOL
    
    __shared__ T shmemX[N*numRHS]; // column major (lol)
    __shared__ T shmemY[numRHS*N]; // row major (lol)
    __shared__ T shmemA[N * rowStride * tileHeight];  // row major (no lol) 
    
    for (unsigned i = tIdx; i < N*numRHS; i+=numThreads) {
        const unsigned iRhs = i / N;
        const unsigned n    = i % N;
        shmemX[colm(n, iRhs, numRHS, N)] = g_d_xs[iRhs][N*site + n];
    }
    
    for (unsigned iDir = 0; iDir < 9; iDir++) {
    // for (unsigned iDir = 0; iDir < 1; iDir++) { // DEBUG
        const unsigned targetSite = g_indexmap[iDir*numSites + site];
        for (unsigned iiRow = 0; iiRow < N; iiRow += rowStride*tileHeight) {
// if (site==0) printf("iiRow loop: iiRow=%u, dRow=%u\n", iiRow, dRow);
            for (unsigned  i = tIdx; i < N*rowStride*tileHeight; i+=numThreads) {
                const unsigned iCol = i % N;
                const unsigned iRow = i / N;
                shmemA[rowm(iRow,iCol,N,rowStride*tileHeight)] = d_A[targetSite*N*N + rowm(iiRow + iRow, iCol, N, N)];
            }
            __syncthreads();

// if (site == 0) printf("--> Checking shmemA\n");
// printMatrixRowMajor(shmemA, rowStride*tileHeight, N, "    ");

            for (unsigned iiRhs = 0; iiRhs < numRHS; iiRhs += rhsStride*tileWidth) {
            // for (unsigned iiRhs = 0; iiRhs < 1; iiRhs += rhsStride*tileWidth) { // DEBUG
                const unsigned iidRhs = iiRhs + dRhs;
                if (iidRhs >= numRHS) continue; // access guard
                const unsigned iidRow = iiRow + dRow;
                // Could add access guard for row as well but would also need to add access guard for loading A if I wanted to relax assert statements
                
// if (site==0) printf("--> iiRhs Loop: iiRhs=%u\n", iiRhs);
                
                T regRes[tileHeight][tileWidth] = {0.0};
                T regX[tileWidth]  = {0.0}; // values of shmemX
                T regA[tileHeight] = {0.0}; // values of shmemA
                for (unsigned _k = 0; _k < N; _k++) {
                    const unsigned k = (tIdx + _k) % N;
// if(site==0) printf("--> --> k Loop: k=%u\n", k);
                    // fill regA and regX
                    for (unsigned iTileRow = 0; iTileRow < tileHeight; iTileRow++) {
                        regA[iTileRow] = shmemA[rowm(dRow+iTileRow, k, N, rowStride*tileHeight)];
//  if(site==0) printf("--> --> --> iTileRow Loop: iTileRow=%u\n", iTileRow);
//  if(site==0) printf("--> --> --> --> accessing shmemA at %u\n", rowm(dRow+iTileRow, k, N, rowStride*tileHeight));
// if(site==0) printf("--> --> --> --> regA[%u]=%f\n", iTileRow, regA[iTileRow]);
                    }
                    for (unsigned iTileRhs = 0; iTileRhs < tileWidth; iTileRhs++) {
                        regX[iTileRhs] = shmemX[colm(k, iidRhs+iTileRhs, numRHS, N)];
// if(site==0) printf("--> --> --> iTileRhs Loop: iTileRhs=%u\n", iTileRhs);
// if(site==0) printf("--> --> --> --> regX[%u]=%f\n", iTileRhs, regX[iTileRhs]);
                    }

                    // perform the arithmetics :)
                    for (unsigned iTileRow = 0; iTileRow < tileHeight; iTileRow++) {
                        for (unsigned iTileRhs = 0; iTileRhs < tileWidth; iTileRhs++) {
                            regRes[iTileRow][iTileRhs] += regA[iTileRow] * regX[iTileRhs];
                        }
                    }
                }

                if (iDir == 0) {
                    for (unsigned iTileRow = 0; iTileRow < tileHeight; iTileRow++) {
                        for (unsigned iTileRhs = 0; iTileRhs < tileWidth; iTileRhs++) {
                            const unsigned iRow = iidRow + iTileRow;
                            const unsigned iRhs = iidRhs + iTileRhs;
                            shmemY[rowm(iRow, iRhs, numRHS, N)]  = regRes[iTileRow][iTileRhs];
                        }
                    }
                } else { 
                    for (unsigned iTileRow = 0; iTileRow < tileHeight; iTileRow++) {
                        for (unsigned iTileRhs = 0; iTileRhs < tileWidth; iTileRhs++) {
                            const unsigned iRow = iidRow + iTileRow;
                            const unsigned iRhs = iidRhs + iTileRhs;
                            shmemY[rowm(iRow, iRhs, numRHS, N)] += regRes[iTileRow][iTileRhs];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
    
    for (unsigned i = tIdx; i < N*numRHS; i+=numThreads) {
        const unsigned iRhs = i / N;
        const unsigned n    = i % N;
        g_d_ys[iRhs][N*site + n] = shmemY[rowm(n, iRhs, numRHS, N)];
    }
}



struct bFullStencil {
    const bGrid grid;
    std::vector<unsigned> targetmap[9];

    bFullStencil(const bGrid &grid) : grid(grid) {
        targetmap[0].resize(grid.numSites);
        for (unsigned site = 0; site < grid.numSites; site++) {
            targetmap[0][site] = site;
        }
        unsigned i_dir = 1;
        for (unsigned mu = 0; mu < 4; mu++) {
            targetmap[i_dir] = grid.calcTargetMap(mu, true);
            i_dir++;
            targetmap[i_dir] = grid.calcTargetMap(mu, false);
            i_dir++;
        }
    }

    template<class T, unsigned stride>
    void fillDevicePointerArray(T ** d_d_field
                                , T * d_field ) const {
        for (unsigned site = 0; site < grid.numSites; site++) {
            d_d_field[site] = d_field + site * stride;
        }
    }

    template<class T, unsigned stride>
    void fillDevicePointerArrayPermute(T ** d_d_field
                                , T * d_field
                                , unsigned i_dir ) const {
        for (unsigned site = 0; site < grid.numSites; site++) {
            d_d_field[site] = d_field + targetmap[i_dir][site] * stride;
        }
    }

    template<class T, unsigned N, unsigned numRHS, unsigned blkSize = 256>
    void execute_blas(cublasHandle_t handle,
                 bVectorField<T, N> * const * const ys,
                 const bMatrixField<T, N> & A,
                 const bVectorField<T, N> * const * const xs) const {
        // check grid compatibility 
        if (not bLatticeHelpers::areGridsCompatible<T, N, numRHS>(ys, A, xs))
            throw std::invalid_argument("Grids not compatible");
        if (not grid.isCompatible(A.grid))
            throw std::invalid_argument("Grids not compatible");

        // prepare device pointers and layout changes
        stopwatch.press();
        T * d_X, * d_Y;
        CCE(cudaMalloc(&d_X, sizeof(T)*numRHS*grid.numSites*N));
        CCE(cudaMalloc(&d_Y, sizeof(T)*numRHS*grid.numSites*N));

        T ** d_d_X, ** d_d_Y, ** d_d_A;
        // TODO: Rewrite pointermapping using uvm, to simplify code and debugging
        CCE(cudaMallocManaged(&d_d_X, sizeof(T*)*grid.numSites));
        CCE(cudaMallocManaged(&d_d_Y, sizeof(T*)*grid.numSites));
        CCE(cudaMallocManaged(&d_d_A, sizeof(T*)*grid.numSites));
        
        stopwatch.press();
        // copy inputs to matrixfield X
        mrhs_helper::fillMatrixfieldFromBatch<T, N, numRHS, blkSize>(d_X, xs);
        stopwatch.press();


        T alpha = 1;
        T beta = 0;
        cublasCCE(gemmStridedBatched::call<T>(handle
                                           , CUBLAS_OP_T
                                           , CUBLAS_OP_N
                                           , N, numRHS, N
                                           , &alpha
                                           , (T*)A.d_data, N, N*N
                                           , d_X, N, N*numRHS
                                           , &beta
                                           , d_Y, N, N*numRHS
                                           , grid.numSites));
        

        fillDevicePointerArray<T, N*numRHS>(d_d_Y, d_Y);
        fillDevicePointerArray<T, N*numRHS>(d_d_X, d_X);
        beta = 1;
        for (unsigned i_dir = 1; i_dir < 9; i_dir++) {
            fillDevicePointerArrayPermute<T, N*N>(d_d_A, (T*)A.d_data, i_dir);
            cublasCCE(gemmBatched::call<T>(handle
                                           , CUBLAS_OP_T
                                           , CUBLAS_OP_N
                                           , N, numRHS, N
                                           , &alpha
                                           , d_d_A, N
                                           , d_d_X, N
                                           , &beta
                                           , d_d_Y, N
                                           , grid.numSites));
            CCE(cudaDeviceSynchronize());
        }
        stopwatch.press(); // cp out
        mrhs_helper::fillBatchFromMatrixfield<T, N, numRHS, blkSize>(ys, d_Y);
        stopwatch.press();

        CCE(cudaFree(d_d_X));
        CCE(cudaFree(d_d_Y));
        CCE(cudaFree(d_d_A));
        CCE(cudaFree(d_X));
        CCE(cudaFree(d_Y));
    }

    template<class T, unsigned N, unsigned numRHS, unsigned blkSize = 256>
    void execute_shmem(bVectorField<T, N> * const * const ys,
                 const bMatrixField<T, N> & A,
                 const bVectorField<T, N> * const * const xs) const {
        // check template parameter compatibility
        static_assert(blkSize%N==0);
        static_assert(N%(blkSize/N)==0);
        // check grid compatibility 
        if (not bLatticeHelpers::areGridsCompatible<T, N, numRHS>(ys, A, xs))
            throw std::invalid_argument("Grids not compatible");
        if (not grid.isCompatible(A.grid))
            throw std::invalid_argument("Grids not compatible");

        stopwatch.press();

        T ** g_d_xs, ** g_d_ys; 
        CCE(cudaMallocManaged(&g_d_xs, sizeof(T*)*numRHS));
        CCE(cudaMallocManaged(&g_d_ys, sizeof(T*)*numRHS));
        for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) {
            g_d_xs[iRHS] = (T*)xs[iRHS]->d_data;
            g_d_ys[iRHS] = (T*)ys[iRHS]->d_data;
        }
        
        unsigned * g_indexmap;
        CCE(cudaMallocManaged(&g_indexmap, sizeof(unsigned*)*9*grid.numSites));
        for (unsigned i_dir = 0; i_dir < 9; i_dir++) {
            memcpy(&g_indexmap[i_dir*grid.numSites]
                   , &targetmap[i_dir][0]
                   , sizeof(unsigned)*grid.numSites);
        }
        
        ker_stencilShmem
            <T, N, numRHS, blkSize>
            <<<grid.numSites, blkSize>>>
            ((T*const*)g_d_ys
                , (const T*)A.d_data
                , (const T*const*)g_d_xs
                , (const unsigned*)g_indexmap
                , grid.numSites);
        CLCE();
        CCE(cudaDeviceSynchronize());

        stopwatch.press();
        
        CCE(cudaFree(g_indexmap));
        CCE(cudaFree(g_d_xs));
        CCE(cudaFree(g_d_ys));
    }

    template<class T, unsigned N, unsigned numRHS, unsigned rowStride, unsigned tileLen>
    void execute_1DBT(bVectorField<T, N> * const * const ys,
                 const bMatrixField<T, N> & A,
                 const bVectorField<T, N> * const * const xs) const {
        // check template parameter compatibility
        static_assert(N%(rowStride * tileLen)==0);
        // check grid compatibility 
        if (not bLatticeHelpers::areGridsCompatible<T, N, numRHS>(ys, A, xs))
            throw std::invalid_argument("Grids not compatible");
        if (not grid.isCompatible(A.grid))
            throw std::invalid_argument("Grids not compatible");

        stopwatch.press();

        T ** g_d_xs, ** g_d_ys; 
        CCE(cudaMallocManaged(&g_d_xs, sizeof(T*)*numRHS));
        CCE(cudaMallocManaged(&g_d_ys, sizeof(T*)*numRHS));
        for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) {
            g_d_xs[iRHS] = (T*)xs[iRHS]->d_data;
            g_d_ys[iRHS] = (T*)ys[iRHS]->d_data;
        }
        
        unsigned * g_indexmap;
        CCE(cudaMallocManaged(&g_indexmap, sizeof(unsigned*)*9*grid.numSites));
        for (unsigned i_dir = 0; i_dir < 9; i_dir++) {
            memcpy(&g_indexmap[i_dir*grid.numSites]
                   , &targetmap[i_dir][0]
                   , sizeof(unsigned)*grid.numSites);
        }
        
        std::cout << "Launching kernel with <<< " << grid.numSites << " , " << N*rowStride << " >>>" << std::endl;
        ker_stencil1DBlocktiling
            <T, N, numRHS, rowStride, tileLen>
            <<<grid.numSites, N*rowStride>>>
            ((T*const*)g_d_ys
                , (const T*)A.d_data
                , (const T*const*)g_d_xs
                , (const unsigned*)g_indexmap
                , grid.numSites);
        CLCE();
        CCE(cudaDeviceSynchronize());

        stopwatch.press();
        
        CCE(cudaFree(g_indexmap));
        CCE(cudaFree(g_d_xs));
        CCE(cudaFree(g_d_ys));
    }

    template<class T, unsigned N, unsigned numRHS
                , unsigned rowStride
                , unsigned rhsStride
                , unsigned tileHeight
                , unsigned tileWidth>
    void execute_2DBT(bVectorField<T, N> * const * const ys,
                 const bMatrixField<T, N> & A,
                 const bVectorField<T, N> * const * const xs) const {
        // check template parameter compatibility
        static_assert(N%(rowStride * tileHeight)==0);
        static_assert(numRHS%tileWidth==0);
        // check grid compatibility 
        if (not bLatticeHelpers::areGridsCompatible<T, N, numRHS>(ys, A, xs))
            throw std::invalid_argument("Grids not compatible");
        if (not grid.isCompatible(A.grid))
            throw std::invalid_argument("Grids not compatible");

        stopwatch.press();

        T ** g_d_xs, ** g_d_ys; 
        CCE(cudaMallocManaged(&g_d_xs, sizeof(T*)*numRHS));
        CCE(cudaMallocManaged(&g_d_ys, sizeof(T*)*numRHS));
        for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) {
            g_d_xs[iRHS] = (T*)xs[iRHS]->d_data;
            g_d_ys[iRHS] = (T*)ys[iRHS]->d_data;
        }
        
        unsigned * g_indexmap;
        CCE(cudaMallocManaged(&g_indexmap, sizeof(unsigned*)*9*grid.numSites));
        for (unsigned i_dir = 0; i_dir < 9; i_dir++) {
            memcpy(&g_indexmap[i_dir*grid.numSites]
                   , &targetmap[i_dir][0]
                   , sizeof(unsigned)*grid.numSites);
        }
        
        std::cout << "Launching kernel with <<< " << grid.numSites << " , " << rowStride*rhsStride << " >>>" << std::endl;
        ker_stencil2DBlocktiling
            <T, N, numRHS, rowStride, rhsStride, tileHeight, tileWidth>
            <<<grid.numSites, rowStride*rhsStride>>>
            ((T*const*)g_d_ys
                , (const T*)A.d_data
                , (const T*const*)g_d_xs
                , (const unsigned*)g_indexmap
                , grid.numSites);
        CLCE();
        CCE(cudaDeviceSynchronize());

        stopwatch.press();
        
        CCE(cudaFree(g_indexmap));
        CCE(cudaFree(g_d_xs));
        CCE(cudaFree(g_d_ys));
    }
};







