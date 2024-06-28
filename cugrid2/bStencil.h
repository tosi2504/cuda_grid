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
    const unsigned flat = iRow * lenRows + iCol;
    if (flat >= lenRows * lenCols) printf("--FROM-GPU--: OUT OF BOUNDS");
    return flat;
}
__device__ inline unsigned colm(unsigned iRow, unsigned iCol, unsigned lenRows, unsigned lenCols) {
    const unsigned flat = iCol * lenCols + iRow;
    if (flat >= lenRows * lenCols) printf("--FROM-GPU--: OUT OF BOUNDS");
    return flat;
}

template<class T, unsigned N, unsigned numRHS, unsigned blkSize>
__global__ void ker_stencilShmemProperlyDone(T * const * const g_d_ys
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
    __shared__ T shmemY[N*numRHS]; // column major (lol)
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
                    tempRes += shmemA[rowm(iRow, k, N, N)] * shmemX[colm(k, iRhs, numRHS, N)]; 
                }
                if (iDir == 0) shmemY[colm(iRow, iRhs, numRHS, N)]  = tempRes;
                else           shmemY[colm(iRow, iRhs, numRHS, N)] += tempRes;
            }
            __syncthreads();
        }
    }

    for (unsigned i = tIdx; i < N*numRHS; i+=blkSize) {
        const unsigned iRhs = i / N;
        const unsigned n    = i % N;
        g_d_ys[iRhs][N*site + n] = shmemY[colm(n, iRhs, numRHS, N)];
    }
}


template<class T, unsigned N, unsigned numRHS, unsigned blkSize>
__global__ void ker_stencilShmemDebug(T * const * const g_d_ys
                            , const T * const d_A
                            , const T * const * const g_d_xs
                            , const unsigned * const g_indexmap
                            , unsigned numSites) {
    const unsigned site = blockIdx.x;
    const unsigned tIdx = threadIdx.x;
    constexpr unsigned rowStrideA = blkSize/N;
    const unsigned dRow = tIdx/N;
    const unsigned iCol = tIdx%N;
    const unsigned dRhs = dRow;
    
    __shared__ T shmemX[N*numRHS]; // column major (lol)
    __shared__ T shmemY[N*numRHS]; // column major (lol)
    __shared__ T shmemA[blkSize];  // row major (no lol) 
    for (unsigned i = tIdx; i < N*numRHS; i+=blkSize) {
        const unsigned iRhs = i / N;
        const unsigned n    = i % N;
        shmemX[colm(n, iRhs, numRHS, N)] = g_d_xs[iRhs][N*site + n];
    }
    
    for (unsigned iDir = 0; iDir < 9; iDir++) {
        const unsigned targetSite = g_indexmap[iDir*numSites + site];

        for (unsigned iiRow = 0; iiRow < N; iiRow+=rowStrideA) {

            shmemA[rowm(dRow,iCol,N,rowStrideA)] = d_A[targetSite*N*N + rowm(iiRow+dRow, iCol, N, N)];
            __syncthreads();

            for (unsigned iiRhs = 0; iiRhs < numRHS; iiRhs+=rowStrideA) {

                if (iiRhs + dRow < numRHS) { // access guard saves one static assert statement
                    T tempRes = 0;

                    for (unsigned k = 0; k < N; k++) {
                        tempRes += shmemA[rowm(dRow,k,N,rowStrideA)] * shmemX[colm(k, iiRhs+dRhs, numRHS, N)];
                    }
                    if (iDir == 0) {
                        shmemY[colm(iiRow+dRow, iiRhs+dRhs, numRHS, N)] = tempRes; // WARNING: dRow == dRhs!!!!! THIS CANT BE CORRECT
                    } else {
                        shmemY[colm(iiRow+dRow, iiRhs+dRhs, numRHS, N)] += tempRes; // WARNING: dRow == dRhs!!!!! THIS CANT BE CORRECT
                    }
                }
                __syncthreads();
            }
            __syncthreads();
        }
    }

    for (unsigned i = tIdx; i < N*numRHS; i+=blkSize) {
        const unsigned iRhs = i / N;
        const unsigned n    = i % N;
        g_d_ys[iRhs][N*site + n] = shmemY[colm(n, iRhs, numRHS, N)];
    }
}



template<class T, unsigned N, unsigned numRHS, unsigned blkSize>
__global__ void ker_stencilShmem(T * const * const g_d_ys
                            , const T * const d_A
                            , const T * const * const g_d_xs
                            , const unsigned * const g_indexmap
                            , unsigned numSites) {
    const unsigned site = blockIdx.x;
    const unsigned tIdx = threadIdx.x;
    constexpr unsigned rowStrideA = blkSize/N;
    const unsigned dRow = tIdx/N;
    const unsigned iCol = tIdx%N;
    const unsigned dRhs = dRow;
    // const unsigned iVec = iCol;
    
    __shared__ T shmemX[N*numRHS]; // column major (lol)
    __shared__ T shmemY[N*numRHS]; // column major (lol)
    __shared__ T shmemA[blkSize];  // row major (no lol) 
    for (unsigned i = tIdx; i < N*numRHS; i+=blkSize) {
        const unsigned iRhs = i / N;
        const unsigned n    = i % N;
        shmemX[i] = g_d_xs[iRhs][N*site + n];
    }
    
    for (unsigned iDir = 0; iDir < 9; iDir++) {

        const unsigned targetSite = g_indexmap[iDir*numSites + site];

        for (unsigned iiRow = 0; iiRow < N; iiRow+=rowStrideA) {

            // __syncthreads(); // DEBUG
            shmemA[tIdx] = d_A[targetSite*N*N + (iiRow+dRow)*N + iCol];
            __syncthreads();

            for (unsigned iiRhs = 0; iiRhs < numRHS; iiRhs+=rowStrideA) {

                if (iiRhs + dRow < numRHS) { // access guard saves one static assert statement
                    T tempRes = 0;

                    // __syncthreads(); // DEBUG

                    for (unsigned k = 0; k < N; k++) {
                        tempRes += shmemA[dRow*N + k] * shmemX[(iiRhs+dRhs)*N + k];
                    }
                    // TODO: CONTINUE DEBUGGING HERE: WHY IS THE STATEMENT IMPORTANT
                    // __syncthreads(); // DEBUG -> THIS IS THE IMPORTANT ONE FOR SOME BLOODY REASON
                    if (iDir == 0)
                        shmemY[(iiRhs+dRhs)*N + iiRow + dRow] = tempRes;
                    else
                        shmemY[(iiRhs+dRhs)*N + iiRow + dRow] += tempRes;
                    __syncthreads();
                }
            }
        }
    }

    // __syncthreads(); // DEBUG
    // need to write out results
    __syncthreads(); // DEBUG
    for (unsigned i = tIdx; i < N*numRHS; i+=blkSize) {
        const unsigned iRhs = i / N;
        const unsigned n    = i % N;
        g_d_ys[iRhs][N*site + n] = shmemY[i];
    }
    __syncthreads(); // DEBUG
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
        
        ker_stencilShmemProperlyDone
            <T, N, numRHS, blkSize>
            <<<grid.numSites, blkSize>>>
            ((T*const*)g_d_ys
                , (const T*)A.d_data
                , (const T*const*)g_d_xs
                , (const unsigned*)g_indexmap
                , grid.numSites);
        CLCE();
        CCE(cudaDeviceSynchronize());
        
        CCE(cudaFree(g_indexmap));
        CCE(cudaFree(g_d_xs));
        CCE(cudaFree(g_d_ys));
    }
};







