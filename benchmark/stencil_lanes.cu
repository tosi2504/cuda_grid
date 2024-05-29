#include <random>
#include "cugrid/lattice.h"
#include "cugrid/grid.h"
#include "cugrid/matmul.h"
#include "cugrid/stopwatch.h"
#include "cugrid/datatypes.h"
#include "cugrid/stencil.h"


constexpr unsigned reps = 100; 
using T = realF;
constexpr unsigned N = 128;
constexpr unsigned numRHS = 60;
const unsigned mu = 0;
const unsigned isForward = true;


VectorBatch<Lane<T,32>, N, numRHS> xs, ys;

SimpleStencil stencil(mu, isForward);

const Grid<32> grids[] = {Grid<32>(4,4,4,4)
                        , Grid<32>(4,4,8,8)
                        , Grid<32>(8,8,8,8)
                        , Grid<32>(16,16,16,16)};

const Grid grid = grids[sizeof(grids)/sizeof(Grid<32>)-1];
matrixField<T,N> A(grid);


template<class T, unsigned N, unsigned numRHS>
void runBenchmark() {
    for (unsigned i_grid = 0; i_grid < sizeof(grids)/sizeof(Grid<32>); ++i_grid) {
        VectorBatch<Lane<T,32>,N,numRHS> xs_temp;
        VectorBatch<Lane<T,32>,N,numRHS> ys_temp;
        for (unsigned i_rhs = 0; i_rhs < numRHS; ++i_rhs) {
            ys_temp[i_rhs] = new vectorField<T,N>(grids[i_grid], *(ys[i_rhs]));
            xs_temp[i_rhs] = new vectorField<T,N>(grids[i_grid], *(xs[i_rhs]));
        }
        matrixField<T,N> A_temp(grids[i_grid], A);
        for (unsigned i = 0; i<reps; i++) {
            stopwatch.reset();
            // perform the call 
            stencil.run_mrhs<Lane<T,32>,N,numRHS>(ys_temp
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
            std::cout << 256 << ",";
            std::cout << stopwatch.getdiff(1) << std::endl;
        }
        for (unsigned i_rhs = 0; i_rhs < numRHS; ++i_rhs) {
            delete ys_temp[i_rhs];
            delete xs_temp[i_rhs];
        }
    }
}

template<class T, unsigned N>
void iterate_over_numRHS() {
    runBenchmark<T, N, 1>();
    runBenchmark<T, N, 12>();
    runBenchmark<T, N, 24>();
    runBenchmark<T, N, 36>();
    runBenchmark<T, N, 48>();
    runBenchmark<T, N, 60>();
}

template<class T>
void iterate_over_N() {
    iterate_over_numRHS<T, 32>();
    iterate_over_numRHS<T, 64>();
    iterate_over_numRHS<T, 128>();
}


int main () {
    // first setup the largest fields for this benchmark
	std::mt19937 gen(0);

    // aquire fields
    for (unsigned b = 0; b < numRHS; b++) xs[b] = new vectorField<T,N>(grid);
    for (unsigned b = 0; b < numRHS; b++) ys[b] = new vectorField<T,N>(grid);

    // fill fields randomly
    for (unsigned b = 0; b < numRHS; b++) xs[b]->fill_benchmark(141, 0, 1);
    for (unsigned b = 0; b < numRHS; b++) xs[b]->upload();
    A.fill_benchmark(4321, 0, 1);   
    A.upload();

    // prepare fields
	A.fill_random(12321, 0, 1);
	A.upload();
	std::cout << "Fields allocated and randomly filled" << std::endl;

	// run benchmark
    iterate_over_N<T>();

    for (unsigned b = 0; b < numRHS; b++) delete xs[b];
    for (unsigned b = 0; b < numRHS; b++) delete ys[b];
}
