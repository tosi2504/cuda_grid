#include "../cugrid/cugrid.h"

#include <cuda/std/complex>
#include <stdlib.h>
#include <chrono>
using namespace std::chrono;


constexpr unsigned lenLane = 32;
constexpr unsigned N = 32;
constexpr unsigned batchsize = 32;
constexpr unsigned mu = 0;
constexpr unsigned isForward = true;

// using T_arithm = cuda::std::complex<double>;
using T_arithm = float;
using lane_t = Lane<T_arithm, lenLane>;
using vec_t = iVector<lane_t, N>;
using mat_t = iMatrix<lane_t, N>;
using VecField = Lattice<vec_t>;
using MatField = Lattice<mat_t>;
using VecBatch = VectorBatch<lane_t,N,batchsize>;

int main () {
	Grid<lenLane> grid(16,16,16,16);
	VecBatch res, rhs;
	for (unsigned b = 0; b < batchsize; b++) res[b] = new VecField(grid);
	for (unsigned b = 0; b < batchsize; b++) rhs[b] = new VecField(grid);
	MatField mfield(grid);

    // mfield.fill_random(123, T_arithm(0,0), T_arithm(1,1));
	std::cout << "Filling with random numbers" << std::endl;
    mfield.fill_benchmark(123, 0, 1);
	for (unsigned b = 0; b < batchsize; b++) rhs[b]->fill_benchmark(b*111, 0, 1);
	std::cout << "DONE" << std::endl;

	std::cout << "Uploading fields onto GPU" << std::endl;
	mfield.upload();
	for (unsigned b = 0; b < batchsize; b++) rhs[b]->upload();
	std::cout << "DONE" << std::endl;

	// define stencil
	SimpleStencil stencil(mu, isForward);

    // TIME IT!
    unsigned reps = 50;
    std::cout << "TIMING STARTED FOR STENCIL_MRHS" << std::endl;
    auto start = high_resolution_clock::now();
	for (unsigned i = 0; i < reps; i++) {
	    stencil.run_mrhs<lane_t, N, batchsize>(res, mfield, rhs);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "DURATION: " << duration.count() << std::endl;
	unsigned long long bytes = batchsize;
	bytes *= grid.vol;
	bytes *= (N*N + 2*N);
	bytes *= sizeof(T_arithm);
	bytes *= reps;
    std::cout << "BANDWIDTH: " << bytes/(float)duration.count() << " MBytes/sec" << std::endl;
	std::cout << "ARITHMETICS: " << batchsize*grid.vol * (2*N*N) * reps / (float)duration.count() << " Mflops" << std::endl;

	// should delete the vector batches ...
}
