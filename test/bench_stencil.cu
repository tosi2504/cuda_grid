#include "../cugrid/cugrid.h"

#include <cuda/std/complex>
#include <chrono>
using namespace std::chrono;


constexpr unsigned lenLane = 32;
constexpr unsigned N = 50;
// using T_arithm = cuda::std::complex<double>;
using T_arithm = double;
using lRealD = Lane<T_arithm, lenLane>;
using iVecRealD = iVector<lRealD, N>;
using iMatRealD = iMatrix<lRealD, N>;

int main () {
	Grid<lenLane> grid(16,16,16,32);
	Lattice<iVecRealD> vfield1(grid), vfield2(grid);
	Lattice<iMatRealD> mfield(grid);

    // mfield.fill_random(123, T_arithm(0,0), T_arithm(1,1));
	std::cout << "Filling with random numbers" << std::endl;
    mfield.fill_benchmark(123, 0, 1);
    vfield2.fill_benchmark(456, 0, 1);
	std::cout << "DONE" << std::endl;

	std::cout << "Uploading fields onto GPU" << std::endl;
	mfield.upload();
	vfield2.upload();
	std::cout << "DONE" << std::endl;

	// define the desired stencil
	SimpleStencil stencil(0, false);

    // TIME IT!
    unsigned reps = 50;
    std::cout << "TIMING STARTED" << std::endl;
    auto start = high_resolution_clock::now();
	for (unsigned i = 0; i < reps; i++) {
	    stencil.run(vfield1, mfield, vfield2);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "DURATION: " << duration.count() << std::endl;
    std::cout << "BANDWIDTH: " << grid.vol*(N*N + 2*N)*sizeof(T_arithm)*reps/(float)duration.count() << " MBytes/sec" << std::endl;
    std::cout << "ARITHMETICS: " << grid.vol * (2*N*N) * reps / (float)duration.count() << " Mflops" << std::endl;

	vfield1.download();
}
