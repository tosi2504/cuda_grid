#include "../cugrid/lattice.h"
#include "../cugrid/grid.h"
#include "../cugrid/tensor.h"
#include "../cugrid/lane.h"

#include <cuda/std/complex>
#include <chrono>
using namespace std::chrono;


constexpr unsigned lenLane = 32;
constexpr unsigned N = 64;
// using T_arithm = cuda::std::complex<double>;
using T_arithm = float;
using lRealD = Lane<T_arithm, lenLane>;
using iVecRealD = iVector<lRealD, N>;
using iMatRealD = iMatrix<lRealD, N>;

int main () {
	Grid<lenLane> grid(16,16,16,16);
	Lattice<iVecRealD> vfield1(grid), vfield2(grid);
	Lattice<iMatRealD> mfield(grid);

    // mfield.fill_random(123, T_arithm(0,0), T_arithm(1,1));
    mfield.fill_random(123, 0, 1);
    vfield2.fill_random(456, 0, 1);

	mfield.upload();
	vfield2.upload();

    // TIME IT!
    unsigned reps = 50;
    std::cout << "TIMING STARTED" << std::endl;
    auto start = high_resolution_clock::now();
	for (unsigned i = 0; i < reps; i++) {
	    matmul_opt(vfield1, mfield, vfield2);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "DURATION: " << duration.count() << std::endl;
    std::cout << "BANDWIDTH: " << grid.vol*(N*N + 2*N)*sizeof(T_arithm)*reps/(float)duration.count() << " MBytes/sec" << std::endl;
    std::cout << "ARITHMETICS: " << grid.vol * (2*N*N) * reps / (float)duration.count() << " Mflops" << std::endl;

	vfield1.download();
}
