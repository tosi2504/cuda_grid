#include "../cugrid/lattice.h"
#include "../cugrid/grid.h"
#include "../cugrid/tensor.h"
#include "../cugrid/lane.h"

#include <cuda/std/complex>


constexpr unsigned lenLane = 32;
constexpr unsigned N = 64;
// using T_arithm = cuda::std::complex<double>;
using T_arithm = double;
using lRealD = Lane<T_arithm, lenLane>;
using iVecRealD = iVector<lRealD, N>;
using iMatRealD = iMatrix<lRealD, N>;

int main () {
	Grid<lenLane> grid(8,16,16,32);
	Lattice<iVecRealD> vfield1(grid), vfield2(grid);
	Lattice<iMatRealD> mfield(grid);

    // mfield.fill_random(123, T_arithm(0,0), T_arithm(1,1));
    mfield.fill_random(123, 0, 1);
    vfield2.fill_random(456, 0, 1);

	mfield.upload();
	vfield2.upload();
	for (int i = 0; i < 3; i++)
	matmul(&vfield1, &mfield, &vfield2);
	vfield1.download();
}
