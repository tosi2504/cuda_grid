#include "../cugrid/lattice.h"
#include "../cugrid/grid.h"
#include "../cugrid/tensor.h"
#include "../cugrid/lane.h"


constexpr unsigned lenLane = 32;
constexpr unsigned N = 64;
using lRealD = Lane<double, lenLane>;
using iVecRealD = iVector<lRealD, N>;
using iMatRealD = iMatrix<lRealD, N>;

int main () {
	Grid<lenLane> grid(16,16,16,32);
	Lattice<iVecRealD> vfield1(grid), vfield2(grid);
	Lattice<iMatRealD> mfield(grid);
	mfield.upload();
	vfield2.upload();
	for (int i = 0; i < 50; i++)
	matmul(&vfield1, &mfield, &vfield2);
	vfield1.download();
}
