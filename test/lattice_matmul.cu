#include "../cugrid/cugrid.h"

#include <cuda/std/complex>
#include <iostream>


constexpr unsigned lenLane = 32;
constexpr unsigned N = 64;
// using T_arithm = cuda::std::complex<double>;
using T_arithm = double;
using lRealD = Lane<T_arithm, lenLane>;
using iVecRealD = iVector<lRealD, N>;
using iMatRealD = iMatrix<lRealD, N>;

int main () {
	Grid<lenLane> grid(16,16,16,16);
	Lattice<iVecRealD> vfield1(grid), vfield2(grid);
	Lattice<iMatRealD> mfield(grid);

    std::cout << "Starting with filling" << std::endl;

    // fill the vector and matrix
    for (unsigned x = 0; x < grid.calcSizeVNodes(); x++) {
        std::cout << "x=" << x << std::endl;
        for (unsigned l = 0; l < lenLane; l++) {
            // vector
            for (unsigned i = 0; i < N; i++) {
                vfield2[x][i][l] = 1;
            }
            // matrix
            for (unsigned i = 0; i < N; i++) {
                for (unsigned j = 0; j < N; j++) {
                    mfield[x][i][j][l] = (x*32.0 + l + 0.01*i)/N + 0.0001*j;
                }
            }
        }
    }
    // this means -> res[x][i][l] = x*32 + l + 0.01*i + 0.00005*(N)*(N-1)

	mfield.upload();
	vfield2.upload();
	std::cout << "lattices uploaded -> starting matmul" << std::endl;
	for (int i = 0; i < 1; i++);
	matmul_opt(vfield1, mfield, vfield2);
	std::cout << "matmul finished -> downloading lattice" << std::endl;
	vfield1.download();


    unsigned x, l, i;
    x = 200;
    l = 31;
    i = 63;
    std::cout << "expected: " << x*32 + l + 0.01*i + 0.00005*(N)*(N-1) << std::endl;
    std::cout << "result:   " << vfield1[x][i][l] << std::endl;
}
