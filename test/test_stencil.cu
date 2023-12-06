#include "../cugrid/cugrid.h"

#include <cuda/std/complex>
#include <chrono>
using namespace std::chrono;


constexpr unsigned lenLane = 32;
constexpr unsigned N = 40;
// using T_arithm = cuda::std::complex<double>;
using T_arithm = unsigned;
using lRealD = Lane<T_arithm, lenLane>;
using iVecRealD = iVector<lRealD, N>;
using iMatRealD = iMatrix<lRealD, N>;

const unsigned mu = 1;
const bool isForward = true;

int main () {
	Grid<lenLane> grid(6,6,6,8);
	Lattice<iVecRealD> vfield1(grid), vfield2(grid);
	Lattice<iMatRealD> mfield(grid);

	auto laneIdxMap = grid.getLaneIdxMap(mu, isForward);
	auto stinfo = grid.getStencilTargetInfoMap(mu, isForward);
	// filling fields with respective numbers from formula
	for (cart coords = {0,0,0,0}; not grid.isEnd(coords); grid.increment(coords)) {
		flat f = grid.toFlat(coords);
		for (unsigned i = 0; i < N; i++) {
			for (unsigned j = 0; j < N; j++) {
				mfield.get(coords, i, j) = coords.y*coords.y + i*N + j;
			}
			vfield2.get(coords, i) = coords.x*1000/(double)N + i;
		}
	}

	std::cout << "Uploading fields onto GPU" << std::endl;
	mfield.upload();
	vfield2.upload();
	std::cout << "DONE" << std::endl;

	// define the desired stencil
	SimpleStencil stencil(mu, isForward);

    // RUN IT!
	stencil.run(vfield1, mfield, vfield2);

	// download it back to host :)
	vfield1.download();

	// check the TRUTH
	cart coords = {1,2,3,4};
	printf("source-coords: x:%u, y:%u, z:%u, t:%u\n", coords.x, coords.y, coords.z, coords.t);
	printf("  -> flat coords: n:%u, l:%u \n", grid.toFlat(coords).n,  grid.toFlat(coords).l);
	unsigned i = 0;
	unsigned res = 0;
	for (unsigned j = 0; j < N; j++) {
		res += ((coords.y+1)*(coords.y+1) + i*N + j)*(coords.x*1000/(double)N + j);
	}
	std::cout << "Theoretical result: " << res << std::endl;
	std::cout << "From Stencil function: " << vfield1.get(coords, i) << std::endl;
}
