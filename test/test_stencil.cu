#include "../cugrid/cugrid.h"

#include <cuda/std/complex>
#include <chrono>
using namespace std::chrono;


constexpr unsigned lenLane = 32;
constexpr unsigned N = 2;
// using T_arithm = cuda::std::complex<double>;
using T_arithm = unsigned;
using lRealD = Lane<T_arithm, lenLane>;
using iVecRealD = iVector<lRealD, N>;
using iMatRealD = iMatrix<lRealD, N>;

const unsigned mu = 0;
const bool isForward = true;

int main () {
	Grid<lenLane> grid(6,2,2,4);
	Lattice<iVecRealD> vfield1(grid), vfield2(grid);
	Lattice<iMatRealD> mfield(grid);

	auto laneIdxMap = grid.getLaneIdxMap(mu, isForward);
	auto stinfo = grid.getStencilTargetInfoMap(mu, isForward);
	// filling fields with respective numbers from formula
	for (cart coords = {0,0,0,0}; not grid.isEnd(coords); grid.increment(coords)) {
		flat f = grid.toFlat(coords);
		//printf("coords: x:%u, y:%u, z:%u, t:%u <--> n:%u, l:%u\n", coords.x, coords.y, coords.z, coords.t, f.n, f.l);
		//printf("    target: n:%u, isBorder:%s, l:%u\n", stinfo[f.n].n_target, stinfo[f.n].isBorder ? "yes" : "no", laneIdxMap[f.l]);
		for (unsigned i = 0; i < N; i++) {
			for (unsigned j = 0; j < N; j++) {
				mfield.get(coords, i, j) = coords.x;//*1000;// + i*N + j;
			}
			vfield2.get(coords, i) = 1;//coords.x;//*1000;// /(double)N + i;
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
	cart coords = {2,0,0,0};
	printf("source-coords: x:%u, y:%u, z:%u, t:%u\n", coords.x, coords.y, coords.z, coords.t);
	printf("  -> flat coords: n:%u, l:%u \n", grid.toFlat(coords).n,  grid.toFlat(coords).l);
	unsigned i = 0;
	// unsigned res = 0;
	// for (unsigned j = 0; j < N; j++) {
	// 	res += ((coords.x+1)*1000 + i*N + j)*(coords.x*1000/(double)N + j);
	// }
	// std::cout << "Theoretical result: " << res << std::endl;
	cart f_coords = coords;
	f_coords.x += 1;
	printf("target-coords: x:%u, y:%u, z:%u, t:%u\n", f_coords.x, f_coords.y, f_coords.z, f_coords.t);
	printf("  -> flat coords: n:%u, l:%u \n", grid.toFlat(f_coords).n,  grid.toFlat(f_coords).l);
	std::cout << "Row of matrix: ";
	for (unsigned j = 0; j < N; j++) std::cout << mfield.get(f_coords, i, j) << " ";
	std::cout << std::endl;
	std::cout << "The vector: ";
	for (unsigned j = 0; j < N; j++) std::cout << vfield2.get(coords, j) << " ";
	std::cout << std::endl;
	std::cout << "From Stencil function: " << vfield1.get(coords, i) << std::endl;
}
