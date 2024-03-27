#include "bench_params.h"
#include "../cugrid/cugrid.h"

constexpr unsigned lenLane = 32;


constexpr unsigned N = BENCH_PARAM_N;
constexpr unsigned batchsize = BENCH_PARAM_numRHS;

using T_arithm = BENCH_PARAM_T;
using lane_t = Lane<T_arithm, lenLane>;
using vec_t = iVector<lane_t, N>;
using mat_t = iMatrix<lane_t, N>;
using VecField = Lattice<vec_t>;
using MatField = Lattice<mat_t>;
using VecBatch = VectorBatch<lane_t,N,batchsize>;

int main (int argc, char * argv[]) {
	unsigned Lx, Ly, Lz, Lt;
	unsigned mu;
	bool isForward;
	parseArgs(argc, argv, &Lx, &Ly, &Lz, &Lt, &mu, &isForward);
	Grid<lenLane> grid(Lx,Ly,Lz,Lt);


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

	double resTime = 0;
	BENCHMARK(resTime, 100, stencil.run_mrhs<lane_t COMMA N COMMA batchsize>, res, mfield, rhs);

	print_results<T_arithm, lenLane>("stencil_lanes", resTime, N, batchsize, 999, grid, mu, isForward);
}
