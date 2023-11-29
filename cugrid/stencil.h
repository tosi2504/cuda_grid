#pragma once

#include "errorcheck.h"

#include "lattice.h"
#include "grid.h"
#include "lane.h"

template<class lobj, unsigned N>
using GaugeField = std::array< Lattice<iMatrix<lobj, N>> * , 4 >;

template<class lobj, unsigned N>
__global__ void ker_SimpleStencil (const StencilTargetInfo * d_stinfo
				, const unsigned * d_laneIdxMap
				, unsigned sizeVNodes
				, iVector<lobj,N> * d_res
				, const iMatrix<lobj,N> * d_targetfield
				, const iVector<lobj,N> * d_rhs) 
{
	warpInfo w;
	unsigned n = w.warpIdxGlobal / N; // vNode index
	unsigned i = w.warpIdxGlobal % N; // iTensor index
	if (n < sizeVNodes) { // access guard; no divergence
		if (d_stinfo[n].isBorder) { // border checking; no divergence
			lobj::mul(w
				, &d_res[n][i]
				, &d_targetfield[d_stinfo[n].n_target][i][0]
				, &d_rhs[n][0]
				, d_laneIdxMap);
			for (unsigned j = 1; j < N; j++) {
				lobj::mac(w
					, &d_res[n][i]
					, &d_targetfield[d_stinfo[n].n_target][i][j]
					, &d_rhs[n][j]
					, d_laneIdxMap);
			}
		} else {
			lobj::mul(w
				, &d_res[n][i]
				, &d_targetfield[n][i][0]
				, &d_rhs[n][0]);
			for (unsigned j = 1; j < N; j++) {
				lobj::mac(w
					, &d_res[n][i]
					, &d_targetfield[n][i][j]
					, &d_rhs[n][j]);
			}
		}
	}
}

class SimpleStencil {
	private:
	unsigned mu;
	bool isForward;
	
	public:
	SimpleStencil(unsigned mu, bool isForward):
		mu(mu), isForward(isForward)
	{}

	template<class lobj, unsigned N, unsigned blocksize = 256>
	void run(Lattice<iVector<lobj,N>> & res
			, const Lattice<iMatrix<lobj,N>> & targetfield
			, const Lattice<iVector<lobj,N>> & rhs)
	{
		// check if grids are compatible
		if (res.grid != targetfield.grid or res.grid != rhs.grid) {
			throw std::logic_error("Grids not compatible!");
		}
		Grid<lobj::_lenLane> grid = res.grid;

		// create laneIdxMap
		auto laneIdxMap = grid.getLaneIdxMap(mu, isForward);
		unsigned * d_laneIdxMap;
		CCE(cudaMalloc(&d_laneIdxMap, sizeof(unsigned)*lobj::_lenLane));
		CCE(cudaMemcpy(d_laneIdxMap
					, laneIdxMap.data()
					, sizeof(unsigned)*lobj::_lenLane
					, cudaMemcpyHostToDevice));

		// create StencilTargetInfo map
		auto stinfo = grid.getStencilTargetInfoMap(mu, isForward);
		StencilTargetInfo * d_stinfo;
		CCE(cudaMalloc(&d_stinfo, sizeof(StencilTargetInfo)*grid.calcSizeVNodes()));
		CCE(cudaMemcpy(d_stinfo
					, stinfo.data()
					, sizeof(StencilTargetInfo)*grid.calcSizeVNodes()
					, cudaMemcpyHostToDevice));

		// kernel call :)
		static_assert(blocksize % lobj::_lenLane == 0, "Length of lane does not divide the blocksize. Change blocksize or lane length!");
		unsigned lanes_per_block = blocksize / lobj::_lenLane;
		unsigned blocks = (grid.calcSizeVNodes()*N + lanes_per_block - 1)/lanes_per_block;
		std::cout << "calling ker_SimpleStencil with:" << std::endl;
		std::cout << "    blocks  : " << blocks << std::endl;
		std::cout << "    threads : " << blocksize << std::endl;
		std::cout << "    #lpb    : " << lanes_per_block << std::endl;
		ker_SimpleStencil <lobj,N> <<<blocks,blocksize>>> (
				d_stinfo
				, d_laneIdxMap
				, grid.calcSizeVNodes()
				, res.d_data
				, targetfield.d_data
				, rhs.d_data);
		CLCE();
		CCE(cudaDeviceSynchronize());

		cudaFree(d_laneIdxMap);
		cudaFree(d_stinfo);
	}
};


