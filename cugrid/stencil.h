#pragma once

#include "errorcheck.h"

#include "lattice.h"
#include "grid.h"
#include "lane.h"
#include "matmul.h"
#include "stopwatch.h"

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
				, &d_targetfield[d_stinfo[n].n_target][i][0]
				, &d_rhs[n][0]);
			for (unsigned j = 1; j < N; j++) {
				lobj::mac(w
					, &d_res[n][i]
					, &d_targetfield[d_stinfo[n].n_target][i][j]
					, &d_rhs[n][j]);

			}
		}
	}
}

template<class lobj, unsigned N, unsigned batchsize>
__global__ void ker_SimpleStencil_mrhs (const StencilTargetInfo * d_stinfo
				, const unsigned * d_laneIdxMap
				, unsigned sizeVNodes
				, iVector<lobj,N> * const * d_batch_res
				, const iMatrix<lobj,N> * d_targetfield
				, const iVector<lobj,N> * const * d_batch_rhs) 
{
	warpInfo w;
	unsigned n = w.warpIdxGlobal / (N*batchsize); // vNode index
	unsigned i = (w.warpIdxGlobal % (N*batchsize)) / batchsize; // iTensor index
	unsigned b = w.warpIdxGlobal % batchsize; // batch index


	if (n < sizeVNodes) { // access guard; no divergence
		if (d_stinfo[n].isBorder) { // border checking; no divergence
			lobj::mul(w
				, &d_batch_res[b][n][i]
				, &d_targetfield[d_stinfo[n].n_target][i][0]
				, &d_batch_rhs[b][n][0]
				, d_laneIdxMap);
			for (unsigned j = 1; j < N; j++) {
				lobj::mac(w
					, &d_batch_res[b][n][i]
					, &d_targetfield[d_stinfo[n].n_target][i][j]
					, &d_batch_rhs[b][n][j]
					, d_laneIdxMap);
			}
		} else {
			lobj::mul(w
				, &d_batch_res[b][n][i]
				, &d_targetfield[d_stinfo[n].n_target][i][0]
				, &d_batch_rhs[b][n][0]);
			for (unsigned j = 1; j < N; j++) {
				lobj::mac(w
					, &d_batch_res[b][n][i]
					, &d_targetfield[d_stinfo[n].n_target][i][j]
					, &d_batch_rhs[b][n][j]);

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
		// std::cout << "calling ker_SimpleStencil with:" << std::endl;
		// std::cout << "    blocks  : " << blocks << std::endl;
		// std::cout << "    threads : " << blocksize << std::endl;
		// std::cout << "    #lpb    : " << lanes_per_block << std::endl;
		ker_SimpleStencil <lobj,N> <<<blocks,blocksize>>> (
				d_stinfo
				, d_laneIdxMap
				, grid.calcSizeVNodes()
				, res.d_data
				, targetfield.d_data
				, rhs.d_data);
		CCE(cudaDeviceSynchronize());

		cudaFree(d_laneIdxMap);
		cudaFree(d_stinfo);
	}

	template<class lobj, unsigned N, unsigned batchsize, unsigned blocksize = 256>
	void run_mrhs(VectorBatch<lobj,N,batchsize> & batch_res
		, const Lattice< iMatrix<lobj,N> > & targetfield
		, const VectorBatch<lobj,N,batchsize> & batch_rhs) 
	{
		// check if grids are compatible
		if ((not check_grid_compatible<lobj, N, batchsize>(batch_res, targetfield.grid)) or (not check_grid_compatible<lobj, N, batchsize>(batch_rhs, targetfield.grid))) {
			throw std::logic_error("Grids not compatible");
		}

        stopwatch.press();

		Grid<lobj::_lenLane> grid = targetfield.grid;

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

		// prepare arrays of pointers to batch data
		iVector<lobj, N> * h_batch_res[batchsize], * h_batch_rhs[batchsize];
		iVector<lobj, N> ** d_batch_res, ** d_batch_rhs;
		for (unsigned b = 0; b < batchsize; b++) {
			h_batch_res[b] = batch_res[b]->d_data;
			h_batch_rhs[b] = batch_rhs[b]->d_data;
		}

		// copy those arrays to device memory
		CCE(  cudaMalloc(&d_batch_res, sizeof(iVector<lobj,N>*)*batchsize)  );
		CCE(  cudaMalloc(&d_batch_rhs, sizeof(iVector<lobj,N>*)*batchsize)  );
		CCE(  cudaMemcpy(d_batch_res, h_batch_res, sizeof(iVector<lobj,N>*)*batchsize, cudaMemcpyHostToDevice)  );
		CCE(  cudaMemcpy(d_batch_rhs, h_batch_rhs, sizeof(iVector<lobj,N>*)*batchsize, cudaMemcpyHostToDevice)  );

		// kernel call :)
		static_assert(blocksize % lobj::_lenLane == 0, "Length of lane does not divide the blocksize. Change blocksize or lane length!");
		unsigned lanes_per_block = blocksize / lobj::_lenLane;
		unsigned blocks = (grid.calcSizeVNodes()*N*batchsize + lanes_per_block - 1)/lanes_per_block;
		ker_SimpleStencil_mrhs <lobj,N,batchsize> <<<blocks,blocksize>>> (
				d_stinfo
				, d_laneIdxMap
				, grid.calcSizeVNodes()
				, d_batch_res
				, targetfield.d_data
				, d_batch_rhs);
		CCE(cudaDeviceSynchronize());

		cudaFree(d_laneIdxMap);
		cudaFree(d_stinfo);

        stopwatch.press();
	}
};


