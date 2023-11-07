#include "../cugrid/lane.h" 
#include "../cugrid/tensor.h" 
#include "../cugrid/errorcheck.h" 

#include <iostream>

constexpr unsigned lenLane = 32;
constexpr unsigned N = 128; // 8*16
using vRealD = Lane<double, lenLane>;
using iVecRealD = iVector<vRealD, N>;

__global__ void addKernel(iVecRealD * res, const iVecRealD * lhs, const iVecRealD * rhs) {
	warpInfo w;
	iVecRealD::add(w, res, lhs, rhs);
}

int main () {
	// define the lane type we are using
	
	// create vector objects on the host and fill them
	iVecRealD h_vecs[3];
	for (unsigned i = 0; i < N; i++) {
		for (unsigned j = 0; j < lenLane; j++) {
			h_vecs[1][i][j] = i + 0.01*j;	
			h_vecs[2][i][j] = i + 0.01*j + 1000;	
		}
	}

	// allocate space on device
	iVecRealD * d_vecs;
	cudaMalloc(&d_vecs, 3*sizeof(iVecRealD));

	// copy to device
	CCE(cudaMemcpy(d_vecs, h_vecs, 3*sizeof(iVecRealD), cudaMemcpyHostToDevice));

	// run the add kernel
	// N = 8*16 warps (32 threads) in total
	using prms = iVecRealD::add_prms<512>;
	addKernel<<<prms::numBlocks, prms::blocksize>>>(&d_vecs[0], &d_vecs[1], &d_vecs[2]);
	CLCE();
	cudaDeviceSynchronize();
	std::cout << "addition finished" << std::endl;

	// copy to device
	CCE(cudaMemcpy(h_vecs, d_vecs, 3*sizeof(iVecRealD), cudaMemcpyDeviceToHost));

	// print result
	h_vecs[0].print(1);
}
