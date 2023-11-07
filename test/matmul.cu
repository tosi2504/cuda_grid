#include "../cugrid/lane.h" 
#include "../cugrid/tensor.h" 
#include "../cugrid/errorcheck.h" 

#include <iostream>

constexpr unsigned lenLane = 32;
constexpr unsigned N = 128; // 8*16
using vRealD = Lane<double, lenLane>;
using iVecRealD = iVector<vRealD, N>;
using iMatRealD = iMatrix<vRealD, N>;

__global__ void matmulKernel(iVecRealD * d_res, const iMatRealD * d_lhs, const iVecRealD * d_rhs) {
	warpInfo w;
	iMatRealD::matmul(w, d_res, d_lhs, d_rhs);
}


int main () {
	// create vector objects on the host and fill them
	iVecRealD res;
	iVecRealD rhs;
	for (unsigned i = 0; i < N; i++) {
		for (unsigned l = 0; l < lenLane; l++) {
			rhs[i][l] = i+0.01*l;	
		}
	}

	// create matrix object on host and fill it as unit matrix on all laneIdx
	iMatRealD lhs;
	for (unsigned i = 0; i < N; i++) {
		for (unsigned j = 0; j < N; j++) {
			for (unsigned l = 0; l < lenLane; l++) {
				lhs[i][j][l] = i==j ? l : 0;
			}
		}
	}

	// allocate space on device
	iVecRealD * d_res;
	iVecRealD * d_rhs;
	iMatRealD * d_lhs;
	CCE(cudaMalloc(&d_res, sizeof(iVecRealD)));
	CCE(cudaMalloc(&d_rhs, sizeof(iVecRealD)));
	CCE(cudaMalloc(&d_lhs, sizeof(iMatRealD)));

	// copy to device
	CCE(cudaMemcpy(d_res, &res, sizeof(iVecRealD), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_rhs, &rhs, sizeof(iVecRealD), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_lhs, &lhs, sizeof(iMatRealD), cudaMemcpyHostToDevice));

	// run the matmul kernel
	// N = 8*16 warps (32 threads) in total
	using prms = iMatRealD::matmul_prms<512>;
	matmulKernel<<<prms::numBlocks, prms::blocksize>>>(d_res, d_lhs, d_rhs);
	CLCE();
	cudaDeviceSynchronize();
	std::cout << "matmul finished" << std::endl;

	// copy to device
	CCE(cudaMemcpy(&res, d_res, sizeof(iVecRealD), cudaMemcpyDeviceToHost));
	CCE(cudaMemcpy(&rhs, d_rhs, sizeof(iVecRealD), cudaMemcpyDeviceToHost));
	CCE(cudaMemcpy(&lhs, d_lhs, sizeof(iMatRealD), cudaMemcpyDeviceToHost));

	// print result
	res.print(12);
}
