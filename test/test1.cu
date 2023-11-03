#include "../cugrid/lane.h" 
#include "../cugrid/errorcheck.h" 

#include <iostream>

template<class T>
__global__ void fillLanes(Lane<T> * d_lanes) {
	warpInfo w;
	d_lanes[w.warpIdx].setByThread(w, w.laneIdx*0.1);
}

template<class T>
__global__ void addLanes(Lane<T> * d_ret, const Lane<T> * d_lhs, const Lane<T> * d_rhs) {
	Lane<T>::add(warpInfo(), d_ret, d_lhs, d_rhs);
}

int main () {
	// create objects on host
	Lane<double> * h_lanes = new Lane<double>[3];

	// allocate space on device
	Lane<double> * d_lanes;
	cudaMalloc(&d_lanes, 3*sizeof(Lane<double>));

	// copy to device
	CCE(cudaMemcpy(d_lanes, h_lanes, 3*sizeof(Lane<double>), cudaMemcpyHostToDevice));

	// run fillLanes kernel
	fillLanes<double><<<1,32*3>>>(d_lanes);
	CLCE();
	cudaDeviceSynchronize();


	std::cout << "Lanes filled!" << std::endl;

	// run addLanes kernel
	addLanes<double><<<1,32>>>(d_lanes+0, d_lanes+1, d_lanes+2);
	CLCE();
	cudaDeviceSynchronize();

	// copy to device
	CCE(cudaMemcpy(h_lanes, d_lanes, 3*sizeof(Lane<double>), cudaMemcpyDeviceToHost));

	// print result
	for (int i = 0; i < 32; i++) {
		std::cout << h_lanes[0][i] << std::endl;
	}
}
