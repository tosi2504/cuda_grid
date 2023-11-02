#include "../cugrid/lane.h" 

#include <iostream>

#define CCE(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CLCE() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}


template<class T>
__global__ void fillLanes(Lane<T> * d_lanes) {
	warpInfo w;
	printf("%u", w.laneIdx);
	d_lanes[w.warpIdx].setByThread(threadIdx.x*0.1);
}

template<class T>
__global__ void addLanes(Lane<T> * d_ret, const Lane<T> * d_lhs, const Lane<T> * d_rhs) {
	
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

	// run addLanes kernel
	addLanes<double><<<1,32*3>>>(d_lanes+0, d_lanes+1, d_lanes+2);
	CLCE();
	cudaDeviceSynchronize();

	// copy to device
	CCE(cudaMemcpy(h_lanes, d_lanes, 3*sizeof(Lane<double>), cudaMemcpyDeviceToHost));

	// print result
	for (int i = 0; i < 32; i++) {
		std::cout << h_lanes[0][i] << std::endl;
	}
}
