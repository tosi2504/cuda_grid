#include "../cugrid2/cugrid2.h"

#include <random>
#include "cublas_v2.h"

using T = float;
constexpr unsigned N = 60;

int main () {
	bGrid grid(4,4,4,4);
	bVectorField<T,N> x(grid), y(grid);
	bMatrixField<T,N> A(grid);

	std::mt19937 gen(0);
	A.fill_random(gen, 0, 1);
	x.fill_random(gen, 0, 1);

	y.upload();
	A.upload();
	x.upload();

	cublasHandle_t handle;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&handle);
	matmul_srhs::cublas<T, N>(handle, y, A, x);
	cublasStatus = cublasDestroy(handle);
}
