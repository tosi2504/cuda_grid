Hi Peter,

thanks for your fast answer!

So far all my work has been done optimizing from the GPU-local (CUDA-lang: device memory) level upwards to the register level.
Therefore all my benchmarks are single-GPU.
Here an in-depth description of my kernel:

Let's assume we have some coarse grid lattice with complex square matrix fields `A1, A2 ... A{nPoints}` and vector fields `X1, Y1, X2, Y2 ... X{M}, Y{M}`.
We have some stencil geometry with `nPoints` points, which for each point `i` maps a site `s` to `s_i`.
The matrices and vectors are of size `N`.
`M` denotes the number of right hand sides.

Each site `s` is uniquely mapped to a CUDA block.
The main observation which motivated my approach was, that for reasonable parameters of `M` and `N` all `X{m}_s`, `Y{m}_s` and a `A{i}_{s_i}` fit onto the `__shared__` memory (shmem) of Ampere GPUs and upwards.
So, in a first step shmem is populated with `X{m}_s` from global memory (gmem).
Then, load `A{i}_{s_i}` into shmem and matrix-multiply-accumulate the result into shmem for each {i}. 
Finally, write the result out onto gmem into the respective `Y`'s.
This is the whole core idea.

This implementation is L1-cache optimal.
No element is loaded twice from gmem, i.e., we have `nPoints*N**2 + 2*M*N` loads per site.
Assuming that this operation is bandwidth-bound (somewhat valid for `N,M <= 64`) one can calculate the performance gains over the BLAS implementation:
BLAS needs `nPoints*(N**2 + 2*M*N)` loads per site.
I.e. for sufficiently large  `nPoints` we get a `1 + 2M/N` performance factor.

While the idea is simple enough, the devil is in the details.
Getting the matrix-multiply-accumulate from shmem level to register level right, was tricky and I mostly just applied what I learned from reading CUTLASS code and from reverse engineering PTX from cuBLAS.
I was able to optimize this enough for float32's as to show the expected performance behaviour, but I guess there is room for improvement.
Especially, if one would consider using tensorfloat32's.

While presenting the various optimizations from shmem level upwards to the FMA instructions is enough to fill a lengthy talk, here is a rough breakdown.
One needs to harness the maximum bandwidth ( sizeof(transaction) / t_cycle * numStreamMPs ) of shmem, where sizeof(transaction) is 128 bytes for almost all recent GPUs.
To this end, two parameters can be optimized (note: one transaction allows for 128 bytes):
1. Reducing the number of transactions: This is achieved by first loading small blocks (e.g. 2x2) of the respective matrices into the (thread-local) register file. This greatly improves the arithmetic density in respect to the shmem loads.
2. Maximizing the number of bytes each transaction provides to the threads. This is a tricky one. One needs to avoid certain strided access pattern and memory bank conflicts and at the same time make use of the vectorized load instructions and coalesce shmem accesses over the threads ("map thread indices to memory banks"). Exploiting the multicast transaction operation newer architectures have to offer also helps.

Since AMD's and NVIDIA's architectures are very similar with the exception of naming conventions, I have the suspicion that the aforementioned optimizations work just as well on AMD devices.
Unfortunately, I currently do not have access to such a machine and therefore could not test it.
So much to my kernel.

I have to admit, I am not entirely sure what is going on in the Perfetto plot as I am just starting to learn about both multigrid and multi-GPU optimizations, but I will discuss it with Christoph.
I suppose the Grid software kernels are the red sections and perform the necessary part of the Halo exchange for one point, on which the BLAS calls are waiting.
In that case I can follow your point about a low-hanging factor of 2x.


Best,
Tobias
