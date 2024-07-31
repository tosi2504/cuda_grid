## 9-point stencil

### shmem implementation

#### Uncoalesced shmem access (memory bank conflicts)
Performance is horrible, most likely due to memory bank conflicts of shmem in the dot-product line.
Need to think about options to solve this problem.
Maybe first need to understand how these memory bank conflicts really emerge.
See `https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/` as source.
Not that the transposeCoalesced kernel of this blog shows the exact same symptom in nsight-compute(under 'Details'):
'Uncoalesced shared accesses'.
So, what do we do now?
The blog suggests adding a padding to the shared memory to offset memory bank accesses.
That's smart but I am not sure whether that works.
Let's dive deeper into the exact access pattern.
First for the transpose kernel, then for the stencil.

##### Transpose Kernel
```
  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
```

In the first iteration (`j = 0`) only `BLOCK_ROWS(= 8)` memory banks are accessed.
Why does padding (`__shared__ float tile[TILE_DIM][TILE_DIM+1]`) fix this problem?
It can be displayed like this (let's assume 8 memory banks and `TILE_DIM = 8` and `BLOCK_ROWS = 4`):
```
     b0  b1  b2  b3  b4  b5  b6  b7
0x00 x   x   x   x   o   o   o   o
0x08 p   x   x   x   x   o   o   o
0x10 o   p   x   x   x   x   o   o
0x18 o   o   p   x   x   x   x   o
0x20 o   o   o   p   x   x   x   x
0x28 o   o   o   o   p   x   x   x
0x30 x   o   o   o   o   p   x   x
0x38 x   x   o   o   o   o   p   x
0x40 x   x   x   o   o   o   o   p
```
Where `x's` are accesses and `o's` are not.
`p` are padding addresses without any meaningful content.
Very nice, I actually understand what is happening.

##### 9-point Stencil
I was able to get massive improvements done using two optimizations regarding memory bank conflicts:


Firstly line `tempRes += shmemA[rowm(dRow, k, N, N)] * shmemX[colm(k, iRhs, numRHS, N)]`:
I modified it to
```
const unsigned _k = (iRhs + k) % N;
tempRes += shmemA[rowm(dRow, _k, N, N)] * shmemX[colm(_k, iRhs, numRHS, N)]; 
```
which gave a performance of factor 2.


Secondly, I made `shmemY` row-major, i.e., `shmemY[colm(...)]` -> `shmemY[rowm(...)]`
Which was a further 25% improvement and nsight-compute is happy now :)


#### Another benchmark
Let's benchmark the code so far on my machine.
We use: 
- type: realF
- N: 32
- numRHS: 60
- grid: 8.8.8.8
Result: 5200 usec.
Compared to the blas implementation: 872+409+6681+410 = 8372 usec.
That is a 35% improvement!!!!!!!!

#### Calculating bandwidth and arithmetic performance
Here are the basic formulas (for realF):
- data_srhs  = numSites (9 N^2 + 18 N numRHS) sizeof(T)
- data_mrhs  = numSites (9 N^2 +  2 N numRHS) sizeof(T)
- flops_real = numSites (18 N^2 numRHS)
- flops_cplx = numSites (72 N^2 numRHS)

#### Further optimizations: 
See
- https://siboehm.com/articles/22/CUDA-MMM
- https://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf
- https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
- https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#pipelining

##### Global mem coalescing
Threads in a warp should access consecutive gmem addresses during one lockstep.
The threads do not need to be consecutive.
This has implications for the filling of shmem in my case.
But in my code, the gmem accesses are already coalesced (thanks to the column-majorness of vec batches).

##### Shmem
Okay, this is obvious and we have fixed it.
Using `T=realF`, `N=32`, `numRHS=60`, `grid=8.8.8.8` and `tileLen=4`:
```
Kernel-Stats:
    Bandwidth(MB/s): 56724.9
    Flops(MFlops): 1.20123e+06
```

##### 1D Blocktiling
To implement 1D Blocktiling efficiently into my framework, I had to change up the geometry a bit.
With tileLen=4 we get the best performance.
Using `T=realF`, `N=32`, `numRHS=60`, `grid=8.8.8.8` and `tileLen=4`:
```
Kernel-Stats:
    Bandwidth(MB/s): 69497.8
    Flops(MFlops): 1.47172e+06
```
This is almost twice as good as cuBLAS!
But I am sure we can get even better performance using 2D blocktiling, so lets get to it.

##### 2D Blocktiling
Let's consider `N=32`. 
If I am correct in the assumption, that we need to split the `N*N` area into quadratic tiles, we might have a problem.
If we do `tileLen=4` we get only 64 threads!
That's really bad occupancy :/
If we do `tileLen=2` we get 256 threads which is okayish.
Whatever, let's try.
I have successfully implemented it and performance is very good.
We got another factor of 2!
I had to enable dynamic shared memory to increase the maximally possible size.
Now we can comfortably do N=64, numRHS=64 and T=complexF without running into memory issues
Anyway: Using `T=realF`, `N=32`, `numRHS=64`, `grid=8.8.8.8`, `rowStride=4`, `rhsStride=8`, `tileHeight=4`, `tileWidth=4`:
```
Kernel-Stats:
    Bandwidth(MB/s): 108841
    Flops(MFlops): 2.84965e+06
```
That is getting very good, though technically were are no-where close to what my GPU should be capable of (but why????).



#### Complex numbers
The performance is horrific for complex numbers, and I am not sure why.
I have a feeling that one might be able to fix this by storing real and imaginary parts in seperate arrays.
That could potentially help with both the strided-gmem access and shared memory bank conflicts.
Maybe I try this in a more controlled environment, namely, the sgemm repo.
Lets do it.

Alright, after a lot of testing, I found a set of optimizations, that could help to get the performance for complex numbers to an acceptable level.

TODO: I need to implement the following optimizations:
- 2D blocktiling
- implement own version of complex numbers
- using fma intrinsics to write a multiply accumulate routine for complex numbers
- optimize blocktiling parameters for max number of threads (ideal: 1024)

HOW's it going:
- started to make Xs row-major. Not sure what that will bring
- the 2dbt and 2dbtv2 kernels are not working correctly with complex numbers ...
- but the shmem-naive kernel does work, which means that the complex number implementation is correct







