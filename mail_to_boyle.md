Dear Prof. Boyle,

I am a PhD student at the University of Regensburg where I work with Christoph Lehner and Andreas Schaefer to optimise Wilson fermion calculations. In the past year I have worked with Grid extensively.

I write to you regarding performance optimisations of coarse-grid multi-rhs stencil operations on GPUs. In Grid this operation is implemented using the available GPU BLAS routines. We found, however, that user managed cache (cuda lang: shared memory) can be exploited to improve performance by 100% and more. Therefore, I was wondering if you would be interested in joining a discussion with Prof. Lehner, Prof. Schaefer and myself regarding these results.

I have appended a jupyter notebook compiled as html which includes the benchmarks comparing Grid, my own BLAS implementation (to reconstruct Grids performance characteristics) and my handcrafted CUDA kernel (2dbtv2). Additionally, you will find a slightly more detailed description of what is going on.
In principal this implementation also allows for latency hiding the Halo exchange, though I have not tested this.

I had to slightly modify GeneralCoarsenedMatrixMultiRHS.h, since it is hardcoded to ComplexD. My benchmarks are using single precision, however. The modified file is appended in case you are interested in these changes.

So far my work has been done exclusively in CUDA. Theoretically, a port to HIP should be possible without sacrificing performance. This has yet to be seen, however.

Best,
Tobias Sizmann
