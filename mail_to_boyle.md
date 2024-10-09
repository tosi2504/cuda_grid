Dear Prof. Boyle,

I am a PhD student at the University of Regensburg where I work with Christoph Lehner and Andreas Schaefer to optimise Wilson fermion calculations. In the past year I have worked with Grid extensively.

I write to you regarding performance optimisations of coarse-grid multi-rhs stencil operations on GPUs. In Grid this operation is implemented using the available GPU BLAS routines. We found, however, that user managed cache (cuda lang: shared memory) can be exploited to improve performance by 100% and more. Therefore, I was wondering if you would be interested in a joint paper with Prof. Lehner, Prof. Schaefer and myself to publish these results and push to production in Grid.

I have appended a jupyter notebook compiled as html which includes the benchmarks comparing Grid, my own BLAS implementation and the highly optimized kernel (2dbtv2). Additionally, you will find a slightly more detailed description of what is going on. I would love to initiate an online meeting with you, Prof. Lehner and Prof. Schaefer to go into further detail.

A quick side note: I had to slightly modify GeneralCoarsenedMatrixMultiRHS.h, since it is hardcoded to ComplexD. My benchmarks are using single precision, however. The modified file is appended in case you want to double check that performance is not reduced by this.

Best,
Tobias Sizmann
