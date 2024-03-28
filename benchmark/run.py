from pyBenchmark import *
import itertools
import pprint


# Benchmark parameters
Ns = [32, 64, 128]
numRHSs = [1, 12, 24, 36, 48, 60]
blkSizes = [32, 64, 128, 256, 512]
all_comb = list(itertools.product(Ns, numRHSs, blkSizes))
targets = {'copy_noMalloc'      : all_comb
           , 'copy_withMalloc'  : all_comb
           , 'mrhs_blas'        : [(N,numRHS,blkSize) for N,numRHS,blkSize in all_comb if (blkSize==128 or blkSize==256)]
           , 'mrhs_lanes'       : [(N,numRHS,9999) for N,numRHS,blkSize in all_comb if (blkSize==32)]
           , 'mrhs_sharedmem'   : [(N,numRHS,blkSize) for N,numRHS,blkSize in all_comb if (blkSize%N==0 and numRHS%(blkSize//N)==0)]
           , 'stencil_blas'     : [(N,numRHS,blkSize) for N,numRHS,blkSize in all_comb if (blkSize==128 or blkSize==256)]
           , 'stencil_lanes'    : [(N,numRHS,9999) for N,numRHS,blkSize in all_comb if (blkSize==32)]}
Ts = ['realF', 'realD', 'complexF', 'complexD']
grids = [
    (4, 4, 4, 4),
    (4, 4, 8, 8),
    (8, 8, 8, 8),
    (16, 16, 16, 16)
]

numCompilations = 0
for key, value in targets.items():
    numCompilations+=len(value)
print("Total number of compilations is : ", numCompilations)


# Problem: we need to reduce the amount of compilations happening as much as possible
# For some binaries, the blkSize is irrelevant or not very important (copy and blas binaries)
# How do I implement this? -> Outer loop: binaries
# Okay solved.

# Now I need to think about how to save the data to make it accessible later on.
# I could use some database, but that might be overengineering 
# I will use JSON, but how exactly? -> Nested dictionary:
# order of keys: target : compile_params : runtime_params : bench_keyword : value 

# okay so lets write a routine that performs these calculations for a specific target

def run_benchmark(target: str, useSrun: bool = False):
    # prepare list of compilation parameters
    compile_params = list(itertools.product(Ts, targets[target]))[:1]
    print(compile_params)

    # prepare dictionary to write results into
    results = dict()
    i = 0
    max_i = len(compile_params) * len(grids)
    for T, (N, numRHS, blkSize) in compile_params:
        temp_results_dict = dict()
        for grid in grids[:1]:
            i+=1
            print(f"Currently working on (i = {i}/{max_i}): ", T, N, numRHS, blkSize, grid)
            set_meson_bench_params(T=T, N=N, numRHS=numRHS, blkSize=blkSize)
            try:
                print("    reconfigure()")
                reconfigure()
                print("    compile_target()")
                compile_target(target=target, force_recompile=False)
                print("    run_binary()")
                output = run_binary(target=target, args=[str(x) for x in grid]+['0', 'true'], useSrun=useSrun)
            except CommandFailedError as err:
                print("PANIC: shell command failed:")
                print(err.stdout)
                print(err.stderr)
            temp_results_dict[grid] = parseBenchOutput(output.stdout.decode('utf-8'))
        results[(T,N,numRHS,blkSize)] = temp_results_dict
    return results

res = run_benchmark('stencil_blas')

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(res)

writePickle(res, filepath='stencil_blas.pickle')
pp.pprint(loadPickle(filepath='stencil_blas.pickle'))


# TODO: save results for one specific target to json
# TODO: build tool to plot the json file :) (4dim plot?)
# TODO: send results to christoph







