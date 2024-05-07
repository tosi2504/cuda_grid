import subprocess
import os
import stat
import sys
import pickle
import pandas as pd
import itertools
import pathlib
import shutil

def get_project_root_path():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), '..'))

def read_meson_file():
    meson_file_path = os.path.join(get_project_root_path(), 'meson.build')
    with open(meson_file_path, "r") as file:
        content = file.read()
    return content

def write_meson_file(content):
    meson_file_path = os.path.join(get_project_root_path(), 'meson.build')
    with open(meson_file_path, "w") as file:
        file.write(content)

def set_meson_bench_params(T: str, N: int, numRHS: int, blkSize: int):
    # modify the meson scipt accordingly
    lines = read_meson_file().splitlines(keepends=True)
    index = int()
    for i, line in enumerate(lines):
        if line.startswith("bench_params"):
            index = i
            break
    lines[index] = 'bench_params = ' + [ f"-DBENCH_PARAM_T={T}"
                                        , f"-DBENCH_PARAM_N={N}"
                                        , f"-DBENCH_PARAM_numRHS={numRHS}"
                                        , f"-DBENCH_PARAM_blkSize={blkSize}" ].__str__() + "\n"
    write_meson_file("".join(lines))



class CommandFailedError(Exception):
    def __init__(self, command, output):
        self.stdout = output.stdout.decode('utf-8')
        self.stderr = output.stderr.decode('utf-8')
        super().__init__(f"Command $>{command}<$ failed with errorcode {output.returncode}")

def reconfigure():
    res = subprocess.run(args=['meson', 'setup', 'build', '--reconfigure']
                   , cwd=get_project_root_path()
                   , stdout=subprocess.PIPE
                   , stderr=subprocess.PIPE)
    if (res.returncode != 0):
        raise CommandFailedError('meson setup build --reconfigure', res)
    return res

def compile_target(target: str, force_recompile: bool = False):
    if (force_recompile):
        res = subprocess.run(args=['ninja', 'clean']
                       , cwd=os.path.join(get_project_root_path(), 'build')
                       , stdout=subprocess.PIPE
                       , stderr=subprocess.PIPE)
        if (res.returncode != 0):
            raise CommandFailedError('ninja clean', res)
        
    res = subprocess.run(args=['ninja', target]
                   , cwd=os.path.join(get_project_root_path(), 'build')
                   , stdout=subprocess.PIPE
                   , stderr=subprocess.PIPE)
    if (res.returncode != 0):
        raise CommandFailedError(f'ninja {target}', res)
    return res

def create_bin_directory():
    pathlib.Path(os.path.join(get_project_root_path(), 'bin')).mkdir(parents=True, exist_ok=True)

def save_target_binary(target: str, compile_params: dict):
    cwd = get_project_root_path()
    binname = target+f"T-{compile_params['T']}_N-{compile_params['N']}_numRHS-{compile_params['numRHS']}_blkSize-{compile_params['blkSize']}"
    shutil.copyfile(src = os.path.join(cwd, 'build/', target), dst = os.path.join(cwd, 'bin/', binname))

def run_binary(target: str, args: list, subdir: str, useSrun: bool = False):
    binary = list()
    if useSrun:
        binary.append('srun')
    binary.append('./'+target)

    res = subprocess.run(args=binary + args
                    , cwd=os.path.join(get_project_root_path(), subdir)
                    , stdout=subprocess.PIPE
                    , stderr=subprocess.PIPE)
    if (res.returncode != 0):
        raise CommandFailedError(f'./{target}'+' '+' '.join(args), res)
    return res

def parseValueString(valueStr: str):
    if (valueStr.startswith('(') and valueStr.endswith(')')):
        temp = valueStr[1:-1].split(',')
        return (int(temp[0]), int(temp[1]), int(temp[2]), int(temp[3]))
    # else:
    try:
        return int(valueStr)
    except ValueError:
        try:
            return float(valueStr)
        except ValueError:
            return valueStr

def parseBenchOutput(output: str):
    lines = (line for line in output.splitlines())

    # skip until header of benchmark results
    for line in lines:
        if line.startswith("========= BENCHMARK RESULTS ========="):
            break

    result = dict()
    for line in lines:
        # check if end of benchmark results has been found
        if line.startswith("======================"):
            break
        # remove spaces
        tempdata = line.strip().replace(' ', '').split(':')
        key = tempdata[0]
        value = parseValueString(tempdata[1])
        result[key] = value
    
    return result



# Benchmark parameters
Ns = [32, 64, 128]
numRHSs = [1, 12, 24, 36, 48, 60]
blkSizes = [32, 64, 128, 256, 512]
all_comb = list(itertools.product(Ns, numRHSs, blkSizes))
targets = {'copy_noMalloc'      : [(N,numRHS,blkSize) for N,numRHS,blkSize in all_comb if (blkSize==128 or blkSize==256)]
           , 'copy_withMalloc'  : [(N,numRHS,blkSize) for N,numRHS,blkSize in all_comb if (blkSize==128 or blkSize==256)]
           , 'mrhs_blas'        : [(N,numRHS,blkSize) for N,numRHS,blkSize in all_comb if (blkSize==128 or blkSize==256)]
           , 'mrhs_lanes'       : [(N,numRHS,9999) for N,numRHS,blkSize in all_comb if (blkSize==32 and numRHS >= 12)]
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

def compile_binaries(target: str):
    # get compile params
    compile_params = list(itertools.product(Ts, targets[target]))
    print(compile_params)

    # prepare bin directory
    create_bin_directory()

    i = 0
    max_i = len(compile_params)
    for T, (N, numRHS, blkSize) in compile_params:
        i+=1
        print(f"Currently working on (i = {i}/{max_i}): ", T, N, numRHS, blkSize)
        set_meson_bench_params(T=T, N=N, numRHS=numRHS, blkSize=blkSize)
        try:
            reconfigure()
            compile_target(target)
        except CommandFailedError as err:
            print("PANIC: shell command failed:")
            print(err.stdout)
            print(err.stderr)
        save_target_binary(target=target, compile_params={'T': T, 'N':N, 'numRHS':numRHS, 'blkSize':blkSize})

def run_benchmark_on_precompiled_binaries(target: str, useSrun: bool = False):
    compile_params = list(itertools.product(Ts, targets[target]))
    print(compile_params)
    results = dict()
    i = 0
    max_i = len(compile_params) * len(grids)

    for T, (N, numRHS, blkSize) in compile_params:
        binname = target+f"T-{T}_N-{N}_numRHS-{numRHS}_blkSize-{blkSize}"
        os.chmod(os.path.join(get_project_root_path(), 'bin', binname), stat.S_IEXEC)
        temp_results_dict = dict()
        for grid in grids:
            i += 1
            print(f"Currently working on (i = {i}/{max_i}): ", T, N, numRHS, blkSize, grid)
            try:
                output = run_binary(target=binname, args=[str(x) for x in grid]+['0', 'true'], subdir='bin', useSrun=useSrun)
                print(output.stdout.decode('utf-8'))
                temp_results_dict[grid] = parseBenchOutput(output.stdout.decode('utf-8'))
            except CommandFailedError as err:
                print("PANIC: shell command failed:")
                print(err.stdout)
                print(err.stderr)
        results[(T,N,numRHS,blkSize)] = temp_results_dict

    return results



def run_benchmark(target: str, useSrun: bool = False):
    # prepare list of compilation parameters
    compile_params = list(itertools.product(Ts, targets[target]))
    print(compile_params)

    # prepare dictionary to write results into
    results = dict()
    i = 0
    max_i = len(compile_params) * len(grids)
    for T, (N, numRHS, blkSize) in compile_params:
        temp_results_dict = dict()
        for grid in grids:
            i+=1
            print(f"Currently working on (i = {i}/{max_i}): ", T, N, numRHS, blkSize, grid)
            set_meson_bench_params(T=T, N=N, numRHS=numRHS, blkSize=blkSize)
            try:
                print("    reconfigure()")
                reconfigure()
                print("    compile_target()")
                compile_target(target=target, force_recompile=False)
                print("    run_binary()")
                output = run_binary(target=target, args=[str(x) for x in grid]+['0', 'true'], subdir='build', useSrun=useSrun)
                temp_results_dict[grid] = parseBenchOutput(output.stdout.decode('utf-8'))
            except CommandFailedError as err:
                print("PANIC: shell command failed:")
                print(err.stdout)
                print(err.stderr)
        results[(T,N,numRHS,blkSize)] = temp_results_dict
    return results

def data_to_df(data: dict):
    # what are we doing here
    temp = dict()
    for target in data.keys():
        for compile_params in data[target].keys():
            for grid, valueDict in data[target][compile_params].items():
                tempNested = dict()
                for key, value in data[target][compile_params][grid].items():
                    if type(value) == tuple:
                        tempNested[key] = str(value) 
                    else:
                        tempNested[key] = value 
                temp[(target, *compile_params, str(grid))] = tempNested

    #return pandas.DataFrame.from_dict(temp, orient='index')
    df = pd.DataFrame.from_dict(temp)#,columns=['target', 'T', 'N', 'numRHS', 'blkSize', 'grid'], orient='index')
    df.index.name = 'result'
    df.columns.names = ['target', 'T', 'N', 'numRHS', 'blkSize', 'grid']
    return df



def writePickle(data: dict, filepath: str):
    with open(filepath, 'wb') as file:
        pickle.dump(data, file=file, protocol=pickle.HIGHEST_PROTOCOL)

def loadPickle(filepath: str):
    with open(filepath, 'rb') as file:
        return pickle.load(file)



def eliminate_blkSize_by_max(data: pd.Series):
    reduced_names = [name for name in data.index.names if name != "blkSize"]
    return data.groupby(level=reduced_names).max()

def plot_data(df, ax, xlabel: str, valuelabel: str, selectors: dict, kind: str = 'bar'):
    series = df.loc[valuelabel, ]
    series = eliminate_blkSize_by_max(series)
    for key, val in selectors.items():
        series = series.xs(val, level=key)
    series.plot(ax = ax, kind = kind
                , ylabel = valuelabel
                , xlabel = xlabel
                , title = str(selectors))





