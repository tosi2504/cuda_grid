import subprocess
import os
import sys
import json
import pickle

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

def run_binary(target: str, args: list, useSrun: bool = False):
    binary = './'+target
    if useSrun:
        binary = 'srun ' + binary

    res = subprocess.run(args=[binary]+args
                    , cwd=os.path.join(get_project_root_path(), 'build')
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



class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tuple):
            return list(obj)
        else:
            return json.JSONEncoder(self, obj)

def writeJSON(data: dict, filepath: str):
    with open(filepath, "w") as file:
        json.dump(data, file, cls=CustomEncoder)

def loadJSON(filepath: str):
    with open(filepath, "r") as file:
        return json.load(file)

def writePickle(data: dict, filepath: str):
    with open(filepath, 'wb') as file:
        pickle.dump(data, file=file, protocol=pickle.HIGHEST_PROTOCOL)

def loadPickle(filepath: str):
    with open(filepath, 'rb') as file:
        return pickle.load(file)

if __name__ ==  "__main__": 
    set_meson_bench_params(T='realF', N=128, numRHS=32, blkSize=256)
    reconfigure()
    try:
        compile_target(target='stencil_lanes', force_recompile=True)
    except CommandFailedError as cfe:
        print(cfe.stdout)
        print(cfe.stderr)
    res = run_binary(target='stencil_lanes', args=['8', '8', '8', '8', '0', 'true'])
    print(parseBenchOutput(res.stdout.decode('utf-8')))