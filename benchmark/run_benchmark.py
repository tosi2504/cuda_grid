import subprocess
import os
import sys

def get_project_root_path():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), '..'))

# TODO: modify meson.build
# open the meson.build file
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


def reconfigure(print_output: bool = False):
    res_string = subprocess.run(args=['meson', 'setup', 'build', '--reconfigure']
                   , cwd=get_project_root_path()
                   , stdout=subprocess.PIPE
                   , stderr=subprocess.STDOUT).stdout.decode('utf-8')
    if print_output:
        print(res_string)


def compile_target(target: str, print_output: bool = False):
    res_string = subprocess.run(args=['ninja', target]
                   , cwd=os.path.join(get_project_root_path(), 'build')
                   , stdout=subprocess.PIPE
                   , stderr=subprocess.STDOUT).stdout.decode('utf-8')
    if print_output:
        print(res_string)


def run_binary(target: str, args: list):
    return  subprocess.run(args=['./'+target]+args
                   , cwd=os.path.join(get_project_root_path(), 'build')
                   , stdout=subprocess.PIPE
                   , stderr=subprocess.STDOUT).stdout.decode('utf-8')
    

if __name__ ==  "__main__": 
    set_meson_bench_params(T='realF', N=64, numRHS=32, blkSize=128)
    reconfigure(print_output=True)
    compile_target(target='mrhs_blas', print_output=True)
    print(run_binary(target='mrhs_blas', args=['8', '8', '8', '8', '0', 'true']))