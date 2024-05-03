#!/bin/bash -x
#SBATCH --account=gm2dwf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:1
#SBATCH --output=stdout-%j.out
#SBATCH --error=error-%j.out

module purge

ml Stages/2024
module load GCCcore/.12.3.0
module load Python/3.11.3
ml NVHPC/23.7-CUDA-12
ml Meson/1.1.1

/p/software/juwelsbooster/stages/2024/software/Python/3.11.3-GCCcore-12.3.0/bin/python3 /p/home/jusers/sizmann1/juwels/phd/cuda_grid/benchmark/run.py
