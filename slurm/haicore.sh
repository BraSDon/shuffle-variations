#!/bin/bash

#SBATCH --job-name=haicore-default
#SBATCH --output=haicore-default.out
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:full:4

module load compiler/gnu/11
module load devel/cuda/11.8
module load mpi/openmpi/4.1

source /hkfs/work/workspace_haic/scratch/tz6121-paper/paper/venv/bin/activate

cd ../src/

srun python -u main.py
