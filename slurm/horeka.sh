#!/bin/bash

#SBATCH --job-name=horeka-default
#SBATCH --output=horeka-default.out
#SBATCH --time=01:00:00
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --account="hk-project-madonna"

module load compiler/gnu/11
module load devel/cuda/11.8
module load mpi/openmpi/4.1

source /hkfs/work/workspace/scratch/tz6121-paper/paper/venv/bin/activate

cd ..

srun python -u src/main.py