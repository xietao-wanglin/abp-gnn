#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3700

export TQDM_DISABLE=1
module load GCCcore/13.3.0 Python/3.12.3

srun python ./create_data.py
