#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3700
#SBATCH --array=4-13

export TQDM_DISABLE=1
module load GCCcore/13.2.0 Python/3.11.5

srun python launcher_input.py $SLURM_ARRAY_TASK_ID
