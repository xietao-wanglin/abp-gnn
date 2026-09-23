#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3700
#SBATCH --array=0-80

# Usage: sbatch parallel_inputs.slurm <script.py> [script args...]
#   e.g. sbatch -J chiral parallel_inputs.slurm lattice_ml.py --experiment chiral --model_step 1000000
export TQDM_DISABLE=1
module load GCCcore/13.2.0 Python/3.11.5

srun python "$1" $SLURM_ARRAY_TASK_ID "${@:2}"
