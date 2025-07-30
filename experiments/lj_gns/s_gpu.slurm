#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3700
#SBATCH --partition=gpu
#SBATCH --gres=gpu:quadro_rtx_6000:1

export TQDM_DISABLE=1
module load GCCcore/13.2.0 Python/3.11.5

cd $SLURM_SUBMIT_DIR/../../
srun python -m experiments.lj_gns.train