#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3700

export TQDM_DISABLE=1
module load GCCcore/13.2.0 Python/3.11.5

python -m wandb login $WANDB_API_KEY

srun python -m wandb agent xietao-wang-lin-university-of-warwick/abp-gnn/tfw82obf
