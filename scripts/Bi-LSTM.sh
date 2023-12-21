#!/bin/sh
#SBATCH --mail-user=mohammed.alruqimi@univr.it
#SBATCH --mail-type=BEGIN,END
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16g
#SBATCH --account=LRQMMM87
#SBATCH --output=%x.o%A_%a
#SBATCH --array 1-10

SEED=$((SLURM_ARRAY_TASK_ID))
echo $SEED
srun python3 Bi_LSTM.py